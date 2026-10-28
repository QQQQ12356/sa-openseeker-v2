"""Random ratios affect real layouts/losses while fixed/evaluation remain deterministic."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from search_comp.models.beacon_config import BeaconConfig
from search_comp.trainer.loss_component_trainer import LossComponentTrainer
from test_beacon_qwen35_linear_memory import _tiny_model
from test_qwen3_whole_information import tiny_qwen3


def config(**kwargs):
    return BeaconConfig(beacon_information_mode='whole', beacon_window_enabled=False,
                        beacon_keep_window=0, beacon_ratio=4,
                        beacon_compression_mode='random', beacon_compression_ratios=[2, 4],
                        beacon_readout_distill_weight=0, **kwargs)


def inputs():
    ids = torch.arange(10, 28).unsqueeze(0)
    labels = ids.clone()
    labels[:, 2:8] = labels[:, 11:15] = -100
    return dict(input_ids=ids, labels=labels, regions=[(2, 8), (11, 15)])


@pytest.mark.parametrize('overrides', [
    {'beacon_compression_mode': 'invalid'},
    {'beacon_compression_ratios': []},
    {'beacon_compression_ratios': [0, 4]},
    {'beacon_compression_ratios': [-1]},
    {'beacon_compression_ratios': [True, 4]},
    {'beacon_compression_ratios': [2.0, 4]},
    {'beacon_compression_ratios': '2,4'},
    {'beacon_compression_ratios': [2, 2]},
])
def test_invalid_compression_configuration_fails_before_training(overrides):
    values = dict(beacon_information_mode='whole', beacon_compression_mode='random')
    values.update(overrides)
    with pytest.raises(ValueError, match='beacon_compression_'):
        BeaconConfig(**values)


def test_window_mode_validates_every_random_candidate():
    with pytest.raises(AssertionError, match='整除'):
        BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2,
                     beacon_compression_mode='random', beacon_compression_ratios=[2, 3])
    with pytest.raises(ValueError, match='intersect'):
        BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=4,
                     beacon_continuation_per_beacon=True, beacon_pos='append',
                     beacon_compression_mode='random', beacon_compression_ratios=[2, 4])


def test_seeded_uniform_sampling_and_config_round_trip():
    value = config()
    def sequence():
        return [value.sample_compression_ratio(training=True) for _ in range(32)]
    torch.manual_seed(12); first = sequence()
    torch.manual_seed(12); assert sequence() == first
    assert set(first) == {2, 4}
    assert value.beacon_ratio == 4
    restored = BeaconConfig.from_dict(json.loads(json.dumps(value.to_dict())))
    assert restored == value
    hf_config = SimpleNamespace()
    value.merge_into_config(hf_config)
    assert BeaconConfig.from_model_config(hf_config) == value
    legacy = BeaconConfig.from_model_config(SimpleNamespace(beacon_ratio=16))
    assert legacy.beacon_compression_mode == 'fixed'
    assert legacy.sample_compression_ratio(training=True) == 16


@pytest.mark.parametrize('factory', [tiny_qwen3, _tiny_model])
@pytest.mark.parametrize('position', ['append', 'intersect'])
@pytest.mark.parametrize('ratio', [2, 4])
def test_sampled_ratio_matches_fixed_forward_cache_and_gradients(factory, position, ratio):
    random_model = factory().train()
    value = config(beacon_pos=position, beacon_linear_writer_rank=8)
    random_model.set_beacon_config(value)
    fixed_model = copy.deepcopy(random_model)
    fixed_model.set_beacon_config(replace(value, beacon_compression_mode='fixed', beacon_ratio=ratio))
    batch = inputs()
    with patch('torch.randint', return_value=torch.tensor([2, 4].index(ratio))) as sample:
        random_output = random_model(**batch)
    sample.assert_called_once()  # 两个 observation 共用一个采样值
    fixed_output = fixed_model(**batch)
    torch.testing.assert_close(random_output.loss, fixed_output.loss)
    random_output.loss.backward(); fixed_output.loss.backward()
    for (_, first), (_, second) in zip(random_model.named_parameters(), fixed_model.named_parameters()):
        if first.grad is None:
            assert second.grad is None
        else:
            torch.testing.assert_close(first.grad, second.grad)
    expected = 8 + (6 + ratio - 1) // ratio + (4 + ratio - 1) // ratio
    for kind, (key, _) in zip(random_model._mem.layer_types, random_model._mem._cache):
        if kind == 'full_attention':
            assert key.shape[2] == expected
    assert random_model._last_compression_ratio == ratio
    assert random_model.beacon_config.beacon_ratio == random_model.config.beacon_ratio == 4
    assert set(random_model._last_loss_parts) == {'ce_loss', 'cont_loss', 'readout_loss', 'kd_loss'}


@pytest.mark.parametrize('factory', [tiny_qwen3, _tiny_model])
def test_eval_no_labels_and_empty_regions_never_sample(factory):
    model = factory(); model.set_beacon_config(config(beacon_linear_writer_rank=8))
    batch = inputs()
    with patch('torch.randint', side_effect=AssertionError('must not sample')):
        model.eval(); model(**batch)  # 验证有 labels 也必须固定
        assert model._mem.compression_ratio == 4
        model.train(); model(input_ids=batch['input_ids'], regions=batch['regions'])
        assert model._mem.compression_ratio == 4
        model(**{**batch, 'regions': []})
        assert model._mem.compression_ratio == 4
        model.prefill_and_get_cache(batch['input_ids'][:, :10], [(2, 8)])
        model.prefill_and_get_cache(batch['input_ids'][:, 10:], [(1, 5)], reuse_cache=True)
        assert model._mem.compression_ratio == 4


def test_fixed_sampling_does_not_consume_rng():
    value = replace(config(), beacon_compression_mode='fixed')
    state = torch.get_rng_state().clone()
    assert value.sample_compression_ratio(training=True) == 4
    assert torch.equal(torch.get_rng_state(), state)


def test_random_ratio_log_is_a_microbatch_mean_and_excludes_eval(tmp_path):
    from transformers import TrainingArguments
    from test_loss_component_logging import _lora_wrapped
    base = _tiny_model(); base.set_beacon_config(config(beacon_linear_writer_rank=8))
    model = _lora_wrapped(base).train()
    trainer = LossComponentTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True,
        remove_unused_columns=False))
    with patch('torch.randint', side_effect=[torch.tensor(0), torch.tensor(1)]):
        trainer.compute_loss(model, inputs())
        trainer.compute_loss(model, inputs())
    model.eval(); trainer.compute_loss(model, inputs())
    logs = {'loss': 1.0}; trainer.log(logs)
    assert logs['compression_ratio_mean'] == 3.0
    assert trainer._loss_part_count == 0
