"""Repetition penalty follows ordinary tokens surviving Beacon compression."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from search_comp.models.beacon_common import BeaconCausalLMMixin, apply_visible_repetition_penalty
from test_beacon_qwen35_linear_memory import _tiny_model
from test_qwen3_whole_information import tiny_qwen3


@pytest.mark.parametrize('layout', ['append', 'intersect'])
def test_visible_ids_track_compression_decode_reuse_and_reset(layout):
    model = _tiny_model(layout)
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30]])
    mem = model.prefill_and_get_cache(ids, regions=[(2, 6)])
    assert mem._visible_token_ids == {10, 11, 30}
    model.decode_step(torch.tensor([[31]]))
    assert mem._visible_token_ids == {10, 11, 30, 31}
    mem = model.prefill_and_get_cache(torch.tensor([[40, 41, 42]]), [(1, 2)], reuse_cache=True)
    assert mem._visible_token_ids == {10, 11, 30, 31, 40, 42}
    mem = model.prefill_and_get_cache(torch.tensor([[50]]), [])
    assert mem._visible_token_ids == {50}


def test_qwen3_visible_ids_and_default_penalty_are_unchanged():
    model = tiny_qwen3()
    ids = torch.tensor([[10, 20, 21, 30]])
    mem = model.prefill_and_get_cache(ids, regions=[(1, 3)])
    assert mem._visible_token_ids == {10, 30}
    default = model.beacon_generate(ids, regions=[(1, 3)], max_new_tokens=3)
    explicit = model.beacon_generate(ids, regions=[(1, 3)], max_new_tokens=3, repetition_penalty=1.0)
    assert torch.equal(default, explicit)


def test_penalty_uses_visible_not_compressed_tokens_and_includes_generated():
    memory = SimpleNamespace(_visible_token_ids={1})
    class Fake:
        config = SimpleNamespace(eos_token_id=9)
        def prefill_and_get_cache(self, *args, **kwargs):
            memory._visible_token_ids = {1}  # token 2 belongs only to compressed observation
            return memory, torch.tensor([[0., 3., 2.5, 0., 0., 0., 0., 0., 0., 0.]])
        def decode_step(self, token):
            memory._visible_token_ids.add(token.item())
            return torch.tensor([[0., 0., 3., 2.5, 0., 0., 0., 0., 0., 0.]])
    args = (torch.tensor([[1, 2]]), [(1, 2)], 2, False, 1.0, 1.0, [9], None, None)
    baseline = BeaconCausalLMMixin._beacon_generate_loop(Fake(), *args, repetition_penalty=1.0)
    penalized = BeaconCausalLMMixin._beacon_generate_loop(Fake(), *args, repetition_penalty=2.0)
    assert baseline.tolist() == [[1, 2]]
    assert penalized.tolist() == [[2, 3]]


def test_visible_penalty_matches_hf_for_positive_negative_and_duplicate_ids():
    from transformers import RepetitionPenaltyLogitsProcessor
    logits = torch.tensor([[0.0, -3.0, 4.0, -2.0, 5.0]])
    ids = torch.tensor([[1, 2, 1]])
    expected = RepetitionPenaltyLogitsProcessor(1.5)(ids, logits.clone())
    actual = apply_visible_repetition_penalty(logits, {1, 2}, 1.5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, torch.tensor([[0.0, -3.0, 4.0, -2.0, 5.0]]))


def test_model_generator_passes_beacon_penalty_and_records_scope():
    from search_comp.evaluation.tool_agent import ModelGenerator
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(to_dict=lambda: {}, eos_token_id=2)
            self.beacon_config = SimpleNamespace(to_dict=lambda: {})
            self.beacon_generate = Mock(return_value=torch.tensor([[2]]))
    model = Model()
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: '<eos>')
    generator = ModelGenerator(model, tokenizer, beacon=True, repetition_penalty=1.1)
    generator({'input_ids': [5, 6], 'regions': [(1, 2)]}, 5)
    assert model.beacon_generate.call_args.kwargs['repetition_penalty'] == 1.1
    assert generator.metadata['repetition_scope'] == 'beacon_visible_tokens'
