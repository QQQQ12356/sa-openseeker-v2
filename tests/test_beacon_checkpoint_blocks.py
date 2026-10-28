"""CPU equivalence of recomputation across changing Beacon window states."""
import copy

import pytest
import torch

from test_beacon_qwen35_linear_memory import _tiny_model
from test_qwen3_whole_information import tiny_qwen3


@pytest.mark.parametrize("factory", [tiny_qwen3, _tiny_model])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("layout", ["whole", "window"])
def test_checkpoint_preserves_loss_gradients_and_memory(factory, offload, layout):
    torch.manual_seed(19)
    reference = factory().train()
    config = copy.deepcopy(reference.beacon_config)
    config.beacon_information_mode = layout
    config.beacon_window_enabled = layout == "window"
    config.beacon_window = config.beacon_stride = 4
    config.beacon_ratio = 2
    config.beacon_keep_window = 3
    config.beacon_attention_backend = "sdpa"
    config.beacon_cpu_offload_activations = offload
    config.beacon_cpu_offload_threshold = 0
    config.beacon_readout_distill_weight = 0
    config.beacon_output_distill_weight = 0
    reference.set_beacon_config(config)
    # Frozen inputs must still permit gradients for trainable projections.
    reference.model.embed_tokens.weight.requires_grad_(False)
    recomputed = copy.deepcopy(reference)
    config = copy.deepcopy(config)
    config.beacon_checkpoint_blocks = True
    recomputed.set_beacon_config(config)
    ids = torch.arange(10, 34).unsqueeze(0)
    labels = ids.clone()
    labels[:, 2:10] = labels[:, 13:19] = -100
    batch = dict(input_ids=ids, labels=labels, regions=[(2, 10), (13, 19)])
    expected = reference(**batch).loss
    actual = recomputed(**batch).loss
    torch.testing.assert_close(expected, actual)
    expected.backward()
    actual.backward()
    for (name, first), (_, second) in zip(reference.named_parameters(), recomputed.named_parameters()):
        assert (first.grad is None) == (second.grad is None), name
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-4, msg=name)
    for first, second in zip(reference._mem._cache, recomputed._mem._cache):
        for left, right in zip(first, second):
            if left is not None:
                torch.testing.assert_close(left, right)
    for idx, state in getattr(reference._mem, "_linear_recurrent", {}).items():
        torch.testing.assert_close(state, recomputed._mem._linear_recurrent[idx])
