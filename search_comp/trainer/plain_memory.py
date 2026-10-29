"""Training-only, chunked LM head for native Qwen3/Qwen3.5 causal models.

Install before PEFT/DeepSpeed wrapping so the outer distributed forward still
owns the complete loss graph. No parameters or checkpoint keys are changed.
Unlabelled forwards keep the native implementation (including generation).
"""
from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..loss_segments import LossSegmentConfig, segment_ids_to_weights


def install_plain_memory_forward(model, cfg, loss_segments=None):
    chunk_size = cfg.get("plain_loss_chunk_size", 32)
    threshold = cfg.get("plain_cpu_offload_threshold", 2048)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("plain_loss_chunk_size 必须为正整数")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError("plain_cpu_offload_threshold 必须为非负整数")
    segments = LossSegmentConfig.from_dict(loss_segments)
    native_forward = model.forward
    ds = cfg.get("deepspeed") or {}
    zero3 = isinstance(ds, dict) and ds.get("zero_optimization", {}).get("stage") == 3

    def memory_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                       past_key_values=None, inputs_embeds=None, labels=None,
                       use_cache=None, logits_to_keep=0, loss_segment_ids=None, **kwargs):
        if labels is None:
            return native_forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs,
            )
        num_items = kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_dict", None)
        offload = (self.training and torch.is_grad_enabled()
                   and cfg.get("plain_cpu_offload_activations", True)
                   and labels.shape[1] >= threshold)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        context = torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda") if offload else nullcontext()
        with context:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=False, return_dict=True, **kwargs,
            )
            hidden = outputs.last_hidden_state[:, :-1]
            targets = labels[:, 1:].to(hidden.device)
            if loss_segment_ids is None:
                weights = torch.ones_like(targets, dtype=torch.float32)
            else:
                weights = segment_ids_to_weights(loss_segment_ids[:, 1:], segments).to(hidden.device)
            weights = weights * (targets != -100)
            active = weights > 0
            selected_hidden = hidden[active]
            targets, weights = targets[active], weights[active]
            # Keep a differentiable zero for an entirely masked microbatch.
            loss = hidden.sum() * 0.0
            chunk_count = (targets.numel() + chunk_size - 1) // chunk_size
            if zero3 and torch.distributed.is_initialized():
                # ZeRO-3 gathers head weights at every call. All ranks must call
                # it equally often even with different supervision lengths.
                count = torch.tensor(chunk_count, device=hidden.device, dtype=torch.long)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX)
                chunk_count = int(count.item())
            for index in range(chunk_count):
                start = index * chunk_size
                def chunk_loss(states, current_targets, current_weights):
                    logits = self.lm_head(states).float()
                    token_loss = F.cross_entropy(logits, current_targets, reduction="none")
                    return (token_loss * current_weights).sum()

                args = (selected_hidden[start:start + chunk_size],
                        targets[start:start + chunk_size], weights[start:start + chunk_size])
                if args[0].shape[0] == 0:
                    args = (hidden.reshape(-1, hidden.shape[-1])[:1],
                            labels.new_zeros(1).to(hidden.device), weights.new_zeros(1))
                if self.training and torch.is_grad_enabled() and cfg.get("plain_checkpoint_loss", True):
                    loss = loss + checkpoint(chunk_loss, *args, use_reentrant=False)
                else:
                    loss = loss + chunk_loss(*args)
            # Preserve Trainer's token normalization across accumulation for
            # ordinary CE; custom segment weights retain their existing local
            # weighted-mean normalization.
            denominator = weights.sum()
            if segments.is_default and num_items is not None:
                denominator = torch.as_tensor(num_items, device=hidden.device)
            loss = loss / denominator.clamp(min=1e-6)
        # Training/evaluation only needs loss; never retain full sequence logits.
        return CausalLMOutputWithPast(loss=loss)

    model.forward = MethodType(memory_forward, model)


class PlainMemoryTrainer(Trainer):
    """Retain segment IDs and the original Trainer normalization semantics."""

    def __init__(self, *args, loss_segments=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = LossSegmentConfig.from_dict(loss_segments).is_default
