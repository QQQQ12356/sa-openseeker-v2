"""Chunked native SFT must preserve full/weighted CE and all gradients."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from search_comp.loss_segments import LossSegmentConfig, SEGMENT_IDS
from search_comp.trainer.plain_memory import install_plain_memory_forward, PlainMemoryTrainer
from search_comp.trainer.segmented_sft_trainer import segmented_causal_loss


def tiny_model(family):
    kwargs = dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                  tie_word_embeddings=False)
    if family == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**kwargs))
    config = Qwen3_5TextConfig(**kwargs, layer_types=["linear_attention", "full_attention"],
                              linear_num_key_heads=2, linear_num_value_heads=2,
                              linear_key_head_dim=8, linear_value_head_dim=8,
                              linear_conv_kernel_dim=4)
    return Qwen3_5ForCausalLM(config)


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_loss_gradients_and_head_chunk_bound(family, offload, weighted):
    torch.manual_seed(19)
    original = tiny_model(family).train()
    chunked = copy.deepcopy(original)
    segments = LossSegmentConfig.from_dict({"answer_content": 3.0} if weighted else {})
    install_plain_memory_forward(chunked, dict(plain_loss_chunk_size=3,
        plain_cpu_offload_activations=offload, plain_cpu_offload_threshold=0), segments)
    # Exercise decoder checkpointing with frozen embeddings, as in LoRA training.
    for model in (original, chunked):
        model.model.embed_tokens.weight.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.arange(2, 14).unsqueeze(0)
    labels = ids.clone()
    labels[:, :4] = -100
    segment_ids = torch.full_like(ids, SEGMENT_IDS["other"])
    segment_ids[:, 7:] = SEGMENT_IDS["answer_content"]
    sizes = []
    hook = chunked.lm_head.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[0]))
    if weighted:
        expected = segmented_causal_loss(original(ids, use_cache=False).logits, labels, segment_ids, segments)
    else:
        expected = original(ids, labels=labels, use_cache=False).loss
    actual = chunked(ids, labels=labels, **({"loss_segment_ids": segment_ids} if weighted else {}))
    assert actual.logits is None
    torch.testing.assert_close(actual.loss, expected)
    actual.loss.backward()
    expected.backward()
    hook.remove()
    assert sizes and max(sizes) <= 3
    for (name, p), (_, q) in zip(original.named_parameters(), chunked.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
    # Inference still returns the native logits.
    original.eval()
    chunked.eval()
    torch.testing.assert_close(original(ids).logits, chunked(ids).logits)


@pytest.mark.parametrize("masked", [False, True])
def test_trainer_lora_accumulation_eval_and_save(tmp_path, masked):
    from peft import LoraConfig, get_peft_model, PeftModel
    model = tiny_model("qwen3")
    install_plain_memory_forward(model, {"plain_loss_chunk_size": 2, "plain_cpu_offload_threshold": 0})
    model = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = torch.arange(2, 12)
    labels = ids.clone() if not masked else torch.full_like(ids, -100)
    data = [{"input_ids": ids, "labels": labels}] * 4
    trainer = PlainMemoryTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), per_device_train_batch_size=1,
        gradient_accumulation_steps=2, max_steps=1, report_to=[],
        use_cpu=True, save_strategy="no", prediction_loss_only=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
        train_dataset=data, eval_dataset=data)
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))
    trainer.save_model(str(tmp_path / "adapter"))
    restored = PeftModel.from_pretrained(tiny_model("qwen3"), tmp_path / "adapter")
    assert restored(ids.unsqueeze(0)).logits.shape == (1, 10, 64)


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_saved_native_model_loads_sdpa(family, tmp_path):
    from search_comp.models.plain_qwen3 import load_sft_model
    original = tiny_model(family)
    original.save_pretrained(tmp_path)
    restored = load_sft_model(tmp_path, torch_dtype=torch.float32, device_map="cpu",
                              attn_implementation="sdpa")
    assert restored.config._attn_implementation == "sdpa"
    install_plain_memory_forward(restored, {})
    ids = torch.arange(2, 12).unsqueeze(0)
    assert torch.isfinite(restored(ids, labels=ids).loss)


def test_two_process_plain_training(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    pytest.importorskip("peft")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(root / "tests/helpers/distributed_training_worker.py"), str(tmp_path), "plain",
    ], cwd=root, env={**os.environ, "OMP_NUM_THREADS": "1",
                         "ACCELERATE_USE_CPU": "true", "CUDA_VISIBLE_DEVICES": ""}, check=True, timeout=120)
    assert (tmp_path / "checkpoint-2/adapter_config.json").is_file()


def test_zero3_padding_keeps_collective_head_call_count(monkeypatch):
    model = tiny_model("qwen3").train()
    cfg = {"plain_loss_chunk_size": 3, "deepspeed": {"zero_optimization": {"stage": 3}}}
    install_plain_memory_forward(model, cfg)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda count, op: count.fill_(4))
    calls = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    ids = torch.arange(2, 8).unsqueeze(0)
    labels = torch.full_like(ids, -100)
    labels[:, -1] = ids[:, -1]
    loss = model(ids, labels=labels).loss
    assert calls == [1, 1, 1, 1]
    handle.remove()
    loss.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize("checkpoint_loss", [False, True])
def test_all_segments_disabled_yields_differentiable_zero(checkpoint_loss):
    from search_comp.loss_segments import SEGMENT_NAMES
    model = tiny_model("qwen3").train()
    segments = {name: False for name in SEGMENT_NAMES}
    install_plain_memory_forward(model, {"plain_checkpoint_loss": checkpoint_loss}, segments)
    ids = torch.arange(2, 8).unsqueeze(0)
    loss = model(ids, labels=ids, loss_segment_ids=torch.full_like(ids, SEGMENT_IDS["other"])).loss
    assert loss.item() == 0
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None


@pytest.mark.parametrize("weighted", [False, True])
def test_accumulated_optimizer_update_matches_existing_trainer(tmp_path, weighted):
    from transformers import Trainer
    from search_comp.trainer.segmented_sft_trainer import SegmentedSFTTrainer
    torch.manual_seed(3)
    original = tiny_model("qwen3")
    optimized = copy.deepcopy(original)
    segments = LossSegmentConfig.from_dict({"answer_content": 3.0} if weighted else {})
    install_plain_memory_forward(optimized, {"plain_loss_chunk_size": 3}, segments)
    rows = []
    for length in (7, 12):
        ids = torch.arange(2, length + 2)
        row = {"input_ids": ids, "labels": ids.clone()}
        if weighted:
            row["loss_segment_ids"] = torch.full_like(ids, SEGMENT_IDS["answer_content"])
        rows.append(row)
    for index, model in enumerate((original, optimized)):
        cls = (SegmentedSFTTrainer if weighted else Trainer) if index == 0 else PlainMemoryTrainer
        trainer = cls(model=model, args=TrainingArguments(
            output_dir=str(tmp_path / str(index)), per_device_train_batch_size=1,
            gradient_accumulation_steps=2, max_steps=1, report_to=[], use_cpu=True,
            save_strategy="no", learning_rate=0.01, max_grad_norm=0, optim="sgd",
            remove_unused_columns=False),
            train_dataset=rows,
            **({"loss_segments": segments} if weighted or index == 1 else {}))
        trainer.train()
    for (name, p), (_, q) in zip(original.named_parameters(), optimized.named_parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5, msg=name)
