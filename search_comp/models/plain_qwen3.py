"""Qwen3 原生纯文本模型加载（无 Beacon）。"""
import torch
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
PLAIN_SFT_ARCH = "Qwen3ForCausalLM"


def load_plain_qwen3(model_name_or_path=DEFAULT_MODEL, torch_dtype=torch.bfloat16,
                     device_map="auto", trust_remote_code=True, attn_implementation=None):
    from transformers import AutoConfig, Qwen3ForCausalLM
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if "BeaconQwen3ForCausalLM" in (config.architectures or []):
        raise ValueError("Beacon checkpoint 请使用 load_beacon_model")
    return Qwen3ForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=trust_remote_code,
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
    ).eval()


def load_sft_model(model_name_or_path=DEFAULT_MODEL, torch_dtype=torch.bfloat16,
                   device_map="auto", trust_remote_code=True, attn_implementation=None):
    """保留统一 SFT API，按模型类型分派到两个独立的架构加载模块。"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if config.model_type == "qwen3":
        loader = load_plain_qwen3
    elif config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from .plain_qwen35 import load_sft_model as loader
    else:
        raise ValueError(f"不支持的纯文本模型类型: {config.model_type}")
    return loader(model_name_or_path, torch_dtype, device_map, trust_remote_code,
                  attn_implementation=attn_implementation)


def load_sft_tokenizer(model_name_or_path=DEFAULT_MODEL):
    from .model_loader import load_tokenizer
    return load_tokenizer(model_name_or_path)


def resolve_plain_evaluation(model_path, base_model_path=None, tokenizer_path=None):
    """Resolve a full checkpoint or a Plain PEFT adapter without loading weights.

    Explicit base > sibling training config > adapter config. A tokenizer saved
    alongside the adapter takes precedence over the base tokenizer.
    """
    path = Path(model_path).expanduser()
    adapter_config = path / "adapter_config.json"
    if not adapter_config.is_file():
        if base_model_path:
            raise ValueError("BASE_MODEL_PATH only applies to a Plain adapter directory")
        return dict(model=str(model_path), base=None, tokenizer=tokenizer_path or str(model_path), adapter=False)
    if (path / "beacon_state.pt").exists():
        raise ValueError("Beacon adapter is not a Plain adapter; merge with 35_beacon_merge_adapter.sh")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"adapter weights missing: {path}")
    with adapter_config.open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("only Plain LoRA adapters are supported")
    resolved_path = path.parent / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
    base = base_model_path or resolved.get("model_name_or_path") or config.get("base_model_name_or_path")
    if not base:
        raise ValueError("adapter base model missing; set BASE_MODEL_PATH to the exact training base")
    # Checkpoint-N usually includes tokenizer files. If it does not, use the
    # base; TOKENIZER_PATH permits an explicitly preserved training tokenizer.
    saved_tokenizer = (path / "tokenizer_config.json").is_file() or (path / "tokenizer.json").is_file()
    tokenizer = tokenizer_path or (str(path) if saved_tokenizer else str(base))
    return dict(model=str(path), base=str(base), tokenizer=str(tokenizer), adapter=True)


def load_plain_evaluation_model(plan, **kwargs):
    """Load the base text model and attach the LoRA adapter without merging."""
    model = load_sft_model(plan["base"] or plan["model"], **kwargs)
    if plan["adapter"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, plan["model"], is_trainable=False)
    return model.eval()
