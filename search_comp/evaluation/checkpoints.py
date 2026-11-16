"""Resolve and load the same model artifacts for online and offline benchmarks."""
import json
from pathlib import Path

BEACON_ARCHITECTURES = {"BeaconQwen3ForCausalLM", "BeaconQwen3_5ForCausalLM"}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_checkpoint(path, kind):
    if not (path / "adapter_config.json").is_file():
        return False
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"LoRA checkpoint 缺少 adapter 权重: {path}")
    beacon_state = (path / "beacon_state.pt").is_file()
    if kind == "beacon" and not beacon_state:
        raise ValueError(f"Beacon LoRA checkpoint 缺少 beacon_state.pt: {path}")
    if kind == "plain" and beacon_state:
        raise ValueError(f"Beacon LoRA checkpoint 必须由 Beacon 入口测试: {path}")
    return True


def _has_model_weights(path):
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file():
            return True
        index = path / f"{name}.index.json"
        if index.is_file():
            shards = set(_read_json(index)["weight_map"].values())
            if shards and all((path / shard).is_file() for shard in shards):
                return True
    return False


def resolve_checkpoint(run_dir, model, kind, *, require_weights=False):
    if run_dir:
        run_dir = Path(run_dir)
        spec = _read_json(run_dir / "experiment.json")
        kind = kind or spec.get("kind") or spec.get("method")
        if not model:
            model = _read_json(run_dir / "models/model/train_summary.json")["final_model_path"]
    if not model:
        raise ValueError("set --run-dir or --model for checkpoint inference")
    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    config_path = path / "config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    beacon_architecture = BEACON_ARCHITECTURES.intersection(config.get("architectures") or ())
    text_config = config.get("text_config") or {}
    is_beacon = (beacon_architecture or config.get("enable_beacon") or text_config.get("enable_beacon")
                 or (path / "beacon_state.pt").is_file())
    kind = kind or ("beacon" if is_beacon else "plain")
    if kind not in {"plain", "beacon"} or (is_beacon and kind != "beacon"):
        raise ValueError(f"checkpoint kind 不匹配: {kind} ({path})")
    is_adapter = _adapter_checkpoint(path, kind)
    if require_weights and not is_adapter:
        if not config_path.is_file() or not _has_model_weights(path):
            raise ValueError(f"checkpoint 缺少模型配置或完整权重（包括分片）: {path}")
        if kind == "beacon" and not beacon_architecture:
            raise ValueError(f"Beacon 全量 checkpoint 缺少已训练的 Beacon 架构配置: {path}")
    return str(path), kind


def _training_config(run_dir):
    if run_dir is None:
        return {}
    run_dir = Path(run_dir)
    resolved = run_dir / "models/model/resolved_config.json"
    if resolved.is_file():
        return _read_json(resolved)
    return _read_json(run_dir / "experiment.json").get("training_config", {})


def load_evaluation_model(model_path, kind, origin, run_dir):
    """Load full weights or attach adapters to the original training base."""
    from search_comp.models.model_loader import load_beacon_model, load_tokenizer

    tokenizer_path = model_path
    if origin == "base-qwen35":
        if kind != "plain":
            raise ValueError("Qwen3.5 baseline requires kind=plain")
        from search_comp.models.base_qwen35 import load_base_qwen35
        model = load_base_qwen35(model_path)
    elif (Path(model_path) / "adapter_config.json").is_file():
        config = _training_config(run_dir)
        base = config.get("model_name_or_path")
        if kind == "plain":
            from search_comp.models.plain_qwen3 import load_plain_evaluation_model, resolve_plain_evaluation
            plan = resolve_plain_evaluation(model_path, base_model_path=base)
            model = load_plain_evaluation_model(plan)
            tokenizer_path = plan["tokenizer"]
        else:
            from search_comp.models.model_loader import load_beacon_adapter
            if not base or "beacon" not in config:
                raise ValueError("Beacon LoRA checkpoint 需要训练实验的基座路径和 Beacon 配置")
            model = load_beacon_adapter(base, model_path, config["beacon"], device_map="auto")
            checkpoint = Path(model_path)
            if not any((checkpoint / name).is_file() for name in ("tokenizer_config.json", "tokenizer.json")):
                tokenizer_path = base
    elif kind == "beacon":
        model = load_beacon_model(model_path)
    else:
        from search_comp.models.plain_qwen3 import load_sft_model
        model = load_sft_model(model_path)
    model.eval()
    return model, load_tokenizer(tokenizer_path)
