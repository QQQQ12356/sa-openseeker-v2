"""把原生 SFT 的 LoRA adapter 合并为独立推理 checkpoint。

原生训练（``use_lora: true``）保存的 ``final/`` 里只有 adapter
（``adapter_model.safetensors`` + ``adapter_config.json``），而评测入口
``search_comp.evaluation.native_interactive_eval`` 经
:func:`search_comp.milestones.qwen35_text.load_text_causal_model` 加载完整模型，
不认识 adapter，因此评测前需先合并。

比 Beacon 版（:mod:`search_comp.trainer.merge_beacon_lora`）简单：原生模型没有
额外的 Beacon 压缩参数，单独一个 ``merge_and_unload`` 即可。

用法::

    python -m search_comp.trainer.merge_lora \
        --base_model_path Qwen/Qwen3.5-2B \
        --adapter_path outputs/models/native_qwen3_searchr1_v1/final \
        --output_path outputs/models/native_qwen3_searchr1_v1/final_merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_paths(args):
    """Resolve Plain artifacts before loading any model weights."""
    root = Path(args.exp_dir) if args.exp_dir else None
    adapter = Path(args.adapter_path) if args.adapter_path else root / "final" if root else None
    if adapter is None or not (adapter / "adapter_config.json").is_file():
        raise ValueError("请提供完整 Plain LoRA adapter：--adapter_path 或 --exp_dir（其 final/）；全参数模型无需合并")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"adapter 权重缺失: {adapter}")
    if (adapter / "beacon_state.pt").exists():
        raise ValueError("Beacon adapter 请使用 scripts/sh/beacon/35_beacon_merge_adapter.sh")
    config_path = (root or adapter.parent) / "resolved_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    base = args.base_model_path or config.get("model_name_or_path")
    if not base:
        raise ValueError("缺少 resolved_config.json 中的基座身份；请显式设置 --base_model_path")
    output = Path(args.output_path) if args.output_path else adapter.with_name(adapter.name + "_merged")
    if output.resolve() == adapter.resolve() or adapter.resolve() in output.resolve().parents or output.resolve() in adapter.resolve().parents:
        raise ValueError("输出与 adapter 目录不能相同或互相包含")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出路径已存在且非空，请使用新目录: {output}")
    return str(base), str(adapter), str(output)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="合并原生 SFT 的 LoRA adapter")
    parser.add_argument("--exp_dir", help="含 resolved_config.json 和 final/ 的模型产物目录")
    parser.add_argument("--base_model_path", help="基础模型或 HF ID；默认读取 resolved_config.json")
    parser.add_argument("--adapter_path", help="训练保存的 adapter；默认 <exp_dir>/final")
    parser.add_argument("--output_path", help="合并后目录；默认 <adapter>_merged，不覆盖非空目录")
    parser.add_argument(
        "--device-map",
        choices=("cpu", "auto"),
        default="cpu",
        help="合并设备映射；默认 CPU，避免训练进程结束后再次占满 GPU 显存。",
    )
    args = parser.parse_args(argv)
    try:
        base, adapter, output = resolve_paths(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    from peft import PeftModel

    from ..milestones.qwen35_text import load_text_causal_model, load_text_tokenizer

    model = load_text_causal_model(base, device_map=args.device_map)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(output)
    load_text_tokenizer(adapter).save_pretrained(output)
    print(f"[plain] 已合并 LoRA adapter -> {output}", flush=True)


if __name__ == "__main__":
    main()
