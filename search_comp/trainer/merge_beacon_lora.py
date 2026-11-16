"""将一个已保存的 Beacon LoRA adapter 合并为独立推理 checkpoint。"""

from __future__ import annotations

import argparse
import json

from ..models.model_loader import load_beacon_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 Beacon LoRA adapter")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--beacon_config_json", required=True)
    parser.add_argument(
        "--device-map",
        choices=("cpu", "auto"),
        default="cpu",
        help="合并设备映射；默认 CPU，避免训练进程结束后再次占满 GPU 显存。",
    )
    args = parser.parse_args()

    from ..milestones.qwen35_text import load_text_tokenizer

    model = load_beacon_adapter(args.base_model_path, args.adapter_path,
                                json.loads(args.beacon_config_json), device_map=args.device_map)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_path)
    load_text_tokenizer(args.adapter_path).save_pretrained(args.output_path)
    print(f"[beacon] 已合并 final checkpoint -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
