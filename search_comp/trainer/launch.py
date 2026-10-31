"""Single-node training launcher shared by shell scripts and experiment.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys


def training_command(module, arguments, python=sys.executable, deepspeed_configs=None, configured_deepspeed=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    default_count = len([value for value in visible.split(",") if value.strip() and value != "-1"])
    count = int(os.environ.get("NPROC_PER_NODE", str(max(1, default_count))))
    if count < 1:
        raise ValueError("NPROC_PER_NODE 必须大于 0")
    if count == 1:
        return [python, "-u", "-m", module, *arguments]
    backend = os.environ.get("DISTRIBUTED_BACKEND", "zero2")
    if backend not in {"ddp", "zero2", "zero3"}:
        raise ValueError("DISTRIBUTED_BACKEND 必须为 ddp / zero2 / zero3")
    if backend == "zero3" and module.endswith("beacon_trainer"):
        raise ValueError("Beacon 动态分段训练请使用 zero2 或 ddp；zero3 仅支持 plain")
    if backend == 'ddp' and configured_deepspeed:
        raise ValueError('DISTRIBUTED_BACKEND=ddp 与 deepspeed 配置冲突；请将 deepspeed 设为 null')
    overrides = []
    if backend != "ddp":
        if deepspeed_configs is None:
            # Compatibility for legacy Search-R1 / external configuration callers.
            config = Path(__file__).resolve().parents[2] / "configs" / "train" / f"ds_{backend}.json"
            value = str(config)
        else:
            if not isinstance(deepspeed_configs, dict) or not isinstance(deepspeed_configs.get(backend), dict):
                raise ValueError(f"脚本 distributed 必须提供 {backend} DeepSpeed 配置")
            config = deepspeed_configs[backend]
            expected = 2 if backend == 'zero2' else 3
            zero = config.get('zero_optimization')
            if not isinstance(zero, dict) or zero.get('stage') != expected:
                raise ValueError(f"distributed.{backend}.zero_optimization.stage 必须为 {expected}")
            value = json.dumps(config, ensure_ascii=False, separators=(',', ':'))
        overrides = ["--set", f"deepspeed={value}"]
    return [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={count}", "--module", module, *arguments, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print resolved configuration and command only")
    args, arguments = parser.parse_known_args()
    distributed = None
    configured_deepspeed = None
    if '--config' in arguments:
        index = arguments.index('--config')
        if index + 1 < len(arguments) and Path(arguments[index + 1]).is_file():
            import yaml
            from ..utils.runtime import load_yaml_config
            overrides = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == '--set']
            config = load_yaml_config(arguments[index + 1], overrides)
            if isinstance(config, dict):
                distributed = config.get('distributed')
                configured_deepspeed = config.get('deepspeed')
                if config.get('self_contained_training') and not isinstance(distributed, dict):
                    raise ValueError('自包含配置必须提供 distributed 映射；不会回退到外部 DeepSpeed 文件')
    command = training_command(args.module, arguments, deepspeed_configs=distributed,
                               configured_deepspeed=configured_deepspeed)
    print("[train-launch]", shlex.join(command), flush=True)
    if args.dry_run:
        if '--config' in arguments:
            from ..utils.runtime import load_yaml_config
            index = arguments.index('--config')
            overrides = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == '--set']
            resolved = load_yaml_config(arguments[index + 1], overrides)
            if len(command) >= 2 and command[-2] == '--set' and command[-1].startswith('deepspeed='):
                resolved['deepspeed'] = yaml.safe_load(command[-1].split('=', 1)[1])
            if args.module.endswith('beacon_trainer'):
                from ..models.beacon_config import configure_sft_layout
                configure_sft_layout(resolved)
            if resolved.get('deepspeed'):
                # setup_training makes DeepSpeed's configured optimizer authoritative.
                resolved['optim'] = 'adamw_torch'
            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), flush=True)
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
