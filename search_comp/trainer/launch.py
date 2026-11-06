"""Build a single-node trainer command from the script-owned distributed configuration."""
from __future__ import annotations

import json
import os
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
            raise ValueError("多卡 ZeRO 需要训练脚本中的 DISTRIBUTED 配置")
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
