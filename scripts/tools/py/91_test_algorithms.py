#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import os
import platform
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'TEST_PATH': {'env': 'SA_TEST_PATH', 'type': 'str'},
 'TOKENIZER_PATH': {'env': 'SA_TOKENIZER_PATH', 'type': 'str'},
 'VERBOSE': {'env': 'SA_VERBOSE', 'type': 'bool'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    if platform.system() != "Linux":
        raise RuntimeError("完整算法回归请在安装训练依赖的 Linux 环境运行")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", BEACON_TEST_DEVICE="cpu", TEST_TOKENIZER_PATH=TOKENIZER_PATH)
    subprocess.run([sys.executable, "-m", "pytest", TEST_PATH, "-v" if VERBOSE else "-q"], cwd=ROOT, check=True)
