#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import os
import platform
import subprocess
import sys
import venv
ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT_DIR': {'env': 'SA_ENVIRONMENT_DIR', 'type': 'path'},
 'TORCH_SPEC': {'env': 'SA_TORCH_SPEC', 'type': 'str'},
 'TORCH_INDEX': {'env': 'SA_TORCH_INDEX', 'type': 'str'},
 'PIP_INDEX': {'env': 'SA_PIP_INDEX', 'type': 'str'},
 'HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'INSTALL_DEEPSPEED': {'env': 'SA_INSTALL_DEEPSPEED', 'type': 'bool'},
 'DEEPSPEED_SPEC': {'env': 'SA_DEEPSPEED_SPEC', 'type': 'str'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("请使用 Linux 的 Python 3.11 运行此安装脚本")
    os.environ.update(HF_ENDPOINT=HF_ENDPOINT, PIP_INDEX_URL=PIP_INDEX)
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts/cache"))
    venv.EnvBuilder(with_pip=True).create(ENVIRONMENT_DIR)
    python = str(ENVIRONMENT_DIR / "bin/python")
    subprocess.run([python, "-m", "pip", "install", TORCH_SPEC, "--index-url", TORCH_INDEX], check=True)
    subprocess.run([python, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    if INSTALL_DEEPSPEED:
        os.environ["DS_BUILD_OPS"] = "0"
        subprocess.run([python, "-m", "pip", "install", DEEPSPEED_SPEC, "--no-build-isolation"], check=True)
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    (ENVIRONMENT_DIR / "installed_requirements.txt").write_text(freeze, encoding="utf-8")
    print(f"环境已安装。后续脚本使用此解释器：{python}")
