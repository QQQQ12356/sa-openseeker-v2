#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'REPO_ID': {'env': 'SA_REPO_ID', 'type': 'str'},
 'REVISION': {'env': 'SA_REVISION', 'type': 'str'},
 'MODEL_NAME': {'env': 'SA_MODEL_NAME', 'type': 'str'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.data import download_model
    download_model(repo_id=REPO_ID, revision=REVISION, model_name=MODEL_NAME, environment=ENVIRONMENT)
