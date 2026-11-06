#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'PREPARED': {'env': 'SA_PREPARED', 'type': 'str'},
 'PROFILE': {'env': 'SA_PROFILE', 'type': 'str'},
 'TOKENIZER': {'env': 'SA_TOKENIZER', 'type': 'str'},
 'MIN_LENGTH': {'env': 'SA_MIN_LENGTH', 'type': 'int'},
 'MAX_LENGTH': {'env': 'SA_MAX_LENGTH', 'type': 'int'},
 'EXACT_LENGTH': {'env': 'SA_EXACT_LENGTH', 'type': 'optional'},
 'CORRECTNESS': {'env': 'SA_CORRECTNESS', 'type': 'str'},
 'OBSERVATIONS': {'env': 'SA_OBSERVATIONS', 'type': 'str'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.data import select_dataset
    select_dataset(dataset_name="openseeker", prepared=PREPARED, profile=PROFILE, tokenizer=TOKENIZER,
                   min_length=MIN_LENGTH, max_length=MAX_LENGTH, exact_length=EXACT_LENGTH,
                   correctness=CORRECTNESS, observations=OBSERVATIONS, environment=ENVIRONMENT)
