#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'SOURCE': {'env': 'SA_SOURCE', 'type': 'str'},
 'SNAPSHOT_DIR': {'env': 'SA_SNAPSHOT_DIR', 'type': 'str'},
 'INPUTS': {'env': 'SA_INPUTS', 'type': 'list'},
 'PROFILE': {'env': 'SA_PROFILE', 'type': 'str'},
 'MODELS': {'env': 'SA_MODELS', 'type': 'list'},
 'MIN_LENGTH': {'env': 'SA_MIN_LENGTH', 'type': 'int'},
 'MAX_LENGTH': {'env': 'SA_MAX_LENGTH', 'type': 'int'},
 'CORRECTNESS': {'env': 'SA_CORRECTNESS', 'type': 'str'},
 'OBSERVATIONS': {'env': 'SA_OBSERVATIONS', 'type': 'str'},
 'WORKERS': {'env': 'SA_WORKERS', 'type': 'int'},
 'SEED': {'env': 'SA_SEED', 'type': 'int'},
 'VAL_FRACTION': {'env': 'SA_VAL_FRACTION', 'type': 'float'},
 'TEST_FRACTION': {'env': 'SA_TEST_FRACTION', 'type': 'float'},
 'OFFLINE': {'env': 'SA_OFFLINE', 'type': 'bool'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.data import prepare_dataset
    prepare_dataset(dataset_name="openseeker", source=SOURCE, snapshot_dir=SNAPSHOT_DIR, inputs=INPUTS,
                    profile=PROFILE, models=MODELS, min_length=MIN_LENGTH, max_length=MAX_LENGTH,
                    correctness=CORRECTNESS, observations=OBSERVATIONS, workers=WORKERS, seed=SEED,
                    val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, offline=OFFLINE,
                    environment=ENVIRONMENT)
