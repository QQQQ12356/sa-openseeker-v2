#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'ENVIRONMENT__SCORER_URLS': {'env': 'SCORER_URLS', 'type': 'str'},
 'ENVIRONMENT__SCORER_API_KEY': {'env': 'SCORER_API_KEY', 'type': 'str'},
 'ENVIRONMENT__SCORER_MODEL_NAME': {'env': 'SCORER_MODEL_NAME', 'type': 'str'},
 'EVALUATION_DIR': {'env': 'SA_EVALUATION_DIR', 'type': 'str'},
 'RUN_NAME': {'env': 'SA_RUN_NAME', 'type': 'str'},
 'JUDGE__JUDGE_WORKERS': {'env': 'SA_JUDGE__JUDGE_WORKERS', 'type': 'int'},
 'JUDGE__JUDGE_ATTEMPTS': {'env': 'SA_JUDGE__JUDGE_ATTEMPTS', 'type': 'int'},
 'JUDGE__JUDGE_MAX_TOKENS': {'env': 'SA_JUDGE__JUDGE_MAX_TOKENS', 'type': 'int'},
 'JUDGE__JUDGE_TIMEOUT': {'env': 'SA_JUDGE__JUDGE_TIMEOUT', 'type': 'int'},
 'JUDGE__JUDGE_ENABLE_THINKING': {'env': 'SA_JUDGE__JUDGE_ENABLE_THINKING', 'type': 'bool'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.benchmarks import score
    score(method="baseline", benchmark="browsecomp", evaluation_dir=EVALUATION_DIR,
          run_name=RUN_NAME, judge=JUDGE, environment=ENVIRONMENT)
