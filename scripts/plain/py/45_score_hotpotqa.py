#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'EVALUATION_DIR': {'env': 'SA_EVALUATION_DIR', 'type': 'str'},
 'RUN_NAME': {'env': 'SA_RUN_NAME', 'type': 'str'},
 'JUDGE__JUDGE_MODEL': {'env': 'SA_JUDGE__JUDGE_MODEL', 'type': 'str'},
 'JUDGE__LLM_JUDGE': {'env': 'SA_JUDGE__LLM_JUDGE', 'type': 'bool'},
 'JUDGE__JUDGE_ATTEMPTS': {'env': 'SA_JUDGE__JUDGE_ATTEMPTS', 'type': 'int'},
 'JUDGE__JUDGE_MAX_TOKENS': {'env': 'SA_JUDGE__JUDGE_MAX_TOKENS', 'type': 'int'},
 'JUDGE__JUDGE_MAX_CONTEXT': {'env': 'SA_JUDGE__JUDGE_MAX_CONTEXT', 'type': 'int'},
 'JUDGE__JUDGE_ENABLE_THINKING': {'env': 'SA_JUDGE__JUDGE_ENABLE_THINKING', 'type': 'bool'},
 'JUDGE__JUDGE_TEMPERATURE': {'env': 'SA_JUDGE__JUDGE_TEMPERATURE', 'type': 'float'},
 'JUDGE__JUDGE_TOP_P': {'env': 'SA_JUDGE__JUDGE_TOP_P', 'type': 'float'},
 'JUDGE__JUDGE_TOP_K': {'env': 'SA_JUDGE__JUDGE_TOP_K', 'type': 'int'},
 'JUDGE__SEED': {'env': 'SA_JUDGE__SEED', 'type': 'int'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.benchmarks import score
    score(method="plain", benchmark="hotpotqa", evaluation_dir=EVALUATION_DIR,
          run_name=RUN_NAME, judge=JUDGE, environment=ENVIRONMENT)
