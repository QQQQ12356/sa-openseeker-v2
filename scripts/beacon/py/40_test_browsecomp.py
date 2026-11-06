#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_API_URL': {'env': 'SUMMARY_API_URL', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_API_KEY': {'env': 'SUMMARY_API_KEY', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_MODEL_NAME': {'env': 'SUMMARY_MODEL_NAME', 'type': 'str'},
 'ENVIRONMENT__JINA_API_KEYS': {'env': 'JINA_API_KEYS', 'type': 'str'},
 'ENVIRONMENT__SERPER_API_KEY': {'env': 'SERPER_API_KEY', 'type': 'str'},
 'ENVIRONMENT__TAVILY_API_KEY': {'env': 'TAVILY_API_KEY', 'type': 'str'},
 'ENVIRONMENT__SEARCH_PROVIDER': {'env': 'SEARCH_PROVIDER', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_ATTEMPTS': {'env': 'SUMMARY_ATTEMPTS', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_MAX_INPUT_CHARS': {'env': 'SUMMARY_MAX_INPUT_CHARS', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_MAX_TOKENS': {'env': 'SUMMARY_MAX_TOKENS', 'type': 'str'},
 'ENVIRONMENT__SUMMARY_TIMEOUT': {'env': 'SUMMARY_TIMEOUT', 'type': 'str'},
 'ENVIRONMENT__JINA_TIMEOUT': {'env': 'JINA_TIMEOUT', 'type': 'str'},
 'BENCHMARK_DIR': {'env': 'SA_BENCHMARK_DIR', 'type': 'str'},
 'TRAINING_RUN': {'env': 'SA_TRAINING_RUN', 'type': 'str'},
 'MODEL_PATH': {'env': 'SA_MODEL_PATH', 'type': 'optional'},
 'PROMPT_DATA': {'env': 'SA_PROMPT_DATA', 'type': 'optional'},
 'RUN_NAME': {'env': 'SA_RUN_NAME', 'type': 'str'},
 'GENERATION__LIMIT': {'env': 'SA_GENERATION__LIMIT', 'type': 'int'},
 'GENERATION__MAX_CONTEXT': {'env': 'SA_GENERATION__MAX_CONTEXT', 'type': 'int'},
 'GENERATION__MAX_NEW_TOKENS': {'env': 'SA_GENERATION__MAX_NEW_TOKENS', 'type': 'int'},
 'GENERATION__MAX_TURNS': {'env': 'SA_GENERATION__MAX_TURNS', 'type': 'int'},
 'GENERATION__MAX_TOOL_TOKENS': {'env': 'SA_GENERATION__MAX_TOOL_TOKENS', 'type': 'int'},
 'GENERATION__TOOL_COUNT': {'env': 'SA_GENERATION__TOOL_COUNT', 'type': 'int'},
 'GENERATION__TOOL_CALL_RETRIES': {'env': 'SA_GENERATION__TOOL_CALL_RETRIES', 'type': 'int'},
 'GENERATION__DECODING': {'env': 'SA_GENERATION__DECODING', 'type': 'str'},
 'GENERATION__REPETITION_PENALTY': {'env': 'SA_GENERATION__REPETITION_PENALTY', 'type': 'float'},
 'GENERATION__TEMPERATURE': {'env': 'SA_GENERATION__TEMPERATURE', 'type': 'float'},
 'GENERATION__TOP_P': {'env': 'SA_GENERATION__TOP_P', 'type': 'float'},
 'GENERATION__GENERATION_SEED': {'env': 'SA_GENERATION__GENERATION_SEED', 'type': 'int'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.benchmarks import evaluate
    evaluate(method="beacon", benchmark="browsecomp", benchmark_dir=BENCHMARK_DIR,
             training_run=TRAINING_RUN, model_path=MODEL_PATH, prompt_data=PROMPT_DATA,
             run_name=RUN_NAME, generation=GENERATION, environment=ENVIRONMENT)
