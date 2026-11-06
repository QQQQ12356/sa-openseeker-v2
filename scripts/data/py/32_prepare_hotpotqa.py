#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'ENVIRONMENT__CUDA_VISIBLE_DEVICES': {'env': 'CUDA_VISIBLE_DEVICES', 'type': 'str'},
 'VERSION': {'env': 'SA_VERSION', 'type': 'str'},
 'PHASE': {'env': 'SA_PHASE', 'type': 'str'},
 'DATA': {'env': 'SA_DATA', 'type': 'optional'},
 'CORPUS': {'env': 'SA_CORPUS', 'type': 'optional'},
 'CORPUS_REVISION': {'env': 'SA_CORPUS_REVISION', 'type': 'optional'},
 'DECRYPTED': {'env': 'SA_DECRYPTED', 'type': 'bool'},
 'TITLE_WEIGHT': {'env': 'SA_TITLE_WEIGHT', 'type': 'float'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.benchmarks import prepare_benchmark
    prepare_benchmark(benchmark="hotpotqa", version=VERSION, phase=PHASE, environment=ENVIRONMENT,
                      data=DATA, corpus=CORPUS, corpus_revision=CORPUS_REVISION,
                      decrypted=DECRYPTED, title_weight=TITLE_WEIGHT)
