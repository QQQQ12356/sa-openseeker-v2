#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {
    "INPUTS": {"env": "SA_INPUTS", "type": "list"},
    "OUTPUT_DIR": {"env": "SA_OUTPUT_DIR", "type": "path"},
    "RUN_NAME": {"env": "SA_RUN_NAME", "type": "str"},
    "INCLUDE_DETAILS": {"env": "SA_INCLUDE_DETAILS", "type": "bool"},
    "INCLUDE_INCOMPLETE": {"env": "SA_INCLUDE_INCOMPLETE", "type": "bool"},
}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.runtime import configure
    from search_comp.workflows.export_benchmark_excel import export_benchmarks
    configure({})
    export_benchmarks(inputs=INPUTS, output_dir=OUTPUT_DIR, run_name=RUN_NAME,
                      include_details=INCLUDE_DETAILS, include_incomplete=INCLUDE_INCOMPLETE)
