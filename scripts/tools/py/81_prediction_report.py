#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'METHOD': {'env': 'SA_METHOD', 'type': 'str'},
 'BENCHMARK': {'env': 'SA_BENCHMARK', 'type': 'str'},
 'EVALUATION_DIR': {'env': 'SA_EVALUATION_DIR', 'type': 'str'},
 'TITLE': {'env': 'SA_TITLE', 'type': 'str'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.runtime import configure, job, path, stamp
    from search_comp.experiments.benchmarks import resolve_evaluation
    from search_comp.workflows.render_predictions_html import render_report
    configure(ENVIRONMENT)
    directory, source = resolve_evaluation(METHOD, BENCHMARK, EVALUATION_DIR)
    with job(directory / "reports" / stamp("html"), "prediction_report", dict(evaluation_dir=directory)) as output:
        render_report(path(source["predictions"]), output / "predictions.html", title=TITLE)
