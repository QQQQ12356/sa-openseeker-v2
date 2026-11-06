#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'ENVIRONMENT__HF_ENDPOINT': {'env': 'HF_ENDPOINT', 'type': 'str'},
 'ENVIRONMENT__HF_HUB_OFFLINE': {'env': 'HF_HUB_OFFLINE', 'type': 'str'},
 'DATASET_DIR': {'env': 'SA_DATASET_DIR', 'type': 'str'},
 'SPLIT': {'env': 'SA_SPLIT', 'type': 'str'},
 'LIMIT': {'env': 'SA_LIMIT', 'type': 'int'},
 'OFFSET': {'env': 'SA_OFFSET', 'type': 'int'},
 'TITLE': {'env': 'SA_TITLE', 'type': 'str'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    from search_comp.experiments.runtime import configure, job, path, stamp
    from search_comp.workflows.render_dataset_html import render_report
    configure(ENVIRONMENT)
    dataset = path(DATASET_DIR)
    with job(dataset / "reports" / stamp("html"), "dataset_report",
             dict(dataset_dir=dataset, split=SPLIT, limit=LIMIT, offset=OFFSET)) as output:
        render_report(dataset, output / "dataset.html", split=SPLIT, limit=LIMIT, offset=OFFSET, title=TITLE)
