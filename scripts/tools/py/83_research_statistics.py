#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCHEMA = {
    "ENVIRONMENT__HF_HUB_OFFLINE": {"env": "HF_HUB_OFFLINE", "type": "str"},
    "SNAPSHOT_DIR": {"env": "SA_SNAPSHOT_DIR", "type": "str"},
    "TOKENIZER": {"env": "SA_TOKENIZER", "type": "str"},
    "MAX_LENGTH": {"env": "SA_MAX_LENGTH", "type": "int"},
    "OBSERVATIONS": {"env": "SA_OBSERVATIONS", "type": "str"},
    "WORKERS": {"env": "SA_WORKERS", "type": "int"},
}

if __name__ == "__main__":
    from search_comp.experiments.runtime import ROOT, configure, job, path, stamp
    from search_comp.experiments.shell_config import load_settings
    from search_comp.workflows.research_statistics import run

    settings = load_settings(__file__, SCHEMA)
    configure(settings["ENVIRONMENT"])
    if not settings["SNAPSHOT_DIR"]:
        raise ValueError("请填写 SA_SNAPSHOT_DIR，指向含 download_manifest.json 的 Research 下载目录")
    options = dict(snapshot_dir=path(settings["SNAPSHOT_DIR"]), tokenizer=settings["TOKENIZER"],
                   max_length=settings["MAX_LENGTH"], observations=settings["OBSERVATIONS"],
                   workers=settings["WORKERS"])
    reports = ROOT / "artifacts/datasets/research/reports"
    with job(reports / stamp("statistics"), "research_statistics", options) as output:
        run(**options, output=output)
