#!/usr/bin/env python3
"""执行适配器；参数只在同名 sh 文件中修改。"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import re
import runpy
import sys
import unittest
import warnings
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
def validate():
    files = [p for folder in ("search_comp", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # 固定上游参考保留原文，不改其转义格式
        for filename in files:
            compile(filename.read_bytes(), str(filename), "exec")
    entries = list((ROOT / "scripts").rglob("*.py"))
    for filename in entries:
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if filename != Path(__file__).resolve():
            namespace = runpy.run_path(str(filename), run_name="inspect_parameters")
            from search_comp.experiments.shell_config import launcher_defaults
            launcher = filename.parent.parent / "sh" / (filename.stem + ".sh")
            launcher_defaults(launcher, namespace["SCHEMA"])
            if b"\r" in launcher.read_bytes():
                raise AssertionError(f"SH 必须使用 LF 换行: {launcher}")
        if any(isinstance(node, ast.Import) and any(a.name == "argparse" for a in node.names)
               or isinstance(node, ast.ImportFrom) and node.module == "argparse" for node in ast.walk(tree)):
            raise AssertionError(f"公共脚本不应要求拼接命令行参数: {filename}")
    for filename in [ROOT / "00_README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        if not re.match(r"\d{2}_", filename.name):
            raise AssertionError(f"文档缺少序号前缀: {filename}")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", filename.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                if not (filename.parent / target.split("#")[0]).exists():
                    raise AssertionError(f"文档链接不存在: {filename.name}: {target}")
    reference = ROOT / "search_comp/offline/reference"
    metadata = json.loads((reference / "00_provenance.json").read_text(encoding="utf-8"))
    for benchmark, record in metadata.items():
        for name, expected in record["sha256"].items():
            actual = hashlib.sha256((reference / benchmark / name).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"官方参考摘要不符: {benchmark}/{name}")
    summary = dict(python_files=len(files), python_adapters=len(entries), shell_launchers=len(list((ROOT / "scripts").rglob("*.sh"))), official_references="verified")
    if RUN_UNIT_TESTS:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite(loader.discover(str(ROOT / "tests"), pattern=p) for p in TEST_PATTERNS)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            result = unittest.TextTestRunner(stream=captured, verbosity=1).run(suite)
        summary.update(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))
        if not result.wasSuccessful():
            print(captured.getvalue())
            raise AssertionError("标准库测试未通过")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

# 类型定义，无运行默认值；由同名 SH 传入。
SCHEMA = {'RUN_UNIT_TESTS': {'env': 'SA_RUN_UNIT_TESTS', 'type': 'bool'},
 'TEST_PATTERNS': {'env': 'SA_TEST_PATTERNS', 'type': 'tuple'}}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    globals().update(load_settings(__file__, SCHEMA))
    validate()
