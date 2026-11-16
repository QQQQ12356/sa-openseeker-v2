"""Completion counts follow flushed results, including quiet subprocesses."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from search_comp.evaluation import tool_agent
from search_comp.utils.progress import CompletionProgress
from search_comp.workflows import benchmark_openseeker, offline_benchmark
from search_comp.workflows.benchmark_openseeker import run_logged


def test_counts_across_shards_and_waits_for_complete_utf8_line(tmp_path):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    lines = []
    progress = CompletionProgress(paths, 8, "all", emit=lines.append)
    progress.poll()  # Workers have not created their output files yet.
    data = (json.dumps(dict(id="中文"), ensure_ascii=False) + "\n").encode()
    paths[0].write_bytes(data[:9])
    progress.poll()
    assert len(lines) == 1 and "done=0/8" in lines[-1]
    with paths[0].open("ab") as target:
        target.write(data[9:])
    paths[1].write_text('{"id":"other","status":"tool_error","seconds":3.5}\n')
    progress.poll()
    assert "done=1/8" in lines[1]
    assert "done=2/8" in lines[2]
    assert "status=tool_error sample=3.5s" in lines[2]
    with paths[0].open("ab") as target:
        target.write(data)  # A retry of the same question does not advance progress.
    progress.poll()
    assert len(lines) == 3


def test_judge_counts_only_unique_items(tmp_path):
    path = tmp_path / "judge.jsonl"
    lines = []
    progress = CompletionProgress([path], 2, "judge", key="item_index", item_type="item", emit=lines.append)
    path.write_text('\n'.join(json.dumps(row) for row in [
        dict(type="summary"), dict(type="item", item_index=0, score=1),
        dict(type="item", item_index=0, score=1),
        dict(type="item", item_index=1, score=None)]) + '\n')
    progress.poll()
    assert len(lines) == 3
    assert "done=2/2" in lines[-1]


def test_quiet_subprocess_reports_before_exit(tmp_path):
    path, log = tmp_path / "predictions.jsonl", tmp_path / "generation.log"
    script = tmp_path / "worker.py"
    script.write_text('''import sys, time
from pathlib import Path
path, log = map(Path, sys.argv[1:])
path.write_text('{"query":"q"}\\n')
deadline = time.monotonic() + 5
while 'done=1/8' not in log.read_text():
    if time.monotonic() > deadline:
        raise RuntimeError('completion was not reported while worker was running')
    time.sleep(0.01)
''')
    run_logged([sys.executable, str(script), str(path), str(log)], tmp_path, log,
               path, 8, "benchmark generation")
    assert "done=0/8" in log.read_text()
    assert "done=1/8" in log.read_text()


def test_native_benchmarks_report_each_flushed_prediction(tmp_path):
    from contextlib import nullcontext, redirect_stdout
    from io import StringIO

    rows = [dict(id="a", query="first", answer="a"), dict(id="b", query="second", answer="b")]
    for benchmark in ("hotpotqa", "browsecomp-plus", "browsecomp"):
        output = tmp_path / benchmark
        output.mkdir()
        predictions = output / "predictions.jsonl"

        def load_model(*args):
            assert f"[benchmark {benchmark} generation] done=0/2" in log.getvalue()
            return object(), object()

        def generate(*args, **kwargs):
            count = len(predictions.read_text(encoding="utf-8").splitlines())
            if count:
                assert f"[benchmark {benchmark} generation] done=1/2 status=completed" in log.getvalue()
            return dict(prediction="answer", events=[], status="tool_error" if count else "completed")

        with redirect_stdout(StringIO()) as log, patch.object(tool_agent, "run_agent", generate), \
                patch.object(tool_agent, "ModelGenerator", lambda *a, **k: SimpleNamespace(metadata={})):
            if benchmark == "browsecomp":
                args = SimpleNamespace(
                    official_dir=tmp_path, benchmark="BrowseComp", run_dir=None, model="model",
                    kind="plain", tool_count=3, max_context=100, max_tokens=10, max_tool_tokens=10, max_turns=2)
                with patch.dict(sys.modules, {"llm_tool_openseeker": SimpleNamespace(Search=object, Visit=object)}), \
                        patch.object(benchmark_openseeker, "resolve_checkpoint", return_value=("model", "plain")), \
                        patch.object(benchmark_openseeker, "load_evaluation_model", load_model), \
                        patch.object(sys, "path", sys.path.copy()):
                    benchmark_openseeker.native_generate(args, rows, predictions)
            else:
                args = offline_benchmark.make_parser().parse_args(["run", "--benchmark", benchmark, "--root", str(tmp_path)])
                tools = SimpleNamespace(reset=lambda: None, retrieved=set(), visited=set(), trace=[], counts={})
                with patch.object(offline_benchmark, "LocalIndex", lambda *a: nullcontext()), \
                        patch.object(offline_benchmark, "LocalTools", return_value=tools), \
                        patch.object(offline_benchmark, "verify_index"), \
                        patch.object(offline_benchmark, "resolve_checkpoint", return_value=("model", "plain")), \
                        patch.object(offline_benchmark, "load_evaluation_model", load_model), \
                        patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}):
                    offline_benchmark.run_phase(args, output, rows, {})
        assert f"[benchmark {benchmark} generation] done=2/2 status=tool_error sample=" in log.getvalue()
