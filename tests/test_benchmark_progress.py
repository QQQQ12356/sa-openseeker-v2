"""Completion counts follow flushed results, including quiet subprocesses."""
import json
import sys

from search_comp.utils.progress import CompletionProgress
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
    paths[1].write_text('{"id":"other","status":"tool_error"}\n')
    progress.poll()
    assert "done=1/8" in lines[1]
    assert "done=2/8" in lines[2]
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
