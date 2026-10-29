"""Offline regression for data encoding, upstream contracts and complete scoring."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from search_comp.evaluation.benchmark_data import decrypt, prepare_browsecomp, read_questions
from search_comp.workflows import benchmark_openseeker as benchmark
from search_comp.workflows import serve_openseeker


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def encrypt(text, password):
    key = hashlib.sha256(password.encode()).digest()
    return base64.b64encode(bytes(c ^ key[i % 32] for i, c in enumerate(text.encode()))).decode()


def test_public_browsecomp_csv_decryption_and_manifest(tmp_path):
    source = tmp_path / "encrypted.csv"
    source.write_text("problem,answer,canary\n" + ",".join([
        encrypt("什么是 BrowseComp?", "canary"), encrypt("测试", "canary"), "canary"]) + "\n")
    output = tmp_path / "data.jsonl"
    rows = prepare_browsecomp(output, source)
    assert rows == [dict(id=0, query="什么是 BrowseComp?", answer="测试")]
    assert read_questions(output) == rows
    metadata = json.loads(output.with_suffix(".manifest.json").read_text())
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        prepare_browsecomp(output, source)
    assert decrypt(encrypt("a" * 100, "c"), "c") == "a" * 100


@pytest.mark.parametrize("rows", [[], [dict(query="q", answer="")],
    [dict(query="q", answer="a"), dict(query="q", answer="b")], [dict(query="q", answer=["a"])]])
def test_invalid_dataset_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        read_questions(jsonl(tmp_path / "data.jsonl", rows))


def test_reader_drops_training_trajectory_and_evidence(tmp_path):
    path = jsonl(tmp_path / "data.jsonl", [dict(question="q", answer="a", messages=["LEAK"], evidence="LEAK")])
    assert read_questions(path) == [dict(id=0, query="q", answer="a")]


def test_align_retries_boxed_query_and_missing_denominator(tmp_path):
    rows = [dict(query="q1", answer="a"), dict(query="q2", answer="b")]
    raw = jsonl(tmp_path / "raw.jsonl", [dict(query="q1 BOX", answer="a", final_response=r)
                                        for r in ("", "prediction")])
    output = tmp_path / "aligned.jsonl"
    coverage = benchmark.align_results(rows, raw, output, " BOX")
    aligned = [json.loads(line) for line in output.read_text().splitlines()]
    assert coverage == dict(expected=2, unique_predictions=1, attempts=2, missing_predictions=1, empty_predictions=1)
    assert [row["final_response"] for row in aligned] == ["prediction", ""]
    assert aligned[1]["status"] == "generation_missing"


@pytest.mark.parametrize("row", [dict(query="other", answer="a", final_response="x"),
    dict(query="q", answer="LEAK", final_response="x"), dict(query="q", answer="a", final_response=None)])
def test_foreign_or_invalid_prediction_rejected(tmp_path, row):
    with pytest.raises(ValueError):
        benchmark.align_results([dict(query="q", answer="a")],
            jsonl(tmp_path / "raw.jsonl", [row]), tmp_path / "aligned.jsonl")


def test_judge_errors_do_not_inflate_full_denominator(tmp_path):
    rows = [dict(type="summary", accuracy=1.0), dict(type="item", item_index=0, score=1),
            dict(type="item", item_index=1, score="EVAL_ERROR"), dict(type="item", item_index=2, score=True)]
    result = benchmark.audit_scores(jsonl(tmp_path / "eval.jsonl", rows), 3)
    assert result["official_accuracy"] == 1.0
    assert result["accuracy_all_questions"] == pytest.approx(1 / 3)
    assert result["judge_unknown_or_error"] == 2
    assert not result["complete"]


def test_actual_upstream_cli_spelling():
    args = SimpleNamespace(official_dir=Path("/official"), mode="official-v2", workers=2,
                           max_tokens=32768, tool_count=200)
    command = benchmark.generation_command(args, Path("/data"), Path("/out"))
    assert "/official/eval/generate_answer_v2.py" in command
    assert "--max_worker" in command
    assert "--max_workers" not in command
    assert command[command.index("--tool_count_max") + 1] == "200"


def test_environment_preflight_and_judge_only(monkeypatch):
    for key in ("SCORER_URLS", "SCORER_API_KEY", "SCORER_MODEL_NAME", "OPENSEEKER_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="SCORER_URLS"):
        benchmark.require_environment("official-v2")
    for key in ("SCORER_URLS", "SCORER_API_KEY", "SCORER_MODEL_NAME"):
        monkeypatch.setenv(key, "configured")
    benchmark.require_environment("official-v2", generate=False)


def test_checkpoint_resolution_and_beacon_server_rejection(tmp_path, monkeypatch):
    model = tmp_path / "checkpoint with spaces"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(dict(enable_beacon=True)))
    assert benchmark.resolve_checkpoint(None, str(model), None) == (str(model), "beacon")
    with pytest.raises(ValueError, match="Beacon"):
        benchmark.resolve_checkpoint(None, str(model), "plain")
    monkeypatch.setattr(sys, "argv", ["serve", "--model", str(model), "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        serve_openseeker.main()
    assert exc.value.code == 2
    (model / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="merge"):
        benchmark.resolve_checkpoint(None, str(model), None)


def test_training_run_resolves_final_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    run = tmp_path / "run"
    (run / "models/model").mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps(dict(kind="beacon")))
    (run / "models/model/train_summary.json").write_text(json.dumps(dict(final_model_path=str(model))))
    assert benchmark.resolve_checkpoint(run, None, None) == (str(model), "beacon")


def test_dry_run_is_read_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "new_output"
    result = subprocess.run([sys.executable, str(root / "scripts/py/47_benchmark.py"),
        "--dry-run", "--mode", "official-v2", "--output-dir", str(output), "--official-dir", str(tmp_path / "vendor")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "generate_answer_v2.py" in result.stdout
    assert not output.exists()
    assert not (tmp_path / "vendor").exists()


def test_pipeline_runs_generation_and_judge_subprocesses(tmp_path, monkeypatch):
    """Exercise real subprocesses; fake remote models/tools only at script boundary."""
    official = tmp_path / "official"
    (official / "eval").mkdir(parents=True)
    (official / "eval/generate_answer.py").write_text(
        "import argparse,json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for name in ['dataset_path','out_dir','max_worker','max_tokens','tool_count_max','max_retry_rounds']: p.add_argument('--'+name)\n"
        "a=p.parse_args()\n"
        "rows=[json.loads(s) for s in Path(a.dataset_path).read_text().splitlines()]\n"
        "rows=[dict(r,final_response='predicted') for r in rows]\n"
        "Path(a.out_dir,'result_tool'+a.tool_count_max+'.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))\n")
    (official / "eval/eval.py").write_text(
        "import argparse,json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--data_path');p.add_argument('--max_workers');a=p.parse_args()\n"
        "rows=[json.loads(s) for s in Path(a.data_path).read_text().splitlines()]\n"
        "out=[dict(type='item',item_index=i,score=int(r['final_response']=='predicted')) for i,r in enumerate(rows)]\n"
        "Path(a.data_path.replace('.jsonl','_eval.jsonl')).write_text(''.join(json.dumps(r)+'\\n' for r in out))\n")
    data = jsonl(tmp_path / "data.jsonl", [dict(query="q", answer="reference")])
    monkeypatch.setattr(benchmark, "ensure_official", lambda _: benchmark.OFFICIAL_REVISION)
    monkeypatch.setattr(benchmark, "require_environment", lambda *a, **kw: None)
    monkeypatch.setenv("SCORER_MODEL_NAME", "fake-judge")
    monkeypatch.setenv("OPENSEEKER_MODEL", "trained-checkpoint")
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--mode", "official-v1", "--data", str(data),
        "--official-dir", str(official), "--output-dir", str(output)])
    benchmark.main()
    metrics = json.loads((output / "summary.metrics.json").read_text())
    assert metrics["complete"]
    assert metrics["accuracy_all_questions"] == 1
    assert (output / "generation.log").is_file()
    assert (output / "judge.log").is_file()
    assert "reference" not in (output / "generation.log").read_text()
    with pytest.raises(SystemExit):
        benchmark.main()  # no accidental append/resume with stale item_index scores


@pytest.mark.parametrize('kind', ['plain', 'beacon'])
def test_native_uses_question_only_and_preserves_compression_path(tmp_path, monkeypatch, kind):
    rows = [dict(query="question", answer="REFERENCE_DO_NOT_PROMPT")]
    model = SimpleNamespace(eval=lambda: None)
    calls = []
    from search_comp.data.agent_prompt import build_system_prompt
    original_system = build_system_prompt() + "\nOriginal training-specific instruction."
    prompt_data = jsonl(tmp_path / "training.jsonl", [dict(messages=[
        dict(role="system", content=original_system),
        dict(role="user", content="TRAINING_QUESTION_DO_NOT_COPY")], answer="TRAINING_ANSWER_DO_NOT_COPY")])
    class Tool:
        def call(self, params):
            return "observation"
    monkeypatch.setitem(sys.modules, "llm_tool_openseeker", SimpleNamespace(
        Search=Tool, Visit=Tool, developer_prompt="official system"))
    monkeypatch.setitem(sys.modules, "search_comp.models.model_loader", SimpleNamespace(
        load_tokenizer=lambda _: "tokenizer", load_beacon_model=lambda _: model))
    monkeypatch.setitem(sys.modules, "search_comp.models.plain_qwen3", SimpleNamespace(load_sft_model=lambda _: model))
    from search_comp.evaluation import tool_agent
    monkeypatch.setattr(tool_agent, "ModelGenerator", lambda m, t, beacon: calls.append(beacon) or "generate")
    def agent(messages, tokenizer, generate, tools, *budgets):
        assert "REFERENCE" not in json.dumps(messages)
        from search_comp.data.agent_prompt import initial_messages
        assert messages == initial_messages("question", system=original_system)
        assert "TRAINING_QUESTION" not in json.dumps(messages)
        assert "TRAINING_ANSWER" not in json.dumps(messages)
        assert tools.execute([dict(name="search", arguments=dict(query=["q"]))])[0]["content"] == "<tool_response>observation</tool_response>"
        return dict(prediction="prediction", status="completed", events=[])
    monkeypatch.setattr(tool_agent, "run_agent", agent)
    monkeypatch.setattr(benchmark, "resolve_checkpoint", lambda *a: ("checkpoint", kind))
    args = SimpleNamespace(official_dir=tmp_path, run_dir=None, model=None, kind=kind, prompt_data=prompt_data,
        max_turns=200, max_context=131072, max_tokens=32768, max_tool_tokens=32768)
    output = tmp_path / "native.jsonl"
    benchmark.native_generate(args, rows, output)
    assert calls == [kind == "beacon"]
    assert json.loads(output.read_text())["final_response"] == "prediction"
    assert json.loads(output.read_text())["prompt_source"] == str(prompt_data)
