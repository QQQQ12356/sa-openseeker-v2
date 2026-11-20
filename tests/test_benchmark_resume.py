import json
from types import SimpleNamespace

import pytest

from search_comp.data import agent_prompt
from search_comp.data.agent_prompt import prompt_metadata
from search_comp.experiments.runtime import digest, read_json, write_json
from search_comp.workflows import offline_benchmark as workflow
from search_comp.workflows import benchmark_openseeker as online


def continuation(tmp_path):
    previous = tmp_path / "original"
    results = previous / "results"
    results.mkdir(parents=True)
    output = tmp_path / "continued"
    output.mkdir()
    rows = [dict(id=str(i), query=f"Question {i}", answer=f"Answer {i}") for i in range(4)]
    prompt = "original system"

    def write(path, questions):
        with path.open("w", encoding="utf-8") as target:
            for i, row in enumerate(questions):
                record = dict(row, sample_index=i, status="completed", **prompt_metadata(prompt))
                target.write(json.dumps(record) + "\n")
        return path

    source = write(results / "predictions.jsonl", rows[:2])
    metadata = dict(benchmark="browsecomp-plus", model="/model", kind="plain",
                    questions_sha256="questions", index=dict(index_sha256="index"))
    manifest = results / "manifest.json"
    write_json(manifest, dict(metadata, selected_ids=["0", "1"], status="complete",
               predictions_sha256=digest(source)))
    args = SimpleNamespace(resume_from=previous, resume_manifest_sha256=digest(manifest),
                           resume_predictions_sha256=digest(source))
    return args, output, rows, metadata, prompt, write


@pytest.mark.parametrize("benchmark", ["browsecomp-plus", "hotpotqa"])
def test_resume_generates_only_suffix_and_preserves_original(tmp_path, monkeypatch, benchmark):
    args, output, rows, metadata, prompt, write = continuation(tmp_path)
    metadata["benchmark"] = benchmark
    original = args.resume_from / "results/manifest.json"
    write_json(original, dict(read_json(original), benchmark=benchmark))
    args.resume_manifest_sha256 = digest(original)
    monkeypatch.setattr(workflow, "agent_prompt", lambda args: (prompt, "training prompt"))

    def generate(worker_args, extension, remaining, manifest, index, resolved_prompt):
        assert worker_args.question_offset == 2
        assert remaining == rows[2:]
        assert resolved_prompt == (prompt, "training prompt")
        return write(extension / "predictions.jsonl", remaining)

    monkeypatch.setattr(workflow, "run_phase", generate)
    result = workflow.continue_run(args, output, rows, {}, metadata)
    predictions = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in predictions] == ["0", "1", "2", "3"]
    assert [row["sample_index"] for row in predictions] == [0, 1, 2, 3]
    assert metadata["resume"]["reused"] == metadata["resume"]["newly_generated"] == 2
    assert read_json(output / "generation.metrics.json")["statuses"] == {"completed": 4}
    assert digest(args.resume_from / "results/predictions.jsonl") == args.resume_predictions_sha256


@pytest.mark.parametrize("change", ["system", "index", "prefix", "limit"])
def test_resume_rejects_incompatible_history(tmp_path, monkeypatch, change):
    args, output, rows, metadata, prompt, _ = continuation(tmp_path)
    monkeypatch.setattr(workflow, "agent_prompt",
                        lambda args: ("changed system" if change == "system" else prompt, "prompt"))
    if change == "index":
        metadata["index"] = dict(index_sha256="different")
    elif change == "prefix":
        rows.reverse()
    elif change == "limit":
        rows = rows[:2]

    def unexpected(*args):
        pytest.fail("continuation must validate the original run before generation")

    monkeypatch.setattr(workflow, "run_phase", unexpected)
    with pytest.raises(ValueError):
        workflow.continue_run(args, output, rows, {}, metadata)
    assert not (output / "predictions.jsonl").exists()


def test_online_resume_accepts_legacy_predictions_without_sample_index(tmp_path, monkeypatch):
    results = tmp_path / "old" / "results"
    results.mkdir(parents=True)
    output = tmp_path / "new"
    output.mkdir()
    rows = [dict(id=i, query=f"Question {i}", answer=f"Answer {i}") for i in range(3)]
    prompt = "online system"
    name = "result_tool5.jsonl"

    def write(target, questions):
        records = [dict(row, status="completed", **prompt_metadata(prompt)) for row in questions]
        target.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    (results / "questions.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    write(results / name, rows[:1])
    metadata = dict(benchmark="BrowseComp", mode="native", official_revision="rev",
                    data_sha256="data", model="/model", kind="plain",
                    official_script_sha256={}, search_provider="serper", summary_model="summary")
    write_json(results / "manifest.json", dict(metadata, count=1))
    args = SimpleNamespace(resume_from=results.parent, resume_manifest_sha256=digest(results / "manifest.json"),
                           resume_predictions_sha256=digest(results / name), run_dir=None)
    monkeypatch.setattr(agent_prompt, "resolve_system_prompt", lambda *args: (prompt, "source"))

    def generate(args, remaining, target, resolved_prompt):
        assert remaining == rows[1:]
        write(target, remaining)

    monkeypatch.setattr(online, "native_generate", generate)
    online.resume_native(args, rows, output, metadata, output / name)
    records = [json.loads(line) for line in (output / name).read_text(encoding="utf-8").splitlines()]
    assert [row["sample_index"] for row in records] == [0, 1, 2]
    assert digest(results / name) == args.resume_predictions_sha256
