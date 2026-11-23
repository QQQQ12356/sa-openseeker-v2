import json

import pytest

from search_comp.experiments import browsecomp_plus_recovery as recovery
from search_comp.experiments.runtime import digest, read_json, write_json


def write_predictions(path, *rows, tail=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows) + tail)


def test_recover_worker_predictions_uses_original_question_scope(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark"
    prepared = benchmark / "prepared"
    prepared.mkdir(parents=True)
    rows = [dict(id=str(i), query=f"Question {i}", answer="Answer", evidence_docs=["doc"], gold_docs=["doc"])
            for i in range(4)]
    questions = prepared / "questions.jsonl"
    write_predictions(questions, *rows)
    dataset = dict(benchmark="browsecomp-plus", questions=len(rows), questions_sha256=digest(questions),
                   corpus_sha256="corpus", canonical_corpus=False, corpus_scope="fixture")
    write_json(prepared / "manifest.json", dataset)
    evaluation = tmp_path / "evaluation"
    results = evaluation / "results"
    results.mkdir(parents=True)
    write_json(evaluation / "job.json", dict(operation="evaluate", status="running", settings=dict(
        benchmark="browsecomp-plus", benchmark_dir=benchmark, generation={"limit": 3}, overlap="filter",
    )))
    write_json(evaluation / "overlap.json", {"excluded_ids": ["1"]})
    write_json(results / "manifest.json", dict(benchmark="browsecomp-plus", selected_ids=["0", "2", "3"],
               questions_sha256=dataset["questions_sha256"], prepared_manifest=dataset, arguments={}))
    for worker, position in enumerate((0, 2)):
        prediction = dict(id=str(position), query=rows[position]["query"], status="completed",
                          prediction="Answer", retrieved_docids=["doc"])
        write_predictions(results / f"workers/worker_{worker}/predictions.jsonl", prediction,
                          tail=b'{"id": "3"' if worker == 0 else b"")
    monkeypatch.setattr(recovery, "configure", lambda environment: None)
    monkeypatch.setattr(recovery, "enforce_offline", lambda: None)
    source_sha256 = digest(evaluation / "job.json")

    output = recovery.recover_browsecomp_plus(evaluation_dir=str(evaluation), run_name="fixture",
                                             retrieval_only=True, judge={"judge_model": ""}, environment={})

    assert digest(evaluation / "job.json") == source_sha256
    report = read_json(output / "recovery.json")
    assert report["missing_ids"] == ["3"]
    assert [source["predictions"] for source in report["prediction_sources"]] == [1, 1]
    assert report["prediction_sources"][0]["skipped_unterminated_line"] is True
    metrics = read_json(output / "results/summary.metrics.json")
    assert metrics["prediction_count"] == 2 and metrics["missing_predictions"] == 1
    assert metrics["recovery"]["coverage"] == metrics["evidence_recall"] == 2 / 3


def test_recover_predictions_rejects_duplicates_without_publishing(tmp_path):
    prediction = dict(id="one", query="Question", status="completed")
    first, second, merged = (tmp_path / f"{name}.jsonl" for name in ("first", "second", "merged"))
    write_predictions(first, prediction)
    write_predictions(second, prediction)
    with pytest.raises(ValueError, match="预测题目重复"):
        recovery.recover_predictions([(first, None), (second, None)], [prediction], merged)
    assert not merged.exists()
    assert not merged.with_suffix(".building.jsonl").exists()


def test_recovery_finds_resume_sources_from_initial_manifest(tmp_path):
    previous = tmp_path / "previous/results"
    write_predictions(previous / "predictions.jsonl", dict(id="a"))
    previous_hash = digest(previous / "predictions.jsonl")
    provenance = dict(benchmark="browsecomp-plus", model="checkpoint", kind="plain",
                      questions_sha256="questions", index={})
    write_json(previous / "manifest.json", dict(**provenance, status="complete", selected_ids=["a"],
                                                predictions_sha256=previous_hash))
    current = tmp_path / "current/results"
    worker = current / "extension/workers/worker_0/predictions.jsonl"
    write_predictions(worker, dict(id="b"))
    manifest = dict(**provenance, arguments=dict(
        resume_from=str(previous.parent), resume_manifest_sha256=digest(previous / "manifest.json"),
        resume_predictions_sha256=previous_hash,
    ))
    rows = [{"id": "a"}, {"id": "b"}]
    assert recovery.prediction_sources(current, rows, manifest) == [
        (previous / "predictions.jsonl", previous_hash), (worker, None),
    ]
