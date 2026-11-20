import json
from contextlib import nullcontext
from types import SimpleNamespace

from search_comp.data.agent_prompt import DEFAULT_SYSTEM_FILE
from search_comp.data.openseeker import VERSION
from search_comp.experiments import training_data_evaluation as evaluation
from search_comp.experiments.runtime import digest, write_json


def test_evaluation_uses_fresh_inputs_and_scores_answer_sources_separately(tmp_path, monkeypatch):
    system = DEFAULT_SYSTEM_FILE.read_text(encoding="utf-8")
    messages = [
        dict(role="system", content=system),
        dict(role="user", content="Repeated question?"),
        dict(role="assistant", content="old tool call"),
        dict(role="user", content="<tool_response>old evidence</tool_response>"),
        dict(role="assistant", content="<answer>old answer</answer>"),
    ]
    rows = [
        dict(id="ref", protocol=VERSION, answer="Paris", messages=messages),
        dict(id="demo", protocol=VERSION, answer="London", messages=messages,
             answer_source="demonstration_final_response"),
    ]
    filename = tmp_path / "train.jsonl"
    filename.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(tmp_path / "experiment.json", dict(training_config=dict(train_data_path=str(filename)),
               data_fingerprints=dict(train_data_path=digest(filename))))
    monkeypatch.setattr(evaluation, "configure", lambda environment: None)
    monkeypatch.setattr(evaluation, "resolve_target",
                        lambda *args: ("mixed", tmp_path, None, "plain", "checkpoint"))
    monkeypatch.setattr(evaluation, "resolve_checkpoint", lambda *args: ("checkpoint", "plain"))
    monkeypatch.setattr(evaluation, "load_evaluation_model", lambda *args: (None, None))
    monkeypatch.setattr(evaluation, "ModelGenerator", lambda *args, **kwargs: SimpleNamespace(metadata={}))
    monkeypatch.setattr(evaluation, "LocalIndex", lambda *args: nullcontext(SimpleNamespace(manifest={})))

    def predict(prompt, tokenizer, generate, tools, *args, **kwargs):
        assert len(prompt) == 2
        assert prompt[1] == messages[1]
        assert "old evidence" not in prompt[0]["content"]
        assert not tools.allow_visit
        return dict(status="completed", prediction="Paris", messages=prompt)

    monkeypatch.setattr(evaluation, "run_agent", predict)
    generation = dict(max_context=128, max_new_tokens=16, max_turns=2, max_tool_tokens=24,
                      tool_count=2, topk=2, snippet_tokens=8, read_tokens=16,
                      tool_call_retries=1, limit=-1, allow_visit=False, verify_index=True)
    output = evaluation.evaluate_training_data(method="plain", training_run=str(tmp_path), checkpoint=None,
               dataset_dir=None, split="train", index_dir=tmp_path, run_name="smoke",
               generation=generation, environment={})
    summary = json.loads((output / "summary.metrics.json").read_text(encoding="utf-8"))
    assert summary["by_answer_source"] == {
        "reference": {"count": 1, "em": 1.0, "f1": 1.0},
        "demonstration_final_response": {"count": 1, "em": 0.0, "f1": 0.0},
    }
    manifest = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert manifest["matches_training_train_sha256"]
