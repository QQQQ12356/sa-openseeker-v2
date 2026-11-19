import json

from search_comp.data.browsecomp_sft import normalize_browsecomp_sft
from search_comp.data.openseeker import parse_calls, question_key
from search_comp.data.question_overlap import QuestionIndex
from search_comp.evaluation.training_overlap import audit_training_overlap, select_questions
from search_comp.experiments.runtime import digest
from search_comp.workflows.prepare_openseeker import prepare_sft_input


QUESTION = "Which historic building was designed by the same architect as the palace?"


def test_main_rewrites_share_canonical_question_and_respect_protection():
    official = {"id": "42", "query": QUESTION, "answer": "reference"}
    messages = [
        {"role": "system", "content": "Original system"},
        {"role": "user", "content": f"We need to answer this puzzle: {QUESTION}"},
        {"role": "reasoning", "content": "Search for the architect."},
        {"role": "tool_call", "content": '<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>'},
        {"role": "tool_output", "content": "<tool_response>historical evidence</tool_response>"},
        {"role": "answer", "content": "<answer>demonstration</answer>"},
    ]
    raw = {"messages": json.dumps(messages), "category": "browsecomp-openai/gpt-oss-20b"}
    index = QuestionIndex([official])
    assert prepare_sft_input(raw, index, index) == "protected_question"
    assert prepare_sft_input(raw, index, QuestionIndex([])) is None

    result = normalize_browsecomp_sft(raw, "browsecomp-sft/data.parquet", 0)
    assert result["question_id"] == question_key(QUESTION)
    assert result["original_question"] == messages[1]["content"]
    assert result["benchmark_question_id"] == "42"
    assert result["answer"] == "demonstration"
    assert parse_calls(result["messages"][2]["content"]) == [
        {"name": "search", "arguments": {"query": ["architect"]}},
    ]


def test_overlap_filters_before_limit_and_audits_the_training_fingerprint(tmp_path):
    questions = [
        {"id": 10, "query": QUESTION, "answer": "reference"},
        {"id": 20, "query": "What is the capital of Japan?", "answer": "Tokyo"},
    ]
    question_file = tmp_path / "questions.jsonl"
    question_file.write_text("".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8")
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(dict(id="sample", question="Rewritten question",
                                         original_question=f"Find the answer: {QUESTION}")) + "\n", encoding="utf-8")
    report = audit_training_overlap(question_file, train_file, digest(train_file), 0.30)
    assert report["excluded_ids"] == ["10"]
    assert report["matches"]["10"]["method"] == "embedded"
    report_file = tmp_path / "overlap.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    assert select_questions(questions, 1, report_file) == questions[1:]
