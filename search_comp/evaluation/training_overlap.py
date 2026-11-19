"""Audit the actual training split before generating benchmark predictions."""
import hashlib
import json

from search_comp.data.question_overlap import QuestionIndex
from search_comp.evaluation.benchmark_data import read_questions
from search_comp.experiments.runtime import digest


def audit_training_overlap(questions_file, train_file, expected_digest, threshold):
    index = QuestionIndex(read_questions(questions_file), threshold)
    matches = {}
    rows = 0
    fingerprint = hashlib.sha256()
    with train_file.open("rb") as stream:
        for line in stream:
            fingerprint.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for question in dict.fromkeys((row["question"], row.get("original_question"))):
                if not question:
                    continue
                for target, score, method in index.matches(question):
                    key = str(target["id"])
                    if key not in matches or score > matches[key]["score"]:
                        matches[key] = dict(train_id=row["id"], score=score, method=method)
    actual_digest = fingerprint.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("训练 JSONL 与实验指纹不一致；请恢复训练时的文件")
    return dict(train_file=str(train_file), train_sha256=actual_digest, train_rows=rows,
                questions_file=str(questions_file), questions_sha256=digest(questions_file),
                threshold=threshold, benchmark_questions=len(index.questions),
                excluded_ids=sorted(matches), matches=matches)


def select_questions(rows, limit, overlap_report=None):
    if overlap_report:
        with overlap_report.open(encoding="utf-8") as stream:
            excluded = set(json.load(stream)["excluded_ids"])
        rows = [row for row in rows if str(row["id"]) not in excluded]
        if not rows:
            raise ValueError("训练重合筛选后无可评测题目；请检查 overlap.json")
    return rows[:limit] if limit > 0 else rows
