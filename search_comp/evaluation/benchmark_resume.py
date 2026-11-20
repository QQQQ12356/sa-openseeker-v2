"""Validate and combine a completed evaluation prefix with new predictions."""
from collections import Counter
import hashlib
import json

from search_comp.experiments.runtime import digest, read_json


def resume_prefix(directory, rows, current, manifest_sha256, predictions_sha256):
    result_dir = directory / "results"
    manifest_path = result_dir / "manifest.json"
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("原测试结果清单已改变")
    previous = read_json(manifest_path)
    if previous.get("status") != "complete" or previous.get("predictions_sha256") != predictions_sha256:
        raise ValueError("续测需要完整且可校验的原测试结果")
    for key in ("benchmark", "model", "kind", "questions_sha256", "index"):
        if previous.get(key) != current[key]:
            raise ValueError(f"续测与原测试的结果来源不一致: {key}")
    selected = previous["selected_ids"]
    if not selected or selected != [row["id"] for row in rows[:len(selected)]]:
        raise ValueError("原测试题目须为目标题目列表的有序前缀")
    if len(selected) == len(rows):
        raise ValueError("没有新增题目；请调高 LIMIT 或使用 -1")
    return result_dir / "predictions.jsonl", len(selected)


def checked_predictions(filename, rows, *, prompt_sha256, file_sha256=None):
    fingerprint = hashlib.sha256()
    with filename.open("rb") as stream:
        for position, row in enumerate(rows):
            line = stream.readline()
            if not line:
                raise ValueError(f"预测文件缺少第 {position + 1} 条记录: {filename}")
            fingerprint.update(line)
            prediction = json.loads(line)
            if (prediction.get("sample_index") != position or prediction.get("id") != row["id"]
                    or prediction.get("query") != row["query"] or prediction.get("answer") != row["answer"]
                    or prediction.get("system_prompt_sha256") != prompt_sha256):
                raise ValueError(f"预测题目、顺序或提示词与本次配置不一致: {filename}, row={position + 1}")
            yield prediction
        if stream.read().strip():
            raise ValueError(f"预测文件含额外记录: {filename}")
    if file_sha256 and fingerprint.hexdigest() != file_sha256:
        raise ValueError(f"原预测文件 SHA256 已改变: {filename}")


def write_predictions(target, predictions, offset=0):
    statuses = Counter()
    for position, prediction in enumerate(predictions, offset):
        prediction["sample_index"] = position
        target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        statuses[prediction["status"]] += 1
    return statuses
