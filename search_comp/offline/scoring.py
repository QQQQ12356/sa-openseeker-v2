"""Pinned HotpotQA metrics and offline BrowseComp-Plus semantic/retrieval scoring."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

from .assets import digest, dump, verify_official
from .data import load_official_functions, records


def load_predictions(path, questions):
    expected = {row["id"]: row for row in questions}
    result = {}
    for row in records(path):
        qid = str(row.get("id", row.get("query_id", "")))
        if qid not in expected or qid in result:
            raise ValueError(f"foreign or duplicate prediction ID: {qid}")
        if "query" in row and row["query"] != expected[qid]["query"]:
            raise ValueError("prediction query differs from benchmark")
        if not isinstance(row.get("status"), str):
            raise ValueError("predictions require explicit status")
        response = row.get("final_response", row.get("prediction"))
        if not isinstance(response, str):
            raise ValueError("predictions require text final_response/prediction")
        row["final_response"] = response
        for name in ("retrieved_docids", "visited_docids"):
            if not isinstance(row.get(name, []), list):
                raise ValueError(f"{name} must be a list")
            row[name] = [str(value) for value in row.get(name, [])]
        counts = row.get("tool_call_counts", {})
        if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("tool_call_counts must map tool names to nonnegative integers")
        result[qid] = row
    return result


def extract_hotpot_response(response):
    """Only generated JSON may supply sentence facts; never infer them from gold/retrieval."""
    value = response.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        parsed = json.loads(value)
    except ValueError:
        return response, None, None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        return response, None, None
    facts = parsed.get("sp")
    if facts is not None and (not isinstance(facts, list) or any(
            not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str)
            or type(fact[1]) is not int or fact[1] < 0 for fact in facts)):
        return parsed["answer"], None, "invalid generated sp; omitted from official supporting-fact predictions"
    return parsed["answer"], facts, None


def score_hotpot(root, questions, predictions, output):
    verify_official(root, "hotpotqa")
    script = Path(root) / "official/hotpot_evaluate_v1.py"
    tree = ast.parse(script.read_text())
    # Only substitute the optional JSON accelerator; all official scoring code is unchanged.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ujson":
                    alias.name = "json"
    namespace = {"__name__": "official_hotpot_metrics"}
    exec(compile(tree, str(script), "exec"), namespace)
    answers, supporting, details = {}, {}, []
    for row in questions:
        prediction = predictions.get(row["id"])
        detail = dict(id=row["id"], completed=False, answer_parse_error=None)
        if prediction and prediction["status"] == "completed" and prediction["final_response"].strip():
            try:
                answer, facts, sp_error = extract_hotpot_response(prediction["final_response"])
                detail["supporting_parse_error"] = sp_error
            except ValueError as error:
                detail["answer_parse_error"] = str(error)
                answer, facts = "", None
            answers[row["id"]] = answer
            if facts is not None:
                supporting[row["id"]] = facts
            detail.update(completed=True, prediction=answer,
                          em=float(namespace["exact_match_score"](answer, row["answer"])),
                          f1=namespace["f1_score"](answer, row["answer"])[0])
        else:
            detail.update(em=0.0, f1=0.0)
        details.append(detail)
    dump(output / "hotpot_predictions.json", dict(answer=answers, sp=supporting))
    dump(output / "hotpot_gold.json", [dict(_id=row["id"], answer=row["answer"],
                                             supporting_facts=row["supporting_facts"]) for row in questions])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        namespace["eval"](output / "hotpot_predictions.json", output / "hotpot_gold.json")
    (output / "official_hotpot.log").write_text(buffer.getvalue())
    metrics = ast.literal_eval(buffer.getvalue().strip().splitlines()[-1])
    dump(output / "per_question.json", details)
    return dict(metrics, metric="official HotpotQA v1 (fractions 0..1)",
                supporting_fact_predictions=len(supporting), count=len(questions), complete=True,
                official_evaluator_sha256=digest(script))


class LocalJudge:
    """Local HF generation with the official Qwen3-32B judge sampling defaults."""
    def __init__(self, model_path, max_tokens=4096, max_context=32768, temperature=0.7,
                 top_p=0.8, top_k=20, seed=42):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("judge model must be an existing local directory")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.max_context, self.max_tokens = max_context, max_tokens
        self.sampling = dict(do_sample=temperature > 0)
        if temperature > 0:
            self.sampling.update(temperature=temperature, top_p=top_p, top_k=top_k)

    def __call__(self, prompt):
        import torch
        # Match upstream user-only chat; preserve the local model's default thinking template.
        text = self.tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(next(self.model.parameters()).device)
        length = inputs["input_ids"].shape[1]
        if length + self.max_tokens > self.max_context:
            raise ValueError("judge context budget exceeded (no silent truncation)")
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id), **self.sampling)
        if ids.shape[1] - length >= self.max_tokens and ids[0, -1].item() != self.tokenizer.eos_token_id:
            raise ValueError("judge generation exhausted token budget")
        return self.tokenizer.decode(ids[0, length:], skip_special_tokens=True)


def retrieval_metrics(questions, predictions):
    recalls = {"evidence": [], "gold": []}
    per_query = {}
    for row in questions:
        prediction = predictions.get(row["id"], {})
        retrieved = set(prediction.get("retrieved_docids", []))
        values = {}
        for kind in recalls:
            relevant = set(row[kind + "_docs"])
            if not relevant:
                raise ValueError(f"question lacks {kind} qrels: {row['id']}")
            score = len(retrieved & relevant) / len(relevant)
            recalls[kind].append(score)
            values[kind + "_recall"] = score
        per_query[row["id"]] = values
    return {kind + "_recall": sum(values) / len(values) for kind, values in recalls.items()}, per_query


def score_browsecomp(root, questions, predictions, output, judge=None, retrieval_only=False):
    verify_official(root, "browsecomp-plus")
    script = Path(root) / "official/scripts_evaluation/evaluate_run.py"
    funcs = load_official_functions(script, {"GRADER_TEMPLATE", "create_judge_prompt", "parse_judge_response",
        "extract_citations_from_response", "compute_citation_metrics", "calib_err", "calculate_calibration_error"})
    import numpy as np
    funcs["calib_err"].__globals__["np"] = np
    recalls, per_query = retrieval_metrics(questions, predictions)
    details = []
    errors = correct = 0
    confidences, correctness, citation_scores = [], [], []
    counts = Counter()
    exports = output / "official_runs"
    exports.mkdir()
    with (output / "judge.jsonl").open("x", encoding="utf-8") as log:
        for index, row in enumerate(questions):
            prediction = predictions.get(row["id"], {})
            response = prediction.get("final_response", "")
            completed = prediction.get("status") == "completed" and bool(response.strip())
            tool_counts = prediction.get("tool_call_counts", {})
            counts.update(tool_counts)
            dump(exports / f"{index:06}.json", dict(query_id=row["id"], status=prediction.get("status", "missing"),
                 retrieved_docids=prediction.get("retrieved_docids", []), tool_call_counts=tool_counts,
                 result=[dict(type="output_text", output=response)]))
            result = dict(id=row["id"], completed=completed, correct=False, **per_query[row["id"]])
            if completed and not retrieval_only:
                if judge is None:
                    raise ValueError("full BrowseComp-Plus scoring requires a local judge model")
                prompt = funcs["create_judge_prompt"](row["query"], response, row["answer"])
                try:
                    raw = judge(prompt)
                    parsed = funcs["parse_judge_response"](raw)
                    result.update(judge_response=raw, judge_result=parsed, correct=parsed["correct"] is True)
                    if parsed["parse_error"]:
                        errors += 1
                except Exception as error:
                    result.update(judge_error=str(error))
                    errors += 1
            if completed:
                cited = funcs["extract_citations_from_response"](response)
                citation = funcs["compute_citation_metrics"](cited, row["evidence_docs"])
                result["citations"] = dict(cited_docids=cited, **citation)
                if cited:
                    citation_scores.append(citation)
            parsed = result.get("judge_result", {})
            if not parsed.get("parse_error", True) and parsed.get("confidence") is not None:
                confidences.append(parsed["confidence"])
                correctness.append(parsed["correct"])
            correct += result["correct"]
            details.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[score] {index + 1}/{len(questions)}; judge_errors={errors}", flush=True)
    metrics = dict(recalls, count=len(questions), judge_errors=errors, complete=errors == 0,
                   score_mode="retrieval-only" if retrieval_only else "official prompt/parser with local HF judge",
                   metric="BrowseComp-Plus", official_evaluator_sha256=digest(script),
                   avg_tool_stats={key: value / len(questions) for key, value in counts.items()})
    if not retrieval_only:
        metrics.update(accuracy=correct / len(questions), **{"Accuracy (%)": round(100 * correct / len(questions), 2),
                                                           "Recall (%)": round(100 * recalls["evidence_recall"], 2)})
    metrics["citation_coverage"] = len(citation_scores) / len(questions)
    for name in ("precision", "recall"):
        metrics["citation_" + name] = (sum(item[name] for item in citation_scores) / len(citation_scores)
                                              if citation_scores else 0.0)
    if not retrieval_only:
        metrics["calibration_sample_count"] = len(confidences)
        metrics["calibration_available"] = len(confidences) >= 100
        # Preserve upstream's 0.0 sentinel for <100 valid confidence values; flag it explicitly.
        metrics["Calibration Error (%)"] = round(funcs["calculate_calibration_error"](confidences, correctness), 2) if len(confidences) >= 100 else 0.0
    dump(output / "per_question.json", details)
    return metrics
