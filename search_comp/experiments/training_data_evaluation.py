"""Generate fresh answers for questions from a prepared SFT split."""
from collections import Counter, defaultdict
import hashlib
import json
import re
import time
from types import SimpleNamespace

from search_comp.data.agent_prompt import TOOLS_RE, build_system_prompt, render_system
from search_comp.data.openseeker import VERSION
from search_comp.evaluation.answer_metrics import answer_metrics
from search_comp.evaluation.checkpoints import load_evaluation_model, resolve_checkpoint
from search_comp.evaluation.tool_agent import ModelGenerator, decoding_options, run_agent
from search_comp.offline.index import LocalIndex
from search_comp.offline.tools import LocalTools
from search_comp.utils.progress import CompletionProgress
from .benchmarks import resolve_target
from .runtime import configure, digest, job, path, read_json, stamp, write_json


SPLIT_KEYS = {"train": "train_data_path", "validation": "val_data_path", "test": "test_data_path"}


def read_samples(filename, limit, allow_visit):
    """Retain only the prompt, question and scoring metadata from each selected row."""
    samples, ids = [], set()
    fingerprint = hashlib.sha256()
    with filename.open("rb") as stream:
        for line in stream:
            fingerprint.update(line)
            if not line.strip() or (limit > 0 and len(samples) >= limit):
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if row.get("protocol") != VERSION or [m["role"] for m in messages[:2]] != ["system", "user"]:
                raise ValueError("评测数据须为当前协议的 prepared JSONL")
            sample_id, answer, question = row["id"], row["answer"], messages[1]["content"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError(f"评测记录 ID 为空或重复: {sample_id!r}")
            if not all(isinstance(value, str) and value.strip() for value in (answer, question)):
                raise ValueError(f"评测记录缺少问题或比较答案: {sample_id}")
            ids.add(sample_id)
            system = render_system(messages[0]["content"], row.get("tools"))
            tool_block = TOOLS_RE.search(system)
            visit = bool(allow_visit and tool_block and re.search(r'"name"\s*:\s*"visit"', tool_block[1]))
            system = build_system_prompt(system=system, local=True, allow_visit=visit)
            samples.append(dict(id=sample_id, query=question, answer=answer, system=system,
                                allow_visit=visit, source=row.get("source"),
                                answer_source=row.get("answer_source") or "reference",
                                correctness=row.get("correctness")))
    if not samples:
        raise ValueError(f"所选评测划分为空: {filename}")
    return samples, fingerprint.hexdigest()


def evaluate_training_data(*, method, training_run, checkpoint, dataset_dir, split,
                           index_dir, run_name, generation, environment):
    configure(environment)
    if method not in {"plain", "beacon"} or split not in SPLIT_KEYS:
        raise ValueError("METHOD 须为 plain/beacon，SPLIT 须为 train/validation/test")
    args = SimpleNamespace(**generation)
    positive = ("max_context", "max_new_tokens", "max_turns", "max_tool_tokens",
                "tool_count", "topk", "snippet_tokens", "read_tokens")
    if (any(getattr(args, name) < 1 for name in positive)
            or args.tool_call_retries < 0 or args.limit == 0 or args.limit < -1):
        raise ValueError("生成与读取预算须为正数，重试次数须非负，LIMIT 须为 -1 或正数")
    decoding = decoding_options(args, method == "beacon")
    dataset_name, run, model, kind, origin = resolve_target(method, training_run, None, checkpoint)
    spec = read_json(run / "experiment.json")
    key = SPLIT_KEYS[split]
    original = spec["training_config"].get(key)
    if not dataset_dir and not original:
        raise ValueError(f"训练记录缺少 {split} 划分；请使用已有划分或填写 DATASET_DIR")
    filename = path(dataset_dir) / f"{split}.jsonl" if dataset_dir else path(original)
    samples, fingerprint = read_samples(filename, args.limit, args.allow_visit)
    training_fingerprints = spec["data_fingerprints"]
    if original and filename == path(original) and fingerprint != training_fingerprints[key]:
        raise ValueError("原实验数据指纹已改变；请恢复原文件")

    index_file = path(index_dir) / "index.sqlite"
    output = run / "evaluation" / "training-data" / stamp(run_name)
    settings = dict(method=method, dataset_name=dataset_name, checkpoint=checkpoint or None,
                    split=split, dataset_file=str(filename), dataset_sha256=fingerprint,
                    matches_training_train_sha256=fingerprint == training_fingerprints["train_data_path"],
                    index_file=str(index_file), generation=generation)
    with job(output, "evaluate_training_data", settings), LocalIndex(index_file, args.verify_index) as index:
        model_path, kind = resolve_checkpoint(run, model, kind)
        loaded, tokenizer = load_evaluation_model(model_path, kind, origin, run)
        generate = ModelGenerator(loaded, tokenizer, kind == "beacon", **decoding)
        generate.metadata.update(checkpoint=model_path, model_origin=origin)
        progress = CompletionProgress((), len(samples), "training data evaluation")
        predictions = output / "predictions.jsonl"
        counts = Counter()
        totals = defaultdict(Counter)
        with predictions.open("x", encoding="utf-8") as stream:
            for number, sample in enumerate(samples):
                started = time.monotonic()
                tools = LocalTools(index, tokenizer, args.topk, args.snippet_tokens, args.read_tokens,
                                   sample["allow_visit"], args.tool_count)
                messages = [dict(role="system", content=sample["system"]),
                            dict(role="user", content=sample["query"])]
                result = run_agent(messages, tokenizer, generate, tools, args.max_turns,
                                   args.max_context, args.max_new_tokens, args.max_tool_tokens,
                                   max_call_retries=args.tool_call_retries)
                result.update({k: v for k, v in sample.items() if k != "system"})
                scores = answer_metrics([sample], [result])
                group = totals[sample["answer_source"]]
                group.update(scores)
                group["count"] += 1
                counts[result["status"]] += 1
                result.update(sample_index=number, seconds=time.monotonic() - started,
                              retrieved_docids=sorted(tools.retrieved), visited_docids=sorted(tools.visited),
                              retrieval_trace=tools.trace, tool_call_counts=dict(tools.counts), **scores)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                progress.record(result)
        by_source = {name: dict(count=group["count"], em=group["em"] / group["count"],
                               f1=group["f1"] / group["count"]) for name, group in sorted(totals.items())}
        write_json(output / "summary.metrics.json", dict(count=len(samples), statuses=dict(counts),
                   completed_rate=counts["completed"] / len(samples), by_answer_source=by_source))
        write_json(output / "evaluation.json", dict(settings, index=index.manifest, model_path=model_path,
                   predictions=str(predictions), predictions_sha256=digest(predictions)))
    return output
