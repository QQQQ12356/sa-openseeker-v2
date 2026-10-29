"""Prepare, index, run and score fixed-corpus benchmarks with no network during evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from search_comp.offline.assets import DEV_URL, WIKI_URL, digest, download_assets, dump, enforce_offline, verify_official
from search_comp.offline.data import prepare, records
from search_comp.offline.index import LocalIndex, build_index
from search_comp.offline.scoring import LocalJudge, load_predictions, score_browsecomp, score_hotpot
from search_comp.offline.tools import BCP_INSTRUCTION, HOTPOT_INSTRUCTION, LocalTools
from search_comp.workflows.benchmark_openseeker import resolve_checkpoint
from search_comp.workflows.evaluate_openseeker import worker_devices

ROOT = Path(__file__).resolve().parents[2]


def prepared_questions(root, benchmark, limit=-1):
    verify_official(root, benchmark)
    folder = root / "prepared"
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["benchmark"] != benchmark or digest(folder / "questions.jsonl") != manifest["questions_sha256"]:
        raise ValueError("prepared question checksum/benchmark changed")
    rows = list(records(folder / "questions.jsonl"))
    if len(rows) != manifest["questions"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("prepared question count/IDs invalid")
    return rows[:limit] if limit > 0 else rows, manifest


def prepare_phase(args):
    data = args.data or args.root / ("raw/queries" if args.benchmark == "browsecomp-plus" else "raw/hotpot_dev_fullwiki_v1.json")
    corpus = args.corpus or args.root / ("raw/corpus" if args.benchmark == "browsecomp-plus" else "raw/wiki-abstracts.tar.bz2")
    encrypted = args.benchmark == "browsecomp-plus" and not args.decrypted
    if (args.root / "prepared").exists() and args.action == "all":
        _, manifest = prepared_questions(args.root, args.benchmark)
        if manifest["encrypted"] != encrypted or manifest["corpus_limit"] != args.corpus_limit:
            raise ValueError("preparation options changed; use a new ROOT")
        for name, value in (("data", args.data), ("corpus", args.corpus)):
            if value and str(value) != manifest.get("source_" + name + "_path"):
                raise ValueError(f"{name} source changed; use a new ROOT")
        return manifest
    result = prepare(args.benchmark, args.root, data, corpus, encrypted, args.corpus_limit)
    return result


def verify_index(args, manifest):
    with LocalIndex(args.root / "index.sqlite", args.verify_index) as index:
        if index.manifest["benchmark"] != args.benchmark or index.manifest["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("index and dataset have different corpus/benchmark provenance")
        if index.manifest["title_weight"] != args.title_weight:
            raise ValueError("TITLE_WEIGHT differs from the built index; rebuild in a new ROOT")
        return index.manifest


def run_phase(args, output, rows, manifest):
    if args.num_workers > 1:
        return run_parallel(args, output, rows, manifest)
    model_path, kind = resolve_checkpoint(args.run_dir, args.model, args.kind)
    from search_comp.data.agent_prompt import build_system_prompt, resolve_system_prompt
    original_system, prompt_source = resolve_system_prompt(getattr(args, 'prompt_data', None), args.run_dir)
    system = build_system_prompt(system=original_system, local=True, allow_visit=args.allow_visit,
        answer_instruction=HOTPOT_INSTRUCTION if args.benchmark == "hotpotqa" else BCP_INSTRUCTION)
    verify_index(args, manifest)
    from search_comp.evaluation.tool_agent import ModelGenerator, run_agent
    from search_comp.models.model_loader import load_beacon_model, load_tokenizer
    from search_comp.models.plain_qwen3 import load_sft_model

    print(f"[offline] loading {kind} checkpoint={model_path}; questions={len(rows)}", flush=True)
    tokenizer = load_tokenizer(model_path)
    model = (load_beacon_model if kind == "beacon" else load_sft_model)(model_path)
    model.eval()
    generate = ModelGenerator(model, tokenizer, kind == "beacon")
    statuses = Counter()
    start = time.monotonic()
    with LocalIndex(args.root / "index.sqlite") as index, (output / "predictions.jsonl").open("x", encoding="utf-8") as target:
        tools = LocalTools(index, tokenizer, args.topk, args.snippet_tokens, args.read_tokens,
                           args.allow_visit, args.tool_count)
        for position, row in enumerate(rows):
            tools.reset()
            tick = time.monotonic()
            # Only question text, no reference, qrels, supporting facts or source context.
            messages = [dict(role="system", content=system), dict(role="user", content=row["query"])]
            result = run_agent(messages, tokenizer, generate, tools, args.max_turns,
                               args.max_context, args.max_new_tokens, args.max_tool_tokens)
            result.update(id=row["id"], sample_index=args.shard_index + position * args.shard_count,
                          prompt_source=prompt_source,
                          query=row["query"], final_response=result["prediction"],
                          seconds=time.monotonic() - tick, retrieved_docids=sorted(tools.retrieved),
                          visited_docids=sorted(tools.visited), retrieval_trace=tools.trace,
                          tool_call_counts=dict(tools.counts))
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            statuses[result["status"]] += 1
            print(f"[offline] done={position + 1}/{len(rows)} status={result['status']} "
                  f"sample={result['seconds']:.1f}s elapsed={time.monotonic() - start:.1f}s", flush=True)
    dump(output / "generation.metrics.json", dict(statuses=dict(statuses), count=len(rows),
                                                  seconds=time.monotonic() - start))
    del generate, model, tokenizer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output / "predictions.jsonl"


def worker_command(args, output, rank, count):
    """Children only generate: preparation, merging and scoring belong to the parent."""
    model, kind = resolve_checkpoint(args.run_dir, args.model, args.kind)
    command = [sys.executable, str(ROOT / "scripts/py/52_offline_benchmark.py"), "run"]
    excluded = {"action", "output_dir", "num_workers", "shard_index", "shard_count",
                "model", "kind", "judge_model", "predictions", "dry_run"}
    for key, value in vars(args).items():
        if key in excluded or value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    command.extend(["--model", str(model), "--kind", kind, "--output-dir", str(output),
                    "--num-workers", "1", "--shard-index", str(rank), "--shard-count", str(count)])
    return command


def merge_worker_predictions(paths, output, rows, manifest):
    """Validate every shard and atomically publish predictions in original question order."""
    count = len(paths)
    statuses = Counter()

    def checked_rows(path, rank):
        metadata = json.loads((path.parent / "manifest.json").read_text())
        expected = rows[rank::count]
        if (metadata.get("status") != "complete"
                or metadata.get("questions_sha256") != manifest["questions_sha256"]
                or metadata.get("selected_ids") != [row["id"] for row in expected]
                or metadata.get("predictions_sha256") != digest(path)):
            raise ValueError(f"worker provenance/checksum invalid: {path}")
        with path.open(encoding="utf-8") as source:
            for position, row in enumerate(expected):
                line = source.readline()
                if not line:
                    raise ValueError(f"worker missing predictions: {path}")
                prediction = json.loads(line)
                if (prediction.get("id") != row["id"] or prediction.get("query") != row["query"]
                        or prediction.get("sample_index") != rank + position * count):
                    raise ValueError(f"worker prediction order/ID/query invalid: {path}")
                yield prediction
            if source.read().strip():
                raise ValueError(f"worker has extra predictions: {path}")

    target = output / "predictions.jsonl"
    partial = output / "predictions.building.jsonl"
    try:
        with partial.open("x", encoding="utf-8") as stream:
            for prediction in heapq.merge(*(checked_rows(path, rank) for rank, path in enumerate(paths)),
                                          key=lambda row: row["sample_index"]):
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                statuses[prediction["status"]] += 1
        if target.exists():
            raise FileExistsError(target)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target, statuses


def forward_worker_progress(readers, buffers):
    for rank, reader in enumerate(readers):
        buffers[rank] += reader.read()
        lines = buffers[rank].split("\n")
        buffers[rank] = lines.pop()
        for line in lines:
            if line.startswith("[offline"):
                print(f"[worker {rank}] {line}", flush=True)


def run_parallel(args, output, rows, manifest):
    devices = worker_devices(args.num_workers, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    # Avoid loading unused replicas when a smoke test has fewer questions than GPUs.
    devices = devices[:len(rows)]
    if not devices:
        raise ValueError("parallel evaluation requires at least one question")
    directory = output / "workers"
    directory.mkdir()
    processes, paths = [], []
    start = time.monotonic()
    with ExitStack() as stack:
        readers, buffers = [], []
        try:
            for rank, device in enumerate(devices):
                folder = directory / f"worker_{rank}"
                paths.append(folder / "predictions.jsonl")
                log_path = directory / f"worker_{rank}.log"
                log = stack.enter_context(log_path.open("x", encoding="utf-8"))
                command = worker_command(args, folder, rank, len(devices))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                print(f"[offline worker {rank}] GPU={device}; log={log_path}", flush=True)
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  cwd=ROOT))
                readers.append(stack.enter_context(log_path.open(encoding="utf-8", errors="replace")))
                buffers.append("")
            pending = list(processes)
            heartbeat = time.monotonic()
            while pending:
                forward_worker_progress(readers, buffers)
                for process in pending[:]:
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"offline worker {processes.index(process)} exited with code {code}; see {directory}")
                    pending.remove(process)
                if time.monotonic() - heartbeat >= 15:
                    print(f"[offline] active workers={len(pending)}/{len(processes)}; "
                          f"elapsed={time.monotonic() - start:.1f}s; logs={directory}", flush=True)
                    heartbeat = time.monotonic()
                if pending:
                    time.sleep(0.1)
            forward_worker_progress(readers, buffers)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    predictions, statuses = merge_worker_predictions(paths, output, rows, manifest)
    dump(output / "generation.metrics.json", dict(statuses=dict(statuses), count=len(rows),
         seconds=time.monotonic() - start, num_workers=len(devices), devices=devices))
    return predictions


def score_phase(args, output, rows, manifest, prediction_path):
    predictions = load_predictions(prediction_path, rows)
    source_manifest = prediction_path.parent / "manifest.json"
    if source_manifest.is_file():
        source = json.loads(source_manifest.read_text())
        if source.get("questions_sha256") != manifest["questions_sha256"]:
            raise ValueError("prediction provenance differs from prepared questions")
        if source.get("selected_ids") != [row["id"] for row in rows]:
            raise ValueError("prediction selection differs; set the same --limit as the original run")
    if args.benchmark == "hotpotqa":
        summary = score_hotpot(args.root, rows, predictions, output)
    else:
        judge = None
        if not args.retrieval_only:
            if not args.judge_model or not args.judge_model.is_dir():
                raise ValueError("BrowseComp-Plus answer scoring needs --judge-model pointing to local Qwen3-32B (or an explicitly recorded alternative)")
            # Lazy load: failed/empty generation needs no judge invocation.
            loaded = []
            def judge(prompt):
                if not loaded:
                    loaded.append(LocalJudge(args.judge_model, args.judge_max_tokens, args.judge_max_context,
                                             args.judge_temperature, args.judge_top_p, args.judge_top_k, args.seed))
                return loaded[0](prompt)
        summary = score_browsecomp(args.root, rows, predictions, output, judge, args.retrieval_only)
    summary.update(benchmark=args.benchmark, prediction_count=len(predictions),
                   missing_predictions=len(rows) - len(predictions),
                   questions_sha256=manifest["questions_sha256"], corpus_sha256=manifest["corpus_sha256"],
                   subset=args.limit > 0 or not manifest["canonical_corpus"],
                   corpus_scope=manifest["corpus_scope"], offline=True)
    dump(output / "summary.metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["complete"]:
        raise RuntimeError("judge errors/unknown labels remain; inspect judge.jsonl; failed judgements remain in denominator")
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "prepare", "index", "run", "score", "all"])
    parser.add_argument("--benchmark", choices=["browsecomp-plus", "hotpotqa"], required=True)
    parser.add_argument("--root", type=Path, help="benchmark assets, prepared corpus and persistent index")
    parser.add_argument("--data", type=Path, help="local official question JSON/JSONL/Parquet or directory")
    parser.add_argument("--corpus", type=Path, help="local independent corpus; never dev candidate paragraphs")
    parser.add_argument("--decrypted", action="store_true", help="BrowseComp-Plus questions are already decrypted")
    parser.add_argument("--corpus-limit", type=int, help="SMOKE ONLY: index the first N documents")
    parser.add_argument("--official-only", action="store_true", help="download only evaluator/license/qrels assets")
    parser.add_argument("--corpus-revision", help="Hugging Face corpus revision for explicit download")
    parser.add_argument("--dev-url", default=DEV_URL)
    parser.add_argument("--wiki-url", default=WIKI_URL, help="Hotpot abstracts mirror; official MD5 is still required")
    parser.add_argument("--title-weight", type=float, default=1.0)
    parser.add_argument("--verify-index", action="store_true", help="SHA256 scan of full index before run")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prompt-data", type=Path,
                        help="training JSONL providing original system/tools; default: RUN_DIR train_data_path or Polar profile")
    parser.add_argument("--model")
    parser.add_argument("--kind", choices=["plain", "beacon"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="one full model replica per GPU; shard questions across explicit CUDA_VISIBLE_DEVICES")
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--snippet-tokens", type=int, default=512)
    parser.add_argument("--allow-visit", action="store_true")
    parser.add_argument("--read-tokens", type=int, default=4096)
    parser.add_argument("--tool-count", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-tool-tokens", type=int, default=32768)
    parser.add_argument("--judge-model", type=Path)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-context", type=int, default=32768)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.8)
    parser.add_argument("--judge-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.root = args.root or ROOT / "outputs/data/benchmarks" / args.benchmark
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    positive = ("topk", "snippet_tokens", "read_tokens", "tool_count", "max_turns", "max_context",
                "max_new_tokens", "max_tool_tokens", "judge_max_tokens", "judge_max_context", "judge_top_k")
    if any(getattr(args, name) < 1 for name in positive) or args.limit == 0 or args.limit < -1:
        parser.error("budgets must be positive; limit must be -1 or positive")
    if args.num_workers < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("num-workers must be positive; invalid shard index/count")
    if args.shard_count > 1 and (args.num_workers > 1 or args.action != "run"):
        parser.error("worker shards require action=run and num-workers=1")
    if args.num_workers > 1 and args.action in {"run", "all"}:
        try:
            worker_devices(args.num_workers, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        except ValueError as error:
            parser.error(str(error))
    if args.corpus_limit is not None and args.corpus_limit < 1:
        parser.error("corpus-limit must be positive")
    if (not math.isfinite(args.title_weight) or args.title_weight <= 0 or
            not math.isfinite(args.judge_temperature) or args.judge_temperature < 0 or
            not 0 < args.judge_top_p <= 1):
        parser.error("invalid BM25 weight or judge sampling parameters")
    if args.action == "score" and not args.predictions:
        parser.error("score requires --predictions")
    if args.action in {"all", "score"} and args.benchmark == "browsecomp-plus" and not args.retrieval_only and not args.judge_model and not args.dry_run:
        parser.error("BrowseComp-Plus needs --judge-model; use --retrieval-only only for retrieval diagnostics")
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2))
        return
    if args.action in {"run", "all"}:
        resolve_checkpoint(args.run_dir, args.model, args.kind)
    if args.action in {"all", "score"} and args.judge_model and not args.judge_model.is_dir():
        parser.error("judge-model must be an existing local model directory")
    if args.action == "download":
        print(json.dumps(download_assets(args.benchmark, args.root, args.corpus_revision, args.official_only,
                                        args.dev_url, args.wiki_url), indent=2))
        return
    enforce_offline()
    if args.action in {"prepare", "all"}:
        print(json.dumps(prepare_phase(args), indent=2))
        if args.action == "prepare":
            return
    if args.action == "index" or args.action == "all" and not (args.root / "index.sqlite").exists():
        prepared_questions(args.root, args.benchmark)
        print(json.dumps(build_index(args.root, args.title_weight), indent=2))
        if args.action == "index":
            return
    rows, manifest = prepared_questions(args.root, args.benchmark, args.limit)
    rows = rows[args.shard_index::args.shard_count]
    if args.action in {"run", "all"}:
        # Resolve before creating a result directory, and before any large model allocation.
        model, kind = resolve_checkpoint(args.run_dir, args.model, args.kind)
        index_metadata = verify_index(args, manifest)
    else:
        model, kind, index_metadata = None, None, None
    if args.judge_model and not args.judge_model.is_dir():
        parser.error("judge-model must be a local model directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ((args.run_dir / "evaluation" if args.run_dir else ROOT / "outputs/results") /
                                 f"offline_{args.benchmark}_{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    run_manifest = dict(benchmark=args.benchmark, offline=True, model=model, kind=kind,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        created_at=datetime.now(timezone.utc).isoformat(), selected_ids=[row["id"] for row in rows],
        questions_sha256=manifest["questions_sha256"], prepared_manifest=manifest, index=index_metadata,
        judge_model=str(args.judge_model) if args.judge_model else None,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        code_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                     [Path(__file__), *sorted((ROOT / "search_comp/offline").glob("*.py"))]}, status="running")
    dump(output / "manifest.json", run_manifest)
    try:
        predictions = run_phase(args, output, rows, manifest) if args.action in {"run", "all"} else args.predictions
        run_manifest["predictions_sha256"] = digest(predictions)
        source_manifest = predictions.parent / "manifest.json"
        if source_manifest.is_file() and source_manifest.parent != output:
            run_manifest["prediction_source_manifest_sha256"] = digest(source_manifest)
        if args.action in {"all", "score"}:
            score_phase(args, output, rows, manifest, predictions)
        run_manifest["status"] = "complete"
    except BaseException:
        run_manifest["status"] = "failed"
        raise
    finally:
        dump(output / "manifest.json", run_manifest)
    print(f"[offline] artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
