"""Run pinned upstream OpenSeeker generation and judging without changing legacy eval."""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

from search_comp.evaluation.benchmark_data import prepare_browsecomp, read_questions, sha256
from search_comp.utils.progress import CompletionProgress

OFFICIAL_URL = "https://github.com/PolarSeeker/OpenSeeker.git"
OFFICIAL_REVISION = "e3dec1f727d87c5eaa93506e4882c67237647098"
ROOT = Path(__file__).resolve().parents[2]


def ensure_official(path):
    """Fetch into a new directory only; never reset an existing user checkout."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", OFFICIAL_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", OFFICIAL_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if revision != OFFICIAL_REVISION or dirty:
        raise ValueError(f"official checkout must be clean at {OFFICIAL_REVISION}; use a new --official-dir")
    for name in ("eval/eval.py", "eval/prompt.py", "eval/generate_answer_v2.py", "src/tools/visit.py"):
        if not (path / name).is_file():
            raise ValueError(f"missing official source: {name}")
    return revision


def require_environment(mode, generate=True):
    names = ["SCORER_URLS", "SCORER_API_KEY", "SCORER_MODEL_NAME"]
    if generate:
        names += ["SUMMARY_API_URL", "SUMMARY_API_KEY", "SUMMARY_MODEL_NAME", "JINA_API_KEYS"]
        provider = os.environ.get("SEARCH_PROVIDER", "serper")
        if provider not in {"serper", "tavily"}:
            raise ValueError("SEARCH_PROVIDER must be serper or tavily")
        names += ["TAVILY_API_KEY" if provider == "tavily" else
                  ("SERPER_API_KEY" if os.environ.get("SERPER_API_KEY") else "SERPER_KEY_ID")]
        if mode != "native":
            names += ["OPENSEEKER_BASE_URL", "OPENSEEKER_MODEL"]
        if mode == "official-v2":
            names += ["E2B_API_KEY"]
    missing = [name for name in names if not os.environ.get(name, "").strip()
               or os.environ[name].startswith("YOUR_")]
    if missing:
        raise ValueError("configure environment variables: " + ", ".join(missing))


def resolve_checkpoint(run_dir, model, kind):
    spec = {}
    if run_dir:
        spec = json.loads((run_dir / "experiment.json").read_text())
        if not model:
            summary = run_dir / "models/model/train_summary.json"
            model = json.loads(summary.read_text())["final_model_path"]
    kind = kind or spec.get("kind")
    if not model:
        raise ValueError("set --run-dir or --model for checkpoint inference")
    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    if (path / "adapter_config.json").exists():
        raise ValueError("merge the LoRA adapter first (scripts/sh/plain/29_plain_merge_adapter.sh or scripts/sh/beacon/35_beacon_merge_adapter.sh)")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    text_config = config.get("text_config") or {}
    is_beacon = config.get("enable_beacon") or text_config.get("enable_beacon")
    kind = kind or ("beacon" if is_beacon else "plain")
    if is_beacon and kind != "beacon":
        raise ValueError("checkpoint enables Beacon; use kind=beacon and native generation")
    if kind not in {"plain", "beacon"}:
        raise ValueError("checkpoint kind must be plain or beacon")
    return str(path), kind


def run_logged(command, cwd, log, progress_path=None, total=None, label=None, key="query", item_type=None):
    print("[benchmark] " + shlex.join(map(str, command)), flush=True)
    with log.open("x", encoding="utf-8") as target:
        lock = threading.Lock()

        def emit(line):
            with lock:
                print(line, flush=True)
                target.write(line + "\n")
                target.flush()

        progress = (CompletionProgress([progress_path], total, label, key=key,
                                       item_type=item_type, emit=emit) if progress_path else None)
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            with progress.watch() if progress else nullcontext():
                for line in process.stdout:
                    emit(line.rstrip("\n"))
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def generation_command(args, data, output):
    script = "generate_answer_v2.py" if args.mode == "official-v2" else "generate_answer.py"
    return [sys.executable, str(args.official_dir / "eval" / script),
            "--dataset_path", str(data), "--out_dir", str(output),
            "--max_worker", str(args.workers), "--max_tokens", str(args.max_tokens),
            "--tool_count_max", str(args.tool_count), "--max_retry_rounds", "1"]


def boxed_instruction(official_dir):
    tree = ast.parse((official_dir / "eval/generate_answer_v2.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == "BOX_FORMAT_INSTRUCTION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("upstream v2 boxed instruction not found")


def align_results(rows, raw_path, destination, suffix=""):
    """Select the latest attempt per query, retain missing questions in denominator."""
    expected = {row["query"] + suffix: row for row in rows}
    found = {}
    attempts = 0
    if raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            query = result.get("query")
            if query not in expected:
                raise ValueError("prediction query does not belong to this benchmark/protocol")
            if result.get("answer") != expected[query]["answer"]:
                raise ValueError("prediction reference differs from benchmark")
            if not isinstance(result.get("final_response"), str):
                raise ValueError("prediction final_response must be text")
            found[query] = result
            attempts += 1
    missing = empty = 0
    with destination.open("x", encoding="utf-8") as target:
        for query, row in expected.items():
            result = found.get(query)
            if result is None:
                missing += 1
                result = dict(row, query=query, final_response="", status="generation_missing")
            empty += not bool(result["final_response"].strip())
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
    return dict(expected=len(rows), unique_predictions=len(found), attempts=attempts,
                missing_predictions=missing, empty_predictions=empty)


def audit_scores(path, expected):
    labels = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item.get("type") == "item":
                index, score = item["item_index"], item.get("score")
                if type(index) is not int or not 0 <= index < expected:
                    raise ValueError("invalid judge item_index")
                if type(score) in (int, float) and score in (0, 1):
                    labels[index] = score
    correct = sum(labels.values())
    return dict(judged=len(labels), judge_unknown_or_error=expected - len(labels),
                official_accuracy=correct / len(labels) if labels else None,
                accuracy_all_questions=correct / expected, complete=len(labels) == expected)


def native_generate(args, rows, output):
    """Use the shared SFT/native protocol with upstream online search/visit."""
    sys.path.insert(0, str(args.official_dir / "src"))
    from llm_tool_openseeker import Search, Visit
    from search_comp.data.agent_prompt import initial_messages, resolve_system_prompt
    from search_comp.evaluation.tool_agent import ModelGenerator, Registry, run_agent, decoding_options
    from search_comp.models.model_loader import load_beacon_model, load_tokenizer
    from search_comp.models.plain_qwen3 import load_sft_model

    print(f"[benchmark native generation] done=0/{len(rows)}", flush=True)
    system, prompt_source = resolve_system_prompt(getattr(args, 'prompt_data', None), args.run_dir)
    model_path, kind = resolve_checkpoint(args.run_dir, args.model, args.kind)
    generation_options = decoding_options(args, kind == 'beacon')
    tokenizer = load_tokenizer(model_path)
    model = (load_beacon_model if kind == "beacon" else load_sft_model)(model_path)
    model.eval()
    generate = ModelGenerator(model, tokenizer, kind == "beacon", **generation_options)
    search, visit = Search(), Visit()
    tools = Registry({"search": lambda **kw: search.call(kw), "visit": lambda **kw: visit.call(kw)})
    with output.open("x", encoding="utf-8") as target:
        for index, row in enumerate(rows):
            start = time.monotonic()
            messages = initial_messages(row["query"], system=system)
            result = run_agent(messages, tokenizer, generate, tools, args.max_turns,
                               args.max_context, args.max_tokens, args.max_tool_tokens)
            result.update(row, final_response=result["prediction"], prompt_source=prompt_source,
                          elapsed_seconds=time.monotonic() - start,
                          tool_calls=sum(len(e.get("calls", [])) for e in result["events"]))
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"[benchmark] native {index + 1}/{len(rows)} status={result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, default=ROOT / "outputs/vendor/OpenSeeker")
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/data/benchmarks/browsecomp.jsonl")
    parser.add_argument("--csv", type=Path, help="local encrypted official BrowseComp CSV")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", choices=["official-v1", "official-v2", "native"], default="native")
    parser.add_argument("--benchmark", default="BrowseComp", help="record the actual benchmark name for custom data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions", type=Path, help="judge existing query/answer/final_response JSONL")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prompt-data", type=Path,
                        help="training JSONL supplying original system/tools; overrides RUN_DIR training data")
    parser.add_argument("--model")
    parser.add_argument("--kind", choices=["plain", "beacon"])
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--judge-workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--tool-count", type=int, default=200)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-tool-tokens", type=int, default=32768)
    from search_comp.evaluation.tool_agent import add_decoding_arguments, decoding_options
    add_decoding_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        generation_options = decoding_options(args, args.kind == 'beacon')
    except ValueError as error:
        parser.error(str(error))
    if args.mode != 'native' and generation_options:
        parser.error('decoding overrides are supported in native mode only')
    for name in ("workers", "judge_workers", "max_tokens", "tool_count", "max_context", "max_turns", "max_tool_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit != -1 and args.limit < 1:
        parser.error("--limit must be -1 (full benchmark) or positive")
    for name in ("official_dir", "data", "output_dir", "predictions", "run_dir", "csv", "prompt_data"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.mode != "native" and (args.run_dir or args.model or args.kind == "beacon"):
        parser.error("official modes use OPENSEEKER_BASE_URL/MODEL; serve Plain via 48, use --mode native for Beacon")
    if args.mode != "native" and args.prompt_data:
        parser.error("--prompt-data is for native mode; official modes retain upstream prompts")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ((args.run_dir / "evaluation" if args.run_dir else ROOT / "outputs/results")
                                 / f"benchmark_{args.mode}_{stamp}")
    if args.dry_run:
        command = generation_command(args, args.data, output) if args.mode != "native" else [
            "native_generate", "--run-dir", str(args.run_dir), "--model", str(args.model)]
        if args.predictions:
            command = ["import_predictions", str(args.predictions)]
        print(json.dumps(dict(revision=OFFICIAL_REVISION, mode=args.mode, data=str(args.data),
                              output=str(output), prompt_data=str(args.prompt_data) if args.prompt_data else None,
                              generation=command,
                              decoding=dict(mode=args.decoding, repetition_penalty=args.repetition_penalty,
                                            temperature=args.temperature, top_p=args.top_p,
                                            generation_seed=args.generation_seed)), indent=2))
        return
    if output.exists() and not args.prepare_only:
        parser.error("output directory exists; use a new --output-dir (no overwriting or stale judge cache)")
    if not args.prepare_only:
        require_environment(args.mode, generate=args.predictions is None)
        if args.mode == "native" and not args.predictions:
            resolve_checkpoint(args.run_dir, args.model, args.kind)
    revision = ensure_official(args.official_dir)
    if not args.data.exists():
        if args.benchmark != "BrowseComp":
            parser.error("custom benchmark requires an existing --data JSONL")
        prepare_browsecomp(args.data, args.csv)
    rows = read_questions(args.data)
    if args.prepare_only:
        print(f"[benchmark] ready: {len(rows)} questions; official revision={revision}")
        return
    if args.limit > 0:
        rows = rows[:args.limit]
    output.mkdir(parents=True)
    dataset = output / "questions.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = dict(benchmark=args.benchmark, mode=args.mode, official_revision=revision,
                    official_url=OFFICIAL_URL, data_sha256=sha256(args.data),
                    selection="first N in source order" if args.limit > 0 else "full source order",
                    count=len(rows), judge_model=os.environ["SCORER_MODEL_NAME"],
                    search_provider=os.environ.get("SEARCH_PROVIDER", "serper"),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    created_at=datetime.now(timezone.utc).isoformat())
    if args.mode == "native" and not args.predictions:
        model, kind = resolve_checkpoint(args.run_dir, args.model, args.kind)
        manifest.update(model=model, kind=kind, generation_alignment="original training system/schema; shared native serializer; upstream v1 tools")
    elif not args.predictions:
        manifest.update(model=os.environ["OPENSEEKER_MODEL"], generation_alignment="unmodified official script")
    else:
        manifest.update(predictions_sha256=sha256(args.predictions), generation_alignment="imported predictions; verify provenance")
    manifest["local_workflow_sha256"] = sha256(__file__)
    manifest["official_script_sha256"] = {
        str(path.relative_to(args.official_dir)): sha256(path)
        for path in sorted((args.official_dir / "eval").glob("*.py"))}
    data_manifest = args.data.with_suffix(".manifest.json")
    if data_manifest.is_file():
        metadata = json.loads(data_manifest.read_text())
        if metadata.get("data_sha256") != sha256(args.data):
            raise ValueError("benchmark differs from its preparation manifest; restore or import as custom data")
        manifest["dataset_manifest"] = metadata
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    raw = args.predictions or output / f"result_tool{args.tool_count}.jsonl"
    if not args.predictions:
        if args.mode == "native":
            native_generate(args, rows, raw)
        else:
            run_logged(generation_command(args, dataset, output), args.official_dir, output / "generation.log",
                       raw, len(rows), f"benchmark {args.benchmark} generation")
    aligned = output / "result_for_judge.jsonl"
    suffix = boxed_instruction(args.official_dir) if args.mode == "official-v2" else ""
    coverage = align_results(rows, raw, aligned, suffix)
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    run_logged([sys.executable, str(args.official_dir / "eval/eval.py"), "--data_path", str(aligned),
                "--max_workers", str(args.judge_workers)], args.official_dir, output / "judge.log",
               output / "result_for_judge_eval.jsonl", len(rows), f"benchmark {args.benchmark} judge",
               key="item_index", item_type="item")
    scores = audit_scores(output / "result_for_judge_eval.jsonl", len(rows))
    summary = dict(coverage, **scores, mode=args.mode, benchmark=args.benchmark,
                   official_revision=revision, metric="OpenSeeker JUDGE_PROMPT_BC_en A/B accuracy")
    (output / "summary.metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not scores["complete"]:
        raise RuntimeError("judge coverage incomplete; inspect judge.log and summary (not a complete benchmark score)")


if __name__ == "__main__":
    main()
