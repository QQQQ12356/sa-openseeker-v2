"""Three benchmarks with explicit method ownership and separate scoring jobs."""
from __future__ import annotations

from types import SimpleNamespace
from .runtime import ROOT, configure, digest, job, path, read_json, slug, stamp, write_json

BENCHMARKS = {"browsecomp", "browsecomp-plus", "hotpotqa"}


def online_options(benchmark_dir, **values):
    options = dict(official_dir=path(benchmark_dir) / "vendor/OpenSeeker",
                   data=path(benchmark_dir) / "questions.jsonl", csv=None, prepare_only=False,
                   mode="native", benchmark="BrowseComp", output_dir=None, predictions=None,
                   generate_only=False, run_dir=None, prompt_data=None, model=None,
                   model_origin="checkpoint", kind=None, workers=1, judge_workers=1,
                   judge_attempts=3, judge_max_tokens=1024, judge_timeout=120,
                   judge_enable_thinking=False, skip_judge_preflight=False, limit=-1,
                   max_tokens=8192, tool_count=200, max_context=131072, max_turns=200,
                   tool_policy="stable", tool_call_retries=1, max_tool_tokens=32768,
                   decoding="greedy", repetition_penalty=1.0, temperature=1.0, top_p=1.0,
                   generation_seed=42, dry_run=False)
    options.update(values)
    return SimpleNamespace(**options)


def offline_options(benchmark, benchmark_dir, **values):
    options = dict(action="run", benchmark=benchmark, root=path(benchmark_dir), data=None,
                   corpus=None, decrypted=False, corpus_limit=None, official_only=False,
                   corpus_revision=None, dev_url=None, wiki_url=None, title_weight=1.0,
                   verify_index=True, run_dir=None, prompt_data=None, model=None,
                   model_origin="checkpoint", kind=None, output_dir=None, predictions=None,
                   num_workers=1, shard_index=0, shard_count=1, limit=-1, topk=5,
                   snippet_tokens=512, allow_visit=True, tool_call_retries=1, read_tokens=4096,
                   tool_count=200, max_turns=200, max_context=131072, max_new_tokens=8192,
                   max_tool_tokens=32768, judge_model=None, llm_judge=False, retrieval_only=False,
                   judge_max_tokens=4096, judge_max_context=32768, judge_temperature=0,
                   judge_attempts=3, judge_enable_thinking=False, skip_judge_preflight=False,
                   judge_top_p=0.8, judge_top_k=20, seed=42, decoding="greedy",
                   repetition_penalty=1.0, temperature=1.0, top_p=1.0, generation_seed=42,
                   dry_run=False)
    options.update(values)
    return SimpleNamespace(**options)


def prepare_benchmark(*, benchmark, version, phase, environment, data=None, corpus=None,
                      decrypted=False, corpus_revision=None, title_weight=1.0, csv=None):
    configure(environment)
    if benchmark not in BENCHMARKS or phase not in {"download", "prepare", "index"}:
        raise ValueError("无效 benchmark 或 PHASE")
    directory = ROOT / "artifacts/benchmarks" / benchmark / slug(version)
    settings = dict(benchmark=benchmark, version=version, phase=phase, data=data, corpus=corpus,
                    decrypted=decrypted, corpus_revision=corpus_revision, title_weight=title_weight, csv=csv)
    with job(directory / "jobs" / stamp(phase), "prepare_benchmark", settings):
        if benchmark == "browsecomp":
            if phase != "download":
                raise ValueError("BrowseComp 使用 PHASE=download 同时准备题目和固定版本工具，无需本地索引")
            from search_comp.workflows.benchmark_openseeker import ensure_official
            from search_comp.evaluation.benchmark_data import prepare_browsecomp, read_questions, sha256
            ensure_official(directory / "vendor/OpenSeeker")
            questions = directory / "questions.jsonl"
            if questions.exists():
                manifest = read_json(questions.with_suffix(".manifest.json"))
                if sha256(questions) != manifest["data_sha256"]:
                    raise ValueError("已准备的 BrowseComp 题目已改变；请使用新 VERSION")
                read_questions(questions)
            else:
                prepare_browsecomp(questions, path(csv) if csv else None)
        else:
            from search_comp.offline.assets import DEV_URL, WIKI_URL
            from search_comp.workflows.offline_benchmark import run
            run(offline_options(benchmark, directory, action=phase,
                data=path(data) if data else None, corpus=path(corpus) if corpus else None,
                decrypted=decrypted, corpus_revision=corpus_revision, title_weight=title_weight,
                dev_url=DEV_URL, wiki_url=WIKI_URL))


def resolve_target(method, training_run, model_path):
    if method == "baseline":
        if training_run or not model_path:
            raise ValueError("baseline 只接受本地 MODEL_PATH，不接受训练实验")
        return "untrained", None, str(path(model_path)), "plain", "base-qwen35"
    if method not in {"plain", "beacon"} or not training_run or model_path:
        raise ValueError("plain/beacon 必须填写 TRAINING_RUN；模型自动从该实验读取")
    run = path(training_run)
    record = read_json(run / "job.json")
    spec = read_json(run / "experiment.json")
    if record["operation"] != "train" or record["status"] != "complete":
        raise ValueError("只能测试已成功完成训练的实验；plan/failed 不能作为模型来源")
    if spec["method"] != method or spec["dataset_name"] not in {"openseeker", "research"}:
        raise ValueError("训练实验与方法不一致")
    return spec["dataset_name"], run, None, method, "checkpoint"


def evaluate(*, method, benchmark, benchmark_dir, training_run, model_path, prompt_data,
             run_name, generation, environment):
    configure(environment)
    if benchmark not in BENCHMARKS:
        raise ValueError("不支持的 benchmark")
    dataset_name, training_run, model, kind, origin = resolve_target(method, training_run, model_path)
    # Training prompts are always bound to the experiment; baseline may choose an explicit comparison profile.
    if method != "baseline" and prompt_data:
        raise ValueError("训练方法的提示词从 TRAINING_RUN 读取，无需覆盖 PROMPT_DATA")
    if kind == "beacon" and generation["decoding"] != "greedy":
        raise ValueError("Beacon 当前仅支持 greedy 解码")
    base = training_run if training_run else ROOT / "artifacts/experiments/baseline/untrained"
    directory = base / "evaluation" / benchmark / stamp(run_name)
    settings = dict(method=method, benchmark=benchmark, dataset_name=dataset_name,
                    benchmark_dir=str(path(benchmark_dir)), training_run=training_run, model_path=model,
                    prompt_data=str(path(prompt_data)) if prompt_data else None, generation=generation)
    common = dict(run_dir=training_run, model=model, kind=kind, model_origin=origin,
                  prompt_data=path(prompt_data) if prompt_data else None, output_dir=directory / "results")
    with job(directory, "evaluate", settings):
        if benchmark == "browsecomp":
            from search_comp.workflows.benchmark_openseeker import run
            options = dict(generation)
            options["max_tokens"] = options.pop("max_new_tokens")
            # Offline retrieval knobs do not apply to the upstream online tools.
            for key in ("num_workers", "title_weight", "verify_index", "topk", "snippet_tokens", "read_tokens", "allow_visit"):
                options.pop(key, None)
            run(online_options(benchmark_dir, **common, **options, generate_only=True))
            predictions = directory / "results" / f"result_tool{generation['tool_count']}.jsonl"
        else:
            from search_comp.workflows.offline_benchmark import run
            run(offline_options(benchmark, benchmark_dir, **common, **generation))
            predictions = directory / "results/predictions.jsonl"
        write_json(directory / "evaluation.json", dict(settings, predictions=str(predictions),
                   predictions_sha256=digest(predictions),
                   engine_manifest_sha256=digest(directory / "results/manifest.json")))
    return directory


def resolve_evaluation(method, benchmark, evaluation_dir):
    directory = path(evaluation_dir)
    record = read_json(directory / "job.json")
    source = read_json(directory / "evaluation.json")
    if record["operation"] != "evaluate" or record["status"] != "complete":
        raise ValueError("仅可评分已成功完成的测试")
    if source["method"] != method or source["benchmark"] != benchmark:
        raise ValueError("评分脚本的方法/benchmark 与测试记录不一致")
    if digest(source["predictions"]) != source["predictions_sha256"]:
        raise ValueError("预测文件已改变，不能与原测试混合归档")
    if digest(directory / "results/manifest.json") != source["engine_manifest_sha256"]:
        raise ValueError("测试 manifest 已改变")
    if benchmark == "browsecomp":
        manifest = read_json(directory / "results/manifest.json")
        if digest(path(source["benchmark_dir"]) / "questions.jsonl") != manifest["data_sha256"]:
            raise ValueError("BrowseComp 题目在测试后改变，请恢复原数据版本")
    return directory, source


def score(*, method, benchmark, evaluation_dir, run_name, judge, environment):
    configure(environment)
    directory, source = resolve_evaluation(method, benchmark, evaluation_dir)
    output = directory / "scores" / stamp(slug(run_name))
    settings = dict(method=method, benchmark=benchmark, evaluation_dir=directory,
                    predictions_sha256=source["predictions_sha256"], judge=judge)
    with job(output, "score", settings):
        common = dict(predictions=path(source["predictions"]), output_dir=output / "results",
                      limit=source["generation"]["limit"])
        if benchmark == "browsecomp":
            from search_comp.workflows.benchmark_openseeker import run
            options = {key: value for key, value in judge.items() if key in {
                "judge_workers", "judge_attempts", "judge_max_tokens", "judge_timeout", "judge_enable_thinking"}}
            run(online_options(source["benchmark_dir"], **common, **options))
        else:
            from search_comp.workflows.offline_benchmark import run
            options = {key: value for key, value in judge.items() if key not in {"judge_workers", "judge_timeout"}}
            options["judge_model"] = path(options["judge_model"]) if options.get("judge_model") else None
            run(offline_options(benchmark, source["benchmark_dir"], action="score", **common, **options))
    return output
