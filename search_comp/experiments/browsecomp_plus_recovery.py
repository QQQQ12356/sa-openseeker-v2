"""Recover and score saved predictions without changing the source evaluation."""
from collections import Counter
import json
import math

from search_comp.evaluation.benchmark_resume import resume_prefix
from search_comp.evaluation.training_overlap import select_questions
from search_comp.offline.assets import enforce_offline
from search_comp.workflows.offline_benchmark import prepared_questions, score_phase
from .benchmarks import offline_options
from .runtime import configure, digest, job, path, read_json, slug, stamp, write_json


def prediction_sources(results, rows, manifest):
    predictions = results / "predictions.jsonl"
    if predictions.is_file():
        return [(predictions, manifest.get("predictions_sha256"))]

    sources = []
    arguments = manifest["arguments"]
    if arguments.get("resume_from"):
        previous, _ = resume_prefix(
            path(arguments["resume_from"]), rows, manifest,
            arguments["resume_manifest_sha256"], arguments["resume_predictions_sha256"],
        )
        sources.append((previous, arguments["resume_predictions_sha256"]))
        results = results / "extension"

    predictions = results / "predictions.jsonl"
    if predictions.is_file():
        sources.append((predictions, None))
    else:
        sources.extend((file, None) for file in sorted(results.glob("workers/worker_*/predictions.jsonl")))
    return sources


def recover_predictions(sources, rows, target):
    expected_queries = {row["id"]: row["query"] for row in rows}
    seen, statuses, inputs = set(), Counter(), []
    building = target.with_suffix(".building.jsonl")
    try:
        with building.open("xb") as output:
            for source, expected_sha256 in sources:
                if expected_sha256 and digest(source) != expected_sha256:
                    raise ValueError(f"预测文件 SHA256 与原记录不符：{source}")
                count, truncated = 0, False
                with source.open("rb") as stream:
                    for number, line in enumerate(stream, 1):
                        if not line.endswith(b"\n"):
                            truncated = True
                            break
                        try:
                            prediction = json.loads(line)
                        except ValueError as error:
                            raise ValueError(f"预测 JSONL 无法解析：{source} 第 {number} 行") from error
                        qid = prediction.get("id") if isinstance(prediction, dict) else None
                        if (not isinstance(qid, str) or qid not in expected_queries or qid in seen
                                or prediction.get("query") != expected_queries[qid]):
                            raise ValueError(f"预测题目重复或与原测试不符：{source} 第 {number} 行")
                        status = prediction.get("status")
                        if not isinstance(status, str):
                            raise ValueError(f"预测缺少有效的 status：{source} 第 {number} 行")
                        seen.add(qid)
                        statuses[status] += 1
                        output.write(line)
                        count += 1
                inputs.append(dict(path=source, predictions=count, skipped_unterminated_line=truncated))
                print(f"[recovery] {source}: predictions={count}; skipped_unterminated_line={truncated}", flush=True)
        if not seen:
            raise ValueError("没有完整的预测记录；请检查原测试的 results/ 和各 worker 日志")
        building.rename(target)
    finally:
        building.unlink(missing_ok=True)
    return seen, statuses, inputs


def recover_browsecomp_plus(*, evaluation_dir, run_name, retrieval_only, judge, environment):
    configure(environment)
    enforce_offline()
    if not evaluation_dir.strip():
        raise ValueError("请填写 SA_EVALUATION_DIR，指向 42 的测试记录根目录")
    evaluation = path(evaluation_dir)
    record = read_json(evaluation / "job.json")
    settings = record["settings"]
    if record["operation"] != "evaluate" or settings.get("benchmark") != "browsecomp-plus":
        raise ValueError("SA_EVALUATION_DIR 须指向 BrowseComp-Plus 测试记录根目录")
    status = record["status"]
    if status not in {"failed", "running"}:
        raise ValueError("恢复评分适用于中断的测试；已完成的测试请使用 43 评分")

    judge = dict(judge)
    model = judge["judge_model"]
    judge["judge_model"] = path(model) if model and not retrieval_only else None
    if not retrieval_only:
        if not judge["judge_model"] or not judge["judge_model"].is_dir():
            raise ValueError("答案评分须填写有效的本地 SA_JUDGE__JUDGE_MODEL 目录")
        budgets = ("judge_attempts", "judge_max_tokens", "judge_max_context", "judge_top_k")
        if any(judge[key] < 1 for key in budgets):
            raise ValueError("裁判尝试次数、token 预算及 top_k 须为正整数")
        temperature = judge["judge_temperature"]
        if not math.isfinite(temperature) or temperature < 0 or not 0 < judge["judge_top_p"] <= 1:
            raise ValueError("裁判 temperature 须为非负有限数，top_p 须在 (0, 1]")

    results = evaluation / "results"
    manifest = read_json(results / "manifest.json")
    args = offline_options("browsecomp-plus", settings["benchmark_dir"], action="score",
                           limit=settings["generation"]["limit"], retrieval_only=retrieval_only, **judge)
    rows, prepared = prepared_questions(args.root, args.benchmark)
    overlap = evaluation / "overlap.json" if settings.get("overlap") == "filter" else None
    if overlap:
        report_sha256 = settings.get("overlap_report_sha256")
        if report_sha256 and digest(overlap) != report_sha256:
            raise ValueError("原测试重合筛选报告 SHA256 已改变")
    rows = select_questions(rows, args.limit, overlap)
    selected_ids = [row["id"] for row in rows]
    if (not selected_ids or manifest.get("benchmark") != args.benchmark
            or manifest["selected_ids"] != selected_ids or manifest["questions_sha256"] != prepared["questions_sha256"]
            or manifest["prepared_manifest"]["corpus_sha256"] != prepared["corpus_sha256"]):
        raise ValueError("当前题集或语料与原测试清单不一致，请恢复原 benchmark 版本和筛选报告")

    sources = prediction_sources(results, rows, manifest)
    output = evaluation / "recovery" / stamp(slug(run_name))
    recovery_settings = dict(evaluation_dir=evaluation, retrieval_only=retrieval_only, judge=judge,
                             benchmark_dir=args.root, original_status=status, limit=args.limit)
    with job(output, "recover_browsecomp_plus", recovery_settings):
        predictions = output / "predictions.jsonl"
        seen, statuses, inputs = recover_predictions(sources, rows, predictions)
        coverage = dict(original_evaluation=str(evaluation), original_status=status, planned=len(rows),
                        recovered=len(seen), missing=len(rows) - len(seen), coverage=len(seen) / len(rows))
        write_json(output / "recovery.json", dict(
            **coverage, missing_ids=[qid for qid in selected_ids if qid not in seen],
            prediction_sources=inputs, statuses=statuses, predictions_sha256=digest(predictions),
        ))
        print(f"[recovery] recovered={len(seen)}/{len(rows)}; missing={coverage['missing']}", flush=True)
        score_output = output / "results"
        score_output.mkdir()
        score_phase(args, score_output, rows, prepared, predictions, recovery=coverage)
    print(f"[recovery] metrics: {score_output / 'summary.metrics.json'}", flush=True)
    return output
