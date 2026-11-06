"""Export existing benchmark scores without importing model or scoring backends."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from search_comp.experiments.runtime import ROOT, digest, job, path, stamp, write_json


SUMMARY_NAME = "summary.metrics.json"
SUMMARY_COLUMNS = (
    "record", "method", "benchmark", "dataset_name", "training_run", "model_path",
    "benchmark_dir", "limit", "score_run", "job_status", "complete", "created_at",
    "finished_at", "count", "prediction_count", "missing_predictions", "judge_errors",
    "em", "f1", "llm_judge_accuracy", "accuracy", "official_accuracy",
    "accuracy_all_questions", "evidence_recall", "gold_recall", "sp_em", "sp_f1",
    "joint_em", "joint_f1", "citation_precision", "citation_recall", "citation_coverage",
    "subset", "corpus_scope", "score_mode", "metric", "judge_model", "judge_attempts",
    "summary_file", "summary_sha256", "evaluation_dir",
)
DETAIL_COLUMNS = (
    "record", "source_file", "source_kind", "source_row", "id", "item_index", "query",
    "reference_answer", "prediction", "completed", "score", "correct", "em", "f1",
    "evidence_recall", "gold_recall", "score_source", "judge_error", "judge_reasoning",
    "tool_calls",
)
DETAIL_FILES = ("per_question.json", "judge.jsonl", "result_for_judge_eval.jsonl")
ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def decode_json(text, source):
    """Reject nonstandard NaN/Infinity rather than silently exporting bad numbers."""
    def invalid(value):
        raise ValueError(f"{source}: non-finite JSON number {value}")

    def finite_float(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)

    try:
        return json.loads(text, parse_constant=invalid, parse_float=finite_float)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}:{error.lineno}: {error.msg}") from error


def read_json(filename):
    return decode_json(Path(filename).read_text(encoding="utf-8-sig"), filename)


def read_object(filename):
    value = read_json(filename)
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: expected a JSON object")
    return value


def discover_summaries(inputs):
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("INPUTS 必须是非空 JSON 路径列表")
    found = set()
    for value in inputs:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("INPUTS 的每个元素必须是非空路径字符串")
        source = path(value)
        if source.is_dir():
            matches = list(source.rglob(SUMMARY_NAME))
        elif source.is_file() and source.name == SUMMARY_NAME:
            matches = [source]
        else:
            raise ValueError(f"输入不存在或不是目录/{SUMMARY_NAME}: {source}")
        if not matches:
            raise ValueError(f"输入中没有评分摘要: {source}")
        found.update(match.resolve() for match in matches)
    return sorted(found)


def flatten(value, prefix=""):
    """Retain every summary field, including future nested metrics."""
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            yield from flatten(item, name)
        else:
            yield name, item


def load_record(filename):
    """Use adjacent provenance only; copied standalone summaries remain untracked."""
    summary = read_object(filename)
    score_dir = filename.parent.parent
    job_file = score_dir / "job.json"
    tracked = filename.parent.name == "results" and job_file.is_file()
    metadata = read_object(job_file) if tracked else {}
    if tracked and metadata.get("operation") != "score":
        raise ValueError(f"{job_file}: operation 必须为 score")
    settings = metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError(f"{job_file}: settings 必须为对象")
    # The standard layout stores evaluation.json two directories above a score run.
    evaluation_dir = score_dir.parent.parent
    evaluation_file = evaluation_dir / "evaluation.json"
    evaluation = read_object(evaluation_file) if tracked and evaluation_file.is_file() else {}
    manifest_file = evaluation_dir / "results/manifest.json"
    manifest = read_object(manifest_file) if evaluation and manifest_file.is_file() else {}
    for field in ("method", "benchmark"):
        if settings.get(field) and evaluation.get(field) != settings[field]:
            raise ValueError(f"{filename}: 评分与测试的 {field} 不一致或测试身份缺失")
    expected_benchmark = settings.get("benchmark", evaluation.get("benchmark"))
    if (expected_benchmark and summary.get("benchmark") and
            str(summary["benchmark"]).lower() != expected_benchmark):
        raise ValueError(f"{filename}: 摘要与评分的 benchmark 不一致")
    judge = settings.get("judge") or {}
    generation = evaluation.get("generation") or {}
    row = {key: summary.get(key) for key in SUMMARY_COLUMNS}
    row.update(
        method=settings.get("method", evaluation.get("method")),
        benchmark=expected_benchmark or summary.get("benchmark"),
        dataset_name=evaluation.get("dataset_name"), training_run=evaluation.get("training_run"),
        model_path=evaluation.get("model_path") or manifest.get("model"),
        benchmark_dir=evaluation.get("benchmark_dir"),
        limit=generation.get("limit"), score_run=score_dir.name if tracked else None,
        job_status=metadata.get("status", "untracked"), created_at=metadata.get("created_at"),
        finished_at=metadata.get("finished_at"), judge_model=judge.get("judge_model"),
        judge_attempts=summary.get("judge_attempts", judge.get("judge_attempts")),
        summary_file=str(filename), summary_sha256=digest(filename),
        evaluation_dir=str(evaluation_dir) if evaluation else None,
    )
    # Online BrowseComp uses different names for the same coverage counts.
    row["count"] = summary.get("count", summary.get("expected"))
    row["prediction_count"] = summary.get("prediction_count", summary.get("unique_predictions"))
    row["judge_errors"] = summary.get("judge_errors", summary.get("judge_unknown_or_error"))
    eligible = summary.get("complete") is True and (not tracked or metadata.get("status") == "complete")
    return row, summary, eligible


def detail_rows(filename):
    """Keep metric and judge rows separate, never join online rows by file order."""
    if filename.suffix == ".json":
        values = read_json(filename)
        if not isinstance(values, list):
            raise ValueError(f"{filename}: expected a JSON array")
        source = enumerate(values, 1)
        for index, value in source:
            if not isinstance(value, dict):
                raise ValueError(f"{filename}:{index}: expected a JSON object")
            yield index, value
    else:
        with filename.open(encoding="utf-8-sig") as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = decode_json(line, f"{filename}:{index}")
                if not isinstance(value, dict):
                    raise ValueError(f"{filename}:{index}: expected a JSON object")
                if filename.name != "result_for_judge_eval.jsonl" or value.get("type") == "item":
                    yield index, value


class SheetWriter:
    """Bound memory with write-only sheets; split at Excel's row limit."""

    def __init__(self, workbook, title, columns):
        self.workbook, self.title, self.columns = workbook, title, columns
        self.parts = self.rows = self.clipped_cells = self.cleaned_cells = 0
        self._new_sheet()

    def _new_sheet(self):
        from openpyxl.styles import Font, PatternFill
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.utils import get_column_letter

        self.parts += 1
        name = self.title if self.parts == 1 else f"{self.title}_{self.parts}"
        self.sheet = self.workbook.create_sheet(name)
        self.sheet.freeze_panes = "A2"
        for index, name in enumerate(self.columns, 1):
            width = 55 if any(part in name for part in ("file", "path", "query", "prediction", "answer")) else 23
            self.sheet.column_dimensions[get_column_letter(index)].width = width
        header = []
        for name in self.columns:
            cell = WriteOnlyCell(self.sheet, value=name)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="203864")
            header.append(cell)
        self.sheet.append(header)
        self.sheet_rows = 1

    def append(self, values):
        from openpyxl.cell import WriteOnlyCell

        if self.sheet_rows == 1_048_576:
            self.finish()
            self._new_sheet()
        cells = []
        for name in self.columns:
            value = values.get(name)
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{self.title}.{name}: 不支持 NaN/Infinity")
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if type(value) is int and abs(value) >= 10 ** 15:
                value = str(value)
            if isinstance(value, str):
                clean = ILLEGAL_XML.sub("", value)
                self.cleaned_cells += clean != value
                value = clean
                # Excel counts UTF-16 code units, including two for non-BMP characters.
                encoded = value.encode("utf-16-le")
                if len(encoded) > 32767 * 2:
                    suffix = "\n[truncated; see source_file/summary_file]"
                    value = encoded[:(32767 - len(suffix)) * 2].decode("utf-16-le", errors="ignore") + suffix
                    self.clipped_cells += 1
            cell = WriteOnlyCell(self.sheet, value=value)
            if isinstance(value, str):
                cell.data_type = "s"  # Preserve untrusted model text as literal text, including '=...'.
            elif type(value) is float:
                cell.number_format = "0.0000"
            cells.append(cell)
        self.sheet.append(cells)
        self.sheet_rows += 1
        self.rows += 1

    def finish(self):
        from openpyxl.utils import get_column_letter
        self.sheet.auto_filter.ref = f"A1:{get_column_letter(len(self.columns))}{self.sheet_rows}"


def export_benchmarks(*, inputs, output_dir, run_name, include_details, include_incomplete):
    try:
        from openpyxl import Workbook
    except ImportError as error:
        raise RuntimeError("请在 Linux 安装 python -m pip install -r requirements.tools-export.txt") from error

    output_dir = path(output_dir)
    if not output_dir.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("OUTPUT_DIR 必须位于本项目 artifacts/ 下")
    settings = dict(inputs=inputs, output_dir=output_dir, run_name=run_name,
                    include_details=include_details, include_incomplete=include_incomplete)
    with job(output_dir / stamp(run_name), "benchmark_excel", settings) as output:
        sources = discover_summaries(inputs)
        accepted, skipped = [], []
        for filename in sources:
            row, summary, eligible = load_record(filename)
            if not eligible and not include_incomplete:
                skipped.append(dict(file=str(filename), complete=summary.get("complete"),
                                    job_status=row["job_status"]))
                print(f"跳过未完成评分: {filename}", flush=True)
                continue
            row["record"] = f"score_{len(accepted) + 1:06}"
            accepted.append((filename, row, summary))
        write_json(output / "export_manifest.json", dict(settings, skipped=skipped,
                   accepted=[dict(file=str(f), sha256=r["summary_sha256"]) for f, r, _ in accepted]))
        if not accepted:
            raise ValueError("没有可导出的评分结果；检查输入路径和 INCLUDE_INCOMPLETE")

        workbook = Workbook(write_only=True)
        summary_sheet = SheetWriter(workbook, "summary", SUMMARY_COLUMNS)
        metric_sheet = SheetWriter(workbook, "metrics", ("record", "summary_file", "metric", "value"))
        detail_sheet = SheetWriter(workbook, "details", DETAIL_COLUMNS) if include_details else None
        detail_sources = []
        for filename, row, summary in accepted:
            summary_sheet.append(row)
            for name, value in flatten(summary):
                metric_sheet.append(dict(record=row["record"], summary_file=str(filename), metric=name, value=value))
            if detail_sheet is None:
                continue
            for name in DETAIL_FILES:
                detail_file = filename.parent / name
                if not detail_file.is_file():
                    continue
                detail_sources.append(dict(file=str(detail_file), sha256=digest(detail_file)))
                for index, item in detail_rows(detail_file):
                    values = {key: item.get(key) for key in DETAIL_COLUMNS}
                    judge_result = item.get("judge_result") or {}
                    values.update(record=row["record"], source_file=str(detail_file),
                                  source_kind=detail_file.stem, source_row=index,
                                  reference_answer=item.get("answer"),
                                  prediction=item.get("prediction", item.get("candidate_answer", item.get("final_response"))),
                                  correct=item.get("correct", judge_result.get("correct")),
                                  judge_reasoning=judge_result.get("reasoning"))
                    detail_sheet.append(values)
        writers = [summary_sheet, metric_sheet] + ([detail_sheet] if detail_sheet else [])
        for writer in writers:
            writer.finish()
        temporary = output / "benchmark_scores.building.xlsx"
        workbook.save(temporary)
        destination = output / "benchmark_scores.xlsx"
        temporary.replace(destination)
        write_json(output / "export_manifest.json", dict(settings, skipped=skipped,
                   accepted=[dict(file=str(f), sha256=r["summary_sha256"]) for f, r, _ in accepted],
                   detail_sources=detail_sources, workbook_sha256=digest(destination),
                   sheets={w.title: dict(rows=w.rows, parts=w.parts, clipped_cells=w.clipped_cells,
                                        cleaned_cells=w.cleaned_cells) for w in writers}))
        print(f"Excel: {destination}; 评分记录: {len(accepted)}; 跳过: {len(skipped)}", flush=True)
    return destination
