"""Profile downloaded Research trajectories with the training tokenizer."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from html import escape
from itertools import islice
import json

from search_comp.data.selection import Selection
from search_comp.experiments.runtime import path, read_json, write_json
from search_comp.workflows.prepare_openseeker import initialize, measure, rows, stats


BIN_WIDTH = 4096
LAST_EDGE = 65536


def raw_label(raw):
    field = "trajectory correctness"
    return json.dumps(raw[field], ensure_ascii=False, sort_keys=True) if field in raw else "(missing)"


def items(inputs):
    for filename in inputs:
        for index, raw in enumerate(rows(filename)):
            yield f"research/{filename.name}", index, raw


def summarize(inputs, measurements, tokenizer, max_length, observations):
    selections = {
        mode: Selection(max_length=max_length, correctness=mode, observations=observations)
        for mode in ("Correct", "any")
    }
    outcomes = {mode: Counter() for mode in selections}
    seen = {mode: set() for mode in selections}
    raw_labels = Counter()
    measured_labels = Counter()
    within_limit_labels = Counter()
    invalid_reasons = Counter()
    lengths = []
    bins = Counter()
    for label, (row, _, error) in measurements:
        raw_labels[label] += 1
        if error:
            invalid_reasons[error["reason"]] += 1
            for counts in outcomes.values():
                counts["invalid"] += 1
            continue
        length = row["lengths"][tokenizer]["total"]
        lengths.append(length)
        bins[min((max(1, length) - 1) // BIN_WIDTH, LAST_EDGE // BIN_WIDTH)] += 1
        measured_labels[label] += 1
        if 1 <= length <= max_length:
            within_limit_labels[label] += 1
        for mode, selection in selections.items():
            reason = selection.rejection(row)
            if reason is None:
                key = row["question_id"]
                if key in seen[mode]:
                    reason = "duplicate_question"
                else:
                    seen[mode].add(key)
            outcomes[mode][reason or "accepted"] += 1

    histogram = {
        f"{start + 1:,}–{start + BIN_WIDTH:,}": bins[start // BIN_WIDTH]
        for start in range(0, LAST_EDGE, BIN_WIDTH)
    }
    histogram[f">{LAST_EDGE:,}"] = bins[LAST_EDGE // BIN_WIDTH]
    return dict(
        inputs=[str(filename) for filename in inputs],
        tokenizer=tokenizer, max_length=max_length, observations=observations,
        input_count=raw_labels.total(), measured_count=len(lengths),
        invalid_count=invalid_reasons.total(), invalid_reasons=dict(invalid_reasons.most_common()),
        length_tokens=stats(lengths), histogram=histogram,
        raw_correctness=dict(raw_labels.most_common()),
        measured_correctness=dict(measured_labels),
        within_limit_correctness=dict(within_limit_labels),
        selection={mode: {"accepted": counts["accepted"], **counts} for mode, counts in outcomes.items()},
    )


def table(headers, records):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in record) + "</tr>"
        for record in records
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(report):
    histogram = report["histogram"]
    peak = max(1, *histogram.values())
    bars = "\n".join(
        f'<div class="bin"><span>{label}</span>'
        f'<meter min="0" max="{peak}" value="{count}" aria-label="{label} tokens"></meter>'
        f'<span>{count:,}</span></div>'
        for label, count in histogram.items()
    )
    labels = table(
        ("原始标签", "原始条数", "成功测长", "长度合格"),
        ((label, count, report["measured_correctness"].get(label, 0),
          report["within_limit_correctness"].get(label, 0))
         for label, count in report["raw_correctness"].items()),
    )
    correct, any_label = (report["selection"][mode] for mode in ("Correct", "any"))
    selection = table(
        ("结果", "Correct", "any"),
        ((reason, correct.get(reason, 0), any_label.get(reason, 0))
         for reason in sorted(correct.keys() | any_label.keys())),
    )
    lengths = table(("统计量", "值"), report["length_tokens"].items())
    errors = table(("原因", "条数"), report["invalid_reasons"].items())
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Research 数据统计</title>
<style>
body {{font: 16px system-ui,sans-serif; color:#172331; background:#f8fafc; margin:2rem auto;
       padding:0 1rem; max-width:56rem}}
table {{border-collapse:collapse; width:100%; margin-bottom:2rem}}
th,td {{text-align:left; padding:.45rem .7rem; border-bottom:1px solid #ccd6e0;
        overflow-wrap:anywhere}}
.bin {{display:grid; grid-template-columns:9rem 1fr 4rem; gap:.7rem; align-items:center;
       font-variant-numeric:tabular-nums}}
.bin span:last-child {{text-align:right}}
meter {{width:100%; height:1.2rem}}
</style>
<main>
<h1>Research 数据统计</h1>
<p>原始 {report['input_count']:,} 条 · 成功测长 {report['measured_count']:,} 条 ·
无效 {report['invalid_count']:,} 条</p>
<p>Tokenizer <code>{escape(report['tokenizer'])}</code> ·
长度上限 {report['max_length']:,} · 工具观测筛选 <code>{escape(report['observations'])}</code></p>
<h2>完整轨迹 token 长度直方图</h2>
{bars}
{lengths}
<h2>正确性标签分布</h2>
{labels}
<h2>筛选结果</h2>
<p>按问题去重。accepted 是分配训练、验证和测试集之前的总条数。</p>
{selection}
<h2>无效记录原因</h2>
{errors}
</main></html>
"""


def run(*, snapshot_dir, tokenizer, max_length, observations, workers, output):
    if workers < 1:
        raise ValueError("WORKERS 必须至少为 1")
    Selection(max_length=max_length, observations=observations)
    manifest = read_json(snapshot_dir / "download_manifest.json")
    inputs = [path(filename) for filename in manifest["jsonl_inputs"]]
    if not inputs or any(not filename.is_file() for filename in inputs):
        raise ValueError("download_manifest.json 没有可用的分片路径，请检查下载目录及清单")

    def measured_batches(pool):
        stream = items(inputs)
        count = 0
        while batch := list(islice(stream, workers * 8)):
            yield from zip((raw_label(raw) for _, _, raw in batch),
                           pool.map(measure, batch, chunksize=2))
            count += len(batch)
            if count // 256 != (count - len(batch)) // 256:
                print(f"已扫描 {count:,} 条轨迹", flush=True)

    with ProcessPoolExecutor(workers, initializer=initialize, initargs=([tokenizer], True)) as pool:
        report = summarize(inputs, measured_batches(pool), tokenizer, max_length, observations)
    write_json(output / "statistics.json", report)
    (output / "statistics.html").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)
