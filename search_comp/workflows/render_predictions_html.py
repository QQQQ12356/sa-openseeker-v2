#!/usr/bin/env python3
"""Render prediction JSONL files as standalone, offline HTML reports (stdlib only)."""
import json
from pathlib import Path


DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / 'scripts' / 'templates' / '41_predictions.html'


def read_rows(path):
    rows = []
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {exc.msg}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            rows.append(row)
    return rows


REFERENCE_FIELDS = ('answer', 'answers', 'reference_answer', 'ground_truth')


def reference_index(path):
    """Match full questions rather than IDs, which vary between dataset exports."""
    index = {}
    for row in read_rows(path):
        question = row.get('question') or row.get('query')
        answer = next((row[k] for k in REFERENCE_FIELDS if row.get(k) is not None), None)
        if not question or answer is None:
            continue
        if question in index and index[question] != answer:
            raise ValueError(f'{path}: conflicting reference answers for question {question!r}')
        index[question] = answer
    return index


def render_report(source, destination, template=DEFAULT_TEMPLATE, title=None, references=None):
    rows = read_rows(source)
    if references is not None:
        for row in rows:
            if any(row.get(k) is not None for k in REFERENCE_FIELDS):
                continue
            question = row.get('question') or row.get('query')
            if question in references:
                row['answer'] = references[question]
    payload = json.dumps(
        {'title': title or source.name, 'source': str(source), 'rows': rows},
        ensure_ascii=False, allow_nan=False,
    )
    # A JSON script element is still parsed by HTML: escape closing tags and entities.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    markup = template.read_text(encoding='utf-8')
    marker = '__REPORT_DATA__'
    if markup.count(marker) != 1:
        raise ValueError(f'{template}: template must contain exactly one {marker} marker')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markup.replace(marker, payload), encoding='utf-8')
    return len(rows)
