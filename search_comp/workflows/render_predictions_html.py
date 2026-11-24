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


def resolve_score_file(evaluation_dir, score_dir=None):
    """Select a completed matching job; explicit failed jobs allow error inspection."""
    from search_comp.experiments.runtime import read_json, path
    directory = Path(evaluation_dir).resolve()
    evaluation = read_json(directory / 'evaluation.json')
    candidates = [path(score_dir)] if score_dir else sorted((directory / 'scores').glob('*'), reverse=True)
    matches = []
    for candidate in candidates:
        try:
            record = read_json(candidate / 'job.json')
            settings = record.get('settings', {})
            statuses = {'complete', 'failed'} if score_dir else {'complete'}
            if (record.get('operation') != 'score' or record.get('status') not in statuses
                    or settings.get('predictions_sha256') != evaluation['predictions_sha256']
                    or path(settings.get('evaluation_dir', '')) != directory
                    or any(settings.get(k) != evaluation[k] for k in ('method', 'benchmark'))):
                raise ValueError(f'{candidate}: score job does not match this evaluation or is not finalized')
            filename = 'result_for_judge_eval.jsonl' if evaluation['benchmark'] == 'browsecomp' else 'judge.jsonl'
            result = candidate / 'results' / filename
            if not result.is_file():
                raise ValueError(f'{candidate}: no per-question LLM Judge results')
            matches.append((record.get('finished_at', ''), candidate.name, result))
        except (OSError, ValueError, KeyError):
            if score_dir:
                raise
    return max(matches)[2] if matches else None


def resolve_report_inputs(method, benchmark, evaluation_dir, score_dir=None):
    """Resolve a completed evaluation or an existing BrowseComp-Plus recovery."""
    from search_comp.experiments.benchmarks import resolve_evaluation
    from search_comp.experiments.runtime import digest, path, read_json

    directory = path(evaluation_dir)
    record = read_json(directory / 'job.json')
    if record.get('operation') != 'recover_browsecomp_plus':
        directory, evaluation = resolve_evaluation(method, benchmark, directory)
        return directory, path(evaluation['predictions']), resolve_score_file(directory, score_dir)

    if record.get('status') != 'complete':
        raise ValueError(f'{directory}: recovery must finish before generating a prediction report')
    if score_dir and path(score_dir) != directory:
        raise ValueError(f'{directory}: SCORE_DIR must be empty or point to the same recovery directory')

    recovery = read_json(directory / 'recovery.json')
    original = path(record['settings']['evaluation_dir'])
    source = read_json(original / 'job.json')
    if (path(recovery['original_evaluation']) != original
            or source.get('operation') != 'evaluate'
            or benchmark != 'browsecomp-plus'
            or source['settings'].get('benchmark') != benchmark
            or source['settings'].get('method') != method):
        raise ValueError(f'{directory}: recovery does not match the original evaluation, method, or benchmark')

    predictions = directory / 'predictions.jsonl'
    if digest(predictions) != recovery['predictions_sha256']:
        raise ValueError(f'{predictions}: recovered predictions SHA256 does not match recovery.json')
    judge_path = None
    if not record['settings']['retrieval_only']:
        judge_path = directory / 'results' / 'judge.jsonl'
        if not judge_path.is_file():
            raise FileNotFoundError(f'{judge_path}: no per-question LLM Judge results')
    return directory, predictions, judge_path


def judge_summary(row):
    parsed = row.get('judge_result') or {}
    source = row.get('score_source')
    score = row.get('score', parsed.get('correct', row.get('correct')))
    if source == 'judge_error' or row.get('judge_error'):
        status, score = 'error', None
    elif source and score in (0, 1) and score is not None:
        status, score = ('correct' if score else 'incorrect'), int(score)
    else:
        status, score = 'unscored', None
    attempts = row.get('judge_attempts') or []
    raw = row.get('judge_response', row.get('judge_raw'))
    if raw is None and attempts:
        raw = attempts[-1].get('raw')
    return dict(status=status, score=score, source=source,
                reasoning=parsed.get('reasoning'), confidence=parsed.get('confidence'),
                response=raw, attempts=attempts, raw=row)


def merge_judgments(rows, judge_path):
    by_id, by_question = {}, {}
    for index, row in enumerate(rows):
        if row.get('id') is not None:
            key = str(row['id'])
            if key in by_id:
                raise ValueError(f'duplicate prediction id: {key}')
            by_id[key] = index
        question = row.get('query') or row.get('question')
        if question:
            by_question.setdefault(question, []).append(index)
    used = set()
    for judgment in read_rows(judge_path):
        question = judgment.get('query') or judgment.get('question')
        if judgment.get('id') is not None:
            index = by_id.get(str(judgment['id']))
        else:
            # Online BrowseComp exports have item_index but no stable ID.
            choices = by_question.get(question, [])
            index = choices[0] if len(choices) == 1 else None
        if index is None:
            raise ValueError(f'{judge_path}: unmatched or ambiguous judge question')
        if index in used:
            raise ValueError(f'{judge_path}: duplicate judgment for prediction {index}')
        prediction_question = rows[index].get('query') or rows[index].get('question')
        if question and prediction_question and question != prediction_question:
            raise ValueError(f'{judge_path}: judge question does not match prediction')
        used.add(index)
        rows[index]['llm_judge'] = judge_summary(judgment)


def render_report(source, destination, template=DEFAULT_TEMPLATE, title=None, references=None, judge_path=None):
    rows = read_rows(source)
    if references is not None:
        for row in rows:
            if any(row.get(k) is not None for k in REFERENCE_FIELDS):
                continue
            question = row.get('question') or row.get('query')
            if question in references:
                row['answer'] = references[question]
    if judge_path is not None:
        merge_judgments(rows, Path(judge_path))
    payload = json.dumps(
        {'title': title or source.name, 'source': str(source), 'rows': rows,
         'judge_source': str(judge_path) if judge_path is not None else None},
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
