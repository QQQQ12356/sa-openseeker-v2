"""Semantic scoring preserves the selected denominator and audits judge failures."""
import json

import pytest

from search_comp.data.openseeker import normalize
from search_comp.workflows.judge_openseeker import align_predictions, parse_judgment, score
from search_comp.evaluation.judging import JudgeError


def judgment(correct=True):
    return json.dumps(dict(extracted_final_answer='21 years old', reasoning='Equivalent age.',
                           correct=correct, confidence=99))


@pytest.mark.parametrize('raw', ['{}', '{"correct":"yes"}', '<think>unfinished',
                                  judgment().replace('99', '101')])
def test_invalid_judgment(raw):
    with pytest.raises(JudgeError):
        parse_judgment(raw)


def test_full_denominator_failed_responses_and_unknowns(tmp_path):
    base = dict(query='Age?', answer='21')
    rows = [dict(base, id=str(i), sample_index=i, **extra) for i, extra in enumerate([
        dict(status='completed', prediction='21 years old'),
        dict(status='tool_error', prediction='21'),
        dict(status='completed', prediction=''),
        {},
        dict(status='completed', prediction='bad judge output'),
    ])]
    calls = []
    def judge(prompt):
        calls.append(prompt)
        return 'invalid' if 'bad judge output' in prompt else judgment()
    output = tmp_path / 'scores'
    summary = score(rows, judge, output, attempts=1, preflight=False)
    assert summary['accuracy'] == 1 / 5
    assert summary['completed_predictions'] == 2
    assert summary['missing_predictions'] == 1
    assert summary['judge_errors'] == 1 and not summary['complete']
    assert len(calls) == 2
    details = [json.loads(line) for line in (output / 'judge.jsonl').read_text().splitlines()]
    assert details[1]['generation_status'] == 'tool_error'
    assert details[1]['final_response'] == '21'
    assert details[1]['score_source'] == 'empty_or_failed_response_rule'
    assert details[4]['correct'] is None
    with pytest.raises(FileExistsError):
        score(rows, judge, output)


def test_preflight_and_negative_label(tmp_path):
    def judge(prompt):
        return judgment('Response: Tokyo' not in prompt and 'Response: wrong' not in prompt)
    rows = [dict(id='a', sample_index=0, query='Age?', answer='21',
                 status='completed', prediction='wrong')]
    summary = score(rows, judge, tmp_path / 'score')
    assert summary['complete'] and summary['accuracy'] == 0
    assert json.loads((tmp_path / 'score/judge_preflight.json').read_text())['passed']


def test_align_missing_predictions_and_reject_misaligned_records(tmp_path):
    data = tmp_path / 'data.jsonl'
    rows = [dict(answer='21', messages=[dict(role='system', content='Answer.'),
        dict(role='user', content=f'Age {i}?'), dict(role='assistant', content='21')]) for i in range(3)]
    data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    first = normalize(rows[0], index=0)
    prediction = dict(first, sample_index=0, prediction='21 years old', status='completed')
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(json.dumps(prediction) + '\n')
    aligned = align_predictions(data, predictions, 3)
    assert len(aligned) == 3
    assert aligned[1]['query'] == 'Age 1?' and 'status' not in aligned[1]
    prediction['answer'] = 'wrong reference'
    predictions.write_text(json.dumps(prediction) + '\n')
    with pytest.raises(ValueError, match='reference differs'):
        align_predictions(data, predictions, 3)


def test_duplicate_labels_are_rejected():
    raw = judgment().replace('"correct": true', '"correct": true, "correct": false')
    with pytest.raises(JudgeError):
        parse_judgment(raw)


def test_shell_helper_forwards_configuration_and_is_optional(tmp_path):
    import os
    import subprocess
    import sys
    helper = tmp_path / 'capture.py'
    helper.write_text('import json,sys; print(json.dumps(sys.argv[1:]))')
    command = '''
source scripts/sh/benchmark/55_openseeker_judge.sh
PY=capture
capture() { "$TEST_PYTHON" "$CAPTURE_SCRIPT" "$@"; }
OUTPUT='/tmp/predictions with spaces.jsonl'
DATA='/tmp/data with spaces.jsonl'
LIMIT=20
run_openseeker_judge
'''
    env = {**os.environ, 'TEST_PYTHON': sys.executable, 'CAPTURE_SCRIPT': str(helper),
           'JUDGE_MODEL': '/models/judge with spaces', 'JUDGE_ATTEMPTS': '2',
           'JUDGE_ENABLE_THINKING': '1'}
    result = subprocess.run(['bash', '-c', command], env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[:2] == ['-m', 'search_comp.workflows.judge_openseeker']
    assert args[args.index('--judge-model') + 1] == env['JUDGE_MODEL']
    assert args[args.index('--output-dir') + 1] == '/tmp/predictions with spaces.judge'
    assert '--judge-enable-thinking' in args
    env['JUDGE_MODEL'] = ''
    result = subprocess.run(['bash', '-c', command], env=env, text=True, capture_output=True, check=True)
    assert result.stdout == ''
