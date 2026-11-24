"""Standalone reports must preserve records without allowing HTML injection."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'search_comp' / 'workflows' / 'render_predictions_html.py'
spec = importlib.util.spec_from_file_location('render_predictions_html', SCRIPT)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_report_preserves_records_and_escapes_script_termination(tmp_path):
    source = tmp_path / 'results.jsonl'
    row = {'question': '中文 </script><script>alert(1)</script> &',
           'messages': [{'role': 'assistant', 'content': '<think>思考</think>'}],
           'exact_match': False, 'f1': 0.25, 'custom_metric': {'value': 9}}
    source.write_text('\ufeff' + json.dumps(row, ensure_ascii=False) + '\n\n', encoding='utf-8')
    destination = tmp_path / 'report' / 'results.html'
    assert renderer.render_report(source, destination) == 1
    markup = destination.read_text(encoding='utf-8')
    data = markup.split('<script id="report-data" type="application/json">', 1)[1].split('</script>', 1)[0]
    assert '<' not in data and '&' not in data
    assert json.loads(data)['rows'] == [row]
    assert '__REPORT_DATA__' not in markup


@pytest.mark.parametrize('content', ['{}\ninvalid\n', '{}\n[]\n'])
def test_bad_rows_include_source_line_number(tmp_path, content):
    source = tmp_path / 'invalid.jsonl'
    source.write_text(content)
    with pytest.raises(ValueError, match=r'invalid.jsonl:2:'):
        renderer.read_rows(source)


def test_empty_report_and_template_validation(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('\n')
    assert renderer.render_report(source, tmp_path / 'empty.html') == 0
    template = tmp_path / 'bad.html'
    template.write_text('<html></html>')
    with pytest.raises(ValueError, match='exactly one'):
        renderer.render_report(source, tmp_path / 'bad-report.html', template)


def test_missing_references_join_by_question_without_overwriting_existing_answers(tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text(json.dumps({'id': 999, 'query': 'Who?', 'answer': 'Alice'}) + '\n')
    source = tmp_path / 'predictions.jsonl'
    rows = [{'id': 1, 'question': 'Who?'},
            {'query': 'Who?', 'answer': 'Existing'}, {'query': 'Unknown'}]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    destination = tmp_path / 'report.html'
    renderer.render_report(source, destination, references=renderer.reference_index(dataset))
    data = destination.read_text().split('<script id="report-data" type="application/json">')[1].split('</script>')[0]
    rendered = json.loads(data)['rows']
    assert rendered == [dict(rows[0], answer='Alice'), rows[1], rows[2]]
    assert renderer.read_rows(source) == rows


def test_conflicting_reference_answers_are_rejected(tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('\n'.join(json.dumps({'query': 'Who?', 'answer': answer})
                                 for answer in ['Alice', 'Bob']))
    with pytest.raises(ValueError, match='conflicting reference answers'):
        renderer.reference_index(dataset)


def test_client_role_boundaries_and_interactions(tmp_path):
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for offline client execution checks')
    source = tmp_path / 'roles.jsonl'
    rows = [dict(id='first', answer='Alice', prediction='Alice', status='completed',
                 llm_judge=dict(status='correct', score=1, source='model', confidence=92,
                                reasoning='Matches the reference', response='correct: yes', raw={}),
                 exact_match=False, f1=0.1, messages=[
        dict(role='system', content='Tools: <tool_calls_begin><tool_call>{"name":"example"}</tool_call></tool_calls_end>\nFinal format: <answer>Beijing</answer>'),
        dict(role='user', content='Who?'),
        dict(role='assistant', content='<think>Quoted <answer>example inside thought</answer></think>\n<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":["Alice"]}}</tool_call><tool_call>{"name":"visit","arguments":{"url":"https://example.org"}}</tool_call></tool_calls_end><answer>Alice</answer>'),
        dict(role='user', content='<tool_response>Evidence: <answer>literal webpage content</answer></tool_response>')]),
        dict(id='second', question='Second question', status='completed', exact_match=True, f1=1.0,
             llm_judge=dict(status='error', score=None, source='judge_error', raw={}))]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    destination = tmp_path / 'roles.html'
    renderer.render_report(source, destination)
    harness = Path(__file__).with_name('html_report_client_test.js')
    result = subprocess.run([node, str(harness), str(destination)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_merge_judge_formats_by_id_without_changing_predictions(tmp_path):
    source = tmp_path / 'predictions.jsonl'
    original = [dict(id=str(i), query=f'Q{i}', exact_match=False) for i in range(4)]
    source.write_text('\n'.join(map(json.dumps, original)), encoding='utf-8')
    judge = tmp_path / 'judge.jsonl'
    judgments = [dict(id=1, score_source='model', judge_result=dict(correct=True,
                      reasoning='中文 </script> matches', confidence=95), judge_response='correct: yes'),
                 dict(id='0', score=0, score_source='empty_or_failed_response_rule'),
                 dict(id='2', score=None, score_source='judge_error', judge_attempts=[dict(raw='invalid')])]
    judge.write_text('\n'.join(map(json.dumps, judgments)), encoding='utf-8')
    destination = tmp_path / 'report.html'
    renderer.render_report(source, destination, judge_path=judge)
    payload = destination.read_text(encoding='utf-8').split('<script id="report-data" type="application/json">')[1].split('</script>')[0]
    data = json.loads(payload)
    assert data['judge_source'] == str(judge)
    rows = data['rows']
    assert [r.get('llm_judge', {}).get('status') for r in rows] == ['incorrect', 'correct', 'error', None]
    assert rows[1]['llm_judge']['confidence'] == 95
    assert rows[1]['llm_judge']['reasoning'] == '中文 </script> matches'
    assert rows[2]['llm_judge']['response'] == 'invalid'
    assert all(r['exact_match'] is False for r in rows)
    assert renderer.read_rows(source) == original
    assert '<' not in payload
    assert renderer.judge_summary(dict(correct=False))['status'] == 'unscored'  # retrieval-only
    assert renderer.judge_summary(dict(score=1, score_source='model'))['status'] == 'correct'


def test_judge_join_rejects_duplicates_unknown_ids_and_wrong_questions(tmp_path):
    judge = tmp_path / 'judge.jsonl'
    row = dict(id='q', query='Question')
    for judgments in ([dict(id='unknown')], [dict(id='q', query='Wrong')],
                      [dict(id='q'), dict(id='q')]):
        judge.write_text('\n'.join(map(json.dumps, judgments)))
        with pytest.raises(ValueError):
            renderer.merge_judgments([dict(row)], judge)
    judge.write_text(json.dumps(dict(query='Question', score=1, score_source='model')))
    rows = [dict(row)]
    renderer.merge_judgments(rows, judge)
    assert rows[0]['llm_judge']['score'] == 1
    with pytest.raises(ValueError, match='ambiguous'):
        renderer.merge_judgments([dict(query='Question'), dict(query='Question')], judge)


def test_score_selection_checks_provenance_and_uses_latest_complete_job(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    evaluation = dict(method='plain', benchmark='hotpotqa', predictions_sha256='expected')
    write(tmp_path / 'evaluation.json', evaluation)
    settings = dict(evaluation, evaluation_dir=str(tmp_path))
    assert renderer.resolve_score_file(tmp_path) is None
    def score(name, status='complete', sha='expected'):
        directory = tmp_path / 'scores' / name
        write(directory / 'job.json', dict(operation='score', status=status, finished_at=name,
              settings=dict(settings, predictions_sha256=sha)))
        write(directory / 'results/judge.jsonl', dict(id='q', score=1, score_source='model'))
        return directory
    first, latest = score('01'), score('02')
    wrong = score('04', sha='different')
    failed = score('03', status='failed')
    assert renderer.resolve_score_file(tmp_path) == latest / 'results/judge.jsonl'
    assert renderer.resolve_score_file(tmp_path, first) == first / 'results/judge.jsonl'
    assert renderer.resolve_score_file(tmp_path, failed) == failed / 'results/judge.jsonl'
    with pytest.raises(ValueError, match='does not match'):
        renderer.resolve_score_file(tmp_path, wrong)


def test_report_launcher_has_optional_score_selection():
    import runpy
    from search_comp.experiments.shell_config import launcher_defaults
    root = SCRIPT.parents[2]
    schema = runpy.run_path(str(root / 'scripts/tools/py/81_prediction_report.py'))['SCHEMA']
    defaults = launcher_defaults(root / 'scripts/tools/sh/81_prediction_report.sh', schema)
    assert defaults['SCORE_DIR'] is None


@pytest.mark.parametrize('retrieval_only', [False, True])
def test_report_launcher_reads_existing_recovery(tmp_path, monkeypatch, retrieval_only):
    import runpy
    from search_comp.experiments import runtime, shell_config

    evaluation = tmp_path / 'evaluation'
    recovery = evaluation / 'recovery' / 'partial_score'
    results = recovery / 'results'
    results.mkdir(parents=True)
    runtime.write_json(evaluation / 'job.json', dict(operation='evaluate', status='failed',
                       settings=dict(method='beacon', benchmark='browsecomp-plus')))
    runtime.write_json(recovery / 'job.json', dict(operation='recover_browsecomp_plus', status='complete',
                       settings=dict(evaluation_dir=evaluation, retrieval_only=retrieval_only)))
    predictions = recovery / 'predictions.jsonl'
    predictions.write_text(json.dumps(dict(id='q', query='Question', status='completed')) + '\n', encoding='utf-8')
    recovery_manifest = dict(original_evaluation=evaluation, predictions_sha256=runtime.digest(predictions))
    runtime.write_json(recovery / 'recovery.json', recovery_manifest)
    judge = results / 'judge.jsonl'
    judgment = dict(id='q', correct=True, score_source='model') if not retrieval_only else dict(id='q')
    judge.write_text(json.dumps(judgment) + '\n', encoding='utf-8')

    settings = dict(ENVIRONMENT={}, METHOD='beacon', BENCHMARK='browsecomp-plus',
                    EVALUATION_DIR=str(recovery), SCORE_DIR=None, TITLE='Recovery')
    monkeypatch.setattr(shell_config, 'load_settings', lambda *_: settings)
    monkeypatch.setattr(runtime, 'configure', lambda _: None)
    launcher = SCRIPT.parents[2] / 'scripts/tools/py/81_prediction_report.py'
    runpy.run_path(str(launcher), run_name='__main__')

    report, = recovery.glob('reports/*/predictions.html')
    payload = report.read_text(encoding='utf-8').split('<script id="report-data" type="application/json">')[1]
    data = json.loads(payload.split('</script>')[0])
    assert data['source'] == str(predictions)
    assert data['judge_source'] == (None if retrieval_only else str(judge))
    assert data['rows'][0].get('llm_judge', {}).get('status') == (None if retrieval_only else 'correct')
    assert not (recovery / 'evaluation.json').exists()

    with pytest.raises(ValueError, match='method, or benchmark'):
        renderer.resolve_report_inputs('plain', 'browsecomp-plus', recovery)
    with pytest.raises(ValueError, match='same recovery directory'):
        renderer.resolve_report_inputs('beacon', 'browsecomp-plus', recovery, tmp_path / 'other_score')
    predictions.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        renderer.resolve_report_inputs('beacon', 'browsecomp-plus', recovery)
