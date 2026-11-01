"""Regressions for empty-answer false positives, bounded retries and preserved audits."""
import json

import pytest

from search_comp.evaluation.judging import (ChatJudge, JudgeError, attempt_judgment,
                                            calibration, parse_ab, score_online)
from search_comp.workflows.check_judge import audit


@pytest.mark.parametrize('raw,label', [('A', 1), ('B. [INCORRECT]', 0),
    ('<think>A is a possible label</think> B', 0), ('A. [CORRECT]', 1)])
def test_complete_labels_only(raw, label):
    assert parse_ab(raw) == label


@pytest.mark.parametrize('raw', ['', None, '<think>A', 'A or B', 'A. [INCORRECT]',
                                     'B. [CORRECT]', 'Answer: A', 'A\nB'])
def test_reject_ambiguous_or_incomplete_labels(raw):
    with pytest.raises(JudgeError):
        parse_ab(raw)


def test_retry_invalid_not_negative_and_no_secret_in_errors():
    values = iter(['invalid', 'B'])
    label, history = attempt_judgment(lambda: next(values), parse_ab, delay=0)
    assert label == 0 and len(history) == 2
    def unauthorized():
        raise JudgeError('http_401')
    label, history = attempt_judgment(unauthorized, parse_ab, attempts=3, delay=0)
    assert label is None and len(history) == 1
    def error():
        raise RuntimeError('SECRET')
    _, history = attempt_judgment(error, parse_ab, attempts=1)
    assert 'SECRET' not in json.dumps(history)


def test_online_empty_answers_never_call_model_and_unknown_is_retained(tmp_path):
    class Client:
        urls = ['http://localhost']
        def __call__(self, messages, endpoint):
            return 'invalid'
    rows = [dict(query='q', answer='gold', final_response=' \n'),
            dict(query='q2', answer='gold', final_response='nonempty')]
    score_online(rows, '{question} {correct_answer} {response}', Client(), tmp_path,
                 attempts=1, preflight=False)
    result = [json.loads(line) for line in (tmp_path / 'result_for_judge_eval.jsonl').read_text().splitlines()]
    assert result[0]['score'] == 0 and result[0]['judge_attempts'] == []
    assert result[1]['score'] is None and result[1]['score_source'] == 'judge_error'


def test_preflight_catches_always_positive_and_preserves_report(tmp_path):
    with pytest.raises(JudgeError, match='preflight_failed'):
        calibration(lambda *args: (1, []), tmp_path / 'judge_preflight.json')
    report = json.loads((tmp_path / 'judge_preflight.json').read_text())
    assert not report['passed'] and not report['cases'][1]['passed']


def test_chat_rejects_truncated_positive_and_disables_thinking(monkeypatch):
    from search_comp.evaluation import judging
    captured = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps(dict(choices=[dict(finish_reason='length', message=dict(content='A'))])).encode()
    def open_request(request, timeout):
        captured.append(json.loads(request.data))
        assert timeout == 120
        return Response()
    monkeypatch.setattr(judging.urllib.request, 'urlopen', open_request)
    with pytest.raises(JudgeError, match='incomplete'):
        ChatJudge('http://localhost/v1', 'judge', 'secret')([])
    assert captured[0]['temperature'] == 0
    assert captured[0]['chat_template_kwargs']['enable_thinking'] is False


def test_audit_corrects_empty_positive_without_modifying_original(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    row = dict(query='q', answer='a', final_response='')
    (source / 'result_for_judge.jsonl').write_text(json.dumps(row) + '\n')
    old = json.dumps(dict(row, type='item', item_index=0, score=1, judge_raw='A')) + '\n'
    (source / 'result_for_judge_eval.jsonl').write_text(old)
    result = audit(source, tmp_path / 'audit')
    assert result['accuracy_all_questions'] == 0
    assert result['corrections'][0]['original_score'] == 1
    assert (source / 'result_for_judge_eval.jsonl').read_text() == old
    with pytest.raises(FileExistsError):
        audit(source, tmp_path / 'audit')


def test_audit_rejects_misalignment_before_writing(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'result_for_judge.jsonl').write_text(json.dumps(dict(query='q', answer='a', final_response=''))+'\n')
    (source / 'result_for_judge_eval.jsonl').write_text(json.dumps(dict(type='item', item_index=0, query='other'))+'\n')
    with pytest.raises(ValueError, match='mismatch'):
        audit(source, tmp_path / 'audit')
    assert not (tmp_path / 'audit').exists()


def test_agent_retries_malformed_call_without_executing_or_using_gold():
    from search_comp.evaluation.tool_agent import Registry, run_agent
    class Tokenizer:
        def encode(self, text, add_special_tokens=False): return list(text.encode())
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = ''.join(m['content'] for m in messages)
            return list(text.encode()) if tokenize else text
    generated = iter(['<tool_calls_begin><tool_call>{bad json}</tool_call></tool_calls_end>',
                      '<answer>Recovered</answer>'])
    result = run_agent([dict(role='system', content='Use tools'), dict(role='user', content='question')],
                       Tokenizer(), lambda *_: (next(generated), True, 1), Registry({}),
                       max_turns=3, max_call_retries=1)
    assert result['status'] == 'completed' and result['prediction'] == 'Recovered'
    assert result['events'][0]['retry_requested']
    assert 'No tools were executed' in result['messages'][3]['content']


def test_visit_reports_http_auth_failure_without_repeated_requests(monkeypatch):
    from search_comp.evaluation.online_visit import StableVisit
    for key, value in dict(SUMMARY_API_URL='http://localhost/v1', SUMMARY_MODEL_NAME='judge',
                           SUMMARY_API_KEY='secret', JINA_API_KEYS='secret').items():
        monkeypatch.setenv(key, value)
    visit = StableVisit()
    calls = []
    def denied(url):
        calls.append(url)
        raise JudgeError('http_401')
    monkeypatch.setattr(visit, 'read', denied)
    response = visit.call(dict(url='https://example.org', goal='evidence'))
    assert 'page_fetch_failed (http_401)' in response
    assert len(calls) == 1 and 'secret' not in response


def test_real_local_http_chat_preflight_and_scoring(tmp_path):
    """Exercise request serialization, HTTP decoding and the complete scoring loop."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(payload)
            text = payload['messages'][-1]['content']
            label = 'B' if 'Tokyo' in text or 'wrong' in text else 'A'
            data = json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content=label))])).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ChatJudge(f'http://127.0.0.1:{server.server_port}/v1', 'judge', 'EMPTY')
        rows = [dict(query='q', answer='gold', final_response='wrong'),
                dict(query='q2', answer='gold', final_response='')]
        score_online(rows, '{question} {correct_answer} {response}', client, tmp_path, attempts=1)
        assert len(requests) == 3  # two calibration calls and one nonempty prediction
        scored = [json.loads(s) for s in (tmp_path / 'result_for_judge_eval.jsonl').read_text().splitlines()]
        assert [r['score'] for r in scored] == [0, 0]
        assert json.loads((tmp_path / 'judge_preflight.json').read_text())['passed']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('expression,expected', [
    ("'  prompt  '", '  prompt  '),
    ('\"\"\"\\nQuestion: {question}\\nAnswer: {correct_answer}\\nResponse: {response}\\n\"\"\".strip()',
     'Question: {question}\nAnswer: {correct_answer}\nResponse: {response}'),
])
def test_official_prompt_loader_accepts_literal_strip(tmp_path, expression, expected):
    from search_comp.evaluation.judging import load_judge_prompt
    path = tmp_path / 'prompt.py'
    path.write_text('JUDGE_PROMPT_BC_en = ' + expression + '\n')
    assert load_judge_prompt(path) == expected


@pytest.mark.parametrize('source', [
    "JUDGE_PROMPT_BC_en = str('prompt')",
    "JUDGE_PROMPT_BC_en = 'prompt'.replace('p', 'x')",
    "JUDGE_PROMPT_BC_en = 'prompt'.strip('p')",
    "JUDGE_PROMPT_BC_en = ''",
    "JUDGE_PROMPT_BC_en = 'one'\nJUDGE_PROMPT_BC_en = 'two'",
    "OTHER_PROMPT = 'prompt'",
])
def test_prompt_loader_rejects_dynamic_expressions(tmp_path, source):
    from search_comp.evaluation.judging import load_judge_prompt
    path = tmp_path / 'prompt.py'
    path.write_text(source + '\n')
    with pytest.raises(ValueError):
        load_judge_prompt(path)


def test_prompt_loader_never_executes_source(tmp_path):
    from search_comp.evaluation.judging import load_judge_prompt
    marker = tmp_path / 'must_not_exist'
    path = tmp_path / 'prompt.py'
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
                    "JUDGE_PROMPT_BC_en = 'prompt'.strip()\n")
    assert load_judge_prompt(path) == 'prompt'
    assert not marker.exists()


def test_preflight_cli_reads_official_strip_prompt(tmp_path, monkeypatch, capsys):
    import sys
    from search_comp.workflows import check_judge
    official = tmp_path / 'official'
    (official / 'eval').mkdir(parents=True)
    (official / 'eval/prompt.py').write_text(
        "JUDGE_PROMPT_BC_en = '''\\n{question} {correct_answer} {response}\\n'''.strip()\n")
    monkeypatch.setattr(check_judge, 'ensure_official', lambda _: None)
    for name, value in dict(SCORER_URLS='http://localhost/v1', SCORER_MODEL_NAME='judge', SCORER_API_KEY='EMPTY').items():
        monkeypatch.setenv(name, value)
    calls = []
    class Client:
        model = 'judge'
        urls = ['http://localhost/v1']
        def __init__(self, *args): pass
        def __call__(self, messages, endpoint):
            calls.append(messages)
            return 'B' if 'Tokyo' in messages[-1]['content'] else 'A'
    monkeypatch.setattr(check_judge, 'ChatJudge', Client)
    output = tmp_path / 'preflight'
    monkeypatch.setattr(sys, 'argv', ['check', 'preflight', '--official-dir', str(official),
                                    '--output-dir', str(output)])
    check_judge.main()
    assert len(calls) == 2
    report = json.loads((output / 'judge_preflight_0.json').read_text())
    assert report['passed']
