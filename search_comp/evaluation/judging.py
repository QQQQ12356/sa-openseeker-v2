"""Bounded, auditable judging shared by online and offline benchmarks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import urllib.error
import urllib.request

POLICY = 'stable-judge-v1'


class JudgeError(ValueError):
    """A safe error code (never include credentials or HTTP response bodies)."""
    def __init__(self, code, **details):
        super().__init__(code)
        self.details = details


def final_content(raw):
    """Ignore reasoning; never interpret a label from an unfinished thought."""
    if not isinstance(raw, str):
        raise JudgeError('non_text_content')
    text = raw.split('<|message|>')[-1].split('<|return|>')[0].strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text:
        raise JudgeError('unfinished_thinking')
    if not text:
        raise JudgeError('empty_judge_content')
    return text


def parse_ab(raw):
    """Only accept a complete unambiguous label, not a prefix of free-form prose."""
    text = final_content(raw)
    match = re.fullmatch(r'([AB])(?:\s*[.:]?\s*\[(CORRECT|INCORRECT)\])?[.!]?', text)
    if not match or (match[2] and match[2] != ('CORRECT' if match[1] == 'A' else 'INCORRECT')):
        raise JudgeError('invalid_or_conflicting_label')
    return int(match[1] == 'A')


def attempt_judgment(call, parse, attempts=3, delay=0.5):
    """Retry transport/format failures only. A valid negative label is final."""
    if type(attempts) is not int or attempts < 1:
        raise ValueError('judge attempts must be a positive integer')
    history = []
    for number in range(1, attempts + 1):
        entry = dict(attempt=number)
        try:
            raw = call()
            entry['raw'] = raw
            parsed = parse(raw)
            entry['valid'] = True
            history.append(entry)
            return parsed, history
        except Exception as error:
            if isinstance(error, JudgeError):
                entry.update(error.details)
            entry.update(valid=False, error=str(error) if isinstance(error, JudgeError) else type(error).__name__)
            history.append(entry)
            if isinstance(error, JudgeError) and str(error).startswith(('http_400', 'http_401', 'http_403', 'http_404')):
                break
            if number < attempts and delay:
                time.sleep(min(delay * 2 ** (number - 1), 4))
    return None, history


class ChatJudge:
    """Standard-library OpenAI-compatible client; explicit timeout and no SDK retries."""
    def __init__(self, urls, model, key, max_tokens=1024, timeout=120, enable_thinking=False):
        self.urls = [url.strip().rstrip('/') for url in urls.split(',') if url.strip()]
        if not self.urls:
            raise ValueError('judge requires at least one URL')
        if type(max_tokens) is not int or max_tokens < 1 or not 0 < timeout < float('inf'):
            raise ValueError('judge max_tokens/timeout must be positive and finite')
        self.model, self.key = model, key
        self.max_tokens, self.timeout, self.enable_thinking = max_tokens, timeout, enable_thinking

    def __call__(self, messages, endpoint=0):
        payload = dict(model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens,
                       stream=False, skip_special_tokens=False,
                       chat_template_kwargs=dict(enable_thinking=self.enable_thinking))
        request = urllib.request.Request(self.urls[endpoint % len(self.urls)] + '/chat/completions',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise JudgeError(f'http_{error.code}') from None
        try:
            choice = result['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise JudgeError('incomplete_judge_generation', raw=choice['message'].get('content'),
                                 finish_reason=choice['finish_reason'])
            raw = choice['message']['content']
            final_content(raw)
            return raw
        except (KeyError, IndexError, TypeError):
            raise JudgeError('invalid_chat_response') from None


def calibration(check, output):
    """Check obvious positive/negative cases; retain evidence and fail before scoring."""
    cases = [('positive', 'Paris', True), ('negative', 'Tokyo', False)]
    details = []
    for name, response, expected in cases:
        score, history = check('What is the capital of France?', response, 'Paris')
        details.append(dict(case=name, expected=expected, score=score, attempts=history,
                            passed=score is not None and score == expected))
    details.append(dict(case='empty', expected=False, score=False, passed=True, source='empty_response_rule'))
    report = dict(policy=POLICY, passed=all(row['passed'] for row in details), cases=details)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not report['passed']:
        raise JudgeError('judge_preflight_failed: inspect judge_preflight.json')
    return report


def score_online(rows, prompt, client, output, workers=1, attempts=3, preflight=True):
    def check(query, response, answer, endpoint=0):
        messages = [dict(role='system', content='Judge the response objectively.'),
                    dict(role='user', content=prompt.format(question=query, correct_answer=answer, response=response))]
        return attempt_judgment(lambda: client(messages, endpoint), parse_ab, attempts)

    if preflight and any(row['final_response'].strip() for row in rows):
        # All configured endpoints must pass, not only the first randomly selected server.
        for endpoint in range(len(client.urls)):
            calibration(lambda q, r, a: check(q, r, a, endpoint),
                        output / ('judge_preflight.json' if endpoint == 0 else f'judge_preflight_{endpoint}.json'))

    def score(pair):
        index, row = pair
        result = dict(type='item', item_index=index, query=row['query'], answer=row['answer'],
                      final_response=row['final_response'], tool_calls=row.get('tool_calls', 0),
                      generation_status=row.get('status'), judge_policy=POLICY,
                      judge_endpoint=index % len(client.urls))
        if not row['final_response'].strip():
            label, history = 0, []
            result['score_source'] = 'empty_response_rule'
        else:
            label, history = check(row['query'], row['final_response'], row['answer'], index)
            result['score_source'] = 'model' if label is not None else 'judge_error'
        result.update(score=label, is_correct=bool(label) if label is not None else None,
                      judge_raw=history[-1].get('raw', '') if history else '', judge_attempts=history)
        return result

    # Write every result, including unknowns. Never silently drop failures from the denominator.
    with (output / 'result_for_judge_eval.jsonl').open('x', encoding='utf-8') as target, \
            (output / 'judge.log').open('x', encoding='utf-8') as log:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, result in enumerate(executor.map(score, enumerate(rows)), 1):
                target.write(json.dumps(result, ensure_ascii=False) + '\n')
                target.flush()
                message = f'[stable judge] done={done}/{len(rows)} source={result["score_source"]}'
                print(message, flush=True)
                log.write(message + '\n')
                log.flush()
