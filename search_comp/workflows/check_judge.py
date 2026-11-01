"""Audit recorded scores offline or calibrate the configured online judge."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from search_comp.evaluation.benchmark_data import sha256
from search_comp.evaluation.judging import ChatJudge, POLICY, attempt_judgment, calibration, parse_ab
from search_comp.workflows.benchmark_openseeker import ROOT, audit_scores, ensure_official


def audit(source, output):
    """Correct only deterministic empty-response errors, preserving all original evidence."""
    raw_path = source / 'result_for_judge.jsonl'
    eval_path = source / 'result_for_judge_eval.jsonl'
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get('query'), str) or not isinstance(r.get('answer'), str)
                       or not isinstance(r.get('final_response'), str) for r in rows):
        raise ValueError('audit requires nonempty aligned query/answer/final_response JSONL')
    scored = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    corrections = []
    results = []
    seen = set()
    for item in scored:
        if item.get('type') != 'item':
            continue
        index = item.get('item_index')
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError('foreign or duplicate judge item_index')
        seen.add(index)
        original = rows[index]
        for key in ('query', 'answer', 'final_response'):
            if item.get(key) != original.get(key):
                raise ValueError(f'judge/input mismatch: index={index} field={key}')
        if not isinstance(original.get('final_response'), str):
            raise ValueError('prediction final_response must be text')
        if not original['final_response'].strip():
            if item.get('score') != 0 or type(item.get('score')) is bool:
                corrections.append(dict(item_index=index, original_score=item.get('score'), corrected_score=0))
            item.update(original_score=item.get('score'), score=0, is_correct=False,
                        score_source='empty_response_rule', audit_policy=POLICY)
        results.append(item)
    # Validation is complete before creating a new output directory.
    output.mkdir(parents=True, exist_ok=False)
    destination = output / eval_path.name
    destination.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    summary = audit_scores(destination, len(rows))
    summary.update(policy=POLICY, audit_only=True, model_rejudged=False, count=len(rows),
                   empty_predictions=sum(not r['final_response'].strip() for r in rows),
                   corrections=corrections, source=str(source),
                   input_sha256=sha256(raw_path), original_eval_sha256=sha256(eval_path),
                   limitation='Only empty-response labels corrected; nonempty model labels not independently verified.')
    (output / 'summary.metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'preflight'])
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--official-dir', type=Path, default=ROOT / 'outputs/vendor/OpenSeeker')
    parser.add_argument('--judge-attempts', type=int, default=3)
    parser.add_argument('--judge-max-tokens', type=int, default=1024)
    parser.add_argument('--judge-timeout', type=float, default=120)
    parser.add_argument('--judge-enable-thinking', action='store_true')
    args = parser.parse_args()
    if args.judge_attempts < 1 or args.judge_max_tokens < 1 or not 0 < args.judge_timeout < float('inf'):
        parser.error('judge budgets must be positive and finite')
    output = (args.output_dir or ROOT / 'outputs/results' / ('judge_' + args.action + '_' +
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).expanduser().resolve()
    if args.action == 'audit':
        if args.result_dir is None:
            parser.error('audit requires --result-dir')
        result = audit(args.result_dir.expanduser().resolve(), output)
    else:
        from search_comp.workflows.benchmark_openseeker import require_environment
        require_environment('native', generate=False)
        ensure_official(args.official_dir)
        tree = ast.parse((args.official_dir / 'eval/prompt.py').read_text())
        prompt = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'JUDGE_PROMPT_BC_en' for t in node.targets))
        client = ChatJudge(os.environ['SCORER_URLS'], os.environ['SCORER_MODEL_NAME'], os.environ['SCORER_API_KEY'],
                           args.judge_max_tokens, args.judge_timeout, args.judge_enable_thinking)
        output.mkdir(parents=True, exist_ok=False)
        result = dict(policy=POLICY, judge_model=client.model, endpoints=[],
                      judge_max_tokens=args.judge_max_tokens, judge_timeout=args.judge_timeout,
                      judge_enable_thinking=args.judge_enable_thinking,
                      prompt_sha256=sha256(args.official_dir / 'eval/prompt.py'))
        for endpoint in range(len(client.urls)):
            def check(query, response, answer):
                messages = [dict(role='system', content='Judge the response objectively.'),
                    dict(role='user', content=prompt.format(question=query, correct_answer=answer, response=response))]
                return attempt_judgment(lambda: client(messages, endpoint), parse_ab, args.judge_attempts)
            report = calibration(check, output / f'judge_preflight_{endpoint}.json')
            result['endpoints'].append(report)
        (output / 'summary.metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), **result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
