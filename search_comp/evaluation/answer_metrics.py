"""Supplementary answer EM/F1, with missing and failed predictions scoring zero."""
import re

from .em_f1 import compute_em, compute_f1, extract_answer


def answer_metrics(questions, predictions, strip_citations=False):
    """Match by query for online exports or by ID for fixed-corpus exports.

    Scores use the generated final answer, never a judge-extracted answer.
    BrowseComp citation markers are excluded only when explicitly requested.
    """
    if not questions:
        raise ValueError('answer metrics require nonempty questions')
    by_query = {p['query']: p for p in predictions if 'query' in p}
    by_id = {str(p['id']): p for p in predictions if 'id' in p}
    em = f1 = 0.0
    for row in questions:
        pred = by_query.get(row.get('query')) or by_id.get(str(row.get('id'))) or {}
        if pred.get('status', 'completed') != 'completed':
            continue
        response = pred.get('final_response', pred.get('prediction', ''))
        answer = extract_answer(response)
        answer = response.strip() if answer is None else answer
        if strip_citations:
            answer = re.sub(r'\[[^\[\]\n]+\]', '', answer).strip()
        if not answer:
            continue
        golds = row['answer'] if isinstance(row['answer'], list) else [row['answer']]
        em += max(compute_em(answer, gold) for gold in golds)
        f1 += max(compute_f1(answer, gold) for gold in golds)
    return dict(em=em / len(questions), f1=f1 / len(questions))
