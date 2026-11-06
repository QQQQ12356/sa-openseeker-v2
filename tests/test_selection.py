"""Length, correctness and observation filters use actual serialized training lengths."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from search_comp.data.openseeker import VERSION, normalize, encode
from search_comp.data.selection import Selection
from test_openseeker import polar, Tokenizer


def measured():
    row = normalize(polar())
    row['correctness'] = 'Correct'
    row['lengths'] = {'mock': {'total': 100}, 'other': {'total': 101}}
    return row


def test_inclusive_boundaries_exact_length_and_tokenizer_identity():
    row = measured()
    assert Selection(100, 100, tokenizer='mock').rejection(row) is None
    assert Selection(100, 100).rejection(row) == 'over_length'
    assert Selection(101, 200, tokenizer='mock').rejection(row) == 'under_length'
    assert Selection(1, 200, exact_length=101, tokenizer='mock').rejection(row) == 'not_exact_length'
    with pytest.raises(ValueError, match='no length measurements'):
        Selection(tokenizer='missing').rejection(row)


@pytest.mark.parametrize('value', ['Incorrect', None, '', 'correct'])
def test_default_requires_explicit_correct(value):
    row = measured(); row['correctness'] = value
    assert Selection().rejection(row) == 'incorrect'
    assert Selection(correctness='any').rejection(row) is None


def test_observation_filters_only_use_observation_bodies():
    row = measured()
    assert Selection(observations='with').rejection(row) is None
    assert Selection(observations='without').rejection(row) == 'has_observation'
    row['messages'] = row['messages'][:2] + [row['messages'][-1]]
    row['messages'][-1]['content'] += '<tool_response>quoted</tool_response>'
    assert Selection(observations='without').rejection(row) is None
    assert Selection(observations='with').rejection(row) == 'no_observation'
