"""Stop positions match full decoding without decoding each generated prefix."""
import string
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from search_comp.evaluation.tool_agent import ModelGenerator
from search_comp.generation_stopping import GeneratedTextStop
from search_comp.models.beacon_common import BeaconCausalLMMixin


STOPS = ['</tool_calls_end>', '</answer>', '<|im_end|>']


@pytest.fixture
def tokenizer():
    # An offline tokenizer with both split and merged spellings of the markers.
    tokens = list(string.printable) + [
        '[UNK]', 'prefix</ans', 'wer>suffix', 'x</answer>tail',
        '</answer>tail', '<|im_end|>', '中文',
    ]
    backend = Tokenizer(models.BPE(dict(zip(tokens, range(len(tokens)))), [], unk_token='[UNK]'))
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                  additional_special_tokens=['<|im_end|>'],
                                  clean_up_tokenization_spaces=False)


@pytest.mark.parametrize('pieces', [
    list('ordinary text without a marker'),
    list('中文</answer>'),
    list('</tool_calls_end>'),
    list('<|im_end|>'),
    ['prefix</ans', 'wer>suffix'],
    ['x</answer>tail'],
    ['</answer>tail'],
    list('</answe'),
])
def test_first_stop_matches_full_decode(tokenizer, pieces):
    prompt = tokenizer.encode('prompt </answer> <|im_end|>', add_special_tokens=False)
    stop = GeneratedTextStop(tokenizer, STOPS, start_length=len(prompt))
    ids = tokenizer.convert_tokens_to_ids(pieces)
    for length in range(1, len(ids) + 1):
        generated = ids[:length]
        expected = any(s in tokenizer.decode(generated, skip_special_tokens=False) for s in STOPS)
        with patch.object(tokenizer, 'decode', side_effect=AssertionError('per-step decode')):
            actual = bool(stop(torch.tensor([prompt + generated])))
        assert actual == expected
        if expected:
            break


def test_long_generation_uses_bounded_suffix_without_decode(tokenizer):
    stop = GeneratedTextStop(tokenizer, STOPS)
    assert stop.suffix_length <= max(map(len, STOPS))
    ids = tokenizer.encode('a' * 8192, add_special_tokens=False)
    with patch.object(tokenizer, 'decode', side_effect=AssertionError('per-step decode')):
        assert not stop(torch.tensor([ids]))
        ids.extend(tokenizer.encode('</answer>', add_special_tokens=False))
        assert stop(torch.tensor([ids]))


@pytest.mark.parametrize('kind', ['plain', 'beacon'])
@pytest.mark.parametrize('ending', ['</answer>', '</tool_calls_end>', '<|im_end|>', 'budget', 'eos'])
def test_generation_keeps_stop_tokens_eos_and_budget(tokenizer, kind, ending):
    eos = tokenizer.convert_tokens_to_ids('!')
    text = 'hello' + (ending if ending not in ('budget', 'eos') else '!')
    scripted = tokenizer.encode(text + 'unused', add_special_tokens=False)
    limit = 5 if ending == 'budget' else len(scripted)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(eos_token_id=eos, bos_token_id=None, to_dict=lambda: {})
            self.beacon_config = SimpleNamespace(to_dict=lambda: {})
            self.position = 0

        def generate(self, input_ids, generation_config, stopping_criteria, **kwargs):
            for token in scripted[:generation_config.max_new_tokens]:
                input_ids = torch.cat((input_ids, torch.tensor([[token]])), dim=1)
                if token == eos or stopping_criteria(input_ids, None):
                    break
            return input_ids

        def logits(self):
            value = torch.full((1, len(tokenizer)), -100.0)
            value[0, scripted[self.position]] = 100.0
            return value

        def prefill_and_get_cache(self, *args, **kwargs):
            return SimpleNamespace(_visible_token_ids=set()), self.logits()

        def decode_step(self, token):
            self.position += 1
            return self.logits()

        def beacon_generate(self, **kwargs):
            return BeaconCausalLMMixin._beacon_generate_loop(
                self, kwargs['input_ids'], kwargs['regions'], kwargs['max_new_tokens'],
                False, 1.0, 1.0, kwargs['eos_token_ids'], kwargs['stop_texts'], kwargs['tokenizer'])

    generator = ModelGenerator(Model(), tokenizer, beacon=kind == 'beacon', reuse_plain_cache=False)
    prompt = tokenizer.encode('prompt </answer>', add_special_tokens=False)
    # Only the final response decode is allowed; stopping uses tensor matching.
    with patch.object(tokenizer, 'decode', wraps=tokenizer.decode) as decode:
        actual, ended, count = generator({'input_ids': prompt, 'regions': []}, limit)
    expected_tokens = tokenizer.encode('hello' + (ending if ending not in ('budget', 'eos') else ''),
                                       add_special_tokens=False)
    if ending == 'eos':
        expected_tokens.append(eos)
    assert count == len(expected_tokens)
    assert ended == (ending != 'budget')
    assert actual == tokenizer.decode(expected_tokens, skip_special_tokens=False).split('<|im_end|>', 1)[0]
    assert decode.call_count == 1


def test_minimal_tokenizer_preserves_legacy_matching():
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ''.join(map(chr, ids)))
    stop = GeneratedTextStop(tokenizer, STOPS, start_length=1)
    assert not stop(torch.tensor([[0] + list(map(ord, 'unfinished'))]))
    assert stop(torch.tensor([[0] + list(map(ord, 'ok</answer>'))]))
