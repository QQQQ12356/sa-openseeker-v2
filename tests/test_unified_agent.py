"""Shared prompt, data migration and native environment contracts."""
import copy
import json

import pytest

from search_comp.data.agent_prompt import (build_system_prompt, initial_messages, prompt_metadata,
                                           resolve_system_prompt, REPLACED_SYSTEM_VERSION, render_system, TOOLS_RE)
from search_comp.data.openseeker import VERSION, OpenSeekerDataset, encode, normalize
from search_comp.trainer.plain_sft_trainer import PlainSFTDataset
from test_openseeker import Tokenizer, polar


def test_plain_beacon_and_online_share_tokens_and_every_generation_prefix(tmp_path):
    row = normalize(polar())
    path = tmp_path / 'train.jsonl'
    path.write_text(json.dumps(row) + '\n')
    tokenizer = Tokenizer()
    beacon = OpenSeekerDataset(path, tokenizer)[0]
    plain = PlainSFTDataset(path, tokenizer, data_mode='openseeker')[0]
    assert plain['input_ids'].tolist() == beacon['input_ids']
    assert plain['labels'].tolist() == beacon['labels']
    assert row['messages'][:2] == initial_messages(row['question'], system=row['messages'][0]['content'])
    for index, message in enumerate(row['messages']):
        if message['role'] == 'assistant':
            prefix = encode(row['messages'][:index], tokenizer, generation_prompt=True)
            assert prefix['input_ids'] == beacon['input_ids'][:len(prefix['input_ids'])]
            assert prefix['regions'] == [(start, end) for start, end in beacon['regions']
                                         if end <= len(prefix['input_ids'])]


def test_original_prepared_data_is_reusable_and_v4_is_not_guessed(tmp_path):
    row = normalize(polar())
    row['lengths'] = {'original': {'total': 10}}
    original = copy.deepcopy(row)
    assert normalize(row) == original
    assert row['messages'] == polar()['trajectory']
    path = tmp_path / 'train.jsonl'
    path.write_text(json.dumps(row) + '\n')
    assert OpenSeekerDataset(path, Tokenizer())[0]['input_ids'] == encode(row['messages'], Tokenizer())['input_ids']
    row['protocol'] = REPLACED_SYSTEM_VERSION
    with pytest.raises(ValueError, match='rebuild from original'):
        normalize(row)
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='original raw/v3'):
        OpenSeekerDataset(path, Tokenizer())[0]
    with pytest.raises(ValueError, match='original raw/v3'):
        resolve_system_prompt(path)


def test_offline_changes_only_tools_and_required_constraints():
    base = build_system_prompt()
    local = build_system_prompt(local=True, answer_instruction='Cite [docid].')
    assert local[:TOOLS_RE.search(local).start()] == base[:TOOLS_RE.search(base).start()]
    assert local[TOOLS_RE.search(local).end():].startswith(base[TOOLS_RE.search(base).end():])
    assert 'No external websites' in local and local.endswith('Cite [docid].')
    assert 'web searches' not in local and 'top 10 results' not in local
    assert 'according to the goal' not in local
    search_only = build_system_prompt(local=True, allow_visit=False)
    schemas = [json.loads(line) for line in TOOLS_RE.search(search_only).group(1).strip().splitlines()]
    assert [tool['name'] for tool in schemas] == ['search']
    assert schemas[0]['parameters'] == json.JSONDecoder().raw_decode(TOOLS_RE.search(base).group(1).lstrip())[0]['parameters']
    assert 'Do not call visit.' in search_only
    assert prompt_metadata(base)['system_prompt_sha256'] != prompt_metadata(local)['system_prompt_sha256']


def test_training_prompt_resolution_is_exact_and_rejects_ambiguous_sources(tmp_path):
    row = polar()
    path = tmp_path / 'train.jsonl'
    path.write_text(json.dumps(row) + '\n')
    system, source = resolve_system_prompt(path)
    assert system == row['trajectory'][0]['content']
    assert source == str(path)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps({'training_config': {'train_data_path': str(path)}}))
    assert resolve_system_prompt(run_dir=run) == (system, source)
    custom = tmp_path / 'override.jsonl'
    other = copy.deepcopy(row)
    other['trajectory'][0]['content'] = 'Another original system'
    custom.write_text(json.dumps(other) + '\n')
    assert resolve_system_prompt(custom, run)[0] == 'Another original system'
    path.write_text(json.dumps(row) + '\n' + json.dumps(other) + '\n')
    with pytest.raises(ValueError, match='mixed training systems'):
        resolve_system_prompt(path)
    with pytest.raises(FileNotFoundError):
        resolve_system_prompt(tmp_path / 'missing.jsonl')


def test_separate_tool_schemas_render_identically_without_changing_stored_system(tmp_path):
    row = polar()
    row['tools'] = [{'name': 'search', 'parameters': {'type': 'object'}}]
    normalized = normalize(row)
    assert normalized['messages'][0] == row['trajectory'][0]
    system = render_system(normalized['messages'][0]['content'], normalized['tools'])
    path = tmp_path / 'train.jsonl'
    path.write_text(json.dumps(normalized) + '\n')
    assert resolve_system_prompt(path)[0] == system
    full = encode(normalized['messages'], Tokenizer(), tools=normalized['tools'])
    for index, message in enumerate(normalized['messages']):
        if message['role'] == 'assistant':
            prefix = encode(normalized['messages'][:index], Tokenizer(), generation_prompt=True, tools=normalized['tools'])
            assert prefix['input_ids'] == full['input_ids'][:len(prefix['input_ids'])]
    assert render_system(system, normalized['tools']) == system


def test_effective_beacon_layout_is_resolved_before_checkpoint_configuration():
    from search_comp.models.beacon_config import configure_whole_trajectory
    cfg = {'beacon': {'beacon_keep_window': 1024, 'beacon_ratio': 16}}
    configure_whole_trajectory(cfg)
    assert cfg['beacon'] == dict(beacon_information_mode='whole', beacon_window_enabled=False,
                                beacon_doc_region_split=False, beacon_keep_window=0, beacon_ratio=16)


@pytest.mark.parametrize('beacon', [False, True])
def test_native_decoding_uses_same_greedy_policy_and_multiple_eos(beacon):
    import torch
    from transformers import GenerationConfig, Qwen3Config
    from search_comp.evaluation.tool_agent import ModelGenerator
    from search_comp.models.beacon_config import BeaconConfig

    class TextTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return ''.join('A' if token == 65 else '<eos>' for token in tokens)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = Qwen3Config(eos_token_id=[2, 9])
            self.beacon_config = BeaconConfig()
            self.generation_config = GenerationConfig(num_beams=3, repetition_penalty=2.0)

        def generate(self, input_ids, generation_config, **kwargs):
            assert generation_config.num_beams == 1
            assert generation_config.do_sample is False
            assert generation_config.repetition_penalty == 1.0
            assert generation_config.eos_token_id == [2, 9]
            return torch.cat([input_ids, torch.tensor([[65, 9]])], dim=1)

        def beacon_generate(self, eos_token_ids, reuse_cache, **kwargs):
            assert eos_token_ids == [2, 9]
            assert reuse_cache is False
            return torch.tensor([[65, 9]])

    text, ended, count = ModelGenerator(Model(), TextTokenizer(), beacon)(
        {'input_ids': [3, 4], 'regions': []}, 8)
    assert text == 'A<eos>' and ended and count == 2
