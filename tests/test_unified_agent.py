"""Shared prompt, data migration and native environment contracts."""
import copy
import json

import pytest

from search_comp.data.agent_prompt import build_system_prompt, initial_messages, prompt_metadata
from search_comp.data.openseeker import LEGACY_VERSION, VERSION, OpenSeekerDataset, encode, normalize
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
    assert row['messages'][:2] == initial_messages(row['question'])
    for index, message in enumerate(row['messages']):
        if message['role'] == 'assistant':
            prefix = encode(row['messages'][:index], tokenizer, generation_prompt=True)
            assert prefix['input_ids'] == beacon['input_ids'][:len(prefix['input_ids'])]
            assert prefix['regions'] == [(start, end) for start, end in beacon['regions']
                                         if end <= len(prefix['input_ids'])]


def test_legacy_upgrade_requires_remeasurement_and_keeps_original_targets(tmp_path):
    row = normalize(polar())
    row.update(protocol=LEGACY_VERSION, lengths={'old': {'total': 10}})
    row['messages'][0]['content'] = 'Old prompt'
    original = copy.deepcopy(row)
    upgraded = normalize(row)
    assert row == original
    assert upgraded['protocol'] == VERSION
    assert 'lengths' not in upgraded
    assert upgraded['messages'][1:] == original['messages'][1:]
    assert upgraded['messages'][0]['content'] == build_system_prompt()
    assert normalize(upgraded) == upgraded
    path = tmp_path / 'old.jsonl'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='regenerate and remeasure'):
        OpenSeekerDataset(path, Tokenizer())[0]


def test_offline_prompt_changes_only_capability_and_task_extensions():
    base = build_system_prompt()
    local = build_system_prompt(local=True, answer_instruction='Cite [docid].')
    assert local.startswith(base)
    assert 'No external websites' in local and local.endswith('Cite [docid].')
    search_only = build_system_prompt(local=True, allow_visit=False)
    schemas = [json.loads(line) for line in search_only.split('<tools>\n')[1].split('\n</tools>')[0].splitlines()]
    assert [tool['name'] for tool in schemas] == ['search']
    assert 'Do not call visit.' in search_only
    assert prompt_metadata(base)['system_prompt_sha256'] != prompt_metadata(local)['system_prompt_sha256']
    assert prompt_metadata('custom instructions')['prompt_version'] == 'custom'


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
