"""Original Qwen3.5 baseline must not load or dispatch through Plain SFT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_base_loader_validates_architecture_and_calls_raw_text_loader(tmp_path, monkeypatch):
    from transformers import AutoConfig
    from search_comp.models.base_qwen35 import load_base_qwen35
    from search_comp.milestones import qwen35_text
    seen = []
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace(model_type='qwen3_5'))
    monkeypatch.setattr(qwen35_text, 'load_text_causal_model', lambda path: seen.append(path) or 'raw model')
    assert load_base_qwen35('/original-qwen') == 'raw model'
    assert seen == ['/original-qwen']
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace(model_type='qwen3_5_text'))
    with pytest.raises(ValueError, match='original'):
        load_base_qwen35('/trained-text-checkpoint')
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    with pytest.raises(ValueError, match='adapter'):
        load_base_qwen35(str(adapter))






def test_loads_real_tiny_original_qwen35_text_backbone(tmp_path):
    """Load a genuine multimodal-layout checkpoint without SFT or PEFT."""
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from search_comp.models.base_qwen35 import load_base_qwen35
    text = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                layer_types=['full_attention'], max_position_embeddings=64)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                  out_hidden_size=16, num_position_embeddings=16)
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=60,
                           video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    original = Qwen3_5ForConditionalGeneration(config)
    expected = original.model.language_model.embed_tokens.weight.detach().to(torch.bfloat16)
    original.save_pretrained(tmp_path)
    loaded = load_base_qwen35(str(tmp_path))
    assert loaded.config.model_type == 'qwen3_5_text' and not loaded.training
    torch.testing.assert_close(loaded.model.embed_tokens.weight.cpu(), expected.cpu())


def test_offline_base_generation_does_not_load_plain_model(tmp_path, monkeypatch):
    from test_offline_benchmarks import prepared_hotpot
    from search_comp.workflows import offline_benchmark as workflow
    from search_comp.evaluation import tool_agent
    root = prepared_hotpot(tmp_path)
    model = tmp_path / 'raw_model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    output = tmp_path / 'out'
    output.mkdir()
    args = workflow.make_parser().parse_args(['run', '--benchmark', 'hotpotqa', '--root', str(root),
        '--model', str(model), '--kind', 'plain', '--model-origin', 'base-qwen35'])
    rows, manifest = workflow.prepared_questions(root, 'hotpotqa', limit=1)
    monkeypatch.setitem(sys.modules, 'search_comp.models.model_loader', SimpleNamespace(
        load_tokenizer=lambda _: object(), load_beacon_model=None))
    monkeypatch.setitem(sys.modules, 'search_comp.models.plain_qwen3', None)
    monkeypatch.setitem(sys.modules, 'search_comp.models.base_qwen35', SimpleNamespace(
        load_base_qwen35=lambda _: SimpleNamespace(eval=lambda: None)))
    class Generator:
        def __init__(self, *a, **k): self.metadata = {}
    monkeypatch.setattr(tool_agent, 'ModelGenerator', Generator)
    monkeypatch.setattr(tool_agent, 'run_agent', lambda messages, tokenizer, generator, *a, **kwargs:
        dict(status='completed', prediction='yes', generation_config=generator.metadata))
    path = workflow.run_phase(args, output, rows, manifest)
    assert json.loads(path.read_text())['generation_config']['model_origin'] == 'base-qwen35'


def test_online_base_generation_does_not_load_plain_model(tmp_path, monkeypatch):
    from search_comp.workflows import benchmark_openseeker as workflow
    from search_comp.evaluation import tool_agent
    model = tmp_path / 'raw_model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    class Tool:
        def call(self, params): return 'observation'
    monkeypatch.setitem(sys.modules, 'llm_tool_openseeker', SimpleNamespace(Search=Tool, Visit=Tool))
    monkeypatch.setitem(sys.modules, 'search_comp.models.model_loader', SimpleNamespace(
        load_tokenizer=lambda _: object(), load_beacon_model=None))
    monkeypatch.setitem(sys.modules, 'search_comp.models.plain_qwen3', None)
    monkeypatch.setitem(sys.modules, 'search_comp.models.base_qwen35', SimpleNamespace(
        load_base_qwen35=lambda _: SimpleNamespace(eval=lambda: None)))
    class Generator:
        def __init__(self, *a, **k): self.metadata = {}
    monkeypatch.setattr(tool_agent, 'ModelGenerator', Generator)
    monkeypatch.setattr(tool_agent, 'run_agent', lambda messages, tokenizer, generator, *a, **kwargs:
        dict(status='completed', prediction='answer', events=[], generation_config=generator.metadata))
    args = SimpleNamespace(official_dir=tmp_path, prompt_data=None, run_dir=None,
        model=str(model), kind='plain', model_origin='base-qwen35',
        max_turns=2, max_context=100, max_tokens=10, max_tool_tokens=10, tool_count=200)
    output = tmp_path / 'output.jsonl'
    workflow.native_generate(args, [dict(query='question', answer='answer')], output)
    assert json.loads(output.read_text())['generation_config']['model_origin'] == 'base-qwen35'








def test_offline_workers_preserve_base_origin(tmp_path):
    from search_comp.workflows import offline_benchmark as workflow
    model = tmp_path / 'raw_model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    args = workflow.make_parser().parse_args(['run', '--benchmark', 'hotpotqa', '--root', str(tmp_path),
        '--model', str(model), '--kind', 'plain', '--model-origin', 'base-qwen35', '--num-workers', '2'])
    command = workflow.worker_command(args, tmp_path / 'worker', 0, 2)
    assert command[command.index('--model-origin') + 1] == 'base-qwen35'
    assert command[command.index('--model') + 1] == str(model)
