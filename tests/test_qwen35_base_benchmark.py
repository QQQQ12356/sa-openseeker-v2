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


def test_native_bcp_uses_qwen_template_with_browser_roles_and_keeps_full_trace():
    from search_comp.evaluation.qwen_native_browser import run_agent, system_prompt
    from search_comp.offline.native_browser import NativeBrowser

    doc = dict(docid="1", title="Bridge", source_url="https://example.invalid/bridge", text="A red bridge.")
    retrieval = SimpleNamespace(search=lambda query, topk: [doc], get_url=lambda url: doc)
    browser = NativeBrowser(retrieval, retrieval, lambda query, text: text, topk=15)

    class Tokenizer:
        def __init__(self):
            self.contexts = []

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "browser.search"
            assert kwargs["add_generation_prompt"] and not kwargs["tokenize"]
            self.contexts.append(list(messages))
            return "encoded"

        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "encoded" and not add_special_tokens
            return [1, 2]

    class Generator:
        metadata = {}

        def __init__(self):
            self.replies = iter([
                'Need a source.</think><tool_call><function=browser.search>'
                '<parameter=query>bridge</parameter></function></tool_call>',
                'Read it.</think><tool_call>{"name":"browser.open","arguments":{"id":0}}</tool_call>',
                'Explanation: the page says red.\nExact Answer: Red\nConfidence: 95%',
            ])

        def start_question(self, question):
            assert question == "What color?"

        def __call__(self, inputs, budget, *, stop_texts):
            assert inputs["input_ids"] == [1, 2] and budget == 10 and "<tool_response>" in stop_texts
            return next(self.replies), True, 1

    tokenizer = Tokenizer()
    try:
        result = run_agent([dict(role="system", content=system_prompt("2026-11-25")),
                            dict(role="user", content="What color?")],
                           tokenizer, Generator(), browser,
                           max_turns=3, max_context=100, max_new_tokens=10, observation_window=1)
    finally:
        browser.close()
    assert result["status"] == "completed"
    assert "Exact Answer: Red" in result["prediction"]
    assert result["messages"][2]["reasoning_content"] == "Need a source."
    assert result["messages"][3]["tool_call_id"] == "1"
    assert "example.invalid" in result["messages"][3]["content"]
    assert "[Auto-archived browser result]" in tokenizer.contexts[-1][3]["content"]
    assert browser.retrieved == browser.visited == {"1"}


def test_native_bcp_batch_cursors_use_state_from_before_the_turn():
    from search_comp.evaluation.qwen_native_browser import parse_assistant
    from search_comp.offline.native_browser import NativeBrowser

    doc = dict(docid="1", title="Bridge", source_url="https://example.invalid/bridge", text="A red bridge.")
    index = SimpleNamespace(search=lambda query, topk: [doc], get_url=lambda url: doc)
    browser = NativeBrowser(index, index, lambda query, text: text)
    try:
        calls = parse_assistant(
            '<tool_call>{"name":"browser.search","arguments":{"query":["bridge","red"]}}</tool_call>'
            '<tool_call>{"name":"browser.open","arguments":{"id":0}}</tool_call>', 1)["tool_calls"]
        results = browser.execute(calls)
        assert [ok for _, ok in results] == [True, True, False]
        assert results[0][0].startswith("[0] bridge")
        assert results[1][0].startswith("[1] red")
        opened = browser.execute([calls[-1]])
        assert opened[0][1] and "L0: A red bridge." in opened[0][0]
        find = parse_assistant('<tool_call>{"name":"browser.find","arguments":{"pattern":"RED"}}</tool_call>',
                               3)["tool_calls"]
        assert "match at L0" in browser.execute(find)[0][0]
    finally:
        browser.close()


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
