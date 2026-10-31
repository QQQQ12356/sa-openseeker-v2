"""Plain recovery and public scoring entry points."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_merge_recovery_resolves_base_and_rejects_existing_output(tmp_path):
    from argparse import Namespace
    from search_comp.trainer.merge_lora import resolve_paths
    (tmp_path / 'resolved_config.json').write_text(json.dumps({'model_name_or_path': '/base'}))
    adapter = tmp_path / 'final'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    (adapter / 'adapter_model.safetensors').write_bytes(b'dummy')
    args = Namespace(exp_dir=str(tmp_path), adapter_path=None, output_path=None, base_model_path=None)
    base, saved, output = resolve_paths(args)
    assert (base, saved, output) == ('/base', str(adapter), str(tmp_path / 'final_merged'))
    Path(output).mkdir()
    (Path(output) / 'existing').write_text('keep')
    with pytest.raises(ValueError, match='非空'):
        resolve_paths(args)
    assert (Path(output) / 'existing').read_text() == 'keep'


def test_plain_workflow_scores_existing_openseeker_predictions(tmp_path):
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(json.dumps({'status': 'completed', 'prediction': 'Paris', 'answer': 'Paris'}) + '\n')
    output = tmp_path / 'summary.json'
    env = {**os.environ, 'PYTHON': sys.executable, 'TARGET': 'openseeker', 'ACTION': 'score',
           'PREDICTIONS': str(predictions), 'OUTPUT_PATH': str(output), 'LOG_PATH': str(tmp_path / 'log')}
    command = ['bash', 'scripts/sh/plain/27_plain_workflow.sh']
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['experiments'][0]['overall_em'] == 1.0
    again = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert again.returncode != 0
    assert json.loads(output.read_text())['experiments'][0]['overall_em'] == 1.0


@pytest.mark.parametrize('family', ['qwen3', 'qwen35'])
def test_plain_adapter_eval_matches_merged_checkpoint(tmp_path, family):
    """A real PEFT adapter loaded for 21 must reproduce the merged model's logits."""
    import torch
    from peft import LoraConfig, get_peft_model
    from test_plain_memory import tiny_model
    from search_comp.models.plain_qwen3 import resolve_plain_evaluation, load_plain_evaluation_model

    torch.manual_seed(3)
    base_dir = tmp_path / 'base'
    adapter_dir = tmp_path / 'run' / 'checkpoint-200'
    model = tiny_model(family)
    model.save_pretrained(base_dir)
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'],
                                           task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if 'lora_B' in name:
                param.fill_(0.2)
    peft.save_pretrained(adapter_dir)
    (adapter_dir.parent / 'resolved_config.json').write_text(json.dumps({'model_name_or_path': str(base_dir)}))
    merged = peft.merge_and_unload().eval()
    plan = resolve_plain_evaluation(str(adapter_dir))
    assert plan['adapter'] and plan['base'] == str(base_dir) and plan['tokenizer'] == str(base_dir)
    loaded = load_plain_evaluation_model(plan, torch_dtype=torch.float32, device_map='cpu')
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, merged(ids).logits, rtol=1e-5, atol=1e-6)
    assert not loaded.training


def test_plain_adapter_requires_base_and_rejects_beacon(tmp_path):
    from search_comp.models.plain_qwen3 import resolve_plain_evaluation
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{"peft_type":"LORA"}')
    (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
    with pytest.raises(ValueError, match='BASE_MODEL_PATH'):
        resolve_plain_evaluation(str(adapter))
    assert resolve_plain_evaluation(str(adapter), '/base', '/tokenizer')['tokenizer'] == '/tokenizer'
    (adapter / 'beacon_state.pt').write_bytes(b'beacon')
    with pytest.raises(ValueError, match='Beacon'):
        resolve_plain_evaluation(str(adapter), '/base')


def test_base_benchmark_shell_routes_dry_run_without_loading_model(tmp_path):
    env = {**os.environ, 'PYTHON': sys.executable, 'TARGET': 'hotpotqa', 'ACTION': 'run',
           'BASE_CHECKPOINT': str(tmp_path / 'base with spaces'), 'DRY_RUN': '1'}
    result = subprocess.run(['bash', 'scripts/sh/baseline/70_qwen35_base_eval.sh'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert args['benchmark'] == 'hotpotqa' and args['kind'] == 'plain'
    assert args['model'] == str(tmp_path / 'base with spaces')


def test_plain_adapter_shell_passes_base_and_tokenizer_paths(tmp_path):
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    env = {**os.environ, 'PYTHON': sys.executable, 'MODEL_PATH': str(tmp_path / 'adapter'),
           'BASE_MODEL_PATH': str(tmp_path / 'base with space'),
           'TOKENIZER_PATH': str(tmp_path / 'tokenizer with space'),
           'DATA': str(data), 'DRY_RUN': '1'}
    result = subprocess.run(['bash', 'scripts/sh/plain/21_plain_eval.sh'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--base-model' in result.stdout and 'base\\ with\\ space' in result.stdout
    assert '--tokenizer' in result.stdout and 'tokenizer\\ with\\ space' in result.stdout


def test_plain_decoding_options_reach_shell_and_offline_benchmark(tmp_path):
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    env = {**os.environ, 'PYTHON': sys.executable, 'MODEL_PATH': str(tmp_path / 'model'),
           'DATA': str(data), 'DRY_RUN': '1', 'DECODING': 'sample',
           'REPETITION_PENALTY': '1.1', 'TEMPERATURE': '0.8', 'TOP_P': '0.9', 'GENERATION_SEED': '17'}
    result = subprocess.run(['bash', 'scripts/sh/plain/21_plain_eval.sh'], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--decoding sample' in result.stdout and '--repetition-penalty 1.1' in result.stdout
    assert '--generation-seed 17' in result.stdout
    env.update(TARGET='hotpotqa', ACTION='run')
    result = subprocess.run(['bash', 'scripts/sh/baseline/70_qwen35_base_eval.sh'], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert (parsed['decoding'], parsed['repetition_penalty'], parsed['generation_seed']) == ('sample', 1.1, 17)


def test_official_mode_rejects_custom_decoding(tmp_path):
    env = {**os.environ, 'PYTHON': sys.executable, 'MODE': 'official-v1', 'DRY_RUN': '1',
           'DECODING': 'greedy', 'REPETITION_PENALTY': '1.1'}
    result = subprocess.run(['bash', 'scripts/sh/benchmark/47_openseeker_benchmark.sh'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'native mode only' in result.stderr


def test_beacon_entry_accepts_visible_penalty_but_rejects_sampling(tmp_path):
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    env = {**os.environ, 'PYTHON': sys.executable, 'KIND': 'beacon',
           'MODEL_PATH': str(tmp_path / 'model'), 'DATA': str(data),
           'DECODING': 'greedy', 'REPETITION_PENALTY': '1.1', 'DRY_RUN': '1'}
    result = subprocess.run(['bash', 'scripts/sh/beacon/31_beacon_eval.sh'], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--repetition-penalty 1.1' in result.stdout
    # 31 dry-run only prints; Python entry performs the real validation.
    cmd = [sys.executable, 'scripts/py/50_evaluate.py', '--model', str(tmp_path / 'model'),
           '--data', str(data), '--output', str(tmp_path / 'out.jsonl'),
           '--kind', 'beacon', '--decoding', 'sample']
    rejected = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert rejected.returncode != 0 and 'Beacon requires greedy' in rejected.stderr
