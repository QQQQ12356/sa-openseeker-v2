"""Training launchers archive their parameters without reading external configs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from search_comp.models.beacon_config import configure_sft_layout
from search_comp.trainer.launch import training_command
from search_comp.utils.runtime import load_yaml_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('backend,stage', [('zero2', 2), ('zero3', 3)])
def test_inline_deepspeed_overrides_reach_training_config(tmp_path, monkeypatch, backend, stage):
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', backend)
    configs = {backend: {'zero_optimization': {'stage': stage, 'reduce_bucket_size': 12345},
                         'gradient_accumulation_steps': 'auto'}}
    path = tmp_path / 'train.yaml'
    path.write_text('learning_rate: 0.001\n')
    command = training_command('search_comp.trainer.plain_sft_trainer', ['--config', str(path)],
                               deepspeed_configs=configs)
    assert command[-2] == '--set'
    resolved = load_yaml_config(str(path), [command[-1]])
    assert resolved['deepspeed'] == configs[backend]
    assert 'configs/train/ds_' not in command[-1]


def test_inline_deepspeed_missing_or_wrong_stage_fails(monkeypatch):
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', 'zero2')
    with pytest.raises(ValueError, match='distributed'):
        training_command('search_comp.trainer.plain_sft_trainer', [], deepspeed_configs={})
    with pytest.raises(ValueError, match='stage'):
        training_command('search_comp.trainer.plain_sft_trainer', [],
                         deepspeed_configs={'zero2': {'zero_optimization': {'stage': 3}}})
    with pytest.raises(ValueError, match='stage'):
        training_command('search_comp.trainer.plain_sft_trainer', [],
                         deepspeed_configs={'zero2': {'zero_optimization': None}})


def test_ddp_rejects_conflicting_explicit_deepspeed(monkeypatch):
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', 'ddp')
    with pytest.raises(ValueError, match='冲突'):
        training_command('search_comp.trainer.plain_sft_trainer', [],
                         configured_deepspeed={'zero_optimization': {'stage': 2}})


def test_beacon_layout_policy_keeps_compatibility_and_explicit_layout():
    legacy = {'beacon': {'beacon_keep_window': 12}}
    configure_sft_layout(legacy)
    assert legacy['beacon']['beacon_keep_window'] == 0
    configured = {'data_mode': 'openseeker', 'beacon_layout_policy': 'configured',
        'beacon': {'beacon_information_mode': 'window', 'beacon_window_enabled': True,
                   'beacon_window': 64, 'beacon_stride': 64, 'beacon_ratio': 16,
                   'beacon_keep_window': 12, 'beacon_doc_region_split': False}}
    configure_sft_layout(configured)
    assert configured['beacon']['beacon_information_mode'] == 'window'
    assert configured['beacon']['beacon_keep_window'] == 12
    configured['beacon']['beacon_doc_region_split'] = True
    with pytest.raises(ValueError, match='OpenSeeker'):
        configure_sft_layout(configured)


@pytest.mark.parametrize('script', ['scripts/sh/plain/22_plain_searchr1_train.sh',
                                     'scripts/sh/beacon/32_beacon_searchr1_train.sh'])
def test_searchr1_default_plan_uses_only_inline_config(tmp_path, script):
    import os
    env = {**os.environ, 'PYTHON': sys.executable, 'ACTION': 'plan',
           'CUDA_VISIBLE_DEVICES': '0,1', 'NPROC_PER_NODE': '2',
           'DISTRIBUTED_BACKEND': 'zero2', 'TMPDIR': str(tmp_path)}
    result = subprocess.run(['bash', script], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'deepspeed={' in result.stdout and 'configs/train/ds_zero2.json' not in result.stdout
    assert not list(tmp_path.glob('searchr1-train.*'))


def test_searchr1_data_override_supports_paths_with_spaces(tmp_path):
    import os
    path = tmp_path / 'my training.jsonl'
    env = {**os.environ, 'PYTHON': sys.executable, 'ACTION': 'plan', 'SEARCHR1_DATA': str(path),
           'CUDA_VISIBLE_DEVICES': '0', 'NPROC_PER_NODE': '1', 'TMPDIR': str(tmp_path)}
    result = subprocess.run(['bash', 'scripts/sh/plain/22_plain_searchr1_train.sh'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(path) in result.stdout


@pytest.mark.parametrize('kind,script', [
    ('plain', 'scripts/sh/plain/20_plain_train.sh'),
    ('beacon', 'scripts/sh/beacon/30_beacon_train.sh'),
])
def test_inline_experiment_plan_without_any_external_config(tmp_path, monkeypatch, capsys, kind, script):
    from search_comp.workflows import experiment
    from search_comp.data.openseeker import VERSION
    source = (ROOT / script).read_text()
    raw = source.split("cat > \"$CONFIG_FILE\" <<'YAML'\n", 1)[1].split('\nYAML\n', 1)[0]
    spec = yaml.safe_load(raw)
    data = tmp_path / 'prepared'
    data.mkdir()
    for split in ('train', 'validation', 'test'):
        (data / f'{split}.jsonl').write_text('{}\n')
    model = spec['models'][spec['model_family']]
    (data / 'manifest.json').write_text(json.dumps({'protocol': VERSION, 'tokenizers': {model: {}},
                                                   'arguments': {'max_length': 10}}))
    spec['dataset'] = str(data)
    spec['distributed']['zero2']['zero_optimization']['reduce_bucket_size'] = 123456
    if kind == 'beacon':
        spec['training']['beacon_layout_policy'] = 'configured'
        spec['beacon'].update(beacon_information_mode='window', beacon_window_enabled=True,
                              beacon_window=64, beacon_stride=64, beacon_ratio=16,
                              beacon_keep_window=24)
    config = tmp_path / 'self_contained.yaml'
    config.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', 'zero2')
    monkeypatch.setattr(sys, 'argv', ['experiment', 'plan', '--config', str(config)])
    experiment.main()
    plan = json.loads(capsys.readouterr().out)
    assert 'configs/train/' not in ' '.join(plan['train_command'])
    assert json.loads(plan['train_command'][-1].split('=', 1)[1])['zero_optimization']['reduce_bucket_size'] == 123456
    actual = yaml.safe_load((Path(plan['run_dir']) / 'train.yaml').read_text())
    if kind == 'beacon':
        assert actual['beacon']['beacon_information_mode'] == 'window'
        assert actual['beacon']['beacon_keep_window'] == 24


@pytest.mark.parametrize('script', [
    'scripts/sh/beacon/30_beacon_train.sh', 'scripts/sh/beacon/37_beacon_memory_probe.sh',
    'scripts/sh/beacon/34_beacon_searchr1_train_impl.sh',
])
def test_beacon_shell_exposes_all_beacon_config_fields(script):
    from dataclasses import fields
    from search_comp.models.beacon_config import BeaconConfig
    source = (ROOT / script).read_text()
    raw = source.split("<<'YAML'\n", 1)[1].split('\nYAML\n', 1)[0]
    config = yaml.safe_load(raw)
    assert set(config['beacon']) == {field.name for field in fields(BeaconConfig)}
    assert config['distributed']['zero2']['zero_optimization']['stage'] == 2
    assert 'base_configs' not in config
    policy = config.get('beacon_layout_policy', config.get('training', {}).get('beacon_layout_policy'))
    assert policy == 'configured'


def test_script_owned_ds_honors_inline_set_override(tmp_path, monkeypatch, capsys):
    from search_comp.trainer import launch
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'distributed': {'zero2': {'zero_optimization':
        {'stage': 2, 'reduce_bucket_size': 1}}}}))
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', 'zero2')
    monkeypatch.setattr(sys, 'argv', ['launch', '--module', 'search_comp.trainer.plain_sft_trainer',
        '--dry-run', '--config', str(config), '--set', 'distributed.zero2.zero_optimization.reduce_bucket_size=99'])
    launch.main()
    output = capsys.readouterr().out
    assert 'reduce_bucket_size: 99' in output
    assert 'configs/train/ds_zero2.json' not in output


def test_self_contained_launch_never_falls_back_to_external_ds(tmp_path, monkeypatch):
    from search_comp.trainer import launch
    config = tmp_path / 'config.yaml'
    config.write_text('self_contained_training: true\nlearning_rate: 0.001\n')
    monkeypatch.setenv('NPROC_PER_NODE', '2')
    monkeypatch.setenv('DISTRIBUTED_BACKEND', 'zero2')
    monkeypatch.setattr(sys, 'argv', ['launch', '--module', 'search_comp.trainer.plain_sft_trainer',
                                     '--dry-run', '--config', str(config)])
    with pytest.raises(ValueError, match='不会回退'):
        launch.main()


def test_self_contained_experiment_missing_distributed_does_not_use_legacy_files(tmp_path, monkeypatch, capsys):
    from search_comp.workflows import experiment
    config = tmp_path / 'missing_ds.yaml'
    config.write_text(yaml.safe_dump({'model_family': 'qwen35', 'kind': 'plain',
        'dataset': str(tmp_path), 'models': {'qwen35': 'fake'}, 'training': {}}))
    monkeypatch.setattr(sys, 'argv', ['experiment', 'plan', '--config', str(config)])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
    assert '不会回退' in capsys.readouterr().err


def test_plain_shuffle_and_batch_invalid_are_rejected_before_model_load(tmp_path):
    from search_comp.trainer.plain_sft_trainer import main_train
    config = tmp_path / 'invalid.yaml'
    config.write_text('shuffle_train_data: "not-a-bool"\n')
    with pytest.raises(ValueError, match='shuffle_train_data'):
        main_train(str(config))
    config.write_text('per_device_batch_size: 2\n')
    with pytest.raises(ValueError, match='per_device_batch_size'):
        main_train(str(config))
