"""Benchmark entries dispatch without loading models or invoking external services."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('kind', ['plain', 'beacon'])
@pytest.mark.parametrize('number,name', [('47', 'browsecomp'), ('56', 'browsecomp_plus'), ('57', 'hotpotqa')])
def test_independent_benchmark_entry(tmp_path, kind, number, name):
    script = ROOT / f'scripts/sh/{kind}/{number}_{kind}_{name}.sh'
    env = {**os.environ, 'PYTHON': sys.executable, 'KIND': kind, 'ACTION': 'run',
           'MODE': 'native', 'DRY_RUN': '1', 'DECODING': 'greedy',
           'MODEL_PATH': str(tmp_path / 'model with spaces'), 'LIMIT': '3'}
    result = subprocess.run(['bash', str(script)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    if number == '47':
        assert config['mode'] == 'native'
        assert str(tmp_path / 'model with spaces') in config['generation']
    else:
        assert config['kind'] == kind
        assert config['model'] == str(tmp_path / 'model with spaces')
        assert config['limit'] == 3
    env['KIND'] = 'plain' if kind == 'beacon' else 'beacon'
    rejected = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
    assert rejected.returncode == 2 and 'KIND' in rejected.stderr


@pytest.mark.parametrize('script,benchmark', [
    ('74_qwen35_browsecomp_plus.sh', 'browsecomp-plus'),
    ('75_qwen35_hotpotqa.sh', 'hotpotqa'),
])
def test_baseline_dataset_entry_pins_target(script, benchmark):
    env = {**os.environ, 'PYTHON': sys.executable, 'DRY_RUN': '1', 'ACTION': 'run',
           'TARGET': 'invalid', 'BASE_CHECKPOINT': '/models/raw qwen'}
    result = subprocess.run(['bash', str(ROOT / 'scripts/sh/baseline' / script)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config['benchmark'] == benchmark
    assert config['model_origin'] == 'base-qwen35'
    assert config['model'] == '/models/raw qwen'
