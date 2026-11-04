"""Exercise experiment naming without loading model weights or requiring PyYAML."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ExperimentDatasetNameTests(unittest.TestCase):
    def setUp(self):
        # JSON is a YAML subset; isolate serialization from naming/plan behavior.
        yaml = types.ModuleType('yaml')
        yaml.safe_load = json.loads
        yaml.safe_dump = lambda value, **kwargs: json.dumps(value)
        module_spec = importlib.util.spec_from_file_location(
            'experiment_under_test', ROOT / 'search_comp/workflows/experiment.py')
        self.experiment = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, {'yaml': yaml}):
            module_spec.loader.exec_module(self.experiment)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment.ROOT = self.root
        self.data = self.root / 'prepared'
        self.data.mkdir()
        for name, content in [('train.jsonl', '{}\n'), ('validation.jsonl', ''), ('test.jsonl', '')]:
            (self.data / name).write_text(content, encoding='utf-8')

    def plan(self, source, dataset_name=None, kind='plain', method=None):
        manifest = dict(protocol=self.experiment.VERSION,
                        arguments=dict(source=source, max_length=16384),
                        tokenizers={'local-model': {}})
        (self.data / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        spec = dict(model_family='qwen35', kind=kind, data_mode='openseeker',
                    dataset='prepared', models={'qwen35': 'local-model'}, distributed={},
                    training=dict(max_length=16384, learning_rate=5e-5, seed=42))
        if dataset_name is not None:
            spec['dataset_name'] = dataset_name
        if method is not None:
            spec['method'] = method
        config = self.root / 'source.json'
        config.write_text(json.dumps(spec), encoding='utf-8')
        # Import actual Beacon config validation without models/__init__ loading torch.
        models = types.ModuleType('search_comp.models')
        models.__path__ = [str(ROOT / 'search_comp/models')]
        with patch.object(sys, 'argv', ['experiment', 'plan', '--config', str(config)]), \
                patch.dict(sys.modules, {'search_comp.models': models}), \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(self.experiment, 'training_command', return_value=['python', 'train']):
            self.experiment.main()
        runs = list((self.root / 'outputs/experiments').iterdir())
        self.assertEqual(len(runs), 1)
        return runs[0], json.loads((runs[0] / 'experiment.json').read_text())

    def test_research_manifest_sets_dataset_name(self):
        run, archived = self.plan('research')
        self.assertTrue(run.name.startswith('qwen35_plain_research_L16384_'))
        self.assertEqual(archived['dataset_name'], 'research')
        self.assertEqual(archived['training_config']['data_mode'], 'openseeker')

    def test_removed_training_source_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan('browsecomp')
        self.assertFalse((self.root / 'outputs').exists())

    def test_research_wrong_identity_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan('research', 'openseeker')

    def test_baseline_identity_uses_plain_trainer(self):
        run, archived = self.plan('research', kind='plain', method='baseline')
        self.assertTrue(run.name.startswith('qwen35_baseline_research_L16384_'))
        self.assertEqual(archived['kind'], 'plain')
        self.assertEqual(archived['method'], 'baseline')

    def test_openseeker_source_keeps_original_naming(self):
        run, archived = self.plan('polar')
        self.assertTrue(run.name.startswith('qwen35_plain_openseeker_L16384_'))
        self.assertEqual(archived['dataset_name'], 'openseeker')

    def test_invalid_dataset_name_does_not_create_run(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan('polar', '../outside')
        self.assertFalse((self.root / 'outputs').exists())


if __name__ == '__main__':
    unittest.main()
