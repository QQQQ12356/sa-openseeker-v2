"""Small local fixtures validate documented data stages; no downloads/tokenizers."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from search_comp.workflows import download_openseeker, report_openseeker, audit_openseeker


class DatasetPipelineTests(unittest.TestCase):
    def test_research_snapshot_rejects_missing_shard_before_preprocessing(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash or not Path(bash).is_file():
            self.skipTest('Bash is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            present = root / 'first.jsonl'
            present.write_text('{}\n', encoding='utf-8')
            (root / 'download_manifest.json').write_text(json.dumps(dict(jsonl_inputs=[
                str(present), str(root / 'missing-shard.jsonl')])), encoding='utf-8')
            env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1',
                       SNAPSHOT_DIR=root.as_posix(), DATASET_DIR=(root/'never-created').as_posix())
            result = subprocess.run([bash, 'scripts/sh/data/09_prepare_research.sh'],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('jsonl_inputs', result.stderr)
            self.assertFalse((root/'never-created').exists())

    def test_research_download_resolves_commit_and_keeps_native_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'research' / 'fixed-commit'
            def download(repo, **kwargs):
                snapshot.mkdir(parents=True)
                (snapshot / 'train.jsonl').write_text('{"messages":[]}\n', encoding='utf-8')
                (snapshot / 'train.parquet').write_bytes(b'original parquet')
                self.assertEqual(kwargs['revision'], 'fixed-commit')
            fake = types.ModuleType('huggingface_hub')
            fake.HfApi = lambda **kwargs: types.SimpleNamespace(dataset_info=lambda *a, **k: types.SimpleNamespace(sha='fixed-commit'))
            fake.snapshot_download = download
            with patch.dict(sys.modules, {'huggingface_hub': fake}), \
                    patch.object(sys, 'argv', ['download', '--source', 'research', '--output', str(root)]), \
                    patch.object(download_openseeker, 'export_parquet') as export, \
                    contextlib.redirect_stdout(io.StringIO()):
                download_openseeker.main()
                export.assert_not_called()
            manifest = json.loads((snapshot / 'download_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['revision'], 'fixed-commit')
            self.assertEqual(manifest['repo'], 'AmanPriyanshu/tool-reasoning-sft-RESEARCH-OpenSeeker-v1-Data')
            self.assertEqual(manifest['jsonl_inputs'], [str((snapshot / 'train.jsonl').resolve())])
            self.assertEqual((snapshot / 'train.parquet').read_bytes(), b'original parquet')

    def test_parquet_export_preserves_unicode_and_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'train.parquet'
            target = Path(directory) / 'exports' / 'train.jsonl'
            source.write_bytes(b'original')
            batch = types.SimpleNamespace(to_pylist=lambda: [dict(question='完整数据')])
            pq = types.ModuleType('pyarrow.parquet')
            pq.ParquetFile = lambda path: types.SimpleNamespace(iter_batches=lambda **kwargs: [batch])
            parent = types.ModuleType('pyarrow')
            parent.parquet = pq
            with patch.dict(sys.modules, {'pyarrow': parent, 'pyarrow.parquet': pq}):
                download_openseeker.export_parquet(source, target)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['question'], '完整数据')
            self.assertEqual(source.read_bytes(), b'original')

    def fixture(self, root, models=('local-tokenizer',)):
        prepared = root / 'prepared'
        prepared.mkdir()
        rows = [dict(id=str(i), question='Which city?', correctness=None, declared_tool_calls=None,
                     messages=[dict(role='system', content='system'), dict(role='user', content='Which city?'),
                               dict(role='assistant', content='Paris')],
                     lengths={model: dict(total=length, observations=length//2, assistant=5) for model in models})
                for i, length in enumerate((100, 400))]
        def dump(name, values):
            (prepared / name).write_text(''.join(json.dumps(row) + '\n' for row in values), encoding='utf-8')
        dump('normalized.jsonl', rows)
        dump('train.jsonl', rows[:1])
        dump('validation.jsonl', [])
        dump('test.jsonl', [])
        dump('measurements.jsonl', [dict(id=row['id'], calls=0, tools={}, lengths=row['lengths'],
                                        rejection=None if i == 0 else 'too_long') for i,row in enumerate(rows)])
        dump('rejected.jsonl', [dict(id='1', reason='too_long'), dict(reason='invalid protocol')])
        (prepared / 'manifest.json').write_text(json.dumps(dict(arguments=dict(source='research'),
            counts=dict(input=3, invalid=1, train=1), tokenizers={model: {} for model in models})),encoding='utf-8')
        return prepared

    def test_report_separates_before_retained_and_actual_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prepared=self.fixture(root)
            output=root/'report'
            with patch.object(sys, 'argv', ['report','--prepared',str(prepared),'--output',str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                report_openseeker.main()
            result=json.loads((output/'quality_length.json').read_text(encoding='utf-8'))
            self.assertEqual(result['model'], 'local-tokenizer')
            self.assertEqual(result['populations']['before_filter']['count'], 2)
            self.assertEqual(result['populations']['retained']['count'], 1)
            self.assertEqual(result['populations']['train']['distributions']['total']['max'], 100)
            self.assertEqual(result['populations']['validation']['count'], 0)
            self.assertEqual(result['rejections']['too_long'], 1)
            self.assertEqual(result['input_counts']['invalid'], 1)

    def test_report_rejects_implicit_mixing_of_tokenizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prepared=self.fixture(root, ('model-a','model-b'))
            with patch.object(sys, 'argv', ['report','--prepared',str(prepared),'--output',str(root/'report')]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                report_openseeker.main()
            self.assertFalse((root/'report').exists())

    def test_research_raw_audit_counts_original_roles_and_adapted_tools(self):
        row=dict(messages=[dict(role='system',content='Research.'), dict(role='user',content='Which city?'),
            dict(role='reasoning',content='Search.'),
            dict(role='tool_call',content='<tool_call>{"name":"search","arguments":{"query":["city"]}}</tool_call>'),
            dict(role='tool_output',content='Paris'), dict(role='answer',content='Paris')])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'train.jsonl'
            source.write_text(json.dumps(row)+'\n',encoding='utf-8')
            output=root/'audit.json'
            with patch.object(sys, 'argv', ['audit','--source','research','--input',str(source),'--output',str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                audit_openseeker.main()
            result=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['roles']['reasoning'], 1)
            self.assertEqual(result['tools']['search'], 1)
            self.assertEqual(result['normalization_failures'], {})


if __name__ == '__main__':
    unittest.main()
