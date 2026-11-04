"""Standard-library regressions: run with unittest, without torch/pytest/model weights."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from search_comp.data.browsecomp import normalize_browsecomp
from search_comp.data.agent_prompt import TOOLS_RE, build_system_prompt, resolve_system_prompt
from search_comp.data.openseeker import OpenSeekerDataset, encode, pieces
from search_comp.data.selection import Selection
from search_comp.offline.assets import install_official
from search_comp.offline.scoring import score_hotpot, score_hotpot_judge
from search_comp.workflows import prepare_openseeker

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')


class ByteTokenizer:
    """Boundary/loss tests only: byte counts are not model token length measurements."""
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))


def demonstration():
    return dict(messages=[
        dict(role='system', content='Research carefully.'),
        dict(role='user', content='Which city?'),
        dict(role='assistant', content='<think>Search.</think>', tool_calls=[
            dict(id='call1', function=dict(name='browser.search', arguments='{"query":"city"}'))]),
        dict(role='tool', tool_call_id='call1', content='A page about Paris.'),
        dict(role='assistant', content='<think>Resolved.</think>Paris'),
    ], tools=[dict(type='function', function=dict(name='browser.search', parameters=dict(type='object')))])


class BrowseCompAdaptationTests(unittest.TestCase):
    def test_preserves_source_and_marks_training_target(self):
        raw = demonstration()
        original = copy.deepcopy(raw)
        row = normalize_browsecomp(raw)
        self.assertEqual(raw, original)
        self.assertEqual(row['answer'], 'Paris')
        self.assertEqual(row['answer_source'], 'demonstration_final_response')
        self.assertTrue(row['training_only'])
        self.assertEqual(row['messages'][-1]['content'], raw['messages'][-1]['content'])

    def test_reference_is_preserved(self):
        raw = demonstration()
        raw['answer'] = 'City of Paris'
        row = normalize_browsecomp(raw)
        self.assertEqual(row['answer'], 'City of Paris')
        self.assertEqual(row['answer_source'], 'reference')

    def test_literal_observation_tags_are_data(self):
        raw = demonstration()
        raw['messages'][3]['content'] = 'Page: <tool_response>example</tool_response> end'
        row = normalize_browsecomp(raw)
        self.assertEqual(row['escaped_observation_delimiters'], 2)
        bodies = [text for text, _, compressible in pieces(row['messages']) if compressible]
        self.assertEqual(bodies, ['Page: &lt;tool_response&gt;example&lt;/tool_response&gt; end'])

    def test_only_observation_body_is_compressible_and_masked(self):
        row = normalize_browsecomp(demonstration())
        sample = encode(row['messages'], ByteTokenizer(), tools=row['tools'])
        self.assertEqual(len(sample['regions']), 1)
        for start, end in sample['regions']:
            self.assertEqual(bytes(sample['input_ids'][start:end]).decode(), 'A page about Paris.')
            self.assertTrue(all(v == -100 for v in sample['labels'][start:end]))
        self.assertGreater(sum(label != -100 for label in sample['labels']), 0)
        self.assertEqual(sample['n_searches'], 1)

    def test_complete_length_includes_schema(self):
        row = normalize_browsecomp(demonstration())
        with_schema = encode(row['messages'], ByteTokenizer(), tools=row['tools'])
        without_schema = encode(row['messages'], ByteTokenizer())
        self.assertGreater(len(with_schema['input_ids']), len(without_schema['input_ids']))

    def test_length_boundary_and_no_observation_selection(self):
        row = normalize_browsecomp(demonstration())
        row['lengths'] = {'test': dict(total=100)}
        self.assertIsNone(Selection(max_length=100, correctness='any', observations='with').rejection(row))
        self.assertEqual(Selection(max_length=99, correctness='any').rejection(row), 'over_length')
        self.assertEqual(Selection(max_length=100, correctness='any', observations='without').rejection(row),
                         'has_observation')

    def test_dataset_rejects_overlength_without_truncation(self):
        row = normalize_browsecomp(demonstration())
        size = len(encode(row['messages'], ByteTokenizer(), tools=row['tools'])['input_ids'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            self.assertEqual(len(OpenSeekerDataset(path, ByteTokenizer(), max_length=size)[0]['input_ids']), size)
            with self.assertRaisesRegex(ValueError, 'exceeds max_length'):
                OpenSeekerDataset(path, ByteTokenizer(), max_length=size - 1)[0]

    def test_unfinished_reasoning_and_orphan_tool_rejected(self):
        raw = demonstration()
        raw['messages'][-1]['content'] = '<think>unfinished'
        with self.assertRaises(ValueError):
            normalize_browsecomp(raw)
        raw = demonstration()
        raw['messages'][3]['tool_call_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'orphan'):
            normalize_browsecomp(raw)

    def test_prepare_measure_uses_explicit_adapter(self):
        prepare_openseeker.TOKENIZERS = {'test': ByteTokenizer()}
        row, facts, error = prepare_openseeker.measure(('browsecomp/train.jsonl', 0, demonstration()))
        self.assertIsNone(error)
        self.assertTrue(row['training_only'])
        self.assertEqual(facts['calls'], 1)
        self.assertGreater(row['lengths']['test']['observations'], 0)
        # The original source still requires a genuine reference field.
        row, _, error = prepare_openseeker.measure(('polar/train.jsonl', 0, demonstration()))
        self.assertIsNone(row)
        self.assertIsNotNone(error)

    def test_browsecomp_evaluation_advertises_executable_tools(self):
        row = normalize_browsecomp(demonstration(), source='browsecomp/train.jsonl')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            system, _ = resolve_system_prompt(prompt_data=path)
        self.assertTrue(system.startswith('Research carefully.'))
        schemas = TOOLS_RE.search(system).group(1)
        self.assertNotIn('browser.search', schemas)
        self.assertIn('"search"', schemas)
        self.assertIn('"visit"', schemas)
        local = build_system_prompt(system=system, local=True, allow_visit=False)
        self.assertNotIn('"visit"', TOOLS_RE.search(local).group(1))
        self.assertIn('No external websites', local)
        # Training schema and original calls remain available in stored data.
        self.assertEqual(row['tools'][0]['function']['name'], 'browser.search')
        self.assertIn('browser.search', row['messages'][2]['content'])


class HotpotJudgeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name)
        self.rows = [dict(id='q1', query='Which city?', answer='Paris', supporting_facts=[]),
                     dict(id='q2', query='Which country?', answer='France', supporting_facts=[])]

    def test_missing_failed_answers_stay_in_denominator(self):
        judge = Mock(return_value='A')
        predictions = {'q1': dict(status='completed', final_response='{"answer":"Paris","sp":[]}')}
        metrics = score_hotpot_judge(self.rows, predictions, self.output, judge, preflight=False)
        self.assertEqual(metrics['llm_judge_accuracy'], 0.5)
        self.assertTrue(metrics['complete'])
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(json.loads(judge.call_args.args[0].split('\n', 1)[1])['candidate_answer'], 'Paris')

    def test_invalid_label_is_unknown_and_incomplete(self):
        predictions = {'q1': dict(status='completed', final_response='Paris')}
        metrics = score_hotpot_judge(self.rows, predictions, self.output,
                                     Mock(return_value='maybe'), attempts=1, preflight=False)
        self.assertEqual(metrics['judge_errors'], 1)
        self.assertFalse(metrics['complete'])
        details = [json.loads(line) for line in (self.output / 'judge.jsonl').read_text().splitlines()]
        self.assertIsNone(details[0]['score'])
        self.assertEqual(len(details), 2)

    def test_valid_negative_is_not_retried(self):
        judge = Mock(return_value='B')
        metrics = score_hotpot_judge(self.rows, {'q1': dict(status='completed', final_response='Tokyo')},
                                     self.output, judge, attempts=3, preflight=False)
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(metrics['llm_judge_accuracy'], 0)

    def test_empty_json_answer_skips_judge_and_preflight(self):
        judge = Mock(return_value='A')
        metrics = score_hotpot_judge(self.rows, {'q1': dict(status='completed', final_response='{"answer":""}')},
                                     self.output, judge)
        self.assertEqual(metrics['llm_judge_accuracy'], 0)
        judge.assert_not_called()

    def test_preflight_and_empty_failed_rules(self):
        judge = Mock(side_effect=['A', 'B', 'A'])
        predictions = {'q1': dict(status='completed', final_response='Paris'),
                       'q2': dict(status='error', final_response='France')}
        metrics = score_hotpot_judge(self.rows, predictions, self.output, judge)
        self.assertTrue(json.loads((self.output / 'judge_preflight.json').read_text())['passed'])
        self.assertEqual(metrics['llm_judge_accuracy'], 0.5)
        self.assertEqual(judge.call_count, 3)

    def test_official_metrics_coexist_with_judge(self):
        root = self.output / 'assets'
        install_official(root, 'hotpotqa')
        predictions = {'q1': dict(status='completed', final_response='Paris'),
                       'q2': dict(status='completed', final_response='France')}
        metrics = score_hotpot(root, self.rows, predictions, self.output,
                               Mock(return_value='B'), judge_preflight=False)
        self.assertEqual(metrics['em'], 1)
        self.assertEqual(metrics['f1'], 1)
        self.assertEqual(metrics['llm_judge_accuracy'], 0)


class EntryPointTests(unittest.TestCase):
    @unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash is unavailable')
    def test_all_nine_shell_routes_with_real_python_dry_run(self):
        for method in ('plain', 'beacon', 'base'):
            for benchmark in ('browsecomp', 'browsecomp-plus', 'hotpotqa'):
                with self.subTest(method=method, benchmark=benchmark):
                    env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1',
                               DRY_RUN='1', METHOD=method, BENCHMARK=benchmark, JUDGE_MODEL='local-judge',
                               ACTION='run' if benchmark == 'browsecomp' else 'all')
                    result = subprocess.run([BASH, 'scripts/sh/benchmark/59_run_benchmark.sh'], cwd=ROOT,
                                            env=env, capture_output=True, text=True, encoding='utf-8')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    value = json.loads(result.stdout)
                    if benchmark == 'browsecomp':
                        self.assertEqual(value['judging']['policy'], 'stable-judge-v1')
                    else:
                        self.assertTrue(value['judge_model'])
                        self.assertEqual(value['llm_judge'], benchmark == 'hotpotqa')
                        self.assertEqual(value['kind'], 'beacon' if method == 'beacon' else 'plain')

    def test_hotpot_python_cli_remains_opt_in(self):
        result = subprocess.run([sys.executable, 'scripts/py/52_offline_benchmark.py', 'score',
                                 '--benchmark', 'hotpotqa', '--predictions', 'predictions.jsonl', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value['llm_judge'])
        self.assertIsNone(value['judge_model'])

    def test_requested_judge_requires_local_model_before_loading_data(self):
        result = subprocess.run([sys.executable, 'scripts/py/52_offline_benchmark.py', 'score',
                                 '--benchmark', 'hotpotqa', '--predictions', 'predictions.jsonl', '--llm-judge'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--llm-judge requires --judge-model', result.stderr)

    def test_browsecomp_rejects_evaluation_splits_before_loading_tokenizer(self):
        result = subprocess.run([sys.executable, 'scripts/py/10_prepare.py', '--source', 'browsecomp',
                                 '--input', 'missing.jsonl', '--output', 'unused-output'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--val-fraction 0 --test-fraction 0', result.stderr)


if __name__ == '__main__':
    unittest.main()
