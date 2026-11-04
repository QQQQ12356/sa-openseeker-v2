"""Validate entry routing and scoring contracts without GPU, models or network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search_comp.data.agent_prompt import resolve_system_prompt, build_system_prompt
from search_comp.evaluation.answer_metrics import answer_metrics
from search_comp.workflows.benchmark_openseeker import require_environment, align_results

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') if os.name != 'nt' else 'C:/Program Files/Git/bin/bash.exe'


class AnswerMetricTests(unittest.TestCase):
    def test_missing_failed_and_aliases_keep_full_denominator(self):
        questions = [dict(id='a', answer=['Paris', 'City of Paris']),
                     dict(id='b', answer='France'), dict(id='c', answer='Japan')]
        predictions = [dict(id='a', status='completed', final_response='<answer>Paris</answer>'),
                       dict(id='b', status='error', final_response='France')]
        self.assertEqual(answer_metrics(questions, predictions), dict(em=1/3, f1=1/3))

    def test_citations_removed_only_for_bcp_supplement(self):
        questions = [dict(id='a', answer='Paris')]
        predictions = [dict(id='a', final_response='Paris [doc1]')]
        self.assertEqual(answer_metrics(questions, predictions, True), dict(em=1, f1=1))
        self.assertEqual(answer_metrics(questions, predictions)['em'], 0)

    def test_online_matches_query_even_when_prediction_order_differs(self):
        rows = [dict(query='q1', answer='Paris'), dict(query='q2', answer='France')]
        predictions = [dict(query='q2', final_response='France'), dict(query='q1', final_response='Paris')]
        self.assertEqual(answer_metrics(rows, predictions), dict(em=1, f1=1))

    def test_generation_does_not_require_scorer(self):
        variables = dict(SUMMARY_API_URL='local', SUMMARY_API_KEY='local', SUMMARY_MODEL_NAME='local',
                         JINA_API_KEYS='local', SERPER_API_KEY='local')
        with patch.dict(os.environ, variables, clear=True):
            require_environment('native', generate=True, score=False)
            with self.assertRaisesRegex(ValueError, 'SCORER'):
                require_environment('native', generate=False)

    def test_online_failed_response_is_zero_for_judge_and_answer_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw.jsonl'
            output = Path(directory) / 'aligned.jsonl'
            raw.write_text(json.dumps(dict(query='q', answer='Paris', final_response='Paris', status='error')),
                           encoding='utf-8')
            align_results([dict(query='q', answer='Paris')], raw, output)
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['final_response'], '')
            self.assertEqual(answer_metrics(rows, rows), dict(em=0, f1=0))

    def test_bcp_real_scoring_keeps_all_three_metrics(self):
        from search_comp.offline.assets import install_official
        from search_comp.offline.scoring import score_browsecomp
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / 'assets'
            install_official(root, 'browsecomp-plus')
            questions = [dict(id='q1', query='Which city?', answer='Paris', evidence_docs=['d1'], gold_docs=['d1']),
                         dict(id='q2', query='Which country?', answer='France', evidence_docs=['d2'], gold_docs=['d2'])]
            predictions = {'q1': dict(id='q1', status='completed', final_response='Paris [d1]',
                                      retrieved_docids=['d1'])}
            response = 'extracted_final_answer: Paris\nreasoning: Same city.\ncorrect: yes\nconfidence: 100'
            metrics = score_browsecomp(root, questions, predictions, output,
                                      judge=lambda _: response, judge_preflight=False)
            self.assertEqual(metrics['em'], 0.5)
            self.assertEqual(metrics['f1'], 0.5)
            self.assertEqual(metrics['llm_judge_accuracy'], 0.5)
            self.assertTrue(metrics['complete'])

    def test_research_prompt_preserved_and_local_adaptation_appended(self):
        system = 'Research instructions.\n<tools>{"name":"search","parameters":{}}</tools>'
        row = dict(messages=[dict(role='system', content=system), dict(role='user', content='q')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps(row), encoding='utf-8')
            resolved, _ = resolve_system_prompt(path)
            self.assertEqual(resolved, system)
            adapted = build_system_prompt(system=resolved, local=True, allow_visit=False,
                                          answer_instruction='Return a short answer.')
            self.assertTrue(adapted.startswith('Research instructions.'))
            self.assertIn('Return a short answer.', adapted)
            row['messages'].append(dict(role='reasoning', content='raw export'))
            path.write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'prepared RESEARCH'):
                resolve_system_prompt(path)

    def test_training_prompt_fingerprint_prevents_silent_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            data = run / 'train.jsonl'
            data.write_text(json.dumps(dict(messages=[dict(role='system', content='changed')])), encoding='utf-8')
            (run / 'experiment.json').write_text(json.dumps(dict(
                training_config=dict(train_data_path=str(data)),
                data_fingerprints=dict(train_data_path='incorrect-digest'))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed since training'):
                resolve_system_prompt(run_dir=run)


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash is unavailable')
class EntryTests(unittest.TestCase):
    def run_entry(self, method, filename, **variables):
        env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1',
                   DRY_RUN='1', ACTION='all', RUN_DIR='placeholder-training-run', JUDGE_MODEL='placeholder-judge')
        env.update(variables)
        return subprocess.run([BASH, 'scripts/sh/' + method + '/' + filename], cwd=ROOT,
                              env=env, capture_output=True, text=True, encoding='utf-8')

    def test_all_eighteen_test_and_score_entries_route_independently(self):
        for method in ('plain', 'beacon', 'baseline'):
            for number, bench in ((40, 'browsecomp'), (42, 'browsecomp_plus'), (44, 'hotpotqa')):
                with self.subTest(method=method, benchmark=bench):
                    result = self.run_entry(method, f'{number}_test_{bench}.sh')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    value = json.loads(result.stdout)
                    if bench == 'browsecomp':
                        self.assertTrue(value['generate_only'])
                    else:
                        self.assertEqual(value['action'], 'run')
                        self.assertEqual(value['kind'], 'beacon' if method == 'beacon' else 'plain')
                    result = self.run_entry(method, f'{number+1}_score_{bench}.sh', PREDICTIONS='predictions.jsonl')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    value = json.loads(result.stdout)
                    if bench == 'browsecomp':
                        self.assertFalse(value['generate_only'])
                        self.assertEqual(value['generation'][0], 'import_predictions')
                    else:
                        self.assertEqual(value['action'], 'score')
                        self.assertTrue(value['judge_model'])
                        self.assertEqual(value['llm_judge'], bench == 'hotpotqa')

    def test_all_six_training_entries_have_fixed_dataset_and_complete_config(self):
        for method in ('plain', 'beacon', 'baseline'):
            for number, dataset in ((10, 'openseeker'), (11, 'research')):
                path = ROOT / f'scripts/sh/{method}/{number}_train_{dataset}.sh'
                result = subprocess.run([BASH, '-n', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = path.read_text(encoding='utf-8')
                self.assertIn('dataset_name: ' + dataset, text)
                self.assertIn('method: ' + method, text)
                self.assertIn('distributed:', text)
                self.assertIn('use_lora: false' if method == 'baseline' else 'use_lora: true', text)

    def test_generation_requires_training_prompt_source(self):
        result = self.run_entry('plain', '40_test_browsecomp.sh', RUN_DIR='', PROMPT_DATA='')
        self.assertEqual(result.returncode, 2)
        self.assertIn('RUN_DIR', result.stderr)

    def test_legacy_training_entries_cannot_start_a_third_dataset(self):
        for method, filename in [('plain', '22_plain_searchr1_train.sh'),
                                 ('beacon', '32_beacon_searchr1_train.sh'),
                                 ('beacon', '34_beacon_searchr1_train_impl.sh')]:
            result = self.run_entry(method, filename)
            self.assertEqual(result.returncode, 2)
            self.assertIn('11_train_research.sh', result.stderr)

    def test_removed_browsecomp_sft_source_rejected_before_tokenizer_loading(self):
        result = subprocess.run([sys.executable, 'scripts/py/10_prepare.py', '--source', 'browsecomp',
                                 '--input', 'missing.jsonl', '--output', 'unused-output'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
