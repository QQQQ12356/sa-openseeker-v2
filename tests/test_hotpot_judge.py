"""Local judge regressions without model weights or network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from search_comp.offline.assets import install_official
from search_comp.offline.scoring import score_hotpot, score_hotpot_judge

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


if __name__ == '__main__':
    unittest.main()
