"""No model weights or network: export protocol, supervision and HTML regressions."""
import copy
import json
from pathlib import Path
import unittest

from search_comp.data.research import normalize_research
from search_comp.data.openseeker import encode, parse_calls, pieces
from search_comp.workflows.render_dataset_html import prepare_record


def example():
    return {'messages': json.dumps([
        {'role': 'system', 'content': '## Tool-Use Protocol\nold\nYou are a tool-augmented QA agent.\n<tools>{"name":"search"}</tools>'},
        {'role': 'user', 'content': 'Where?'},
        {'role': 'reasoning', 'content': '<think>Look up.</think>'},
        {'role': 'tool_call', 'content': '<tool_call>{"name":"search","arguments":{"query":["city"]}}</tool_call>'},
        {'role': 'tool_output', 'content': '<tool_response>Paris.</tool_response>'},
        {'role': 'reasoning', 'content': '<think>Done.</think>'},
        {'role': 'answer', 'content': '<answer>Paris</answer>'},
    ])}


class ResearchTests(unittest.TestCase):
    def test_normalized_turns_and_source(self):
        raw = example()
        original = copy.deepcopy(raw)
        row = normalize_research(raw)
        self.assertEqual(raw, original)
        self.assertEqual([m['role'] for m in row['messages']],
                         ['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertEqual(parse_calls(row['messages'][2]['content'])[0]['name'], 'search')
        self.assertEqual(row['answer'], 'Paris')
        self.assertTrue(row['training_only'])
        self.assertEqual(row['answer_source'], 'demonstration_final_response')
        self.assertNotIn('old', row['messages'][0]['content'])
        self.assertIn('<tools>', row['messages'][0]['content'])

    def test_observation_mask_and_region(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
        row = normalize_research(example())
        bodies = [(text, supervised) for text, supervised, compressible in pieces(row['messages']) if compressible]
        self.assertEqual(bodies, [('Paris.', False)])
        encoded = encode(row['messages'], Tokenizer())
        a, b = encoded['regions'][0]
        self.assertEqual(encoded['labels'][a:b], [-100] * (b-a))
        self.assertTrue(any(n != -100 for n in encoded['labels']))

    def test_raw_html_observation(self):
        record = prepare_record(example(), Path('raw.jsonl'), 1)
        self.assertEqual(len(record['observations']), 1)
        self.assertEqual(record['messages'][2]['role'], 'reasoning')

    def test_missing_response_or_answer_is_rejected(self):
        for removed in (4, 6):
            raw = example()
            messages = json.loads(raw['messages'])
            messages.pop(removed)
            raw['messages'] = messages
            with self.assertRaises(ValueError):
                normalize_research(raw)

    def test_reference_preserved(self):
        raw = example()
        raw['answer'] = 'City of Paris'
        self.assertEqual(normalize_research(raw)['answer_source'], 'reference')

    def test_incompatible_tool_is_rejected(self):
        raw = example()
        raw['messages'] = raw['messages'].replace('search', 'unsupported_tool')
        with self.assertRaisesRegex(ValueError, 'unsupported RESEARCH tool'):
            normalize_research(raw)

    def test_trailing_messages_are_rejected(self):
        raw = example()
        raw['messages'] = json.loads(raw['messages'])
        raw['messages'].append({'role': 'reasoning', 'content': '<think>Again.</think>'})
        with self.assertRaisesRegex(ValueError, 'after RESEARCH final answer'):
            normalize_research(raw)


if __name__ == '__main__':
    unittest.main()
