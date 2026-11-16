"""Plain cross-turn caching contracts, with a tiny Qwen integration when installed."""
from importlib.util import find_spec
from types import SimpleNamespace
import unittest

from search_comp.data.openseeker import encode
from search_comp.evaluation.plain_generation import PlainGenerationState
from search_comp.evaluation.tool_agent import ModelGenerator, Registry, run_agent


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class PlainGenerationTests(unittest.TestCase):
    def test_incremental_prompt_matches_full_serialization_without_reencoding_history(self):
        tokenizer = Tokenizer()
        state = PlainGenerationState(tokenizer)
        history = [dict(role="system", content="System rules"),
                   dict(role="user", content="Original question")]
        self.assertEqual(state.encode_turn(history)["input_ids"],
                         encode(history, Tokenizer(), generation_prompt=True)["input_ids"])
        history.extend([dict(role="assistant", content="Call"),
                        dict(role="user", content="<tool_response>Result</tool_response>"),
                        dict(role="user", content="<tool_response>Another result</tool_response>")])
        tokenizer.encoded.clear()
        self.assertEqual(state.encode_turn(history)["input_ids"],
                         encode(history, Tokenizer(), generation_prompt=True)["input_ids"])
        self.assertIn("Result", tokenizer.encoded)
        self.assertNotIn("Original question", tokenizer.encoded)
        self.assertNotIn("System rules", tokenizer.encoded)

    def test_exact_prefix_uses_actual_cache_length_and_mismatch_discards_cache(self):
        state = PlainGenerationState(Tokenizer())
        cache = SimpleNamespace(get_seq_length=lambda: 3)
        prompt, generated = [1, 2], [3, 4]
        state.save_cache(cache, prompt, generated)
        self.assertEqual(state.take_cache([1, 2, 3, 5, 6]), (cache, 3))
        state.save_cache(cache, prompt, generated)
        self.assertEqual(state.take_cache([1, 2, 7, 5, 6]), (None, 0))
        self.assertEqual(state.take_cache([1, 2, 3, 5, 6]), (None, 0))
        state.save_cache(cache, prompt, generated)
        self.assertEqual(state.take_cache([1, 2, 3]), (None, 0))

    def test_agent_reuses_prefix_after_tool_result_and_resets_for_next_question(self):
        tokenizer = Tokenizer()
        tool_call = ('<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":["Paris"]}}'
                     '</tool_call></tool_calls_end>')

        class ScriptedGenerator(ModelGenerator):
            def __call__(self, encoded, max_new_tokens):
                ids = encoded["input_ids"]
                past, self.cached_input_tokens = self.plain_state.take_cache(ids)
                reply = "<answer>Paris</answer>" if past else tool_call
                generated = tokenizer.encode(reply)
                cache = SimpleNamespace(get_seq_length=lambda: len(ids) + len(generated) - 1)
                self.plain_state.save_cache(cache, ids, generated)
                return reply, True, len(generated)

        config = SimpleNamespace(to_dict=lambda: {})
        generate = ScriptedGenerator(SimpleNamespace(config=config), tokenizer)
        for question in ("Where is it?", "What is the capital?"):
            messages = [dict(role="system", content="System"), dict(role="user", content=question)]
            tools = Registry({"search": lambda query: "Paris was found."})
            result = run_agent(messages, tokenizer, generate, tools, max_turns=2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["prediction"], "Paris")
            self.assertEqual(tools.calls, 1)
            self.assertGreater(result["cached_input_tokens"], 0)

    @unittest.skipUnless(find_spec("torch") and find_spec("transformers"), "requires PyTorch and Transformers")
    def test_qwen35_cached_prefill_only_forwards_suffix_and_matches_full_decode(self):
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        torch.manual_seed(7)
        cfg = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"], max_position_embeddings=256,
            pad_token_id=0, eos_token_id=None,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        )
        model = Qwen3_5ForCausalLM(cfg).eval()
        tokenizer = Tokenizer()
        cached = ModelGenerator(model, tokenizer, repetition_penalty=1.1)
        full = ModelGenerator(model, tokenizer, repetition_penalty=1.1, reuse_plain_cache=False)
        prefill_lengths = []

        def record_length(_model, _args, kwargs):
            prefill_lengths.append(kwargs["input_ids"].shape[1])

        hook = model.register_forward_pre_hook(record_length, with_kwargs=True)
        self.addCleanup(hook.remove)
        cached.start_question("first")
        first = {"input_ids": [9, 8, 7, 6]}
        generated, _, _ = cached(first, 3)
        second = {"input_ids": first["input_ids"] + tokenizer.encode(generated) + [5, 4, 3]}
        prefill_lengths.clear()
        actual = cached(second, 2)
        self.assertGreater(cached.cached_input_tokens, 0)
        self.assertEqual(prefill_lengths[0], len(second["input_ids"]) - cached.cached_input_tokens)
        self.assertEqual(actual, full(second, 2))

        cached.start_question("second")
        prefill_lengths.clear()
        self.assertEqual(cached(second, 2), full(second, 2))
        self.assertEqual(cached.cached_input_tokens, 0)
        self.assertEqual(prefill_lengths[0], len(second["input_ids"]))
