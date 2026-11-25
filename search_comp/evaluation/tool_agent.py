"""Full-history OpenSeeker agent: same serializer/regions as SFT, arbitrary tools.

Replay is an exact action match diagnostic, never a substitute for online metrics.
A gateway implements POST {name, arguments} -> {content: str} for live tools.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from collections import deque
from types import SimpleNamespace
from ..data.openseeker import (encode, parse_calls, json_text, CALL_END, RESPONSE_RE)
from ..data.agent_prompt import prompt_metadata, render_system
from .plain_generation import PlainGenerationState


class ToolError(RuntimeError):
    pass


class Registry:
    def __init__(self, handlers, log=None, max_calls=None):
        self.handlers = handlers
        self.log = log
        self.max_calls = max_calls
        self.calls = 0

    def execute(self, calls):
        if self.max_calls is not None and self.calls + len(calls) > self.max_calls:
            raise ToolError('tool-call budget exhausted')
        self.calls += len(calls)
        responses = []
        for index, call in enumerate(calls, 1):
            label = f"tool={call['name']} call={index}/{len(calls)}"
            start = time.monotonic()
            if self.log:
                self.log(f"{label} start")
            if call['name'] not in self.handlers:
                if self.log:
                    self.log(f"{label} error=unknown_tool elapsed={time.monotonic() - start:.2f}s")
                raise ToolError('unknown_tool: ' + call['name'])
            try:
                content = self.handlers[call['name']](**call['arguments'])
            except Exception as error:
                if self.log:
                    self.log(f"{label} error={type(error).__name__} elapsed={time.monotonic() - start:.2f}s")
                raise ToolError(f'tool_execution_error: {error}') from error
            if not isinstance(content, str):
                content = json_text(content)
            if self.log:
                self.log(f"{label} end chars={len(content)} elapsed={time.monotonic() - start:.2f}s")
            responses.append(dict(role='user', content='<tool_response>' + content + '</tool_response>'))
        return responses


class HTTPTools:
    def __init__(self, endpoint, timeout=90):
        self.endpoint, self.timeout = endpoint, timeout

    def execute(self, calls):
        replies = []
        for call in calls:
            request = urllib.request.Request(self.endpoint, data=json_text(call).encode(),
                                            headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body['content']
                if not isinstance(content, str):
                    raise ValueError('gateway content must be text')
            except Exception as error:
                raise ToolError(f'gateway_error: {error}') from error
            replies.append(dict(role='user', content='<tool_response>' + content + '</tool_response>'))
        return replies


class ReplayTools:
    """Consume the next recorded batch only if every name and argument matches."""
    def __init__(self, messages):
        self.batches = []
        for i, msg in enumerate(messages):
            if msg['role'] != 'assistant':
                continue
            calls = parse_calls(msg['content'])
            if not calls:
                continue
            responses = []
            for later in messages[i+1:]:
                if later['role'] != 'user':
                    break
                responses.append(later)
            self.batches.append((calls, responses))
        self.index = 0

    def execute(self, calls):
        if self.index >= len(self.batches) or self.batches[self.index][0] != calls:
            raise ToolError('replay_miss: generated action differs from recorded action')
        replies = self.batches[self.index][1]
        self.index += 1
        return [dict(m) for m in replies]


def add_decoding_arguments(parser):
    parser.add_argument('--decoding', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--generation-seed', type=int, default=42)


def decoding_options(args):
    """Validate shared native options; omit defaults for existing integrations."""
    values = dict(decoding=getattr(args, 'decoding', 'greedy'),
                  repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
                  temperature=getattr(args, 'temperature', 1.0),
                  top_p=getattr(args, 'top_p', 1.0),
                  seed=getattr(args, 'generation_seed', 42))
    if values['decoding'] not in ('greedy', 'sample'):
        raise ValueError('decoding must be greedy or sample')
    if not math.isfinite(values['repetition_penalty']) or values['repetition_penalty'] <= 0:
        raise ValueError('repetition-penalty must be finite and positive')
    if not math.isfinite(values['temperature']) or values['temperature'] <= 0:
        raise ValueError('temperature must be finite and positive')
    if not math.isfinite(values['top_p']) or not 0 < values['top_p'] <= 1:
        raise ValueError('top-p must be in (0, 1]')
    if values['seed'] < 0 or values['seed'] > 2**31 - 1:
        raise ValueError('generation-seed must be between 0 and 2147483647')
    defaults = dict(decoding='greedy', repetition_penalty=1.0, temperature=1.0, top_p=1.0, seed=42)
    return {} if values == defaults else values


class ModelGenerator:
    def __init__(self, model, tokenizer, beacon=False, decoding='greedy', repetition_penalty=1.0,
                 temperature=1.0, top_p=1.0, seed=42, reuse_plain_cache=True):
        self.model, self.tokenizer, self.beacon = model, tokenizer, beacon
        self.reuse_plain_cache = reuse_plain_cache and not beacon
        self.plain_state = None
        self.cached_input_tokens = None
        self.decoding, self.repetition_penalty = decoding, repetition_penalty
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        decoding_options(SimpleNamespace(decoding=decoding, repetition_penalty=repetition_penalty,
                                        temperature=temperature, top_p=top_p, generation_seed=seed))
        config = model.config.to_dict()
        self.metadata = dict(kind='beacon' if beacon else 'plain', do_sample=decoding == 'sample',
                             decoding=decoding, repetition_penalty=repetition_penalty,
                             temperature=temperature, top_p=top_p, generation_seed=seed,
                             checkpoint=getattr(model.config, '_name_or_path', ''),
                             model_config_sha256=hashlib.sha256(
                                 json_text(config).encode()).hexdigest())
        if beacon:
            self.metadata['beacon_config'] = model.beacon_config.to_dict()
        else:
            self.metadata['plain_cross_turn_cache'] = self.reuse_plain_cache
        self.metadata['repetition_scope'] = ('beacon_visible_tokens' if beacon else 'full_input_and_generated')

    def start_question(self, question):
        if self.reuse_plain_cache:
            self.plain_state = PlainGenerationState(self.tokenizer)
            self.cached_input_tokens = 0
        if self.decoding == 'sample':
            import torch
            # Same question/seed receives the same RNG sequence across shards.
            offset = int.from_bytes(hashlib.sha256(question.encode('utf-8')).digest()[:4], 'big')
            torch.manual_seed((self.seed + offset) % (2**32))

    def encode_turn(self, history, *, allow_invalid_calls=False):
        if self.plain_state is not None:
            return self.plain_state.encode_turn(history)
        return encode(history, self.tokenizer, generation_prompt=True, allow_invalid_calls=allow_invalid_calls)

    def invalidate_history(self):
        if self.plain_state is not None:
            self.plain_state = PlainGenerationState(self.tokenizer)

    def __call__(self, encoded, max_new_tokens, *, stop_texts=None):
        import torch
        from transformers import GenerationConfig, StoppingCriteriaList
        from ..generation_stopping import GeneratedTextStop
        tokenizer = self.tokenizer
        ids = torch.tensor([encoded['input_ids']], device=next(self.model.parameters()).device)
        stops = stop_texts if stop_texts is not None else [CALL_END, '</answer>', '<|im_end|>']
        eos = self.model.config.eos_token_id
        eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
        with torch.inference_mode():
            if self.beacon:
                penalty = ({'repetition_penalty': self.repetition_penalty}
                           if self.repetition_penalty != 1.0 else {})
                sampling = (dict(do_sample=True, temperature=self.temperature, top_p=self.top_p)
                            if self.decoding == 'sample' else {})
                out = self.model.beacon_generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    regions=encoded['regions'], max_new_tokens=max_new_tokens, tokenizer=tokenizer,
                    stop_texts=stops, eos_token_ids=eos_ids, reuse_cache=False, **sampling, **penalty)
                generated = out[0].tolist()
            else:
                start = ids.shape[1]
                stop = GeneratedTextStop(tokenizer, stops, start_length=start)
                sampling = dict(temperature=self.temperature, top_p=self.top_p, top_k=0) if self.decoding == 'sample' else {}
                cache_options = {}
                if self.plain_state is not None:
                    past, self.cached_input_tokens = self.plain_state.take_cache(encoded['input_ids'])
                    cache_options.update(use_cache=True, return_dict_in_generate=True)
                    if past is not None:
                        cache_options['past_key_values'] = past
                # Full IDs are used by stopping and repetition penalty. With
                # past_key_values, Transformers forwards only the uncached suffix.
                out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    # Use the benchmark decoding settings instead of checkpoint defaults.
                    generation_config=GenerationConfig(max_new_tokens=max_new_tokens,
                        do_sample=self.decoding == 'sample', num_beams=1, repetition_penalty=self.repetition_penalty,
                        bos_token_id=self.model.config.bos_token_id,
                        eos_token_id=eos_ids or None, pad_token_id=tokenizer.pad_token_id, **sampling),
                    stopping_criteria=StoppingCriteriaList([stop]), **cache_options)
                generated = getattr(out, 'sequences', out)[0, start:].tolist()
                if self.plain_state is not None:
                    self.plain_state.save_cache(getattr(out, 'past_key_values', None), encoded['input_ids'], generated)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        ended = any(s in text for s in stops) or bool(generated and generated[-1] in eos_ids)
        # ChatML assistant boundary is re-serialized by the shared encoder.
        text = text.split('<|im_end|>', 1)[0]
        return text, ended, len(generated)


def extract_response(text):
    """Extract the last assistant reply, following OpenSeeker's local evaluator.

    Remove thinking/tool wrappers; unwrap a closed answer block for EM/F1.
    Structured tool datasets may instead end with plain assistant content.
    """
    text = text.split('<|im_start|>assistant')[-1].split('<|im_end|>', 1)[0]
    if '</think>' in text or '</thinking>' in text:
        text = re.split(r'</(?:think|thinking)>', text)[-1]
    elif '<think>' in text or '<thinking>' in text:
        return ''
    text = re.sub(r'<tool_response>.*?</tool_response>', '', text, flags=re.S)
    text = re.sub(r'<tool_calls_begin>.*?</tool_calls_end>', '', text, flags=re.S)
    from .em_f1 import extract_answer
    answer = extract_answer(text)
    if answer is not None:
        return answer
    if '<answer' in text or '</answer' in text:
        return ''
    return text.strip()


def run_agent(messages, tokenizer, generate, tools, max_turns=32,
              max_context=16384, max_new_tokens=8192, max_tool_tokens=8192, tool_schemas=None,
              log=None, max_call_retries=0, observation_window=0, record_turn_stats=False):
    # Only the original system/question enter inference; no reference is injected.
    history = [dict(message) for message in messages[:2]]
    history[0]['content'] = render_system(history[0]['content'], tool_schemas)
    context = history.copy() if observation_window else history
    recent_observations = deque()
    turn_stats = []
    masked_observations = 0

    def append(replies):
        history.extend(replies)
        if context is not history:
            context.extend(replies)

    if hasattr(generate, 'start_question'):
        generate.start_question(history[1]['content'])
    encode_turn = getattr(generate, 'encode_turn', None)
    reused_tokens = getattr(generate, 'cached_input_tokens', None)
    events = []; status = 'turn_limit'; prediction = ''; generated_tokens = 0
    call_retries = 0
    for turn in range(max_turns):
        if observation_window and len(recent_observations) > observation_window:
            first, calls = recent_observations.popleft()
            for index, call in enumerate(calls, first):
                context[index] = dict(role='user',
                    content=f"<tool_response>[Auto-archived {call['name']} result]</tool_response>")
            masked_observations += len(calls)
            invalidate = getattr(generate, 'invalidate_history', None)
            if invalidate is not None:
                invalidate()
        if encode_turn is None:
            encoded = encode(context, tokenizer, generation_prompt=True, allow_invalid_calls=max_call_retries > 0)
        else:
            encoded = encode_turn(context, allow_invalid_calls=max_call_retries > 0)
        available = max_context - len(encoded['input_ids'])
        if available <= 0:
            status = 'context_limit'; break
        budget = min(max_new_tokens, available)
        start = time.monotonic()
        if log:
            log(f"turn={turn + 1}/{max_turns} generation start input_tokens={len(encoded['input_ids'])} max_new_tokens={budget}")
        try:
            text, ended, n_tokens = generate(encoded, budget)
        except Exception as error:
            if log:
                log(f"turn={turn + 1}/{max_turns} generation error={type(error).__name__} elapsed={time.monotonic() - start:.2f}s")
            raise
        cached = getattr(generate, 'cached_input_tokens', None)
        if reused_tokens is not None:
            reused_tokens += cached or 0
        if record_turn_stats:
            turn_stats.append(dict(round=turn + 1, input_tokens=len(encoded['input_ids']),
                                   output_tokens=n_tokens, cached_input_tokens=cached,
                                   latency_s=time.monotonic() - start))
        if log:
            cache_log = f" cached_input_tokens={cached}" if cached is not None else ""
            log(f"turn={turn + 1}/{max_turns} generation end output_tokens={n_tokens} ended={ended} "
                f"elapsed={time.monotonic() - start:.2f}s{cache_log}")
        generated_tokens += n_tokens
        append([dict(role='assistant', content=text)])
        if not ended:
            status = 'generation_limit'; break
        try:
            calls = parse_calls(text)
        except ValueError as error:
            event = dict(error=str(error), kind='invalid_call', retry_requested=False)
            events.append(event)
            if call_retries < max_call_retries and turn + 1 < max_turns:
                call_retries += 1
                event['retry_requested'] = True
                append([dict(role='user', content='<tool_response>Invalid tool-call format. '
                    'No tools were executed. Retry using valid JSON with name and arguments object, '
                    'inside the required tool-call tags. Escape quotes inside JSON strings.</tool_response>')])
                continue
            status = 'invalid_call'
            break
        if not calls:
            prediction = extract_response(text)
            status = 'completed' if prediction else 'empty_answer'; break
        try:
            if log:
                log(f"turn={turn + 1}/{max_turns} tools start calls={len(calls)}")
            start = time.monotonic()
            replies = tools.execute(calls)
        except ToolError as error:
            if log:
                log(f"turn={turn + 1}/{max_turns} tools error elapsed={time.monotonic() - start:.2f}s")
            status = 'tool_error'; events.append(dict(calls=calls, error=str(error))); break
        if log:
            log(f"turn={turn + 1}/{max_turns} tools end elapsed={time.monotonic() - start:.2f}s")
        lengths = [len(tokenizer.encode(m['content'], add_special_tokens=False)) for m in replies]
        # Do not silently cut observations: filtering during training also preserves bodies.
        if any(n > max_tool_tokens for n in lengths):
            status = 'tool_budget'; events.append(dict(calls=calls, tokens=lengths)); break
        if observation_window:
            recent_observations.append((len(context), calls))
        append(replies)
        events.append(dict(calls=calls, observation_tokens=lengths))
    result = dict(prediction=prediction, status=status, messages=history, events=events,
                  **prompt_metadata(history[0]['content']),
                  generation_config=getattr(generate, 'metadata', {}),
                  generated_tokens=generated_tokens, turns=len(events))
    if reused_tokens is not None:
        result['cached_input_tokens'] = reused_tokens
    if observation_window:
        result.update(final_messages=context, masked_observations=masked_observations)
    if record_turn_stats:
        result.update(turn_stats=turn_stats,
                      total_input_tokens=sum(stats['input_tokens'] for stats in turn_stats))
    return result


class CorpusTools(Registry):
    """Executable offline search/visit over an independent URL-addressed corpus.

    Unlike replay, new queries are supported. This is a local environment,
    not a claim of reproducing a live web provider or goal-conditioned summarizer.
    """
    def __init__(self, corpus_path, topk=10):
        from urllib.parse import quote
        from ..data.retrieval import BM25Retriever
        self.retriever = BM25Retriever(str(corpus_path), tokenize_fn=lambda text:
            re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower()))
        self.topk = topk
        self.urls = {}
        self.documents = {}
        for doc in self.retriever.docs:
            url = doc.get('url') or 'corpus://' + quote(str(doc['id']), safe='')
            if url in self.documents:
                raise ValueError('duplicate corpus URL: ' + url)
            self.urls[doc['id']] = url
            self.documents[url] = doc
        super().__init__({'search': self.search, 'visit': self.visit})

    def search(self, query):
        if not isinstance(query, list) or not query or any(not isinstance(q, str) for q in query):
            raise ValueError('search.query must be a nonempty string array')
        batches = []
        for q in query:
            docs = self.retriever.retrieve(q, topk=self.topk)
            batches.append(f'### A local search for {q!r} found {len(docs)} results:\n\n' + '\n\n'.join(
                f"{i}. [{doc['title']}]({self.urls[doc['id']]})\n{doc['text'][:600]}"
                for i, doc in enumerate(docs, 1)))
        return '\n---\n'.join(batches)

    def visit(self, url, goal):
        urls = [url] if isinstance(url, str) else url
        if not isinstance(urls, list) or not urls or any(not isinstance(u, str) for u in urls):
            raise ValueError('visit.url must be a string or nonempty string array')
        if not isinstance(goal, str):
            raise ValueError('visit.goal must be text')
        if any(u not in self.documents for u in urls):
            raise ValueError('URL is not present in the independent corpus')
        return '\n\n'.join(f"URL: {u}\nGoal: {goal}\n{self.documents[u]['text']}" for u in urls)
