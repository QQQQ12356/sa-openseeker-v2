"""Full-history OpenSeeker agent: same serializer/regions as SFT, arbitrary tools.

Replay is an exact action match diagnostic, never a substitute for online metrics.
A gateway implements POST {name, arguments} -> {content: str} for live tools.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from types import SimpleNamespace
from ..data.openseeker import (encode, parse_calls, json_text, CALL_END, RESPONSE_RE)
from ..data.agent_prompt import prompt_metadata, render_system


class ToolError(RuntimeError):
    pass


class Registry:
    def __init__(self, handlers):
        self.handlers = handlers

    def execute(self, calls):
        responses = []
        for call in calls:
            if call['name'] not in self.handlers:
                raise ToolError('unknown_tool: ' + call['name'])
            try:
                content = self.handlers[call['name']](**call['arguments'])
            except Exception as error:
                raise ToolError(f'tool_execution_error: {error}') from error
            if not isinstance(content, str):
                content = json_text(content)
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


def decoding_options(args, beacon=False):
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
    if beacon and values['decoding'] != 'greedy':
        raise ValueError('native sampling currently supports Plain only; Beacon requires greedy')
    defaults = dict(decoding='greedy', repetition_penalty=1.0, temperature=1.0, top_p=1.0, seed=42)
    return {} if values == defaults else values


class ModelGenerator:
    def __init__(self, model, tokenizer, beacon=False, decoding='greedy', repetition_penalty=1.0,
                 temperature=1.0, top_p=1.0, seed=42):
        self.model, self.tokenizer, self.beacon = model, tokenizer, beacon
        self.decoding, self.repetition_penalty = decoding, repetition_penalty
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        decoding_options(SimpleNamespace(decoding=decoding, repetition_penalty=repetition_penalty,
                                        temperature=temperature, top_p=top_p, generation_seed=seed), beacon)
        config = model.config.to_dict()
        self.metadata = dict(kind='beacon' if beacon else 'plain', do_sample=decoding == 'sample',
                             decoding=decoding, repetition_penalty=repetition_penalty,
                             temperature=temperature, top_p=top_p, generation_seed=seed,
                             checkpoint=getattr(model.config, '_name_or_path', ''),
                             model_config_sha256=hashlib.sha256(
                                 json_text(config).encode()).hexdigest())
        if beacon:
            self.metadata['beacon_config'] = model.beacon_config.to_dict()
        self.metadata['repetition_scope'] = ('beacon_visible_tokens' if beacon else 'full_input_and_generated')

    def start_question(self, question):
        if self.decoding == 'sample':
            import torch
            # Same question/seed receives the same RNG sequence across shards.
            offset = int.from_bytes(hashlib.sha256(question.encode('utf-8')).digest()[:4], 'big')
            torch.manual_seed((self.seed + offset) % (2**32))

    def __call__(self, encoded, max_new_tokens):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        tokenizer = self.tokenizer
        ids = torch.tensor([encoded['input_ids']], device=next(self.model.parameters()).device)
        stops = [CALL_END, '</answer>', '<|im_end|>']
        eos = self.model.config.eos_token_id
        eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
        with torch.inference_mode():
            if self.beacon:
                penalty = ({'repetition_penalty': self.repetition_penalty}
                           if self.repetition_penalty != 1.0 else {})
                out = self.model.beacon_generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    regions=encoded['regions'], max_new_tokens=max_new_tokens, tokenizer=tokenizer,
                    stop_texts=stops, eos_token_ids=eos_ids, reuse_cache=False, **penalty)
                generated = out[0].tolist()
            else:
                start = ids.shape[1]
                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        text = tokenizer.decode(input_ids[0, start:].tolist(), skip_special_tokens=False)
                        return any(s in text for s in stops)
                sampling = dict(temperature=self.temperature, top_p=self.top_p, top_k=0) if self.decoding == 'sample' else {}
                out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    # Do not inherit checkpoint-specific sampling, beam or penalty defaults
                    # that the Beacon argmax loop does not implement.
                    generation_config=GenerationConfig(max_new_tokens=max_new_tokens,
                        do_sample=self.decoding == 'sample', num_beams=1, repetition_penalty=self.repetition_penalty,
                        bos_token_id=self.model.config.bos_token_id,
                        eos_token_id=eos_ids or None, pad_token_id=tokenizer.pad_token_id, **sampling),
                    stopping_criteria=StoppingCriteriaList([Stop()]))
                generated = out[0, start:].tolist()
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
              max_context=16384, max_new_tokens=8192, max_tool_tokens=8192, tool_schemas=None):
    # Only the original system/question enter inference; no reference is injected.
    history = [dict(message) for message in messages[:2]]
    history[0]['content'] = render_system(history[0]['content'], tool_schemas)
    if hasattr(generate, 'start_question'):
        generate.start_question(history[1]['content'])
    events = []; status = 'turn_limit'; prediction = ''; generated_tokens = 0
    for turn in range(max_turns):
        encoded = encode(history, tokenizer, generation_prompt=True)
        available = max_context - len(encoded['input_ids'])
        if available <= 0:
            status = 'context_limit'; break
        text, ended, n_tokens = generate(encoded, min(max_new_tokens, available))
        generated_tokens += n_tokens
        history.append(dict(role='assistant', content=text))
        if not ended:
            status = 'generation_limit'; break
        try:
            calls = parse_calls(text)
        except ValueError as error:
            status = 'invalid_call'; events.append(dict(error=str(error))); break
        if not calls:
            prediction = extract_response(text)
            status = 'completed' if prediction else 'empty_answer'; break
        try:
            replies = tools.execute(calls)
        except ToolError as error:
            status = 'tool_error'; events.append(dict(calls=calls, error=str(error))); break
        lengths = [len(tokenizer.encode(m['content'], add_special_tokens=False)) for m in replies]
        # Do not silently cut observations: filtering during training also preserves bodies.
        if any(n > max_tool_tokens for n in lengths):
            status = 'tool_budget'; events.append(dict(calls=calls, tokens=lengths)); break
        history.extend(replies)
        events.append(dict(calls=calls, observation_tokens=lengths))
    return dict(prediction=prediction, status=status, messages=history, events=events,
                **prompt_metadata(history[0]['content']),
                generation_config=getattr(generate, 'metadata', {}),
                generated_tokens=generated_tokens, turns=len(events))


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
