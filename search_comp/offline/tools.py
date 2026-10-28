"""Local search/visit with bounded observations and retrieval traces."""
from __future__ import annotations

from collections import Counter
import json
from urllib.parse import unquote

SYSTEM_PROMPT = '''You are a research agent answering questions using a fixed local document corpus.
Use the tools to find evidence. No external websites or network services are available.
Call tools using this exact format:
<tool_calls_begin><tool_call>{"name": "search", "arguments": {"query": ["keywords"]}}</tool_call></tool_calls_end>
For multiple calls, put multiple <tool_call>JSON</tool_call> blocks inside one envelope.
search accepts query (array of strings) and returns ranked documents with docid and text.
visit accepts url (a local://docid string or array) and goal (string), and reads a stored document.
Use <think>...</think> for reasoning and <answer>...</answer> for your final answer.
'''
HOTPOT_INSTRUCTION = '''For the final answer, return a JSON object inside <answer>...</answer>:
{"answer": "a short exact answer", "sp": [["Document title", 0]]}.
sp contains ONLY sentence references that support your answer, using the displayed zero-based sentence IDs.
If you cannot identify supporting sentences, use an empty sp array. Do not copy whole passages into answer.
'''
BCP_INSTRUCTION = "Give a clear final answer and cite supporting document IDs using [docid]."


class LocalTools:
    def __init__(self, index, tokenizer, topk=5, snippet_tokens=512, read_tokens=4096,
                 allow_visit=True, max_calls=200):
        self.index, self.tokenizer = index, tokenizer
        self.topk, self.snippet_tokens, self.read_tokens = topk, snippet_tokens, read_tokens
        self.allow_visit, self.max_calls = allow_visit, max_calls
        self.reset()

    def reset(self):
        self.retrieved = set()
        self.visited = set()
        self.counts = Counter()
        self.trace = []

    def clip(self, value, budget):
        tokens = self.tokenizer.encode(value, add_special_tokens=False)
        return self.tokenizer.decode(tokens[:budget], skip_special_tokens=False)

    def render(self, doc, budget):
        if doc["sentences"]:
            body = "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(doc["sentences"]))
        else:
            body = doc["text"]
        return dict(docid=doc["docid"], title=doc["title"], url=doc["url"],
                    source_url=doc["source_url"], text=self.clip(body, budget))

    def search(self, query):
        if not isinstance(query, list) or not query or any(not isinstance(q, str) or not q.strip() for q in query):
            raise ValueError("search.query must be a nonempty array of nonempty strings")
        if len(query) > 16:
            raise ValueError("at most 16 queries per search call")
        batches = []
        for value in query:
            docs = self.index.search(value, self.topk)
            docids = [doc["docid"] for doc in docs]
            self.retrieved.update(docids)
            self.trace.append(dict(query=value, ranked_docids=docids))
            batches.append(dict(query=value, results=[self.render(doc, self.snippet_tokens) for doc in docs]))
        return json.dumps(batches, ensure_ascii=False)

    def visit(self, url, goal):
        if not self.allow_visit:
            raise ValueError("visit disabled in search-only protocol")
        urls = [url] if isinstance(url, str) else url
        if not isinstance(urls, list) or not urls or len(urls) > 16 or not isinstance(goal, str):
            raise ValueError("visit requires 1..16 local URLs and a goal string")
        results = []
        for value in urls:
            if not isinstance(value, str) or not value.startswith("local://"):
                raise ValueError("visit only reads local://docid; external URLs are forbidden")
            docid = unquote(value[len("local://"):])
            doc = self.index.get(docid)
            self.visited.add(docid)
            results.append(self.render(doc, self.read_tokens))
        # goal is accepted for compatibility; no LLM summarization or answer-based filtering.
        return json.dumps(results, ensure_ascii=False)

    def execute(self, calls):
        from search_comp.evaluation.tool_agent import Registry, ToolError
        if sum(self.counts.values()) + len(calls) > self.max_calls:
            raise ToolError("local tool-call budget exhausted")
        handlers = {"search": self.search, "visit": self.visit}
        for call in calls:
            self.counts[call["name"]] += 1
        return Registry(handlers).execute(calls)
