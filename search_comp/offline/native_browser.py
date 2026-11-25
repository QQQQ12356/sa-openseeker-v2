"""The Qwen baseline's local, cursor-based browser."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.util import find_spec
import os
from pathlib import Path
import textwrap
from urllib.parse import quote

from .assets import BCP_HIGHLIGHT_JARS


def _schema(name, description, properties, required=()):
    return dict(type="function", function=dict(
        name=f"browser.{name}", description=description,
        parameters=dict(type="object", properties=properties, required=list(required))))


TOOLS = (
    _schema("search", "Search for information related to query and show ranked results.", {
        "query": {"type": "string"},
        "topn": {"type": "integer", "default": 10},
        "source": {"type": "string"},
    }, ("query",)),
    _schema("open", "Open a URL or a link id on a cursor page. Omit id to scroll the page.", {
        "id": {"type": ["number", "string"], "default": -1},
        "cursor": {"type": "integer", "default": -1},
        "loc": {"type": "integer", "default": -1},
        "num_lines": {"type": "integer", "default": -1},
        "view_source": {"type": "boolean", "default": False},
        "source": {"type": "string"},
    }),
    _schema("find", "Find an exact text pattern on the current or specified page.", {
        "pattern": {"type": "string"},
        "cursor": {"type": "integer", "default": -1},
    }, ("pattern",)),
)


def load_highlighter(root):
    jars = [root / "raw/lucene" / name for name in BCP_HIGHLIGHT_JARS]
    if not all(jar.is_file() for jar in jars):
        raise FileNotFoundError("native browser needs Lucene highlighter JARs; rerun BrowseComp-Plus 31 download")
    try:
        import jnius_config
    except ImportError as error:
        raise RuntimeError("native browser requires Pyserini and Java 21") from error
    if not jnius_config.vm_running:
        spec = find_spec("pyserini")
        if not spec or not spec.submodule_search_locations:
            raise RuntimeError("native browser requires Pyserini and Java 21")
        default = Path(next(iter(spec.submodule_search_locations))) / "resources/jars"
        fat_jars = list(Path(os.environ.get("ANSERINI_CLASSPATH", default)).glob("anserini-*-fatjar.jar"))
        if not fat_jars:
            raise FileNotFoundError("Pyserini Anserini fat JAR not found")
        jnius_config.add_options("-Djava.awt.headless=true")
        jnius_config.set_classpath(str(max(fat_jars, key=lambda jar: jar.stat().st_mtime)),
                                  *(str(jar) for jar in jars))
    from pyserini.pyclass import autoclass
    from jnius import JavaException

    analyzer = autoclass("org.apache.lucene.analysis.standard.StandardAnalyzer")()
    formatter = autoclass("org.apache.lucene.search.uhighlight.DefaultPassageFormatter")("", "", " … ", False)
    highlighter = autoclass("org.apache.lucene.search.uhighlight.UnifiedHighlighter")
    highlighter = highlighter.builderWithoutSearcher(analyzer).withFormatter(formatter).build()
    parser = autoclass("org.apache.lucene.queryparser.classic.QueryParser")("content", analyzer)
    quotes = str.maketrans(dict.fromkeys("“”„«»「」『』", '"'))

    def highlight(query, text):
        query = query.translate(quotes)
        if query.count('"') % 2:
            query = query.replace('"', "")
        try:
            result = highlighter.highlightWithoutSearcher("content", parser.parse(query), text, 50)
        except JavaException:
            result = None
        return (str(result).strip() if result is not None else text)[:180]

    return highlight


def _lines(text):
    return [wrapped for line in text.splitlines() or [""]
            for wrapped in (textwrap.wrap(line, width=80, replace_whitespace=False,
                                          drop_whitespace=False) or [""])]


@dataclass
class Page:
    title: str
    lines: list
    url: str = ""
    links: list = field(default_factory=list)
    query: str | None = None
    docids: list = field(default_factory=list)
    docid: str | None = None

    def render(self, cursor, start, count):
        if not 0 <= start < len(self.lines):
            raise ValueError(f"line {start} is outside this page")
        end = min(len(self.lines), start + count)
        title = f"{self.title} ({self.url})" if self.url else self.title
        header = f"[{cursor}] {title}\n**viewing lines [{start} - {end - 1}] of {len(self.lines) - 1}**\n\n"
        return header + "\n".join(f"L{i}: {self.lines[i]}" for i in range(start, end))


class NativeBrowser:
    def __init__(self, index, documents, highlight, topk=15, max_calls=4096):
        self.index = index
        self.documents = documents
        self.highlight = highlight
        self.topk = topk
        self.max_calls = max_calls
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.reset()

    def reset(self):
        self.pages = []
        self.retrieved = set()
        self.visited = set()
        self.trace = []
        self.counts = Counter()

    def close(self):
        self.executor.shutdown()

    def _search(self, query, topn=None, source=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search requires a nonempty query")
        topn = self.topk if topn is None else int(topn)
        if topn < 1:
            raise ValueError("topn must be positive")
        docs = [d for d in self.index.search(query, topn) if d["source_url"] and d["text"]]
        if not docs:
            raise ValueError(f"no results for {query!r}")
        page = Page(query, ["# Search Results", ""], query=query)
        for i, doc in enumerate(docs):
            url = doc["source_url"]
            page.links.append((url, None))
            page.docids.append(doc["docid"])
            page.lines.append(f"【{i}†{url}】")
            page.lines.extend(_lines(f" Title: {doc['title']}. {self.highlight(query, doc['text'])}"))
        return page, 0, 80

    @staticmethod
    def _page(pages, cursor):
        if not pages:
            raise ValueError("no pages to access")
        if not isinstance(cursor, int) or not -1 <= cursor < len(pages):
            raise ValueError(f"invalid cursor {cursor!r}")
        return pages[cursor]

    def _open(self, pages, id=-1, cursor=-1, loc=-1, num_lines=-1, view_source=False, source=None):
        hint = None
        if isinstance(id, str):
            url = id
        elif id >= 0:
            page = (next((page for page in reversed(pages) if page.query is not None), None)
                    if cursor == -1 else None)
            page = page or self._page(pages, cursor)
            if id >= len(page.links):
                raise ValueError(f"invalid link id {id}")
            url, hint = page.links[id]
        elif not view_source:
            return self._page(pages, cursor), max(loc, 0), num_lines if num_lines > 0 else 50
        else:
            url = self._page(pages, cursor).url
        doc = self.documents.get_url(url)
        page = Page(doc["title"], _lines(doc["text"]), url, docid=doc["docid"])
        start = max((hint or 0) - 4, 0) if loc < 0 else loc
        return page, start, num_lines if num_lines > 0 else 50

    def _find(self, pages, pattern, cursor=-1):
        page = self._page(pages, cursor)
        if page.docid is None or not isinstance(pattern, str) or not pattern:
            raise ValueError("find requires a nonempty pattern and an open document")
        pattern = pattern.lower()
        result = Page(f"Find results for text: {pattern!r} in {page.title!r}", [],
                      f"{page.url}/find?pattern={quote(pattern)}")
        offset = 0
        while offset < len(page.lines) and len(result.links) < 50:
            if pattern not in page.lines[offset].lower():
                offset += 1
                continue
            result.lines.append(f"【{len(result.links)}†match at L{offset}】")
            result.lines.extend(page.lines[offset:offset + 4])
            result.lines.append("")
            result.links.append((page.url, offset))
            offset += 4
        if result.lines:
            result.lines.pop()
        else:
            result.lines.append(f"No find results for pattern: {pattern!r}")
        return result, 0, 50

    def execute(self, calls):
        # Search/encoding is serialized as in the reference search service.
        # Parallel open/find resolve against the same pre-batch snapshot.
        pages = tuple(self.pages)
        handlers = {"browser.open": self._open, "browser.find": self._find}
        pending = []
        for call in calls:
            function = call["function"]
            name = function["name"]
            self.counts[name] += 1
            handler = handlers.get(name)
            if handler is None:
                pending.append(None)
            else:
                pending.append(self.executor.submit(handler, pages, **function["arguments"]))
        prepared = []
        for call, future in zip(calls, pending):
            function = call["function"]
            name = function["name"]
            try:
                if future is not None:
                    value = future.result()
                elif name == "browser.search":
                    value = self._search(**function["arguments"])
                else:
                    raise ValueError(f"unknown tool {name}")
            except Exception as error:
                value = f"Error executing {name}: {error}"
            prepared.append(value)
        results = []
        for call, value in zip(calls, prepared):
            if isinstance(value, str):
                results.append((value, False))
                continue
            page, start, count = value
            try:
                text = page.render(len(self.pages), start, count)
            except ValueError as error:
                results.append((f"Error rendering {call['function']['name']}: {error}", False))
                continue
            self.pages.append(page)
            if page.query is not None:
                self.retrieved.update(page.docids)
                self.trace.append(dict(query=page.query, ranked_docids=page.docids))
            if page.docid is not None:
                self.visited.add(page.docid)
            results.append((text, True))
        return results
