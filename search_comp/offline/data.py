"""Streaming adapters for official BrowseComp-Plus and HotpotQA fullwiki assets."""
from __future__ import annotations

import bz2
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tarfile

from .assets import digest, dump, verify_official


def iter_jsonl(source):
    for number, line in enumerate(source, 1):
        if line.strip():
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid JSONL line {number}") from error


def records(path, *, parquet_columns=None):
    """Read local JSON/JSONL, compressed JSONL, Parquet, tar archives or a directory."""
    path = Path(path)
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*") if p.is_file() and
                       (p.suffix in {".jsonl", ".json", ".bz2", ".gz", ".parquet"}) and
                       ".cache" not in p.relative_to(path).parts)
        if not paths:
            raise ValueError(f"no supported source files: {path}")
        for child in paths:
            yield from records(child, parquet_columns=parquet_columns)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if member.isfile():
                    source = archive.extractfile(member)
                    # The official tar contains bz2-compressed JSONL members.
                    if member.name.endswith(".bz2"):
                        with bz2.open(source, "rt", encoding="utf-8") as text:
                            yield from iter_jsonl(text)
                    elif member.name.endswith((".jsonl", ".json")) or Path(member.name).name.startswith("wiki_"):
                        import io
                        with io.TextIOWrapper(source, encoding="utf-8") as text:
                            yield from iter_jsonl(text)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=parquet_columns):
            yield from batch.to_pylist()
    elif path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("JSON input must be an array")
        yield from values
    else:
        opener = bz2.open if path.suffix == ".bz2" else gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as source:
            yield from iter_jsonl(source)


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(value):
    parser = PlainText()
    parser.feed(value)
    return "".join(parser.parts)


def required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


class EmptyDocumentError(ValueError):
    """A structurally valid Wikipedia record has no body after normalization."""


_BCP_FRONTMATTER = re.compile(
    r"---\s*title:\s*(?P<title>.+?)\s*(?:author:\s*.+?\s*)?"
    r"(?:date:\s*.+?\s*)?---\s*(?P<body>.*?)(?=---|$)", re.S,
)


def document(row, benchmark, *, aligned_parquet=False):
    docid = str(row.get("docid", row.get("id", "")))
    if not docid:
        raise ValueError("document missing docid/id")
    if benchmark == "browsecomp-plus":
        text = required_text(row.get("text"), "document.text")
        title = str(row.get("title") or docid)
        if aligned_parquet:
            header = _BCP_FRONTMATTER.search(text)
            title = header["title"].strip() if header else f"Doc {docid}"
            text = header["body"].strip() if header else text.strip()
        return dict(id=docid, title=title, text=text,
                    source_url=str(row.get("url") or ""), sentences=[])
    title = required_text(row.get("title"), "Wikipedia title")
    text = row.get("text")
    if not isinstance(text, list) or not text:
        raise ValueError("Hotpot fullwiki text must contain sentence arrays")
    if all(isinstance(item, list) for item in text):
        # Skip a standalone heading in the original processed-wiki format.
        paragraphs = text[1:] if text[0] == [title] else text
        sentences = [strip_html(s) for paragraph in paragraphs for s in paragraph]
    elif all(isinstance(item, str) for item in text):
        sentences = [strip_html(s) for s in (text[1:] if text[0] == title else text)]
    else:
        raise ValueError("invalid Wikipedia sentence array")
    body = "".join(sentences)
    if not body.strip():
        raise EmptyDocumentError("Wikipedia document text must be nonempty text")
    return dict(id=docid, title=title, text=body, sentences=sentences,
                source_url=str(row.get("url") or ""))


def load_official_functions(path, names):
    """Compile only selected pure upstream functions/constants; never import vLLM here."""
    import ast
    import re
    from typing import Dict, List
    tree = ast.parse(Path(path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    namespace = dict(re=re, Dict=Dict, List=List)
    # The official decryption functions require these standard-library modules.
    import base64
    import hashlib
    namespace.update(base64=base64, hashlib=hashlib)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    if not names.issubset(namespace):
        raise ValueError(f"missing upstream functions/constants: {names - namespace.keys()}")
    return namespace


def questions(path, benchmark, official, encrypted=False):
    decode = None
    if benchmark == "browsecomp-plus" and encrypted:
        funcs = load_official_functions(official / "scripts_build_index/decrypt_dataset.py",
                                       {"derive_key", "decrypt_string", "DEFAULT_CANARY"})
        decode = lambda value: funcs["decrypt_string"](value, funcs["DEFAULT_CANARY"])
    seen = set()
    # Use the question shard only for IDs, questions and answers. Qrels and
    # searchable documents come from their own assets.
    columns = ["query_id", "query", "answer"] if benchmark == "browsecomp-plus" else None
    for row in records(path, parquet_columns=columns):
        if benchmark == "hotpotqa":
            qid = str(row.get("_id", row.get("id", "")))
            facts = row["supporting_facts"]
            if isinstance(facts, dict):
                if len(facts["title"]) != len(facts["sent_id"]):
                    raise ValueError("mismatched supporting fact fields")
                facts = [list(pair) for pair in zip(facts["title"], facts["sent_id"])]
            if not isinstance(facts, list) or any(not isinstance(f, list) or len(f) != 2
                    or not isinstance(f[0], str) or type(f[1]) is not int or f[1] < 0 for f in facts):
                raise ValueError("invalid Hotpot supporting facts")
            item = dict(id=qid, query=row["question"], answer=row["answer"], supporting_facts=facts)
        else:
            qid = str(row["query_id"])
            item = dict(id=qid, query=decode(row["query"]) if decode else row["query"],
                        answer=decode(row["answer"]) if decode else row["answer"])
        if not qid or qid in seen:
            raise ValueError(f"missing or duplicate question ID: {qid}")
        seen.add(qid)
        required_text(item["query"], "question")
        required_text(item["answer"], "answer")
        yield item


def read_qrels(path):
    result = {}
    with Path(path).open() as source:
        for line in source:
            if not line.strip():
                continue
            qid, _, docid, relevance = line.split()
            if float(relevance) > 0:
                result.setdefault(qid, set()).add(docid)
    return result


def source_fingerprint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"local source not found: {path}; download or copy it before preparation")
    if path.is_file():
        return {path.name: digest(path)}
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(path).parts and p.suffix in {".parquet", ".json", ".jsonl", ".bz2", ".gz"}}


def prepare(benchmark, root, data, corpus, encrypted=False, corpus_limit=None):
    root = Path(root)
    assets = verify_official(root, benchmark)
    output = root / "prepared"
    if output.exists():
        raise FileExistsError(f"prepared directory exists: {output}; choose a new ROOT")
    # Validate source files before creating outputs; no data download occurs here.
    source_data, source_corpus = source_fingerprint(data), source_fingerprint(corpus)
    downloaded_bcp = (benchmark == "browsecomp-plus" and
                      Path(data).resolve() == (root / "raw/queries").resolve() and
                      Path(corpus).resolve() == (root / "raw/corpus").resolve())
    snapshot_hashes = assets.get("raw_parquet_sha256") if downloaded_bcp else None
    if snapshot_hashes and snapshot_hashes != {"queries": source_data, "corpus": source_corpus}:
        raise ValueError("BrowseComp-Plus Parquet files differ from the downloaded snapshots")
    qrels = {}
    if benchmark == "browsecomp-plus":
        qrels = {kind: read_qrels(root / ("official/topics-qrels/qrel_evidence.txt" if kind == "evidence"
                                       else "official/topics-qrels/qrel_golds.txt"))
                 for kind in ("evidence", "gold")}
    output.mkdir(parents=True)
    qcount = 0
    with (output / "questions.jsonl").open("x", encoding="utf-8") as target:
        for row in questions(data, benchmark, root / "official", encrypted):
            for kind, values in qrels.items():
                expected = values.get(row["id"], set())
                if not expected:
                    raise ValueError(f"question {row['id']} lacks official {kind} qrels")
                # Official qrels are authoritative; don't build a corpus from these positives.
                row[kind + "_docs"] = sorted(expected)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            qcount += 1
    count = skipped = 0
    skipped_path = output / "skipped_documents.jsonl"
    with ((output / "corpus.jsonl").open("x", encoding="utf-8") as target,
          skipped_path.open("x", encoding="utf-8") as skip_log):
        columns = ["docid", "url", "text"] if snapshot_hashes else None
        for position, row in enumerate(records(corpus, parquet_columns=columns), 1):
            if corpus_limit is not None and count >= corpus_limit:
                break
            try:
                value = document(row, benchmark, aligned_parquet=bool(snapshot_hashes))
            except EmptyDocumentError:
                skipped += 1
                skip_log.write(json.dumps(dict(record_index=position,
                    id=str(row.get("docid", row.get("id", ""))), title=row["title"],
                    reason="empty Wikipedia body after removing heading/HTML"), ensure_ascii=False) + "\n")
                skip_log.flush()
                if skipped <= 5:
                    print(f"[prepare] skipping empty Wikipedia document: record={position} "
                          f"id={row.get('id')}; skipped={skipped}", flush=True)
                continue
            target.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"[prepare] documents={count}; skipped_empty={skipped}", flush=True)
    print(f"[prepare] documents={count}; skipped_empty={skipped}; skipped_log={skipped_path}", flush=True)
    if not count or not qcount:
        raise ValueError("questions/corpus must be nonempty")
    canonical = ("wiki_sha256" in assets and str(Path(corpus).resolve()) == str((root / "raw/wiki-abstracts.tar.bz2").resolve())
                 and source_corpus.get("wiki-abstracts.tar.bz2") == assets["wiki_sha256"]
                 and qcount == 7405) if benchmark == "hotpotqa" else (
                 "corpus_revision" in assets and downloaded_bcp and qcount == 830 and count == 100195)
    canonical = canonical and corpus_limit is None and skipped == 0
    if snapshot_hashes and corpus_limit is None and not canonical:
        raise ValueError(f"expected 830 official questions and 100195 documents, got {qcount} and {count}")
    manifest = dict(benchmark=benchmark, canonical_corpus=canonical, split="dev" if benchmark == "hotpotqa" else "test",
                    questions=qcount, documents=count, corpus_limit=corpus_limit,
                    skipped_documents=skipped, skipped_documents_sha256=digest(skipped_path),
                    corpus_scope=("official fixed corpus" if canonical else "custom/subset corpus; not a full benchmark claim"),
                    source_data=source_data, source_corpus=source_corpus,
                    source_data_path=str(Path(data).resolve()), source_corpus_path=str(Path(corpus).resolve()),
                    encrypted=encrypted,
                    assets=assets, questions_sha256=digest(output / "questions.jsonl"),
                    corpus_sha256=digest(output / "corpus.jsonl"))
    dump(output / "manifest.json", manifest)
    return manifest
