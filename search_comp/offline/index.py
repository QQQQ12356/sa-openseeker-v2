"""Disk-backed document storage, official Lucene retrieval and legacy FTS5."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote

from .assets import BCP_LUCENE_DIR, digest, dump

VERSION = "sqlite-fts5-bm25-v1"
CORPUS_VERSION = "sqlite-corpus-v1"


def build_index(root, title_weight=1.0):
    root = Path(root)
    prepared = root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    documents_only = (manifest["canonical_corpus"] and
                      bool(manifest.get("assets", {}).get("bm25_files")))
    if documents_only:
        lucene_metadata(root, manifest["assets"])
    corpus = prepared / "corpus.jsonl"
    if digest(corpus) != manifest["corpus_sha256"]:
        raise ValueError("prepared corpus checksum changed")
    output = root / "index.sqlite"
    temporary = root / "index.building.sqlite"
    if output.exists() or temporary.exists():
        raise FileExistsError("index already exists/build was interrupted; use a new ROOT or move partial index aside")
    # Exclusive reservation avoids concurrent builds overwriting each other.
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    connection = sqlite3.connect(temporary)
    count = 0
    start = time.monotonic()
    try:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA cache_size=-65536;
            CREATE TABLE documents(rowid INTEGER PRIMARY KEY, docid TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, source_url TEXT, sentences TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        if not documents_only:
            connection.execute("""CREATE VIRTUAL TABLE corpus_fts USING fts5(title, text, content='documents',
                content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')""")
        with corpus.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                cursor = connection.execute("INSERT INTO documents(docid,title,text,source_url,sentences) VALUES(?,?,?,?,?)",
                    (row["id"], row["title"], row["text"], row["source_url"], json.dumps(row["sentences"], ensure_ascii=False)))
                if not documents_only:
                    connection.execute("INSERT INTO corpus_fts(rowid,title,text) VALUES(?,?,?)",
                                       (cursor.lastrowid, row["title"], row["text"]))
                count += 1
                if count % 10000 == 0:
                    connection.commit()
                    print(f"[index] documents={count}; elapsed={time.monotonic() - start:.1f}s", flush=True)
        if count != manifest["documents"]:
            raise ValueError("index document count differs from prepared manifest")
        missing_relevant = set()
        if manifest["benchmark"] == "browsecomp-plus":
            relevant = set()
            with (prepared / "questions.jsonl").open() as source:
                for line in source:
                    question = json.loads(line)
                    relevant.update(question["gold_docs"])
                    relevant.update(question["evidence_docs"])
            for docid in relevant:
                if connection.execute("SELECT 1 FROM documents WHERE docid=?", (docid,)).fetchone() is None:
                    missing_relevant.add(docid)
            if missing_relevant and manifest["canonical_corpus"]:
                raise ValueError("official corpus is missing qrel documents")
        connection.execute("CREATE INDEX document_urls ON documents(source_url)")
        if not documents_only:
            connection.execute("INSERT INTO corpus_fts(corpus_fts) VALUES('optimize')")
        record = dict(version=CORPUS_VERSION if documents_only else VERSION,
                      benchmark=manifest["benchmark"], count=count,
                      corpus_sha256=manifest["corpus_sha256"], sqlite_version=sqlite3.sqlite_version,
                      missing_relevant_documents=len(missing_relevant), seconds=time.monotonic() - start)
        if not documents_only:
            record.update(title_weight=title_weight, text_weight=1.0, k1=1.2, b=0.75,
                          tokenizer="unicode61 remove_diacritics 2")
        connection.execute("INSERT INTO metadata VALUES('manifest',?)", (json.dumps(record),))
        connection.commit()
    finally:
        connection.close()
    temporary.rename(output)
    # A hash permits explicitly checking the complete transferable index artifact.
    record["index_sha256"] = digest(output)
    dump(root / "index.manifest.json", record)
    return record


class LocalIndex:
    def __init__(self, path, verify=False):
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(f"file:{quote(str(self.path))}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        try:
            row = self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if row is None:
                raise ValueError("index build is incomplete")
            self.manifest = json.loads(row[0])
            sidecar = json.loads(self.path.with_name("index.manifest.json").read_text())
            if self.manifest["version"] not in {VERSION, CORPUS_VERSION}:
                raise ValueError("unsupported index version")
            for name in ("version", "benchmark", "count", "corpus_sha256", "title_weight"):
                if self.manifest.get(name) != sidecar.get(name):
                    raise ValueError(f"index manifest differs: {name}")
            if verify and digest(self.path) != sidecar["index_sha256"]:
                raise ValueError("index checksum changed")
            self.manifest["index_sha256"] = sidecar["index_sha256"]
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def search(self, query, topk=5):
        if self.manifest["version"] == CORPUS_VERSION:
            raise ValueError("this document store requires the official Lucene or FAISS index for search")
        if not isinstance(query, str) or not query.strip() or topk < 1:
            raise ValueError("query must be text and topk must be positive")
        # Treat every query term literally; no FTS operators, URL syntax or SQL injection.
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:128]
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        result = self.connection.execute("""
            SELECT d.docid,d.title,d.text,d.source_url,d.sentences,bm25(corpus_fts,?,1.0) AS rank
            FROM corpus_fts JOIN documents AS d ON corpus_fts.rowid=d.rowid
            WHERE corpus_fts MATCH ? ORDER BY rank,d.rowid LIMIT ?
            """, (self.manifest["title_weight"], match, topk))
        return [self._document(row) for row in result]

    @staticmethod
    def _document(row):
        value = dict(row)
        value["sentences"] = json.loads(value["sentences"])
        value["url"] = "local://" + quote(value["docid"], safe="")
        return value

    def get(self, docid):
        result = self.connection.execute("SELECT * FROM documents WHERE docid=?", (str(docid),)).fetchone()
        if result is None:
            raise ValueError(f"document not found: {docid}")
        return self._document(result)


def lucene_metadata(root, assets, verify=False):
    directory = root / BCP_LUCENE_DIR
    files = assets["bm25_files"]
    actual = {file.relative_to(directory).as_posix() for file in directory.rglob("*")
              if file.is_file() and file.name != "write.lock"}
    if actual != files.keys():
        raise ValueError(f"official Lucene BM25 files are incomplete or differ: {directory}")
    if verify and any(digest(directory / name) != checksum for name, checksum in files.items()):
        raise ValueError(f"official Lucene BM25 checksum changed: {directory}")
    return dict(backend="lucene", revision=assets["index_revision"], files=files)


class LuceneIndex:
    """Rank with the official Lucene index and look up bodies by original docid."""

    def __init__(self, documents, directory):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Pyserini can import optional encoders that expect a key at import time.
            os.environ["OPENAI_API_KEY"] = "local-lucene-unused"
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as error:
            raise RuntimeError("official BM25 requires Pyserini and Java 21; see docs/35_browsecomp_plus_dense.md") from error
        finally:
            if api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            elif not api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        self.documents = documents
        self.searcher = LuceneSearcher(str(directory))
        if self.searcher.num_docs != documents.manifest["count"]:
            self.close()
            raise ValueError("Lucene BM25 document count differs from the prepared corpus")

    def search(self, query, topk):
        return [self.get(hit.docid) for hit in self.searcher.search(query, k=topk)]

    def get(self, docid):
        return self.documents.get(docid)

    def close(self):
        self.searcher.close()
