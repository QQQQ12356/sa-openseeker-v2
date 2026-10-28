"""Disk-backed SQLite FTS5 BM25 index; no pickles, Java or in-memory corpus copy."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote

from .assets import digest, dump

VERSION = "sqlite-fts5-bm25-v1"


def build_index(root, title_weight=1.0):
    root = Path(root)
    prepared = root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
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
            CREATE VIRTUAL TABLE corpus_fts USING fts5(title, text, content='documents',
                content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        with corpus.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                cursor = connection.execute("INSERT INTO documents(docid,title,text,source_url,sentences) VALUES(?,?,?,?,?)",
                    (row["id"], row["title"], row["text"], row["source_url"], json.dumps(row["sentences"], ensure_ascii=False)))
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
        connection.execute("INSERT INTO corpus_fts(corpus_fts) VALUES('optimize')")
        record = dict(version=VERSION, benchmark=manifest["benchmark"], count=count,
                      corpus_sha256=manifest["corpus_sha256"], sqlite_version=sqlite3.sqlite_version,
                      title_weight=title_weight, text_weight=1.0, k1=1.2, b=0.75,
                      missing_relevant_documents=len(missing_relevant),
                      tokenizer="unicode61 remove_diacritics 2", seconds=time.monotonic() - start)
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
            if self.manifest["version"] != VERSION:
                raise ValueError("unsupported index version")
            for name in ("version", "benchmark", "count", "corpus_sha256", "title_weight"):
                if self.manifest[name] != sidecar[name]:
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
