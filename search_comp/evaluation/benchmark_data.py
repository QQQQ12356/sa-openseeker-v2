"""BrowseComp import for the OpenSeeker query/answer JSONL contract."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

BROWSECOMP_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
BROWSECOMP_SOURCE = "https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decrypt(value, canary):
    """The public BrowseComp SHA256-key/XOR encoding (simple-evals)."""
    encrypted = base64.b64decode(value, validate=True)
    key = hashlib.sha256(canary.encode()).digest()
    return bytes(value ^ key[index % len(key)] for index, value in enumerate(encrypted)).decode()


def read_questions(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get("query", row.get("question"))
            answer = row.get("answer")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"line {line_number}: query/question must be nonempty text")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"line {line_number}: answer must be nonempty text")
            if query in seen:
                raise ValueError(f"line {line_number}: duplicate query")
            seen.add(query)
            # Never carry reference trajectories, evidence or answers into the query.
            rows.append(dict(id=row.get("id", line_number - 1), query=query, answer=answer))
    if not rows:
        raise ValueError("benchmark is empty")
    return rows


def prepare_browsecomp(output, source=None):
    output = Path(output)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output}")
    if source:
        payload = Path(source).read_bytes()
    else:
        with urllib.request.urlopen(BROWSECOMP_URL, timeout=120) as response:
            payload = response.read()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not {"problem", "answer", "canary"}.issubset(reader.fieldnames or []):
        raise ValueError("expected encrypted BrowseComp CSV columns: problem, answer, canary")
    rows = [dict(id=index, query=decrypt(row["problem"], row["canary"]),
                 answer=decrypt(row["answer"], row["canary"])) for index, row in enumerate(reader)]
    if not rows or any(not row["query"].strip() or not row["answer"].strip() for row in rows):
        raise ValueError("empty BrowseComp question/answer")
    if len({row["query"] for row in rows}) != len(rows):
        raise ValueError("duplicate BrowseComp questions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.write_text(json.dumps(dict(
        benchmark="BrowseComp", count=len(rows), source=str(source) if source else BROWSECOMP_URL,
        implementation=BROWSECOMP_SOURCE, source_sha256=hashlib.sha256(payload).hexdigest(),
        data_sha256=sha256(output), order="source CSV order; no sampling", split="test",
    ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows
