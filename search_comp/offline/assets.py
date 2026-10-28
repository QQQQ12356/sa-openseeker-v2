"""Explicit download phase and pinned official evaluator provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

BCP_REV = "046949032b0328319cc9a02663a759ec601d9402"
HOTPOT_REV = "3635853403a8735609ee997664e1528f4480762a"
BCP_QUERY_REV = "144cff8e35b5eaef7e526346aa60774a9deb941f"
# Corpus revision is resolved once during explicit download and saved in assets.json.
WIKI_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
WIKI_MD5 = "01edf64cd120ecc03a2745352779514c"
DEV_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_file(url, path, md5=None):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        # An interrupted transfer can be retried; the final name is only published on completion.
        print(f"[download] {url} -> {path}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(8 * 1024 * 1024):
                target.write(chunk)
        temporary.rename(path)
    if md5 and digest(path, "md5") != md5:
        raise ValueError(f"official MD5 mismatch: {path}; move this file aside and download again")
    return path


def official_files(benchmark):
    if benchmark == "browsecomp-plus":
        return "texttron/BrowseComp-Plus", BCP_REV, [
            "scripts_evaluation/evaluate_run.py", "scripts_build_index/decrypt_dataset.py",
            "topics-qrels/qrel_evidence.txt", "topics-qrels/qrel_golds.txt", "LICENSE"]
    return "hotpotqa/hotpot", HOTPOT_REV, ["hotpot_evaluate_v1.py", "LICENSE.txt"]


def install_official(root, benchmark):
    """Materialize bundled, unchanged upstream source/qrels with exact SHA256 checks."""
    import shutil
    root = Path(root)
    reference = Path(__file__).parent / "reference"
    provenance = json.loads((reference / "00_provenance.json").read_text())[benchmark]
    repo, revision, names = official_files(benchmark)
    if provenance["revision"] != revision or provenance["repo"] != repo:
        raise ValueError("bundled official revision differs from code")
    record = dict(benchmark=benchmark, official_repo=repo, official_revision=revision,
                  official_sha256=provenance["sha256"])
    previous = root / "assets.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("benchmark") != benchmark or old.get("official_sha256") != record["official_sha256"]:
            raise ValueError("existing assets differ from pinned reference; choose a new ROOT")
        record = dict(old, **record)
    for name in names:
        source, target = reference / benchmark / name, root / "official" / name
        if digest(source) != provenance["sha256"][name]:
            raise ValueError(f"bundled official source changed: {name}")
        if target.exists():
            if digest(target) != provenance["sha256"][name]:
                raise ValueError(f"existing official source changed: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if not previous.exists():
        dump(previous, record)
    return record


def download_assets(benchmark, root, corpus_revision=None, official_only=False, dev_url=DEV_URL, wiki_url=WIKI_URL):
    root = Path(root)
    records = install_official(root, benchmark)
    previous = root / "assets.json"
    if official_only:
        dump(previous, records)
        return records
    if benchmark == "hotpotqa":
        dev = root / "raw/hotpot_dev_fullwiki_v1.json"
        try:
            download_file(dev_url, dev)
        except (OSError, TimeoutError) as error:
            # The historical CMU host can be unavailable. Use the same publisher's
            # fullwiki validation split, never an unofficial corpus or the test split.
            print(f"[download] dev host unavailable ({type(error).__name__}); trying hotpotqa/hotpot_qa fullwiki validation", flush=True)
            from huggingface_hub import HfApi, snapshot_download
            from .data import records as local_records
            revision = records.get("dev_hf_revision") or HfApi().dataset_info("hotpotqa/hotpot_qa").sha
            raw = root / "raw/dev-hf"
            snapshot_download("hotpotqa/hotpot_qa", repo_type="dataset", revision=revision,
                              allow_patterns=["fullwiki/validation-*.parquet"], local_dir=raw)
            dev.parent.mkdir(parents=True, exist_ok=True)
            temporary = dev.with_name(dev.name + ".part")
            with temporary.open("w", encoding="utf-8") as target:
                target.write("[")
                for index, item in enumerate(local_records(raw)):
                    facts = item["supporting_facts"]
                    if isinstance(facts, dict):
                        facts = list(map(list, zip(facts["title"], facts["sent_id"])))
                    if index:
                        target.write(",")
                    json.dump(dict(_id=item["id"], question=item["question"], answer=item["answer"],
                                   supporting_facts=facts), target, ensure_ascii=False)
                target.write("]")
            temporary.rename(dev)
            records["dev_hf_revision"] = revision
            records["dev_source"] = "hotpotqa/hotpot_qa fullwiki validation"

        if records.get("dev_sha256") and digest(dev) != records["dev_sha256"]:
            raise ValueError("existing Hotpot dev data differs from assets.json; use a new ROOT")
        wiki = download_file(wiki_url, root / "raw/wiki-abstracts.tar.bz2", WIKI_MD5)
        records.update(dev_sha256=digest(dev), wiki_sha256=digest(wiki), wiki_md5=WIKI_MD5,
                       dev_url=dev_url, wiki_url=wiki_url)
    else:
        from huggingface_hub import HfApi, snapshot_download
        if corpus_revision and records.get("corpus_revision") not in (None, corpus_revision):
            raise ValueError("corpus revision changed; use a new ROOT")
        corpus_revision = corpus_revision or records.get("corpus_revision") or HfApi().dataset_info("Tevatron/browsecomp-plus-corpus").sha
        for name, rev, folder in (("Tevatron/browsecomp-plus", BCP_QUERY_REV, "queries"),
                                  ("Tevatron/browsecomp-plus-corpus", corpus_revision, "corpus")):
            snapshot_download(name, repo_type="dataset", revision=rev,
                              allow_patterns=["*.parquet", "data/*.parquet"], local_dir=root / "raw" / folder)
        records.update(query_revision=BCP_QUERY_REV, corpus_revision=corpus_revision)
    dump(previous, records)
    return records


def verify_official(root, benchmark):
    root = Path(root)
    record = install_official(root, benchmark)
    _, revision, names = official_files(benchmark)
    if record["benchmark"] != benchmark or record["official_revision"] != revision:
        raise ValueError("wrong benchmark or official revision in assets.json")
    for name in names:
        if digest(root / "official" / name) != record["official_sha256"][name]:
            raise ValueError(f"official source changed: {name}")
    return record


def enforce_offline():
    """No automatic model downloads, telemetry, or IP socket connections in this process."""
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline")

    def audit(event, arguments):
        if event == "socket.getaddrinfo" or (event == "socket.connect" and arguments[0].family in (socket.AF_INET, socket.AF_INET6)):
            raise RuntimeError("offline evaluation forbids network sockets; use ACTION=download beforehand")
    sys.addaudithook(audit)
