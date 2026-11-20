import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from search_comp.experiments.browsecomp_plus_dense import INDEX_REPO, SHARDS, prepare_vectors
from search_comp.experiments.runtime import digest, write_json
from search_comp.offline import qwen_embedding as dense


def test_import_dense_shards_joins_by_docid_and_validates_provenance(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {})
    write_json(model / "download_manifest.json", dict(repo_id=dense.MODEL_REPO, revision="test"))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    corpus = prepared / "corpus.jsonl"
    ids = ["a", "b", "c", "d"]
    corpus.write_text("".join(json.dumps(dict(id=docid, text=docid)) + "\n" for docid in ids), encoding="utf-8")
    manifest = dict(benchmark="browsecomp-plus", canonical_corpus=True,
                    corpus_sha256=digest(corpus), documents=len(ids), assets={})
    write_json(prepared / "manifest.json", manifest)

    source = tmp_path / "official"
    (source / dense.BACKEND).mkdir(parents=True)
    order = list(reversed(ids))
    vectors = np.eye(len(ids), dense.DIMENSION, dtype=np.float32)
    for i, name in enumerate(SHARDS):
        with (source / name).open("wb") as stream:
            pickle.dump((vectors[i:i + 1], [order[i]]), stream, protocol=4)
    write_json(source / "download_manifest.json", dict(repo=INDEX_REPO, revision="test",
               files={name: digest(source / name) for name in SHARDS}))

    target = tmp_path / "vectors"
    prepare_vectors(tmp_path, target, "official", model, source, "cpu", 1, 128)
    metadata = dense.read_vector_manifest(target, manifest, model, verify=True)
    assert metadata["documents"] == 4

    def encode(queries, max_length):
        assert queries == [dense.QUERY_PREFIX + "test"]
        assert max_length == dense.QUERY_MAX_LENGTH
        return vectors[1:2]

    monkeypatch.setattr(dense, "Encoder", lambda *args: SimpleNamespace(encode=encode))
    index = dense.DenseIndex(SimpleNamespace(get=lambda docid: {"docid": docid}), target, model, "cpu", metadata)
    assert index.search("test", 2) == [{"docid": "c"}, {"docid": "d"}]
    assert index.get("a") == {"docid": "a"}
    with pytest.raises(ValueError, match="different provenance"):
        dense.read_vector_manifest(target, dict(manifest, corpus_sha256="other"), model)
    write_json(Path(target) / "docids.json", ids)
    with pytest.raises(ValueError, match="checksum changed"):
        dense.read_vector_manifest(target, manifest, model, verify=True)
