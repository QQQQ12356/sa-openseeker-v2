import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from search_comp.experiments import browsecomp_plus_dense, runtime
from search_comp.experiments.browsecomp_plus_dense import INDEX_REPO, SHARDS
from search_comp.experiments.runtime import digest, write_json
from search_comp.offline import qwen_embedding as dense


@pytest.mark.parametrize("prepared_dir", ["", "shared/prepared"])
def test_import_dense_shards_joins_by_docid_and_validates_provenance(tmp_path, monkeypatch, prepared_dir):
    pytest.importorskip("faiss")
    monkeypatch.setattr(browsecomp_plus_dense, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(browsecomp_plus_dense, "configure", lambda environment: None)
    benchmark = tmp_path / "artifacts/benchmarks/browsecomp-plus/v1"
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {})
    write_json(model / "download_manifest.json", dict(repo_id=dense.MODEL_REPO, revision="test"))
    prepared = tmp_path / prepared_dir if prepared_dir else benchmark / "prepared"
    prepared.mkdir(parents=True)
    corpus = prepared / "corpus.jsonl"
    ids = ["a", "b", "c", "d"]
    corpus.write_text("".join(json.dumps(dict(id=docid, text=docid)) + "\n" for docid in ids), encoding="utf-8")
    manifest = dict(benchmark="browsecomp-plus", canonical_corpus=True,
                    corpus_sha256=digest(corpus), documents=len(ids), assets={"index_revision": "test"})
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

    target = benchmark / "vector_indexes/official"
    browsecomp_plus_dense.prepare_browsecomp_plus_dense(
        version="v1", index_profile="official", phase="prepare", source="official",
        model_revision="main", index_revision="test", embedding_model_path=model, official_index_dir=source,
        device="cpu", batch_size=1, doc_max_length=128, environment={}, prepared_dir=prepared_dir,
    )
    metadata = dense.read_vector_manifest(target, manifest, model, verify=True)
    assert metadata["documents"] == 4

    def encode(queries, max_length):
        assert queries == [dense.QUERY_PREFIX + "test"]
        assert max_length == dense.QUERY_MAX_LENGTH
        return vectors[1:2] + 0.25 * vectors[0:1]

    monkeypatch.setattr(dense, "Encoder", lambda *args: SimpleNamespace(encode=encode))
    index = dense.DenseIndex(SimpleNamespace(get=lambda docid: {"docid": docid}), target, model, "cpu", metadata)
    assert index.search("test", 2) == [{"docid": "c"}, {"docid": "d"}]
    assert index.get("a") == {"docid": "a"}
    with pytest.raises(ValueError, match="different provenance"):
        dense.read_vector_manifest(target, dict(manifest, corpus_sha256="other"), model)
    with pytest.raises(ValueError, match="share the pinned index revision"):
        dense.read_vector_manifest(target, dict(manifest, assets={"index_revision": "other"}), model)
    write_json(Path(target) / "docids.json", ids)
    with pytest.raises(ValueError, match="checksum changed"):
        dense.read_vector_manifest(target, manifest, model, verify=True)
