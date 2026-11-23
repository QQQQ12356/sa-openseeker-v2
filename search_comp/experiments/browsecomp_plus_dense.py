"""Download official assets or encode the prepared corpus into an exact dense index."""
from itertools import islice
import pickle
from time import monotonic

from search_comp.offline.data import records
from search_comp.offline.qwen_embedding import (
    BACKEND, DIMENSION, MODEL_REPO, QUERY_MAX_LENGTH, QUERY_PREFIX, VERSION, Encoder, model_identity,
)
from .data import download_model
from .runtime import ROOT, configure, digest, job, path, read_json, slug, stamp, write_json


INDEX_REPO = "Tevatron/browsecomp-plus-indexes"
SHARDS = tuple(f"{BACKEND}/corpus.shard{i}_of_4.pkl" for i in range(1, 5))


class VectorUnpickler(pickle.Unpickler):
    """Accept the NumPy array primitives used by the official vector shards."""

    def find_class(self, module, name):
        import numpy as np

        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        allowed = {"multiarray": "_reconstruct", "numeric": "_frombuffer"}
        for submodule, member in allowed.items():
            if module in {f"numpy.core.{submodule}", f"numpy._core.{submodule}"} and name == member:
                return getattr(getattr(np._core, submodule), member)
        raise pickle.UnpicklingError(f"unexpected vector shard global: {module}.{name}")


def download_vectors(root, revision, endpoint):
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi(endpoint=endpoint).dataset_info(INDEX_REPO, revision=revision).sha
    directory = root / "raw" / BACKEND / commit
    snapshot_download(INDEX_REPO, repo_type="dataset", revision=commit, endpoint=endpoint,
                      allow_patterns=list(SHARDS), local_dir=directory)
    write_json(directory / "download_manifest.json", dict(repo=INDEX_REPO, revision=commit,
               files={name: digest(directory / name) for name in SHARDS}))
    print(f"SA_OFFICIAL_INDEX_DIR = {str(directory)!r}", flush=True)


def official_blocks(directory, provenance):
    for name in SHARDS:
        filename = directory / name
        if digest(filename) != provenance["files"][name]:
            raise ValueError(f"downloaded vector shard checksum changed: {filename}")
        with filename.open("rb") as stream:
            vectors, ids = VectorUnpickler(stream).load()
        yield vectors, [str(docid) for docid in ids]


def encode_blocks(corpus, encoder, batch_size, max_length):
    stream = records(corpus)
    while batch := list(islice(stream, batch_size)):
        yield encoder.encode([row["text"] for row in batch], max_length), [row["id"] for row in batch]


def prepare_vectors(root, directory, source, model_dir, official_index_dir, device, batch_size, doc_max_length,
                    *, prepared_dir=None):
    import numpy as np

    prepared = prepared_dir if prepared_dir is not None else root / "prepared"
    corpus = prepared / "corpus.jsonl"
    manifest = read_json(prepared / "manifest.json")
    if manifest["benchmark"] != "browsecomp-plus" or digest(corpus) != manifest["corpus_sha256"]:
        raise ValueError("prepared BrowseComp-Plus corpus checksum or identity changed")
    model = model_identity(model_dir)
    expected_ids = {row["id"] for row in records(corpus)}
    count = manifest["documents"]
    if len(expected_ids) != count:
        raise ValueError("prepared corpus document IDs must be unique")
    provenance = None
    if source == "official":
        if not manifest["canonical_corpus"]:
            raise ValueError("official vectors require the canonical, complete BrowseComp-Plus corpus; use encode for custom corpora")
        provenance = read_json(official_index_dir / "download_manifest.json")
        if provenance.get("repo") != INDEX_REPO or not provenance.get("revision"):
            raise ValueError("OFFICIAL_INDEX_DIR must be a downloaded official vector snapshot")
        blocks = official_blocks(official_index_dir, provenance)
    else:
        print(f"[dense index] loading encoder on {device}; documents={count}", flush=True)
        blocks = encode_blocks(corpus, Encoder(model_dir, device), batch_size, doc_max_length)

    directory.mkdir(parents=True, exist_ok=False)
    vectors_file = directory / "vectors.npy"
    ids = []
    vectors = np.lib.format.open_memmap(vectors_file, mode="w+", dtype=np.float32, shape=(count, DIMENSION))
    try:
        if source == "encode":
            print(f"[dense index] encoding documents=0/{count}; batch_size={batch_size}", flush=True)
        next_report = monotonic()
        for matrix, block_ids in blocks:
            matrix = np.asarray(matrix, dtype=np.float32)
            end = len(ids) + len(block_ids)
            if matrix.shape != (len(block_ids), DIMENSION) or end > count or not np.isfinite(matrix).all():
                raise ValueError("invalid vector shard shape or values")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=0.01, rtol=0):
                raise ValueError("expected unit-normalized document embeddings")
            vectors[len(ids):end] = matrix
            ids.extend(block_ids)
            now = monotonic()
            if source == "official" or now >= next_report or end == count:
                print(f"[dense index] documents={end}/{count}", flush=True)
                next_report = now + 30
        vectors.flush()
    finally:
        del vectors
    if len(ids) != count or set(ids) != expected_ids:
        raise ValueError("vector document IDs do not match the prepared corpus exactly")
    write_json(directory / "docids.json", ids)
    write_json(directory / "manifest.json", dict(version=VERSION, benchmark="browsecomp-plus",
               source=source, upstream=provenance, model=model, documents=count, dimension=DIMENSION,
               corpus_sha256=manifest["corpus_sha256"], corpus_revision=manifest["assets"].get("corpus_revision"),
               similarity="inner_product", pooling="last_token", normalized=True,
               query_prefix=QUERY_PREFIX, query_max_length=QUERY_MAX_LENGTH,
               doc_max_length=doc_max_length if source == "encode" else None,
               files={name: digest(directory / name) for name in ("vectors.npy", "docids.json")}))
    print(f"SA_GENERATION__VECTOR_INDEX_DIR = {str(directory)!r}", flush=True)


def prepare_browsecomp_plus_dense(*, version, index_profile, phase, source, model_revision, index_revision,
                                 embedding_model_path, official_index_dir, device, batch_size, doc_max_length,
                                 environment, prepared_dir=None):
    configure(environment)
    if phase not in {"download", "prepare"} or source not in {"official", "encode"}:
        raise ValueError("PHASE 须为 download/prepare，SOURCE 须为 official/encode")
    root = ROOT / "artifacts/benchmarks/browsecomp-plus" / slug(version)
    output = root / "vector_indexes" / slug(index_profile)
    if phase == "download":
        download_model(repo_id=MODEL_REPO, revision=model_revision, model_name=BACKEND, environment=environment)
        if source == "official":
            with job(root / "jobs" / stamp("download_dense"), "download_dense_vectors", dict(revision=index_revision)):
                download_vectors(root, index_revision, environment["HF_ENDPOINT"])
        return
    if not embedding_model_path or (source == "official" and not official_index_dir):
        raise ValueError("请填写 EMBEDDING_MODEL_PATH，official 模式还须填写 OFFICIAL_INDEX_DIR")
    if batch_size < 1 or not 1 <= doc_max_length <= 32768:
        raise ValueError("BATCH_SIZE 须为正数，DOC_MAX_LENGTH 须在 1 至 32768")
    prepared_dir = path(prepared_dir) if prepared_dir else root / "prepared"
    with job(root / "jobs" / stamp("prepare_dense"), "prepare_dense_vectors",
             dict(source=source, output=output, prepared_dir=prepared_dir, embedding_model_path=embedding_model_path,
                  official_index_dir=official_index_dir, device=device, batch_size=batch_size,
                  doc_max_length=doc_max_length)):
        if output.exists():
            raise FileExistsError(f"向量索引版本已存在；请修改 INDEX_PROFILE: {output}")
        prepare_vectors(root, output, source, path(embedding_model_path),
                        path(official_index_dir) if official_index_dir else None,
                        device, batch_size, doc_max_length, prepared_dir=prepared_dir)
