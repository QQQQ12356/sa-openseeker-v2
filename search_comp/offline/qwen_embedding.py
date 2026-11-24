"""Qwen3-Embedding-8B query encoding and exact vector search over BrowseComp-Plus."""
import json
from pathlib import Path

from .assets import digest


MODEL_REPO = "Qwen/Qwen3-Embedding-8B"
BACKEND = "qwen3-embedding-8b"
VERSION = "browsecomp-plus-qwen3-dense-v1"
DIMENSION = 4096
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QUERY_MAX_LENGTH = 8192


def model_identity(directory):
    directory = Path(directory)
    metadata = directory / "download_manifest.json"
    if not (directory / "config.json").is_file():
        raise ValueError("EMBEDDING_MODEL_PATH must point to a downloaded local embedding model")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != MODEL_REPO or not manifest.get("revision"):
        raise ValueError(f"embedding model must be a versioned {MODEL_REPO} snapshot")
    return dict(repo_id=MODEL_REPO, revision=manifest["revision"], manifest_sha256=digest(metadata))


class Encoder:
    def __init__(self, directory, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("embedding device must be cpu or cuda[:ordinal]")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, padding_side="left")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(directory, local_files_only=True, torch_dtype=dtype)
        if self.model.config.hidden_size != DIMENSION:
            raise ValueError("unexpected Qwen3-Embedding-8B hidden size")
        self.model.to(self.device).eval()

    def encode(self, texts, max_length):
        torch = self.torch
        features = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt").to(self.device)
        # With left padding the final position is the last non-padding token of each input.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            last = self.model(**features, use_cache=False).last_hidden_state[:, -1]
            return torch.nn.functional.normalize(last, p=2, dim=-1).float().cpu().numpy()


def read_vector_manifest(directory, corpus, model_dir, verify=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION or manifest.get("benchmark") != "browsecomp-plus"
            or manifest.get("corpus_sha256") != corpus["corpus_sha256"]
            or manifest.get("documents") != corpus.get("documents", corpus.get("count"))
            or manifest.get("dimension") != DIMENSION):
        raise ValueError("vector index and SQLite corpus have different provenance or dimensions")
    if manifest["model"] != model_identity(model_dir):
        raise ValueError("query encoder snapshot does not match the vector index manifest")
    expected_revision = corpus.get("assets", {}).get("index_revision")
    if manifest.get("source") == "official" and expected_revision and manifest["upstream"]["revision"] != expected_revision:
        raise ValueError("official vector index and Lucene BM25 must share the pinned index revision")
    if manifest.get("query_prefix") != QUERY_PREFIX or manifest.get("query_max_length") != QUERY_MAX_LENGTH:
        raise ValueError("query encoding configuration differs from the vector index manifest")
    if verify:
        for filename in ("vectors.npy", "docids.json"):
            if digest(directory / filename) != manifest["files"][filename]:
                raise ValueError(f"vector index checksum changed: {filename}")
    return manifest


class DenseIndex:
    """Replace only search ranking. SQLite still owns documents and visit."""

    def __init__(self, documents, directory, model_dir, device, manifest):
        import numpy as np
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("dense retrieval requires FAISS; install requirements.browsecomp-plus.txt") from error

        self.documents = documents
        vectors = np.load(Path(directory) / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.docids = json.loads((Path(directory) / "docids.json").read_text(encoding="utf-8"))
        if (vectors.shape != (manifest["documents"], DIMENSION) or vectors.dtype != np.float32
                or len(self.docids) != manifest["documents"] or len(set(self.docids)) != len(self.docids)):
            raise ValueError("invalid dense index shape, dtype or document IDs")
        self.index = faiss.IndexFlatIP(DIMENSION)
        self.index.add(vectors)
        self.encoder = Encoder(model_dir, device)

    def search(self, query, topk=5):
        import numpy as np

        if not isinstance(query, str) or not query.strip() or topk < 1:
            raise ValueError("query must be text and topk must be positive")
        vector = self.encoder.encode([QUERY_PREFIX + query], QUERY_MAX_LENGTH)
        _, ranking = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), min(topk, len(self.docids)))
        return [self.get(self.docids[i]) for i in ranking[0]]

    def get(self, docid):
        return self.documents.get(docid)
