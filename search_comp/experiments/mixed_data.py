"""Mix the selected splits of two immutable prepared dataset versions."""
from collections import Counter
from contextlib import ExitStack
import json
from math import gcd
import random

from search_comp.data.openseeker import VERSION, question_key
from search_comp.data.browsecomp_sft import SOURCE
from .runtime import ROOT, digest, job, path, read_json, slug, stamp, write_json


SPLITS = ("test", "validation", "train")
PARENT_SOURCES = {"openseeker": {"polar", "zephyr", "combined"},
                  "research": {"research", SOURCE}, SOURCE: {SOURCE}}


def read_rows(stream):
    offset = stream.tell()
    while line := stream.readline():
        if line.strip():
            yield offset, json.loads(line)
        offset = stream.tell()


def mix_datasets(*, openseeker_prepared, research_prepared, profile, tokenizer,
                 max_length, ratio, seed, browsecomp_sft_prepared=None):
    parents = {"openseeker": path(openseeker_prepared), "research": path(research_prepared)}
    if browsecomp_sft_prepared:
        parents[SOURCE] = path(browsecomp_sft_prepared)
    if len(set(parents.values())) != len(parents):
        raise ValueError("父版本目录必须互不相同")
    parts = ratio.split(":")
    if len(parts) != len(parents) or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError(f"RATIO 须为 {len(parents)} 个以冒号分隔的正整数")
    weights = tuple(map(int, parts))
    if min(weights) < 1 or max_length < 1:
        raise ValueError("RATIO 两项和 MAX_LENGTH 必须为正数")
    divisor = gcd(*weights)
    weights = tuple(weight // divisor for weight in weights)
    output = ROOT / "artifacts/datasets/mixed/prepared" / slug(profile)
    settings = dict(source="mixed", tokenizer=tokenizer, max_length=max_length, ratio=ratio, seed=seed,
                    ratio_order=list(parents))
    with job(output.parent.parent / "jobs" / stamp("mix"), "mix_datasets",
             dict(settings, parents=parents, output=output)):
        if output.exists():
            raise FileExistsError(f"数据版本已存在；修改 PROFILE: {output}")
        provenance, tokenizers = {}, {}
        for name, directory in parents.items():
            manifest_path = directory / "manifest.json"
            manifest = read_json(manifest_path)
            valid_sources = {"research"} if name == "research" and SOURCE in parents else PARENT_SOURCES[name]
            if manifest.get("protocol") != VERSION or manifest["arguments"]["source"] not in valid_sources:
                raise ValueError(f"父版本来源或轨迹协议不匹配: {directory}")
            metadata = manifest["tokenizers"].get(tokenizer)
            if not metadata or not metadata.get("backend_sha256"):
                raise ValueError(f"父版本缺少指定 tokenizer 或其指纹: {directory}")
            tokenizers[name] = metadata
            provenance[name] = dict(path=str(directory), manifest_sha256=digest(manifest_path),
                                    splits={split: digest(directory / f"{split}.jsonl") for split in SPLITS})
        if any(metadata != tokenizers["openseeker"] for metadata in tokenizers.values()):
            raise ValueError("父版本的 tokenizer 计长身份不一致")

        rows = {split: {name: [] for name in parents} for split in SPLITS}
        assignments, excluded = {}, Counter()
        # Highest priority holds across sources, even if the held-out row is over length.
        for split in SPLITS:
            for name, directory in parents.items():
                with (directory / f"{split}.jsonl").open("rb") as stream:
                    for offset, row in read_rows(stream):
                        key = question_key(row["question"])
                        if assignments.setdefault(key, split) != split:
                            excluded["cross_split"] += 1
                            continue
                        length = row["lengths"][tokenizer]["total"]
                        if type(length) is not int or length < 1:
                            raise ValueError(f"无效的轨迹计长: {directory}, id={row['id']}")
                        if length > max_length:
                            excluded["over_length"] += 1
                            continue
                        rows[split][name].append((offset, length))

        train = rows["train"]
        available = {name: len(pool) for name, pool in train.items()}
        batches = min(available[name] // weight for name, weight in zip(parents, weights))
        if not batches:
            raise ValueError(f"可用训练样本不足以满足比例 {ratio}: {available}")
        rng = random.Random(seed)
        for name, weight in zip(parents, weights):
            train[name] = rng.sample(train[name], batches * weight)

        output.mkdir(parents=True)
        counts = {}
        for split, pools in rows.items():
            counts[split] = {name: dict(samples=len(pool), tokens=sum(length for _, length in pool))
                             for name, pool in pools.items()}
            combined = [(name, offset) for name, pool in pools.items() for offset, _ in pool]
            if split == "train":
                rng.shuffle(combined)
            with ExitStack() as stack:
                inputs = {name: stack.enter_context((directory / f"{split}.jsonl").open("rb"))
                          for name, directory in parents.items()}
                target = stack.enter_context((output / f"{split}.jsonl").open("wb"))
                for name, offset in combined:
                    stream = inputs[name]
                    stream.seek(offset)
                    target.write(stream.readline().rstrip(b"\r\n") + b"\n")
        actual_ratio = ":".join(str(len(train[name])) for name in parents)
        write_json(output / "manifest.json", dict(
            protocol=VERSION, arguments=settings, tokenizers={tokenizer: tokenizers["openseeker"]},
            parents=provenance, counts=counts, available_train=available, excluded=dict(excluded),
            actual_train_ratio=actual_ratio))
        print(f"DATASET_DIR = {str(output)!r}; train {':'.join(parents)} = {actual_ratio}")
