"""Dataset and model preparation APIs called directly by editable scripts."""
from __future__ import annotations

from types import SimpleNamespace
from search_comp.data.browsecomp_sft import REPO, SHARDS, SOURCE
from .runtime import ROOT, configure, digest, job, path, read_json, slug, stamp, write_json


DATASET_SOURCES = {
    "openseeker": {"polar", "zephyr", "combined"},
    "research": {"research"},
    SOURCE: {SOURCE},
}


def download_dataset(*, source, revision, environment):
    configure(environment)
    if source not in {"polar", "zephyr", "research", SOURCE}:
        raise ValueError("source 必须为 polar / zephyr / research / browsecomp-sft")
    from search_comp.workflows.download_openseeker import run
    dataset_name = source if source in {"research", SOURCE} else "openseeker"
    directory = ROOT / "artifacts/datasets" / dataset_name
    with job(directory / "jobs" / stamp("download"), "download_dataset", dict(source=source, revision=revision)):
        run(SimpleNamespace(source=source, revision=revision, endpoint=environment["HF_ENDPOINT"], output=directory / "raw"))


def prepare_dataset(*, dataset_name, source, snapshot_dir, inputs, profile, models, max_length,
                    min_length, correctness, observations, workers, seed, val_fraction,
                    test_fraction, offline, environment, shards=(), canonical_questions=None,
                    protected_questions=(), overlap_threshold=0.30):
    configure(environment)
    if source not in DATASET_SOURCES.get(dataset_name, ()):
        raise ValueError("SOURCE 与 DATASET_NAME 不一致")
    if snapshot_dir and inputs:
        raise ValueError("SNAPSHOT_DIR 和 INPUTS 只能填写一个")
    if snapshot_dir:
        snapshot = read_json(path(snapshot_dir) / "download_manifest.json")
        if source == SOURCE:
            if snapshot["repo"] != REPO or not shards or not set(shards) <= set(SHARDS):
                raise ValueError("请选择正确的 BrowseComp SFT 快照和 SA_SHARDS")
            inputs = [p for p in snapshot["parquet_inputs"] if path(p).name in shards]
        else:
            inputs = snapshot["jsonl_inputs"]
    inputs = [path(p) for p in inputs]
    if not inputs or any(not p.is_file() for p in inputs):
        raise ValueError("请填写下载目录 SNAPSHOT_DIR 或完整的 INPUTS 列表")
    directory = ROOT / "artifacts/datasets" / dataset_name / "prepared" / slug(profile)
    if directory.exists():
        raise FileExistsError(f"数据版本已存在；修改 PROFILE: {directory}")
    options = dict(input=inputs, source=source, output=directory, models=models, offline=offline,
                   workers=workers, max_length=max_length, seed=seed, val_fraction=val_fraction,
                   test_fraction=test_fraction, require_correct=correctness == "Correct", limit=None,
                   min_length=min_length, exact_length=None, correctness=correctness,
                   observations=observations, tokenizer=None)
    if source == SOURCE:
        options.update(canonical_questions=str(path(canonical_questions)) if canonical_questions else None,
                       protected_questions=[str(path(p)) for p in protected_questions],
                       overlap_threshold=overlap_threshold,
                       snapshot_revision=snapshot["revision"] if snapshot_dir else None)
    from search_comp.workflows.prepare_openseeker import run
    with job(directory.parent.parent / "jobs" / stamp("prepare"), "prepare_dataset", options):
        run(SimpleNamespace(**options))


def select_dataset(*, dataset_name, prepared, profile, tokenizer, min_length, max_length,
                   exact_length, correctness, observations, environment):
    configure(environment)
    parent = path(prepared)
    manifest = read_json(parent / "manifest.json")
    if manifest["arguments"]["source"] not in DATASET_SOURCES.get(dataset_name, ()):
        raise ValueError("筛选数据来源与 DATASET_NAME 不一致")
    directory = ROOT / "artifacts/datasets" / dataset_name / "prepared" / slug(profile)
    options = dict(prepared=parent, output=directory, max_length=max_length,
                   require_correct=correctness == "Correct", min_length=min_length,
                   exact_length=exact_length, correctness=correctness, observations=observations,
                   tokenizer=tokenizer, split_mode="grouped")
    from search_comp.workflows.refilter_openseeker import run
    with job(directory.parent.parent / "jobs" / stamp("select"), "select_dataset", options):
        run(SimpleNamespace(**options))


def download_model(*, repo_id, revision, model_name, environment):
    configure(environment)
    from huggingface_hub import HfApi, snapshot_download
    # Resolve main once, then fetch an immutable commit into a versioned local directory.
    commit = HfApi(endpoint=environment["HF_ENDPOINT"]).model_info(repo_id, revision=revision).sha
    directory = ROOT / "artifacts/models" / slug(model_name) / commit
    with job(ROOT / "artifacts/models/jobs" / stamp("download"), "download_model",
             dict(repo_id=repo_id, revision=revision, commit=commit, directory=directory)):
        snapshot_download(repo_id, revision=commit, local_dir=directory, endpoint=environment["HF_ENDPOINT"])
        write_json(directory / "download_manifest.json", dict(repo_id=repo_id, revision=commit,
                   files={str(p.relative_to(directory)): digest(p) for p in directory.rglob("*")
                          if p.is_file() and ".cache" not in p.parts and p.name != "download_manifest.json"}))
        print(f"MODEL_PATH = {str(directory)!r}；预处理和训练请使用相同路径")
