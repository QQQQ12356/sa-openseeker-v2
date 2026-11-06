"""One training interface for Plain and Beacon; no shell or temporary spec layers."""
from __future__ import annotations

from copy import deepcopy
from .runtime import ROOT, configure, digest, execute, job, path, read_json, slug, stamp, write_json


def resolve_training(method, dataset_name, dataset_dir, model_path, model_family, training, beacon):
    if method not in {"plain", "beacon"} or dataset_name not in {"openseeker", "research"}:
        raise ValueError("训练矩阵仅支持 plain/beacon × openseeker/research；baseline 不训练")
    if model_family not in {"qwen3", "qwen35"}:
        raise ValueError("model_family 必须为 qwen3 或 qwen35")
    dataset = path(dataset_dir)
    manifest = read_json(dataset / "manifest.json")
    from search_comp.data.openseeker import VERSION
    if manifest.get("protocol") != VERSION:
        raise ValueError("数据编码协议不兼容，请从原始数据重新预处理")
    source = manifest["arguments"].get("source")
    expected = {"research"} if dataset_name == "research" else {"polar", "zephyr", "combined"}
    if source not in expected:
        raise ValueError(f"数据来源 {source!r} 与 dataset_name={dataset_name} 不符")
    if model_path not in manifest["tokenizers"]:
        raise ValueError("数据没有使用该 MODEL_PATH tokenizer 计长；预处理和训练必须使用相同身份")
    cfg = deepcopy(training)
    if cfg.get("per_device_batch_size", 1) != 1 or cfg.get("per_device_eval_batch_size", 1) != 1:
        raise ValueError("轨迹 collator 只支持单卡 batch=1，请用梯度累积扩大有效 batch")
    if not isinstance(cfg.get("max_length"), int) or cfg["max_length"] < manifest["arguments"]["max_length"]:
        raise ValueError("训练 max_length 必须不小于预处理长度上限")
    cfg.update(model_name_or_path=model_path, data_mode="openseeker", beacon=deepcopy(beacon or {}))
    fingerprints = {}
    for split, key in (("train", "train_data_path"), ("validation", "val_data_path"), ("test", "test_data_path")):
        filename = dataset / f"{split}.jsonl"
        if not filename.is_file() or (split == "train" and filename.stat().st_size == 0):
            raise ValueError(f"数据缺失或训练集为空: {filename}")
        fingerprints[key] = digest(filename)
        cfg[key] = str(filename) if split != "validation" or filename.stat().st_size else None
    if method == "beacon":
        from search_comp.models.beacon_config import BeaconConfig, configure_sft_layout
        configure_sft_layout(cfg)
        BeaconConfig.from_dict(cfg["beacon"])
    return cfg, manifest, fingerprints


def train(*, method, dataset_name, dataset_dir, model_path, model_family, training,
          distributed, run_name, environment, beacon=None, plan_only=False):
    configure(environment)
    cfg, manifest, fingerprints = resolve_training(method, dataset_name, dataset_dir, model_path,
                                                 model_family, training, beacon)
    from search_comp.trainer.launch import training_command
    import yaml
    directory = ROOT / "artifacts/experiments" / method / dataset_name / stamp(slug(run_name))
    cfg.update(output_dir=str(directory), exp_name="model")
    settings = dict(method=method, dataset_name=dataset_name, kind=method, model_family=model_family,
                    dataset=str(path(dataset_dir)), training_config=cfg, distributed=distributed,
                    dataset_manifest=manifest, data_fingerprints=fingerprints, plan_only=plan_only)
    # Validate the process topology before reserving the run.
    command = training_command(f"search_comp.trainer.{method if method == 'beacon' else 'plain_sft'}_trainer",
                               ["--config", str(directory / "train.yaml")],
                               deepspeed_configs=distributed, configured_deepspeed=cfg.get("deepspeed"))
    with job(directory, "plan" if plan_only else "train", settings):
        write_json(directory / "experiment.json", settings)
        (directory / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        write_json(directory / "launch.json", dict(command=command))
        if plan_only:
            print("配置校验完成；未加载模型、未启动训练。正式运行请设置 PLAN_ONLY=False，创建新实验。")
        else:
            execute(command)
    return directory
