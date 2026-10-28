# 20 Plain 训练与评测

修改 `scripts/sh/20_plain_train.sh` 顶部 Python、GPU 列表和脚本中的 YAML，然后直接执行。默认 Qwen3.5-4B、16k、LoRA、batch=1；`model_family` 支持 qwen3 / qwen35。`ACTION="plan"` 只生成实验目录、配置快照、指纹和启动命令，`train` 才加载模型训练。数据路径与 tokenizer 必须在 manifest 中匹配。

每次运行创建新的 `outputs/experiments/<name>/`，保留 `experiment.json`、`train.yaml`、`source_config.yaml`、运行日志与模型产物。根据 `train_summary.json` 的 `final_model_path` 定位推理模型；LoRA 合并与原训练器行为保持一致。计划后的代码或数据指纹改变会被拒绝，应重新创建实验计划。

多卡直接修改 `CUDA_VISIBLE_DEVICES="0,1,2,3"`，默认随可见卡数推断进程数。支持 DDP、ZeRO-2，以及仅 Plain 支持的 ZeRO-3，见 [50 多卡](50_multigpu.md)。

评测修改 `21_plain_eval.sh` 的 `RUN_DIR` 或 `MODEL_PATH`，以及 `SPLIT`、工具后端和预算后执行。`RUN_DIR` 可自动定位模型和测试数据。`DRY_RUN=1` 只检查并打印命令。`26_plain_memory_probe.sh` 是短步数显存实验入口，见 [51 显存测试](51_memory_probe.md)。

22—25 保留原 Search-R1 训练、交互评测和未训练基线入口；本次没有迁移 Search-R1 数据。

训练后可增加独立 BrowseComp 测试：用 48 读取 RUN_DIR 并部署 Plain，配置 API 后运行 47 的官方模式。
原有测试入口继续保留，准备依赖、完整示例与比较边界见 [42 官方 Benchmark](42_openseeker_benchmark.md)。
