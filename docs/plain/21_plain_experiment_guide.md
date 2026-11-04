# 21 Plain 方法实验说明：训练配置、模型合并与诊断

从数据下载到最终评分的完整步骤按训练数据选择 [70 OpenSeeker](../data/70_openseeker_experiment_guide.md) 或 [71 Research](../data/71_research_experiment_guide.md)。两条路线都包含原始/规范化 HTML、全量结构审计、筛选前后分布、训练和三个独立 benchmark。本文件只补充 Plain 的方法特性，不再用旧 27 总调度器描述主流程。

## 21.1 入口和训练特性

| 数据 | 独立训练脚本 |
|---|---|
| OpenSeeker | scripts/sh/plain/10_train_openseeker.sh |
| Research | scripts/sh/plain/11_train_research.sh |

两份 YAML 独立，均为完整上下文 SFT，默认 LoRA；system/schema 和监督协议共用原数据编码，Research 仅在 prepare 时适配角色。学习率、epochs/max_train_steps、LoRA、loss_segments、内存和 DeepSpeed 全部在所选文件中修改，逐项见 [69](../project/69_training_parameters.md)。MODEL_PATH、DATASET_DIR、MAX_LENGTH 是显式环境覆盖；不要仅修改 YAML 的模型路径却保留冲突的 MODEL_PATH。

单卡 batch 固定 1，有效 batch 通过 GPU 进程数和梯度累积扩大。Plain 使用 gradient_checkpointing、分块 loss、可选激活 CPU offload；实际容量须在 Linux GPU 环境检查。默认 LoRA 与 baseline 全参数 SFT 的可训练参数量不同，对照时记录策略。内存选项原理见 [20](20_plain.md)。

## 21.2 审查计划与训练

在 70/71 完成数据检查和分布分析后，给所选训练脚本 ACTION=plan，记录其实际 RUN_DIR 和 next。plan 要求 prepared 已存在，但不加载模型。审阅 source_config.yaml、train.yaml、experiment.json，随后执行 next 的 `51_experiment.py train --run-dir`，沿用同样的分布式环境。

直接运行所选 shell 的 ACTION=train 是另一条“一键创建并训练”路径，会新建实验，不能误认为继续之前的 plan。数据或代码改动后重新 plan，不绕过指纹校验。四卡 OpenSeeker 便捷入口 28 指向当前 10；Research 直接在 11 上设置 CUDA_VISIBLE_DEVICES/NPROC_PER_NODE/DISTRIBUTED_BACKEND。

## 21.3 训练成功与模型产物

```text
RUN_DIR/models/model/
├── resolved_config.json
├── trainer_metrics.jsonl
├── checkpoint-<step>/       周期保存，不一定触发
├── final/                   LoRA adapter；全参数时为完整权重
├── final_merged/             LoRA 自动合并的完整模型
└── train_summary.json        final_model_path 为实际推理路径
```

核对进程退出码、events、日志、最终权重和 tokenizer。LoRA 合并前可能已经写 summary，因此不能只看 summary 是否存在。正常合并后 final_model_path 指向 final_merged；独立离线 benchmark 要求完整模型。

## 21.4 仅在需要时手动合并

```bash
RUN_DIR=/absolute/path/to/the_plain_run
# 目标必须是新目录，主机内存足够。
DEVICE_MAP=cpu bash scripts/sh/plain/29_plain_merge_adapter.sh "$RUN_DIR/models/model" \
  --output_path "$RUN_DIR/models/model/final_recovered"
```

脚本从 resolved_config.json 读取基座，缺失时显式 --base_model_path。中间 checkpoint 可将输入改为 checkpoint-200。手动合并不更新旧 summary；测试显式 MODEL_PATH 指向恢复目录，同时保留 RUN_DIR 读取训练提示。合并具体选项见 [22](../baseline/22_qwen35_baseline_and_adapter.md)。

## 21.5 诊断与正式评测

旧 21_plain_eval.sh 的 replay 是动作复现诊断，replay_miss 不等于在线研究能力失败；可选 corpus/http 后端和数据内部留出诊断见 [40](../benchmark/40_evaluation.md)。内部诊断不能代替三项 benchmark。

正式生成使用本目录 40/42/44，评分使用 41/43/45。三个 benchmark 均独立评分并输出 EM、F1、LLM judge；Hotpot 评分也需要本地裁判。新 BrowseComp 测试不自动调用裁判。命令、结果目录、提示词来源和失败分母按 70/71 的第 8–10 节执行，全部变量见 [68](../project/68_training_benchmark_guide.md)。

生成完成后用 41_predictions_html.py 查看实际预测；先确认工具、最终回答和预算状态，再解释指标。全量运行 LIMIT=-1，使用新目录，固定模型间的题目、工具、预算、生成和裁判参数。
