# 32 Beacon 方法实验说明：压缩、训练与模型恢复

完整实验按训练数据选择 [70 OpenSeeker](../data/70_openseeker_experiment_guide.md) 或 [71 Research](../data/71_research_experiment_guide.md)。数据下载、原始/规范化 HTML、全量审计、筛选前后分布、训练和三项独立测试/评分均在两份指南中按顺序给出。本文件补充 Beacon 方法特性。

## 32.1 独立训练与可比条件

| 数据 | 训练入口 |
|---|---|
| OpenSeeker | scripts/sh/beacon/10_train_openseeker.sh |
| Research | scripts/sh/beacon/11_train_research.sh |

只在 prepare 阶段适配来源角色，训练共用当前轨迹协议。与 Plain/baseline 比较时固定 prepared、tokenizer、split、预算和 benchmark 资产。只压缩 observation 正文，system/问题/协议边界不因此改变。具体字段默认值和说明见 [69](../project/69_training_parameters.md)。

## 32.2 布局、压缩与显存

- beacon_layout_policy=configured 采用当前 YAML；whole 强制兼容整段布局。
- beacon_information_mode、beacon_window_enabled、beacon_window/stride、beacon_keep_window 必须协调；不要只改一个开关就假定转换为窗口布局。
- 固定压缩使用 beacon_ratio；随机压缩按 beacon_compression_ratios 采样，推理仍使用固定 ratio。参数和限制见 [31](31_random_compression.md)。
- Beacon 使用 beacon_checkpoint_blocks 重算无状态块，不以 Plain 的 gradient_checkpointing 替代；激活 offload 消耗主机内存。
- DDP/ZeRO-2 不把多卡显存直接合并成一条轨迹的前向显存；Beacon 拒绝 ZeRO-3。显存排查见 [52](52_beacon_oom.md)。

## 32.3 损失门控

### 默认损失实际参与情况

默认 `beacon_ce_loss_weight=1.0`，各 assistant 分段权重为 1。压缩后的推理仍用真实轨迹前缀进行 teacher forcing，不是训练时自由生成。

默认 `beacon_output_distill_weight=0.0`、`beacon_readout_distill_weight=0.0`，不启用教师模型蒸馏。虽然 continuation 权重写为 0.1，当前 `beacon_information_loss=false` 会关闭该信息续写分支；不能只看一个权重就判断它实际参与训练。

启用蒸馏或信息续写属于新实验，先读 30/62/63，明确 teacher 路径、额外模型资源和损失门控。首次跑通无需开启。


## 32.4 审查计划与训练产物

完成 70/71 的数据统计后，使用对应 10/11 的 ACTION=plan。记录实际 RUN_DIR，检查原始/解析配置、数据/代码指纹、模型、长度、压缩比、损失和分布式参数，再用屏幕 next 启动该 plan。直接 ACTION=train 会创建另一个新实验。四卡便捷入口 38 用于 OpenSeeker；Research 在 11 上直接设置 GPU 和分布式变量。

正常训练结束自动合并 LoRA 和 Beacon 状态；核对退出码、events、模型权重和 train_summary.final_model_path。

```text
RUN_DIR/models/model/
├── resolved_config.json
├── trainer_metrics.jsonl
├── checkpoint-<step>/
├── final_adapter/
├── final/
└── train_summary.json
```

显存试验 12+37 是独立训练实验，默认约 64K/5 步，并不比 16K 基线更轻。要检查特定长度，先修改筛选和试验脚本，不能将试验模型当正式结果。说明见 [51](../tools/51_memory_probe.md)。

## 32.5 仅在需要时手动合并模型

正常收尾不需要额外运行 35。若自动合并中断但 adapter 完整，或要评估中间 checkpoint：

```bash
# 传模型产物目录，不是最外层 RUN_DIR；输出使用新目录。
DEVICE_MAP=cpu bash scripts/sh/beacon/35_beacon_merge_adapter.sh \
  "$RUN_DIR/models/model" \
  --output_path "$RUN_DIR/models/model/final_recovered"

# 评估中间训练步时：
# DEVICE_MAP=cpu bash scripts/sh/beacon/35_beacon_merge_adapter.sh \
#   "$RUN_DIR/models/model/checkpoint-200" \
#   --output_path "$RUN_DIR/models/model/merged_step200"
```

35 从 resolved_config.json 读取基座和 Beacon 配置，默认 CPU 合并，需要足够主机内存。缺少该配置时需要显式给 base_model_path/beacon_config_json，详见 30 与脚本 --help。默认不覆盖已有非空输出目录。

只有 adapter_config.json 的目录不能直接交给 当前独立 benchmark 入口 推理。手动合并不会补写因训练收尾失败而缺失的 train_summary.json；此时评测显式指定 `MODEL_PATH`、`KIND=beacon`，在线/固定语料还应指定 `PROMPT_DATA`。


## 32.6 测试、评分和轨迹检查

正式生成使用本目录 40/42/44，评分使用 41/43/45，全部采用 Beacon 原生推理保留压缩行为。Hotpot 与 BCP 独立评分加载本地裁判，BrowseComp 独立评分使用 SCORER；三项均输出 EM、F1、LLM judge。原始训练提示只针对真实工具和输出格式进行必要调整。

每项先检查少量题再全量，生成与评分保持相同 LIMIT/ROOT_DIR；失败题保留分母，裁判未知不报完整成绩。实际命令与参数按 70/71 第 8–10 节以及 [68](../project/68_training_benchmark_guide.md) 执行。查看预测 HTML，检查工具调用、重复、预算和最终答案，再解释检索、答案、引用/支持事实指标。旧 31 replay 仅用于内部动作诊断。
