# 70 OpenSeeker 完整实验指南：下载、检查、统计、训练、测试与评分

本指南按**数据集**组织，覆盖 Plain、Beacon、baseline 三种方法。所有命令在 Linux 项目根目录的同一个 Bash 会话执行；Windows 只改代码和做无模型检查。本次未下载真实数据、未运行真实预处理、训练、推理或裁判。这里给出的是可执行操作顺序，不把历史数据条数当成本次统计结果。

阅读顺序：本指南完成整条路线 → [68](../project/68_training_benchmark_guide.md) 查脚本变量与 benchmark 参数 → [69](../project/69_training_parameters.md) 查完整训练参数。来源适配与格式细节见 [09](09_research_sft.md)（Research）或 [10](10_data.md)（OpenSeeker）。

## 70.1 实验流程与数据边界

```text
环境/模型与来源固定 → 下载原始快照（或登记本地导出）
→ 原始 HTML 抽查 + 全量结构审计 → 完整轨迹分词与预处理
→ 规范化 HTML 抽查 → 筛选前/保留/train/validation/test 整体分布
→ 三种方法分别 plan 与训练 → 检查最终模型
→ 三项 benchmark 资产准备 → 每项独立测试 → 每项独立评分
→ 预测 HTML 检查 → 全量实验与归档
```

训练数据、训练内部留出集和三项独立 benchmark 是不同对象。原始导出保留，所有过滤和规范化写入新目录。HTML 是人工抽查材料，整体分布必须来自全量遍历。比较三种方法时固定 prepared 数据、tokenizer、split 和 benchmark 设置。

## 70.2 环境与路径先统一

```bash
# 先按 01 环境文档准备已有 Linux 训练环境及模型，再执行下面设置。
conda activate search-comp-qwen35-cu126
export PYTHON="$(command -v python)"
export CONDA_ENV=search-comp-qwen35-cu126
export MODEL_PATH=/models/Qwen3.5-4B
export MAX_LENGTH=16384
export PREPARED=outputs/data/openseeker/prepared_16k_v1
export REPORT_DIR=outputs/reports/openseeker/experiment_v1
```

MODEL_PATH 必须是真实 Linux 模型/tokenizer 目录，预处理与训练使用**完全相同身份字符串**。huggingface_hub 用于下载，Parquet 导出需要 pyarrow，分词需要 transformers/tokenizers，统计需要 numpy，绘图需要 matplotlib；训练另需 torch/peft/DeepSpeed 等，版本和安装见 [01](../common/01_environment.md)。这些命令不负责安装环境。仅 tokenizer 预处理不加载模型权重，HTML 只需标准库；后续训练仍需完整模型。

## 70.3 下载并固定来源

默认来源为 PolarSeeker/OpenSeeker-v1-Data；Zephyr 是同一 OpenSeeker 路线的另一来源，不要与 Research 混为同一训练身份。来源文件见 [Polar 数据仓库](https://huggingface.co/datasets/PolarSeeker/OpenSeeker-v1-Data/tree/main)。

```bash
SOURCE=polar REVISION=main OUTPUT=outputs/data/raw \
  bash scripts/sh/data/16_download.sh
# 正式复现实验把 REVISION=main 换成已记录的实际 commit。
```

下载器先把 branch/tag 解析为实际 commit，再按 commit 下载，实际目录为 `outputs/data/raw/polar/<commit>/`，最后打印该路径。原始 JSONL/Parquet 保留；Parquet 的 JSONL 导出位于 exports 子目录，原源文件不覆盖。download_manifest.json 记录 repo、实际 revision、endpoint、原文件 SHA256/字节数及 `jsonl_inputs` 清单。认证可在服务器使用 HF_TOKEN 或 `hf auth login`；访问失败时先检查权限和网络，不把错误页保存成数据。

```bash
# 填下载器刚刚打印的明确目录，不自动选择“最新目录”。
export SNAPSHOT_DIR=/absolute/path/to/outputs/data/raw/polar/ACTUAL_COMMIT
INPUT_LIST=$("$PYTHON" - "$SNAPSHOT_DIR" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]) / 'download_manifest.json').read_text(encoding='utf-8'))
print('\n'.join(manifest['jsonl_inputs']))
PY
)
mapfile -t RAW_FILES <<< "$INPUT_LIST"
printf '%s\n' "${RAW_FILES[@]}"
```

确认清单中每个文件属于计划使用的完整数据来源，涵盖所有 shard；同一数据若有多种不同命名的镜像/导出，不重复喂入。只有同名 JSONL/Parquet 的情况下载器会优先 JSONL。main 是便捷首次取数方式，不是固定版本；复现实验使用 manifest 内的 commit。

**已有本地 JSONL 时：** 可以跳过下载，显式 `RAW_FILES=(/data/full_polar.jsonl)`，多个输入列在同一数组中。先记录来源 repo/revision（未知则注明未知）、导出方式与文件 SHA256，再继续。只使用一个小样例不能代表完整 OpenSeeker 数据。

## 70.4 原始 HTML 抽查与全量结构审计

```bash
# 每个原始文件抽查前 20 条；不是全量分布统计。换 OFFSET 可检查其他位置。
for i in "${!RAW_FILES[@]}"; do
  LIMIT=20 OFFSET=0 bash scripts/sh/report/40_dataset_html.sh \
    "${RAW_FILES[$i]}" "$REPORT_DIR/raw_${i}.html"
done

# 全量结构审计，不加载 tokenizer 或模型；覆盖所列全部输入。
"$PYTHON" scripts/py/18_audit.py --source openseeker --input "${RAW_FILES[@]}" \
  --output "$REPORT_DIR/raw_audit.json"
```

人工检查角色顺序、system/schema、问题、推理、工具调用、observation、最终回答；对长轨迹和不同文件分批抽查。raw_audit.json 的 roles 为原始角色，tools 为解析出的工具调用，字符分布不是 token 分布。normalization_failures 必须可解释；未知工具或协议失败不能通过放宽验证直接混入训练。

## 70.5 完整轨迹预处理与规范化 HTML

```bash
# 使用全部输入；单个 Polar JSONL 也可以用 data/10_prepare.sh 的 INPUT_FILE 等变量。
INPUT_ARGS=()
for file in "${RAW_FILES[@]}"; do INPUT_ARGS+=(--input "$file"); done
"$PYTHON" scripts/py/10_prepare.py --source polar "${INPUT_ARGS[@]}" \
  --output "$PREPARED" --models "$MODEL_PATH" --offline --workers 4 \
  --min-length 1 --max-length "$MAX_LENGTH" --correctness Correct --observations any \
  --val-fraction 0.02 --test-fraction 0.02 --seed 42
```

切换 Zephyr 时 source=zephyr；组合来源需明确 combined 与所有输入。默认选 Correct，仅保留有对应真实正确性标签的记录；不要把 Research 的示范答案当作同等 Correct 标签。OBSERVATIONS=any/with/without 分别不限制/保留有观测/保留无观测。多文件是否保留原有 split 取决于文件名和预处理规则，不能假定任意 shard 文件名都会自动识别官方验证集。

完整序列用训练共用 encode 计长，包含 system/schema、ChatML、问题、推理/调用、完整工具返回与答案；不是正文字符数，也不是 Beacon 压缩后的物理长度。超长整条过滤，不静默截断。输出目录必须为新目录，不能覆盖旧实验数据。按 question_id 分组去重和划分，默认约 96%/2%/2%，实际数量以 manifest 为准；极小样例可能没有训练行。

```bash
# 检查转换后的训练样例；同时抽查 validation/test（有数据时）。
LIMIT=30 SPLIT=train bash scripts/sh/report/40_dataset_html.sh \
  "$PREPARED" "$REPORT_DIR/prepared_train.html"
# 只查 JSON 是否可解析，不替代下面的总体分析。
"$PYTHON" -m json.tool "$PREPARED/manifest.json" > "$REPORT_DIR/manifest_readable.json"
```

| 产物 | 用途 |
|---|---|
| normalized.jsonl | 规范化成功的筛选前总体，包含被超长/质量/重复等条件拒绝的记录 |
| measurements.jsonl | 每条规范化成功记录的各 tokenizer token 长度、工具数与 rejection |
| train / validation / test.jsonl | 实际保留的完整轨迹；train 用于更新参数，留出集用于训练内检查 |
| rejected.jsonl | invalid 或有效但被拒绝的记录及原因；不能把这些行当训练数据 |
| manifest.json | 输入指纹、tokenizer、参数、计数与规范化成功总体分布 |

HTML 中确认 system 和工具 schema 未丢失，reasoning 与动作合并正确，工具返回与回答边界正常。系统/问题/observation 不参与监督，只有 assistant 输出参与 loss；Beacon 仅压缩 observation 正文。若转换错误，修适配并在新目录重新 prepare，不人工改训练行绕过指纹。

## 70.6 筛选前总体、保留数据与各 split 的整体分布

这一步必须覆盖全部选定原始文件，不能用 HTML 的前 20 条替代。预处理未设置 `--limit` 时会遍历所有输入；`normalized.jsonl` 和 `measurements.jsonl` 覆盖所有规范化成功记录，包括随后被筛除的样本。invalid 原始行单独计数，不能伪造其 token 长度。

```bash
# MODEL 是 manifest 中的 tokenizer 键，与前面的模型路径完全相同。
PREPARED="$PREPARED" MODEL="$MODEL_PATH" OUTPUT="$REPORT_DIR/distribution" \
  bash scripts/sh/report/45_report.sh

# 画原 prepare 的筛选前总体及通过筛选的数据长度图。
MEASUREMENTS="$PREPARED/measurements.jsonl" MODEL="$MODEL_PATH" POPULATION=all \
  OUTPUT="$REPORT_DIR/length_before.svg" bash scripts/sh/report/44_plot.sh
MEASUREMENTS="$PREPARED/measurements.jsonl" MODEL="$MODEL_PATH" POPULATION=retained \
  OUTPUT="$REPORT_DIR/length_retained.svg" bash scripts/sh/report/44_plot.sh
```

| 检查对象 | 文件/字段 | 解读与完成条件 |
|---|---|---|
| 全部原始行 | manifest.counts.input、invalid；raw_audit.json | 确认没有只处理一个 shard 或启用 limit；无效原因可追溯 |
| 规范化成功的筛选前总体 | distribution/quality_length.json → populations.before_filter | 全长、observation、assistant token、调用数和消息数的 n/min/max/mean/p50/p90/p95/p99，以及超 8K/16K/32K/64K 数 |
| 实际保留总体 | populations.retained | 直接读取 train+validation+test，不把筛选前分布误当成实际训练分布 |
| 实际训练及留出集 | populations.train / validation / test | 分别查看条数、长度、质量标签和工具构成，train 必须非空 |
| 筛除原因 | rejections、input_counts | 查看超长、重复、观测筛选与 invalid；核对留存率，不靠 HTML 猜测 |
| 质量/语言分组 | groups | 筛选前按质量/长度分组的 count、cjk、tools、lengths；cjk 是启发式字符检测，不是语种分类器 |
| 长度图 | length_before.svg / length_retained.svg | 正文与外层协议均计长；图例注明所选总体和样本数 |

同一条记录可能先因质量或长度被拒绝，只统计实际记录的首个 rejection，不能把原因计数解释成互不影响的独立筛选实验。`POPULATION=retained` 指原 prepare 的 rejection 为空；若之后用 11_select 重新筛选，重新对最终目录运行 45，以其实际 split 为准。不要用旧目录的 measurements 图冒充新训练集分布。多个 tokenizer 时必须选定一个 MODEL，不把它们的最大长度混成同一分布。

**完成条件：** 数据来源和行数可解释；确认过滤前后各分位数、超长率、保留率、工具构成；train 非空、训练长度上限足够；归档报告后再锁定正式数据目录。若改变模型 tokenizer、编码规则或原始正文，重新 prepare；更改筛选条件应保存到新目录。

## 70.7 三种方法分别配置与训练

三个方法使用同一份 PREPARED，才能比较方法差异。以下选择需要的入口逐一运行，不经过统一 METHOD 调度器。

| 方法 | 独立训练入口 | 关键差异 |
|---|---|---|
| Plain | scripts/sh/plain/10_train_openseeker.sh | 默认 LoRA、完整上下文 |
| Beacon | scripts/sh/beacon/10_train_openseeker.sh | 现有 Beacon 压缩训练；ratio、布局、损失参数独立 |
| baseline | scripts/sh/baseline/10_train_openseeker.sh | 全参数 SFT，底层 kind=plain；区别于旧未训练基座 |

先编辑所选脚本自己的 YAML：学习率、epochs/max_train_steps、梯度累积、LoRA、保存周期；Beacon 另检查压缩比、窗口/whole 布局及损失开关。逐字段见 [69](../project/69_training_parameters.md)。三个方法/两个数据集的 YAML 独立，修改一份不会自动同步其他入口。

```bash
# 三个独立 plan，不加载模型；每次会输出各自的绝对 run_dir 与 next 命令。
DATASET_DIR="$PREPARED" MODEL_PATH="$MODEL_PATH" MAX_LENGTH="$MAX_LENGTH" ACTION=plan \
  bash scripts/sh/plain/10_train_openseeker.sh
DATASET_DIR="$PREPARED" MODEL_PATH="$MODEL_PATH" MAX_LENGTH="$MAX_LENGTH" ACTION=plan \
  bash scripts/sh/beacon/10_train_openseeker.sh
DATASET_DIR="$PREPARED" MODEL_PATH="$MODEL_PATH" MAX_LENGTH="$MAX_LENGTH" ACTION=plan \
  bash scripts/sh/baseline/10_train_openseeker.sh
```

记录三个不同的 RUN_DIR。检查每个目录的 source_config.yaml、train.yaml、experiment.json：dataset_name、method、kind、tokenizer、max_length、步数、有效 batch、指纹都应符合预期。有效 batch = GPU 进程数 × 单卡 batch(固定 1) × 梯度累积步数。正式比较需明确是否允许全参数 baseline 与 LoRA 的训练参数量不同。

```bash
# 将路径换成某个已审阅 plan 的真实目录，执行其屏幕给出的 next 命令。
RUN_DIR=/absolute/path/to/the_reviewed_run
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"
```

若 plan 使用多卡，正式启动必须沿用相同 CUDA_VISIBLE_DEVICES、NPROC_PER_NODE、DISTRIBUTED_BACKEND。Beacon 支持 DDP/ZeRO-2；Plain/baseline 另支持 ZeRO-3。直接运行训练 shell（ACTION=train）会新建并训练另一个实验，不是继续上面的 plan；若希望一键新建并训练，可选择此方式，两个启动方式选一个。

**完成条件：** events/log 的退出码为 0，训练日志与 trainer_metrics 可查，train_summary.json.final_model_path 指向存在的完整模型。Plain LoRA 正常收尾会自动合并；Beacon 正常收尾也会合并 adapter 和 Beacon 状态；baseline 不需 LoRA 合并。不能仅凭 summary 文件存在判定训练成功。自动合并失败或评估中间 checkpoint 的恢复方法见 [21](../plain/21_plain_experiment_guide.md) 和 [32](../beacon/32_beacon_experiment_guide.md)。

## 70.8 独立 Benchmark 准备

三个训练方法共用 benchmark 资产，各基准仅准备一次。训练集内部 validation/test 不是独立 benchmark；尤其不能用 Research 自身示范答案宣称独立研究能力。

```bash
# 在线 BrowseComp：准备固定官方版本及问题；不生成、不评分。
bash scripts/sh/benchmark/46_prepare_benchmark.sh

# BrowseComp-Plus：仅利用旧入口的数据准备能力。
ACTION=download bash scripts/sh/plain/56_plain_browsecomp_plus.sh
ACTION=prepare bash scripts/sh/plain/56_plain_browsecomp_plus.sh
ACTION=index bash scripts/sh/plain/56_plain_browsecomp_plus.sh

# HotpotQA：官方 dev + 独立 Wikipedia 摘要 corpus。
ACTION=download bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=prepare bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=index bash scripts/sh/plain/57_plain_hotpotqa.sh
```

已有正确资产则复用，不重复 prepare/index。完整语料、空间、解密和索引检查见 [43](../benchmark/43_offline_benchmarks.md)、[44](../benchmark/44_offline_sources.md)。download 联网；两个固定语料基准其余阶段严格离线。CORPUS_LIMIT 的小语料检查不等于完整语料成绩。

在线 BrowseComp 需要 search、visit、摘要服务；在线评分另需 SCORER。复制 `scripts/sh/benchmark/49_benchmark_env.example.sh` 到仓库外私有文件并按 [49](../benchmark/49_local_aux_models.md) 配置。两个离线评分需要已缓存的本地裁判目录 JUDGE_MODEL，不依赖在线工具。

## 70.9 三个测试分别生成，再分别评分

同一个训练模型的三个 benchmark 测试必须使用同一个 RUN_DIR。它自动读取原训练 system/schema 和最终模型；单独指定 MODEL_PATH 时必须同时提供对应规范化训练 JSONL 作为 PROMPT_DATA。Research 不能改用 Polar 默认提示；混合 system 的训练数据须显式选择匹配的单一提示配置，不能任取第一行。原训练数据指纹不符时自动读取会拒绝继续。

以下以 Plain 的一个训练实验示范完整六步；分别测试 Beacon/baseline 时，直接运行表中各自目录的同名独立脚本，并更换 RUN_DIR 及结果目录。

| 方法目录 | BrowseComp 测试 / 评分 | BrowseComp-Plus 测试 / 评分 | HotpotQA 测试 / 评分 |
|---|---|---|---|
| scripts/sh/plain | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |
| scripts/sh/beacon | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |
| scripts/sh/baseline | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |

```bash
export RUN_DIR=/absolute/path/to/this_dataset_plain_run
export JUDGE_MODEL=/models/judge
# smoke 只用于检查完整流程。正式实验改成 -1，并换一组新的输出目录。
export LIMIT=8
RESULTS=outputs/results/openseeker/plain_smoke_01

RUN_DIR="$RUN_DIR" ENV_FILE=/private/benchmark.env OUTPUT_DIR="$RESULTS/bc_run" \
  bash scripts/sh/plain/40_test_browsecomp.sh
PREDICTIONS="$RESULTS/bc_run/result_tool200.jsonl" ENV_FILE=/private/benchmark.env \
  OUTPUT_DIR="$RESULTS/bc_score" bash scripts/sh/plain/41_score_browsecomp.sh

RUN_DIR="$RUN_DIR" OUTPUT_DIR="$RESULTS/bcp_run" \
  bash scripts/sh/plain/42_test_browsecomp_plus.sh
PREDICTIONS="$RESULTS/bcp_run/predictions.jsonl" OUTPUT_DIR="$RESULTS/bcp_score" \
  bash scripts/sh/plain/43_score_browsecomp_plus.sh

RUN_DIR="$RUN_DIR" OUTPUT_DIR="$RESULTS/hotpot_run" \
  bash scripts/sh/plain/44_test_hotpotqa.sh
PREDICTIONS="$RESULTS/hotpot_run/predictions.jsonl" OUTPUT_DIR="$RESULTS/hotpot_score" \
  bash scripts/sh/plain/45_score_hotpotqa.sh
```

首次可在各生成命令前加 DRY_RUN=1 审查参数；不会加载模型或访问服务。正式运行时去掉它。LIMIT、题目文件、ROOT_DIR 必须在生成和评分之间保持一致；已有 OUTPUT_DIR 拒绝覆盖。默认 TOOL_COUNT=200，所以在线预测名为 result_tool200.jsonl；更改后以实际生成文件名为准。

原训练提示仅为 benchmark 调整实际工具能力和答案要求：BCP 默认本地 search-only 并追加 docid 引用，Hotpot 用本地 search/visit 并追加 answer/sp JSON；协议、研究指令和工具参数保留。预算、解码、工具调用和全部裁判参数逐项见 [68](../project/68_training_benchmark_guide.md)。旧 baseline/70–75 是未训练原始基座，不用于新 SFT baseline 的本节实验。

## 70.10 预测 HTML、评分解读与实验归档

```bash
# 不加载模型，渲染实际生成轨迹；先确认预测路径存在。
"$PYTHON" scripts/py/41_predictions_html.py "$RESULTS/hotpot_run/predictions.jsonl" \
  --output-dir "$RESULTS/html"
```

抽查成功和失败轨迹，确认最终回答非空、工具调用合法、没有长时间重复和预算耗尽。评分目录的 summary.metrics.json 中读取 `em`、`f1`、`llm_judge_accuracy`，三者均为 0–1。BCP 的 EM/F1 是去方括号引用后的补充答案指标，LLM judge 保留原官方提示；Hotpot 的 EM/F1 是官方实现，支持事实/Joint 单独解释。完整字段和失败口径见 [40](../benchmark/40_evaluation.md)、[59](../benchmark/59_llm_judge_matrix.md)。

缺失/失败/空回答保留分母并计零；裁判传输或格式错误重试耗尽会留下未知判定，complete=false 且非零退出，不能按成功题子集报完整成绩。检查 coverage、judge 日志和 per_question 后，才解释模型差异。

**最终归档清单：** 下载 repo/commit/源文件 SHA256、所有输入文件清单、原始/规范化 HTML、raw audit、manifest/rejected/measurements、筛选前后分布与图、实际训练 split、三个方法各自的 RUN_DIR/配置/日志/模型、提示词来源与 hash、benchmark 资产/索引身份、LIMIT/预算/生成种子/裁判模型和参数、预测及评分目录。正式全量结果必须与 smoke 结果分开。

## 70.11 分阶段完成检查

| 顺序 | 阶段 | 通过条件 |
|---|---|---|
| 1 | 来源与下载 | commit 已记录，输入文件完整可读；本地导出有来源记录 |
| 2 | 原始 HTML 与结构审计 | 抽样仅用于检查格式，全量 audit 用于说明结构和异常 |
| 3 | 预处理 | 完整 tokenizer 计长，manifest 可追溯，train 非空 |
| 4 | 规范化 HTML | 标签、system/schema、调用/观察、最终回答符合训练协议 |
| 5 | 整体分布 | 筛选前后总体口径清楚，实际 train 的长度和工具构成已确认 |
| 6 | 配置与训练 | plan 审阅通过，训练正常退出并得到完整推理模型 |
| 7 | Benchmark 资产 | 三项题集/工具/corpus/索引按协议准备完成 |
| 8 | 测试与评分 | 每方法分别生成并评分，三项指标与失败覆盖可解释 |
| 9 | 轨迹与归档 | HTML 抽查完成，数据、参数、模型、提示词和结果可追溯 |

## 70.12 数据阶段参数与常见修改

| 变量/参数 | 默认或建议 | 作用与修改位置 |
|---|---|---|
| PYTHON / CONDA_ENV | 已激活 Linux Python / search-comp-qwen35-cu126 | 每步使用相同环境；HTML 可仅用标准库 Python |
| REVISION | main 首次取数，正式固定 commit | 下载前设置；解析后的实际 commit 写入 download_manifest |
| OUTPUT | outputs/data/raw | 下载父目录，不是具体源文件；内部再按来源和 commit 分目录 |
| SOURCE | Polar 为 polar；Research 下载固定 research | OpenSeeker 可选 zephyr；预处理按实际来源指定，不能据目录名猜 |
| HF_ENDPOINT | Hugging Face 默认服务 | 可配置可访问 endpoint，实际地址归档；仓库权限仍需单独满足 |
| HF_TOKEN | 可选私有环境变量 | 访问授权，不写入文档、日志或 Git |
| SNAPSHOT_DIR | 下载器打印的确切目录 | Research prepare 非空时从 manifest 加载全部输入，优先于 INPUT |
| RAW_FILES | manifest.jsonl_inputs 的 Bash 数组 | 原始 HTML 和全量 audit 共同使用；本地数据可显式列出所有文件 |
| INPUT / INPUT_FILE | 本地原始 JSONL | 单文件兼容入口，Research 的 INPUT 仅在 SNAPSHOT_DIR 为空时生效 |
| PREPARED / DATASET_DIR | 本文新 prepared 目录 | PREPARED 是指南会话变量，传给训练时使用 DATASET_DIR |
| MODEL_PATH / MODEL_NAME / --models | 本地 tokenizer/模型路径 | Research 和训练用 MODEL_PATH；OpenSeeker shell 用 MODEL_NAME；Python prepare 用 --models；必须同身份 |
| MAX_LENGTH | 16384 | 完整编码上限；更改 tokenizer/正文/编码后重新分词，不截断样本 |
| MIN_LENGTH | OpenSeeker 1 | 下限；Research 可用底层 prepare CLI 的 --min-length |
| CORRECTNESS | OpenSeeker Correct，Research any | 不把 Research 无验证示范自动标成 Correct |
| OBSERVATIONS | OpenSeeker any，Research with | any 不限制、with 有非空观测、without 无非空观测 |
| WORKERS | 4 | CPU 分词并行，每进程加载 tokenizer，注意主机内存 |
| OFFLINE / --offline | 仅本地 tokenizer | Research shell 固定离线；OpenSeeker OFFLINE=1；下载是前面独立联网阶段 |
| VAL_FRACTION / TEST_FRACTION / SEED | 0.02 / 0.02 / 42 | 问题分组划分；实际数量和已有 split 处理以 manifest 为准 |
| LIMIT / OFFSET | HTML 20 或 30 / 0 | 仅控制人工检查抽样；不要把 prepare --limit 用于正式全量统计 |
| REPORT_DIR | 本文新报告父目录 | 原始/规范化 HTML、audit、分布 JSON 与图表分别保存 |
| PREPARED / OUTPUT / MODEL | 45_report 环境变量 | 输入最终数据目录、新报告目录、准确 tokenizer 键；MODEL 多 tokenizer 时必填 |
| MEASUREMENTS / POPULATION | measurements.jsonl / all | 44_plot 输入及总体；all/retained/rejected，空总体明确报错 |

数据改动后，先在新目录重做 HTML 与分布检查，再生成新的训练 plan。只需调整学习率等训练超参数时可复用已确认的数据，但应新建实验目录。不要手改旧 manifest、指纹或模型 summary 来掩盖路径/数据不一致。
