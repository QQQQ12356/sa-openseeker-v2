# 21 Plain 完整实验指南：数据、训练、测试与评分

本文按实际实验顺序组织，以 **OpenSeeker Polar + Qwen3.5-4B + Plain + LoRA** 为基线，训练后分别在 OpenSeeker 留出轨迹、在线 BrowseComp、离线 BrowseComp-Plus 和 HotpotQA 上评测。Plain 使用完整上下文，不进行 Beacon 压缩。本文命令需要在项目根目录执行；真实训练和评测需在有相应资源的目标服务器执行。

复读诊断可通过 `DECODING`、`REPETITION_PENALTY` 等变量设置 Plain native 推理策略；默认保持贪心且不惩罚，完整参数和对照见 [23](../common/23_decoding.md)。

## 21.1 流程与数据边界

```text
准备环境和模型 → 准备 OpenSeeker 训练轨迹 → 检查数据和训练计划
→ 训练 Plain → 检查并在必要时合并 LoRA → OpenSeeker 小样本诊断
→ 选择在线 BrowseComp / 离线 BrowseComp-Plus / HotpotQA
→ 准备基准数据和工具 → 生成 → 评分 → 检查覆盖、轨迹与归档
```

| 数据 | 本文用途 | 入口 |
|---|---|---|
| OpenSeeker Polar 轨迹 | 预处理、训练及留出 split 诊断 | 10、20、21 |
| BrowseComp | 官方题目、在线搜索工具及在线裁判 | 46、47 |
| BrowseComp-Plus | 固定网页语料、离线检索及本地裁判 | 56 |
| HotpotQA fullwiki dev | 独立 Wikipedia 摘要语料、离线检索及官方指标 | 57 |

以上三种 benchmark 题集是评测集，不要作为训练数据。当前仓库支持用 OpenSeeker 轨迹训练 Plain，尚无 BrowseComp 或 Hotpot 专用训练集的轨迹采集、转换和训练流程；仅有问题与参考答案不能等同于可训练的搜索轨迹。若要做数据集专门训练，需要独立、可用的训练划分与真实或经审核的工具轨迹，并先实现相应的数据适配与去重，不能将本文的 benchmark dev/test 当训练集。

`27_plain_workflow.sh` 是可选的统一入口，按 `TARGET` 和 `ACTION` 转发到现有脚本；也可以直接使用各编号脚本。它固定 `KIND=plain`，在线 BrowseComp 固定 `MODE=native`。支持的组合：

| TARGET | ACTION |
|---|---|
| openseeker | download、prepare、plan、train、run、score |
| browsecomp | prepare、run、score |
| browsecomp-plus / hotpotqa | download、prepare、index、run、score、all |

`run` 是生成，OpenSeeker 生成时同时写 EM/F1，在线 BrowseComp 生成后会自动调用在线裁判；离线 `run` 与 `score` 独立，`all` 包含准备、建索引、生成和评分，但不下载。各入口的参数继续由下游脚本解释；`ACTION` 默认 plan，使用 benchmark 时请显式指定。

## 21.2 环境、模型和 Python

```bash
cd /home/huangzj/proj/searchagent/sa-openseeker-v2
conda activate search-comp-qwen35-cu126
export CONDA_ENV=search-comp-qwen35-cu126
export PYTHON="$(command -v python)"
python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)'
# 可选：项目协议和 tiny 模型 CPU 回归；不会训练真实模型
bash scripts/sh/tools/59_check_agent_alignment.sh
# 可选：目标 GPU 环境探针，会实际执行小张量反向及 CPUAdam 检查
bash scripts/sh/tools/03_check_environment.sh
```

新环境安装和资源条件见 [01](../common/01_environment.md)、[67](../tools/67_environment_details.md)。训练需要完整模型权重，数据预处理只加载 tokenizer。默认模型是 `Qwen/Qwen3.5-4B`；如果改成本地路径，prepare 的 `MODEL_NAME` 与 20 脚本 YAML 的 `models.qwen35` 必须使用同一身份字符串。模型支持的上下文、GPU 和主机内存需要在目标机器验证。

## 21.3 OpenSeeker 数据准备

默认 raw 为 `outputs/data/raw/polar/openseeker_v1_data.jsonl`，默认 prepared 为 `outputs/data/openseeker/polar_correct_qwen35_16k/`。已有合适数据可直接复用；先检查 protocol、tokenizer、筛选条件、各 split 和长度：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/data/openseeker/polar_correct_qwen35_16k')
m = json.loads((root / 'manifest.json').read_text())
print('protocol:', m['protocol'])
print('tokenizers:', list(m['tokenizers']))
print('counts:', m.get('counts'))
print('selection:', m['arguments'])
for split in ('train', 'validation', 'test'):
    print(split, sum(bool(line.strip()) for line in (root / f'{split}.jsonl').open()))
PY
```

当前协议为 `openseeker-chatml-v3-original-answer`，保留轨迹的原始 system、工具及回答。旧数据的 644/9/11 是特定筛选下的参考数量，应以实际 manifest 为准。若 raw 尚未准备，检查并修改 `16_download.sh` 中的来源、revision 和输出，再运行 `TARGET=openseeker ACTION=download bash scripts/sh/plain/27_plain_workflow.sh`；不要把可变的 main 视为固定版本。

从 raw 重新准备时使用新目录：

```bash
INPUT_FILE=/data/polar/openseeker_v1_data.jsonl \
OUTPUT_DIR=/data/prepared/plain_polar_qwen35_16k \
MODEL_NAME=/models/Qwen3.5-4B OFFLINE=1 \
TARGET=openseeker ACTION=prepare bash scripts/sh/plain/27_plain_workflow.sh
```

| 变量 | 默认 | 含义 |
|---|---|---|
| INPUT_FILE / SOURCE | 默认 Polar raw / polar | 多文件需编辑 10 脚本的 INPUTS 数组 |
| OUTPUT_DIR | 默认 prepared 目录 | 必须不存在；将其填入训练 YAML 的 dataset |
| MODEL_NAME / OFFLINE | Qwen/Qwen3.5-4B / 1 | tokenizer 身份；0 允许联网下载 tokenizer |
| MIN_LENGTH / MAX_LENGTH | 1 / 16384 | 完整序列长度闭区间，超长过滤而不是截断 |
| CORRECTNESS / OBSERVATIONS | Correct / any | 质量与非空工具观察筛选 |
| VAL_FRACTION / TEST_FRACTION / SEED | 0.02 / 0.02 / 42 | 按问题分组划分；已有输入文件 split 名优先保留 |
| WORKERS | 4 | CPU 分词进程数 |

输出 `train.jsonl`、`validation.jsonl`、`test.jsonl`、`normalized.jsonl`、`measurements.jsonl`、`rejected.jsonl`、`manifest.json`。`normalized.jsonl` 可包含未入训练划分的超长或错误样本。需要观察筛选可运行 `13_filter_observations.sh 输入目录 新输出目录`；长度筛选需先编辑 `11_select.sh` 的参数，默认示例为约 32K，并非 16K 基线。使用 `40_dataset_html.sh` 抽查 system、问题、调用、观察与回答，见 [10](../data/10_data.md)、[11](../data/11_lengths.md)。Plain/Beacon 比较应选择同一数据目录与划分。

## 21.4 配置并审查训练计划

编辑 `scripts/sh/plain/20_plain_train.sh` 的内嵌 YAML，改 `dataset`、`models.qwen35`、`training.max_length`；数据 manifest 的长度上限不得超过训练上限。默认入口的训练参数和 DeepSpeed 均来自该脚本，不读取外部基础配置；完整参数和覆盖规则见 [53](../tools/53_self_contained_training.md)。切换 Qwen3 时同时修改 `model_family`、模型与 tokenizer 数据。

| 参数 | 当前默认 | 说明 |
|---|---|---|
| kind / data_mode | plain / openseeker | 完整轨迹 SFT，assistant 输出监督 |
| learning_rate / num_epochs / max_train_steps | 5e-5 / 1 / null | 正整数 max_train_steps 按 optimizer step 提前停止 |
| per_device_batch_size / grad_accum_steps | 1 / 4 | 四卡有效 batch 约 16；不拆分单条完整轨迹 |
| use_lora / lora_r / lora_alpha | true / 16 / 32 | 使用 LoRA；全参数训练需要更多资源 |
| gradient_checkpointing | true | decoder 层重算 |
| plain_attention_backend | sdpa | 实际融合内核取决于环境 |
| plain_loss_chunk_size / plain_checkpoint_loss | 32 / true | 仅监督位置分块计算并重算输出头 loss |
| plain_cpu_offload_activations / threshold | true / 2048 | 用主机内存换激活显存；0 表示全部样本 |
| save_freq_steps / eval_steps | 200 / 200 | optimizer step 间隔；短训练可能没有周期 checkpoint |
| seed | 42 | 训练随机种子，与数据划分 seed 分别记录 |

脚本中 `training.loss_segments` 用于设置 assistant 输出的分段权重，默认均为 1。Plain 脚本只配置 Plain 模型；资源和限制详见 [20](20_plain.md)。旧实验配置中的 `beacon.loss_segments` 仍由训练器兼容读取。

```bash
# 单卡：只写实验计划，不加载模型
CUDA_VISIBLE_DEVICES=0 TARGET=openseeker ACTION=plan bash scripts/sh/plain/27_plain_workflow.sh

# 四卡：20 的同一份配置，默认 ZeRO-2
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ACTION=plan bash scripts/sh/plain/28_plain_4xa800.sh
```

屏幕输出中记录绝对 `RUN_DIR` 和 `next`。检查 `source_config.yaml`、`train.yaml`、`experiment.json` 的数据、模型、长度、训练步数和指纹。`DISTRIBUTED_BACKEND` 可选 ddp、zero2、zero3，后者仅 Plain 支持，见 [50](../tools/50_multigpu.md)；显式 `NPROC_PER_NODE` 应与可见 GPU 数匹配。修改代码或数据后重新生成 plan。

可选显存试验用 `12_select_memory_probe.sh` 和 `26_plain_memory_probe.sh`；它们默认约 64K / 5 步，并不比 16K 基线轻。先按 [51](../tools/51_memory_probe.md) 调整样本区间、数据路径和长度。实际试验会加载权重并反向，独立保存，不是 dry-run。

## 21.5 训练及产物

使用刚才 plan 打印的命令，以下替换真实路径：

```bash
RUN_DIR=/absolute/path/to/outputs/experiments/your_plain_run
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"
# 四卡 plan 选择下面一条而不是上面单卡命令：
# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 DISTRIBUTED_BACKEND=zero2 \
#   "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"
```

`ACTION=train` 直接调用 20 或 28 会新建并训练另一 run；不能把它当作继续上面的 plan。计划入口不是任意 checkpoint 的优化器恢复入口。训练时检查日志、`events.jsonl`、`trainer_metrics.jsonl`、`failure_rank<N>.json`。默认 LoRA 训练结束后的主要产物：

```text
RUN_DIR/models/model/
├── resolved_config.json
├── checkpoint-<step>/        # 周期 checkpoint，短训练不一定有
├── final/                    # LoRA adapter 和 tokenizer
├── final_merged/             # 自动合并后完整模型
└── train_summary.json        # 成功合并后 final_model_path 指向 final_merged
```

全参数训练时 `final/` 即完整模型。LoRA 合并前也会先写 summary，合并中断时该文件可能仍指向 adapter；必须同时检查进程退出状态以及 `final_model_path` 下的完整 config、tokenizer 和模型权重，不能只以 summary 存在判断成功。

### 21.5.1 需要时手动合并

```bash
# final_merged 不存在或为空时；已有非空目录时指定一个新输出目录。
DEVICE_MAP=cpu bash scripts/sh/plain/29_plain_merge_adapter.sh "$RUN_DIR/models/model" \
  --output_path "$RUN_DIR/models/model/final_recovered"

# 中间 checkpoint：
# DEVICE_MAP=cpu bash scripts/sh/plain/29_plain_merge_adapter.sh "$RUN_DIR/models/model/checkpoint-200" \
#   --output_path "$RUN_DIR/models/model/merged_step200"
```

脚本从模型产物目录的 `resolved_config.json` 读取基座；缺失时额外提供 `--base_model_path`。需要足够 CPU 内存，目标目录非空时拒绝覆盖。手动合并不会更新旧 summary；测试显式设置 `MODEL_PATH="$RUN_DIR/models/model/final_recovered" KIND=plain`，并保留 `RUN_DIR` 以确定原训练数据和提示词。21 可直接挂载 Plain adapter 进行 OpenSeeker 评测；47/56/57 仍要求完整合并模型，详见 [22](../baseline/22_qwen35_baseline_and_adapter.md)。

## 21.6 OpenSeeker 留出轨迹测试和打分

```bash
# 先检查解析后的模型、数据和命令；dry-run 不加载模型验证权重
RUN_DIR="$RUN_DIR" SPLIT=test LIMIT=2 DRY_RUN=1 \
TARGET=openseeker ACTION=run bash scripts/sh/plain/27_plain_workflow.sh

# replay 匹配历史动作，生成时自动计算 EM/F1
RUN_DIR="$RUN_DIR" SPLIT=test LIMIT=2 BACKEND=replay \
TARGET=openseeker ACTION=run bash scripts/sh/plain/27_plain_workflow.sh
```

21 的 `LIMIT` 为正整数，默认 100；`DATA` 可指定其他 JSONL，`SPLIT=train/validation/test` 默认 test。`MAX_CONTEXT=16384`、`MAX_NEW_TOKENS=8192`、`MAX_TOOL_TOKENS=16384`、`MAX_TURNS=32`；比较模型时保持一致。replay 只在动作精确匹配历史记录时返回观察，新查询出现 `replay_miss` 不能解释成真实搜索失败。自由查询可选 `BACKEND=corpus CORPUS=/path/to/corpus.jsonl` 或 `BACKEND=http ENDPOINT=...`，详见 [40](../benchmark/40_evaluation.md)。

21 打印 `predictions_*.jsonl` 和同名 `.metrics.json`。需要对已生成文件重新汇总时：

```bash
PREDICTIONS=/absolute/path/to/predictions_test.jsonl \
OUTPUT_PATH=/absolute/path/to/new_summary.json \
TARGET=openseeker ACTION=score bash scripts/sh/plain/27_plain_workflow.sh
# 或比较多个文件：bash scripts/sh/report/42_summarize_results.sh /new/summary.json /a.jsonl /b.jsonl
```

新汇总目录可创建，但目标 JSON 不能已存在。汇总包括总体 EM/F1、格式正确子集及样本数；查看生成指标和状态来解释失败与分母，不要将 replay 诊断作为独立搜索 benchmark 成绩。

## 21.7 HotpotQA 离线数据、生成与打分

默认目录 `outputs/data/benchmarks/hotpotqa`。检查磁盘及 SQLite FTS5（见 [43](../benchmark/43_offline_benchmarks.md)）；下载需联网，后续阶段严格离线。原始数据或索引已存在时可复用，不重复 prepare：

```bash
TARGET=hotpotqa ACTION=download bash scripts/sh/plain/27_plain_workflow.sh
TARGET=hotpotqa ACTION=prepare bash scripts/sh/plain/27_plain_workflow.sh
TARGET=hotpotqa ACTION=index bash scripts/sh/plain/27_plain_workflow.sh

# 显式分开生成和评分，便于记录各阶段产物
TARGET=hotpotqa ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/plain/27_plain_workflow.sh
```

记录终端的 `[offline] artifacts:` 目录：

```bash
GEN_DIR=/absolute/path/to/hotpot_generation
TARGET=hotpotqa ACTION=score PREDICTIONS="$GEN_DIR/predictions.jsonl" LIMIT=8 \
  bash scripts/sh/plain/27_plain_workflow.sh
```

评分不加载待测模型或裁判，使用官方 EM/F1、支持事实 SP 和 Joint。生成和评分必须使用同一 `ROOT_DIR`、问题顺序和 `LIMIT`；score 默认创建新的结果目录。也可用 `TARGET=hotpotqa ACTION=all RUN_DIR="$RUN_DIR" LIMIT=8 ...` 在一次进程中完成准备、建索引、生成及评分。先检查小样本，再用 `LIMIT=-1` 在新结果目录全量运行。

使用官方有标签 dev 和独立 2017 Wikipedia 摘要语料；dev 问题附带的 context 不能作为全库。输出需包含模型自己给出的 supporting facts 才能获得有意义的 SP/Joint；检索到的文档不会自动变成预测支持事实。缺失或失败题仍计入分母。参考答案和支持标注只用于评分，不进入生成提示。

## 21.8 BrowseComp-Plus 离线数据、生成与打分

默认目录 `outputs/data/benchmarks/browsecomp-plus`，full corpus 需要较大磁盘，来源和准备细节见 [43](../benchmark/43_offline_benchmarks.md)、[44](../benchmark/44_offline_sources.md)。

```bash
TARGET=browsecomp-plus ACTION=download bash scripts/sh/plain/27_plain_workflow.sh
TARGET=browsecomp-plus ACTION=prepare bash scripts/sh/plain/27_plain_workflow.sh
TARGET=browsecomp-plus ACTION=index bash scripts/sh/plain/27_plain_workflow.sh
TARGET=browsecomp-plus ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/plain/27_plain_workflow.sh
```

记录生成目录后，独立加载本地裁判评分：

```bash
GEN_DIR=/absolute/path/to/browsecomp_plus_generation
TARGET=browsecomp-plus ACTION=score PREDICTIONS="$GEN_DIR/predictions.jsonl" \
JUDGE_MODEL=/models/Qwen3-32B LIMIT=8 bash scripts/sh/plain/27_plain_workflow.sh
```

`ROOT_DIR`、`LIMIT` 应与生成一致。`RETRIEVAL_ONLY=1` 可不加载裁判，但只得到检索诊断，不能报告为答案准确率。裁判失败需检查 `judge.jsonl` 与覆盖，修复后以新输出目录重新评分。默认 search-only；此处 SQLite FTS5 BM25 与官方 Lucene 检索器不同，报告时应说明设置。BrowseComp-Plus 结果与在线 BrowseComp 不是同一基准。

### 21.8.1 离线参数及产物

| 变量 | 默认 | 说明 |
|---|---|---|
| ROOT_DIR | 各基准独立目录 | 含 raw、prepared、官方参考文件和索引 |
| RUN_DIR / MODEL_PATH / PROMPT_DATA | 空 | 正常训练用 RUN_DIR；恢复模型显式 MODEL_PATH；PROMPT_DATA 指原训练 JSONL，仅取 system/schema |
| TOPK / SNIPPET_TOKENS / READ_TOKENS | 5 / 512 / 4096 | 检索条数和阅读预算 |
| ALLOW_VISIT | BCP 0，Hotpot 1 | 是否可读取本地文档 |
| MAX_CONTEXT / MAX_NEW_TOKENS | 131072 / 8192 | 完整历史逻辑上限及每轮生成上限，实际模型上下文和资源须支持 |
| MAX_TOOL_TOKENS / MAX_TURNS / TOOL_COUNT | 32768 / 200 / 200 | 工具返回、循环及工具次数限制 |
| LIMIT | -1 | 全量；正整数为前 N 题 |
| NUM_WORKERS | 1 | 每卡一份完整模型，按题并行，不是张量并行 |
| CORPUS_LIMIT | 空 | 正数只用于小语料 smoke，不是全量基准 |
| OUTPUT_DIR | 自动 | 新目录，不能覆盖生成或先前评分结果 |

例如四卡生成：`CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_WORKERS=4 TARGET=hotpotqa ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 bash scripts/sh/plain/27_plain_workflow.sh`。检查 `predictions.jsonl`、`generation.metrics.json`、`manifest.json`，评分后检查 `summary.metrics.json`、`per_question.json`；Hotpot 另有官方格式文件，BCP 另有 `judge.jsonl`。不同模型应共享同一索引和评测配置。

## 21.9 在线 BrowseComp 准备、生成和裁判

在线路线需要真实搜索、网页读取、摘要和裁判服务，与离线基准是并列选择。固定官方代码和题目：

```bash
TARGET=browsecomp ACTION=prepare bash scripts/sh/plain/27_plain_workflow.sh
mkdir -p "$HOME/.config/searchagent"
cp scripts/sh/benchmark/49_benchmark_env.example.sh "$HOME/.config/searchagent/benchmark.env.sh"
```

按 [42](../benchmark/42_openseeker_benchmark.md) 安装在线依赖并编辑私有文件，配置搜索 provider/key、Jina、摘要与裁判地址/key/model。native 不要求 OPENSEEKER 模型服务或 E2B；服务地址须从运行节点可达，真实 key 不写入仓库。

```bash
TARGET=browsecomp ACTION=run RUN_DIR="$RUN_DIR" LIMIT=2 DRY_RUN=1 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" bash scripts/sh/plain/27_plain_workflow.sh

# 实际生成和裁判会调用在线服务
TARGET=browsecomp ACTION=run RUN_DIR="$RUN_DIR" LIMIT=2 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" bash scripts/sh/plain/27_plain_workflow.sh
```

47 的 dry-run 仅打印计划，不验证模型权重或外部服务是否可用。native `MAX_TOKENS=32768` 为每轮上限，`MAX_CONTEXT=131072`、`MAX_TURNS=200`、`MAX_TOOL_TOKENS=32768`；`JUDGE_WORKERS` 控制裁判并发，`WORKERS` 不使 native 自动多卡。完整评测用 `LIMIT=-1` 并新建结果目录。使用手动合并模型时显式 `MODEL_PATH`；训练路径迁移时增加 `PROMPT_DATA=/path/to/original_train.jsonl`。

生成与自动裁判均完成后会输出 `result_tool*.jsonl`、`coverage.json`、`summary.metrics.json`、`judge.log`。需要对既有预测重新在线评分时：

```bash
TARGET=browsecomp ACTION=score PREDICTIONS=/absolute/path/to/result_tool200.jsonl \
LIMIT=2 ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/plain/27_plain_workflow.sh
```

评分的题目数据和 LIMIT 必须与预测对应。若裁判覆盖不完整，检查日志和失败题，不能只报告成功裁判的分母。原样官方 v1/v2 对照可用 48 部署 Plain 并直接调用 47，详见 42；27 固定为 native。

## 21.10 结果检查与归档

1. 确认实际生成题数、completed 和失败状态，查看空回答、工具错误、上下文和生成预算耗尽。
2. 检查真实 search/visit 参数、返回证据和最终答案；`41_predictions_html.sh` 的 INPUTS 要改为实际文件。
3. 对应基准读取正确指标：OpenSeeker EM/F1、Hotpot 官方 EM/F1/SP/Joint、BCP 答案与检索指标、在线 BrowseComp 裁判及覆盖。
4. 保存 RUN_DIR、数据与 tokenizer 身份、训练与代码指纹、模型、提示来源、语料/索引、预算、裁判配置、题数和失败分母。对比 Plain/Beacon 时固定数据、提示、工具与预算，另记压缩设置。

| 顺序 | 执行与完成条件 |
|---|---|
| 1 | 准备环境、完整模型和 tokenizer |
| 2 | 检查已有 OpenSeeker 数据或运行 10；确认 manifest 和划分 |
| 3 | 修改 20，使用 plan 审查；可选 12+26 显存试验 |
| 4 | 用计划启动训练；检查退出码、summary 和完整模型；必要时用 29 合并 |
| 5 | 21 做少量 OpenSeeker 生成及 EM/F1 诊断 |
| 6 | 按需要运行 57 Hotpot、56 BCP 或 46+47 在线 BrowseComp；先小样本再全量 |
| 7 | 检查轨迹、评分和覆盖，保存实验条件与结果 |

Plain 专属 Benchmark 入口：`47_plain_browsecomp.sh`（在线）、`56_plain_browsecomp_plus.sh`、`57_plain_hotpotqa.sh`（离线），均位于 `scripts/sh/plain/`。27 工作流已分发到这些脚本。完整变量、运行示例和路径迁移见 [46 脚本手册](../benchmark/46_benchmark_scripts.md)。
