# 32 Beacon 完整实验指南：数据准备、训练、合并与评测

本文按实际实验顺序组织，以 **OpenSeeker Polar + Qwen3.5-4B + Beacon + LoRA + 固定压缩比 16** 为基线。照本文依次执行即可，不必先读完所有算法文档。文中的真实训练和模型评测命令由使用者在目标服务器执行，本次整理没有启动这些任务。

训练保留数据中的原始 system、工具 schema、推理和回答；Plain/Beacon 共用编码器。native 表示本项目直接加载模型并执行工具循环，可以配在线工具，也可以配本地离线语料，不表示“无压缩”或“只能离线”。

## 32.1 路线图与阅读顺序

```text
环境与模型准备
  → 确认已有 prepared 数据，或从 raw 重新 prepare
  → 检查长度、正确性、观测与划分
  → 检查训练参数并生成 plan
  → 必要时做独立显存试验
  → 执行正式训练
  → 检查 final 和训练总结（需要时手动合并）
  → 少量生成检查
  → 离线 BCP / Hotpot，或在线 BrowseComp
  → 查看轨迹、指标，保存实验条件
```

| 阶段 | 先读 | 何时再读补充文档 |
|---|---|---|
| 环境 | [01 环境](../common/01_environment.md) | 多卡/CPUAdam 问题看 [67 环境细节](../tools/67_environment_details.md) |
| 原始数据与预处理 | [10 数据](../data/10_data.md) | 长度统计口径看 [11 长度](../data/11_lengths.md) |
| 提示词与协议 | [45 统一流程](../common/45_unified_agent_flow.md) | 此文说明训练原 system 如何用于在线/离线测试 |
| Beacon 训练 | [30 Beacon](30_beacon.md) | 随机比例看 [31](31_random_compression.md)，模型结构看 [60](../architecture/60_architecture.md) |
| 四卡/显存 | [50 多卡](../tools/50_multigpu.md) | 资源紧张时读 [51 显存探针](../tools/51_memory_probe.md)、[52 OOM](52_beacon_oom.md) |
| 小样本工具诊断 | [40 评测](../benchmark/40_evaluation.md) | HTML 浏览看 [41](../report/41_html.md) |
| 固定语料离线测试 | [43 离线基准](../benchmark/43_offline_benchmarks.md) | 来源、版本与许可看 [44](../benchmark/44_offline_sources.md) |
| 在线 BrowseComp | [42 在线 Benchmark](../benchmark/42_openseeker_benchmark.md) | API 与部署参数均在该文 |
| 输出核对 | [99 验证记录](../project/99_validation.md) | 这里的 CPU 回归结果不等于真实模型成绩 |

首次实验只需沿正文主线走。62/63 蒸馏、64 线性状态消融、65 压缩分析是后续研究选项；22/23、32—34 等 Search-R1 **脚本**是另一条数据路线，不能混用本文的 OpenSeeker 数据协议。本文编号 32 是文档编号。

## 32.2 第一步：准备环境、模型与路径

在项目根目录执行，先激活已有训练环境：

```bash
cd /home/huangzj/proj/searchagent/sa-openseeker-v2
conda activate search-comp-qwen35-cu126

# 新终端可重新设置。PYTHON 用于支持它的 shell 入口，避免误用系统 Python。
export CONDA_ENV=search-comp-qwen35-cu126
export PYTHON="$(command -v python)"

python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)'
```

新服务器按 01/67 安装项目固定依赖。不要在已有环境中直接无约束升级 torch/transformers。训练需要完整基座权重；prepare 只需要 tokenizer，prepare 成功不代表模型权重已经准备好。

默认模型标识是 `Qwen/Qwen3.5-4B`。可以使用已下载缓存，也可以把数据准备与训练里的模型标识**同时**改为同一个本地路径。实验入口检查 `manifest.json.tokenizers` 的键；prepare 用 HF 名称、训练却改成本地别名，即使底层文件相同也可能因身份键不一致被拒绝。换 tokenizer 必须重新计长。

目标 GPU 服务器可运行以下环境探针；它会做小张量反向与 CPUAdam 检查，不是纯静态检查：

```bash
bash scripts/sh/tools/03_check_environment.sh
```

03 的 PYTHON/CONDA_ENV 在该脚本顶部修改。只检查本项目协议与 tiny 模型一致性、不加载真实权重时运行：

```bash
bash scripts/sh/tools/59_check_agent_alignment.sh
```

GPU 驱动、BF16、融合算子和多卡资源是否适合正式训练，需要在目标服务器确认。CPU 回归通过不能证明 4B 模型一定能在目标显存内训练。

## 32.3 第二步：决定复用已有数据还是重新准备

当前默认数据目录：

```text
outputs/data/raw/polar/openseeker_v1_data.jsonl       原始轨迹
outputs/data/openseeker/polar_correct_qwen35_16k/     默认 prepared 数据
```

如果已有 prepared 数据的 protocol、tokenizer、质量筛选和长度上限符合本次实验，直接复用，不重复运行 prepare。当前协议标识为 `openseeker-chatml-v3-original-answer`，训练原始 system 不被替换。

先只读查看 manifest：

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
    p = root / f'{split}.jsonl'
    print(split, sum(1 for line in p.open() if line.strip()))
PY
```

先前默认数据实测 train/validation/test 为 644/9/11；这是特定筛选与 tokenizer 下的既有结果，不是原始 11,677 条轨迹全部进入训练。重新准备、换长度或换划分后，以新 manifest 和实际文件为准。

原始 Polar 已在本目录时不用下载。新服务器缺少源数据才使用 `16_download.sh`；先修改该脚本的 SOURCE、REVISION、OUTPUT，优先固定实际 revision。不要把 `main` 当作固定数据版本。数据来源细节见 10。

### 32.3.1 从 raw 重新 prepare

输出目录必须不存在；不会覆盖原 prepared 数据。默认目录已存在时跳过此步，或显式用新输出目录：

```bash
# 默认目录尚不存在时：
bash scripts/sh/data/10_prepare.sh

# 自己指定源、输出、tokenizer；不加载基座模型，不训练：
INPUT_FILE=/data/polar/openseeker_v1_data.jsonl \
OUTPUT_DIR=/data/prepared/polar_qwen35_16k_run1 \
MODEL_NAME=/models/Qwen3.5-4B OFFLINE=1 \
  bash scripts/sh/data/10_prepare.sh
```

| 10_prepare.sh 参数 | 默认 | 含义与修改提示 |
|---|---|---|
| INPUT_FILE / INPUTS | 原始 Polar JSONL | 单文件用环境变量；多文件编辑 INPUTS 数组 |
| SOURCE | polar | 数据来源标记，支持 polar/zephyr/combined |
| OUTPUT_DIR | 默认 prepared 目录 | 必须新目录；后续训练 dataset 指向此目录 |
| MODEL_NAME / MODELS | Qwen/Qwen3.5-4B | 分词模型身份；多模型编辑 MODELS 数组 |
| MIN_LENGTH / MAX_LENGTH | 1 / 16384 | 完整 ChatML 序列长度闭区间；超长过滤，不截断 |
| CORRECTNESS | Correct | Correct/Incorrect/any；首次基线保留正确轨迹 |
| OBSERVATIONS | any | any/with/without；是否要求非空工具观察正文 |
| VAL_FRACTION / TEST_FRACTION | 0.02 / 0.02 | 问题分组划分，训练比例为剩余；两者和小于 1 |
| SEED | 42 | 固定哈希划分的种子 |
| WORKERS | 4 | CPU 分词进程数，内存不够时降低 |
| OFFLINE | 1 | 只用本地 tokenizer；0 允许下载 tokenizer，不负责下载完整训练权重 |

末尾 CLI 标量参数可覆盖默认值；`--input` 是追加而不是替换，换单输入应设置 INPUT_FILE。输入文件名是 val/validation/test 时会优先保留既有问题划分。不能用官方 benchmark 测试题来补训练样本。

### 32.3.2 理解 prepare 的输出

| 文件 | 用途 |
|---|---|
| train.jsonl / validation.jsonl / test.jsonl | 筛选和划分后的实际实验输入 |
| normalized.jsonl | 所有结构合法的规范化记录，包含超长或 Incorrect 样本及实测长度，供后续重新筛选 |
| measurements.jsonl | 每条长度、调用统计和拒绝原因 |
| rejected.jsonl | 拒绝记录 |
| manifest.json | 来源指纹、tokenizer、参数、规模与长度分布 |

`MAX_LENGTH=16384` 限制实际训练划分，不意味着 normalized.jsonl 只有 16K 内的记录。因此后续可以从同一个完整 normalized 重新筛选更长样本，无需重新分词；前提是原 prepare 确实留下完整 normalized。

## 32.4 第三步：按实验目标筛选，并检查样本

首次复现基线可直接用默认 prepared 数据。想专门研究压缩，可要求有 observation 的轨迹；无 observation 样本可以训练任务行为，但没有文档压缩区可供学习。比较 Plain/Beacon 时必须使用相同筛选后的训练数据。

### 32.4.1 可选：只保留有观察的记录

```bash
# 后一个路径必须不存在；运行后将训练脚本 dataset 改成该输出目录。
bash scripts/sh/data/13_filter_observations.sh \
  outputs/data/openseeker/polar_correct_qwen35_16k \
  outputs/data/openseeker/polar_correct_qwen35_16k_obs_run1
```

13 不重新分词，保留 split，逐记录保留内容；剔除记录和审计信息写入 removed/、filter_report.json。新目录不一定有完整 normalized.jsonl，因此需要进一步长度筛选时，优先回到原 prepare 目录的 normalized。

### 32.4.2 可选：筛选其他长度或质量区间

编辑 `scripts/sh/data/11_select.sh` 顶部的 PREPARED、OUTPUT、TOKENIZER、MIN_LENGTH、MAX_LENGTH、CORRECTNESS、OBSERVATIONS，再运行：

```bash
bash scripts/sh/data/11_select.sh
```

注意 11 的当前示例默认是 **32768—33792、有 observation**，不是默认 16K 训练集，不能不看参数直接运行再接到 16K 训练。需要正常划分时使用 `SPLIT_MODE=grouped`；`train` 将全部样本写入训练集，只用于显存等诊断。新数据长度上限不能大于训练 max_length。

### 32.4.3 抽查真实轨迹

```bash
DATASET_PATH=outputs/data/openseeker/polar_correct_qwen35_16k \
LIMIT=20 OUTPUT_HTML=outputs/reports/beacon_train_samples.html \
  bash scripts/sh/report/40_dataset_html.sh
```

查看原 system、问题、search/visit 调用、完整工具观察、最终 answer，以及 train/validation/test 数量。只有 `<tool_response>` 正文被标记为可压缩区；工具调用、问题和 assistant 推理不压缩。answer 参考字段不会被追加到测试提示中。

**完成条件：** 选定一份 prepared 目录；确认 manifest/tokenizer、完整序列长度、筛选条件和各 split；把目录记录为本实验的数据来源。

## 32.5 第四步：修改 Beacon 配置，先建立固定比例基线

日常训练参数改 `scripts/sh/beacon/30_beacon_train.sh`，四卡入口 38 也读取它。训练、Beacon 和 DeepSpeed 参数均在脚本临时 YAML 中，不读取外部基础配置；脚本保存的配置及启动器覆盖规则见 [53](../tools/53_self_contained_training.md)。

修改顺序：

1. `dataset` 改成第二/三步最终选定的 prepared 目录。
2. `models.qwen35` 与 prepare 的 tokenizer 身份一致；换 Qwen3 时同时改 model_family、模型和数据。
3. `training.max_length` 至少覆盖 prepared manifest 的上限。
4. 保持 `beacon_compression_mode: fixed`、`beacon_ratio: 16`，先验证默认训练和生成行为。
5. 根据资源选择单卡或多卡，再决定训练步数、保存与验证间隔。

| 训练参数 | 当前基线值 | 解释与修改位置 |
|---|---|---|
| model_family / kind / data_mode | qwen35 / beacon / openseeker | 30 内 YAML；不要用 Search-R1 标签数据替代 OpenSeeker |
| dataset | 默认 16K prepared 目录 | 必须同时有 manifest 和 split 文件；不能只给 train.jsonl |
| learning_rate | 5e-5 | 首次保留基线；评估退化时使用独立实验比较，不在同 run 内改学习率 |
| num_epochs / max_train_steps | 1 / null | null 按 epoch；正整数是 optimizer steps，适合短训练 |
| per_device_batch_size | 1 | Beacon 支持单轨迹 batch；用梯度累积增加有效 batch |
| grad_accum_steps | 4 | 有效 batch = GPU 进程数 × 1 × 累积数；四卡默认约 16 |
| seed | 42 | 训练随机数，与数据分组 seed 分别记录 |
| use_lora / lora_r / lora_alpha | true / 16 / 32 | 基座主体冻结，LoRA 与 Beacon 参数训练；关 LoRA 会改变资源需求 |
| save_freq_steps / eval_steps | 200 / 200 | optimizer step 间隔；总训练不足间隔时不会有对应周期事件，最终仍保存 |
| logging_steps / log_freq_steps | 5 / 5 | 日志间隔；短试验可改 1 |
| beacon_compression_mode | fixed | 在 30 的 `beacon` 段设置 fixed / random |
| beacon_ratio | 16 | 控制固定训练及生成比例 |
| beacon_compression_ratios | 8 16 32 64 | 仅 random 训练采样候选；推理仍用 beacon_ratio |
| beacon_information_mode | whole | 每个 observation 整段读取后压缩 |
| beacon_keep_window | 0 | 默认整段 keep；`beacon_layout_policy: configured` 时可设置正数分块 |
| beacon_attention_backend | sdpa | 减少显式注意力矩阵开销；不保证所有硬件组合都走 FlashAttention |
| beacon_checkpoint_blocks | true | 重算纯 attention/MLP/reader/writer 块，降低保存激活需求 |
| beacon_cpu_offload_activations | true | 用 CPU 内存换 GPU 激活保存，须考虑多 rank 主机内存总量 |
| beacon_loss_chunk_size / checkpoint_loss | 32 / true | 分块词表 loss 与 loss 重算；不同于整层 checkpoint |

### 32.5.1 当前究竟训练哪些损失

默认 `beacon_ce_loss_weight=1.0`，各 assistant 分段权重为 1。压缩后的推理仍用真实轨迹前缀进行 teacher forcing，不是训练时自由生成。

默认 `beacon_output_distill_weight=0.0`、`beacon_readout_distill_weight=0.0`，不启用教师模型蒸馏。虽然 continuation 权重写为 0.1，当前 `beacon_information_loss=false` 会关闭该信息续写分支；不能只看一个权重就判断它实际参与训练。

启用蒸馏或信息续写属于新实验，先读 30/62/63，明确 teacher 路径、额外模型资源和损失门控。首次跑通无需开启。

## 32.6 第五步：生成 plan，检查启动方式

**单卡：**

```bash
CUDA_VISIBLE_DEVICES=0 ACTION=plan bash scripts/sh/beacon/30_beacon_train.sh
```

**四卡 A800：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ACTION=plan \
  bash scripts/sh/beacon/38_beacon_4xa800.sh
```

两者都只生成新实验目录、配置与启动命令，不加载训练模型。每次调用 30/38 都创建一个新 run。屏幕 JSON 中的 run_dir 和 next 必须记录下来。

检查生成的：

```text
outputs/experiments/<本次 run>/
├── source_config.yaml     入口配置
├── train.yaml             脚本内训练参数和实际布局解析后的配置
└── experiment.json        数据/代码指纹、模型、训练与评测配置
```

确认 dataset、tokenizer 身份、max_length、训练步数、ratio 和损失符合预期。默认 `beacon_layout_policy=configured`，布局由脚本字段决定；设为 `whole` 会强制兼容整段布局。DeepSpeed 配置来自脚本 `distributed` 段，通过打印的命令内联传递，详见 [53](../tools/53_self_contained_training.md)。

| 运行变量 | 用法与注意事项 |
|---|---|
| PYTHON / CONDA_ENV | 30/38 支持同名环境变量覆盖；公共解析规则见 01 |
| CUDA_VISIBLE_DEVICES | GPU 列表；单卡一张、多卡列多张 |
| NPROC_PER_NODE | 未设时按可见卡推断；遗留的值可能覆盖推断，手动设置须匹配 GPU 数 |
| DISTRIBUTED_BACKEND | 多卡默认 zero2，可选 ddp；Beacon 拒绝 zero3 |
| ACTION | plan 只归档；train 启动；30/38 默认 train |

单进程 launcher 不因为环境变量写 zero2 就自动变成多卡训练。四卡 DDP/ZeRO-2 都不能把四张卡显存简单相加当作一条轨迹可用显存；单条前向激活仍在某个 rank 上。

**完成条件：** 已有一个审查通过、尚未训练的 RUN_DIR。修改代码或数据后应重新建 plan，不编辑旧指纹绕过一致性检查。

## 32.7 可选：正式训练前做独立显存试验

若资源充足且默认长度已验证，可以跳过此阶段。对于未验证的长轨迹，按 51/52 先试验；显存试验也是实际训练，会加载权重、执行反向并更新小实验的参数，不是 dry-run。

12/37 当前示例是约 64K 的长样本、5 个 optimizer steps，**不应当作比 16K 更轻的测试**。需要验证 16K 时，先把 12 的筛选区间和 37 的 max_length 改为对应长度，并让 12.OUTPUT 与 37.dataset 一致；样本要有 observation，且数量足够。

```bash
# 完成上述配置修改、输出目录不存在时，筛选显存样本：
bash scripts/sh/data/12_select_memory_probe.sh

# 先审查独立试验配置：
CUDA_VISIBLE_DEVICES=0 ACTION=plan bash scripts/sh/beacon/37_beacon_memory_probe.sh

# 再由使用者在目标 GPU 上执行实际短训练：
CUDA_VISIBLE_DEVICES=0 bash scripts/sh/beacon/37_beacon_memory_probe.sh
```

37 与 30 的 YAML 独立，修改 30 不会自动修改 37。显存试验会新建独立 run，不应把它的最终模型当作正式训练结果。出错看对应模型目录 failure.json、训练日志和 GPU/CPU 内存；不要通过静默截断输入来“解决”。

若试验后改了正式配置或代码，回到第五步重新 plan。

## 32.8 第六步：启动正式训练并检查产物

在第五步得到的 run 上开始训练，使用屏幕给出的 next 命令；例如：

```bash
# 替换为刚才 plan 实际生成的绝对目录，不要自动取“最新目录”。
RUN_DIR=/absolute/path/to/outputs/experiments/your_beacon_run

# 单卡，使用该 plan 的固定配置：
CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"

# 若选择的是四卡 plan，执行这一条而非上面的单卡命令：
# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 DISTRIBUTED_BACKEND=zero2 \
#   "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"
```

也可以直接 `bash scripts/sh/beacon/30_beacon_train.sh` 或 `bash scripts/sh/beacon/38_beacon_4xa800.sh` 自动新建 run 并训练，但这不是继续第五步的同一目录。两种启动方式选一种。

`train --run-dir` 是从审查好的 plan 开始，不是支持任意中断 checkpoint 的 optimizer 恢复入口。已有 models/model 的训练目录会拒绝再次启动，避免覆盖；失败后先判断问题，改配置时建立新实验。

训练中查看 run 根目录 train_*.log、events.jsonl，以及：

```text
RUN_DIR/models/model/
├── resolved_config.json         训练器实际解析配置
├── trainer_metrics.jsonl        loss/学习率等日志
├── checkpoint-<step>/           周期 adapter 与 Beacon 参数
├── final_adapter/               训练结束的 LoRA adapter 与 Beacon 参数
├── final/                       自动合并后的完整推理模型
└── train_summary.json           total_steps、final_model_path 等
```

保存/验证周期按 optimizer steps 计算；小训练不一定产生 checkpoint-200。loss 降低不等于自由生成或搜索能力已经正常，应继续做后续评测。

正常训练结束会自动合并 LoRA + Beacon 状态到 final，随后写 train_summary.json。**完成条件：** 进程正常退出、events 返回码正常，final 有完整 config/tokenizer/权重，train_summary.final_model_path 指向该 final。

## 32.9 第七步：仅在需要时手动合并模型

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

只有 adapter_config.json 的目录不能直接交给 31/47/56/57 推理。手动合并不会补写因训练收尾失败而缺失的 train_summary.json；此时评测显式指定 `MODEL_PATH`、`KIND=beacon`，在线/固定语料还应指定 `PROMPT_DATA`。

## 32.10 第八步：先做少量生成与工具诊断

31 的参数都可以通过同名环境变量覆盖。先 dry-run 审查模型和数据路径：

```bash
RUN_DIR="$RUN_DIR" SPLIT=test LIMIT=2 DRY_RUN=1 \
  bash scripts/sh/beacon/31_beacon_eval.sh
```

31 默认 BACKEND=replay。它只有在生成的动作与历史记录精确一致时才返回记录中的 observation；新查询会 replay_miss。因此 replay 可查格式、动作匹配与拟合，不是真实搜索准确率，不能因为 replay_miss 就断言模型不会研究问题。

```bash
# 只做历史动作匹配诊断；输出新 predictions 文件。
RUN_DIR="$RUN_DIR" SPLIT=test LIMIT=2 BACKEND=replay \
  bash scripts/sh/beacon/31_beacon_eval.sh
```

要允许模型自由生成查询，选择下列之一：

| 后端 | 入口 | 使用方式 |
|---|---|---|
| corpus | 31，BACKEND=corpus | 设置独立 CORPUS JSONL；15 可从历史 visit 轨迹导出证据库，须说明来源划分，不能当独立 benchmark |
| HTTP | 31，BACKEND=http | 设置 ENDPOINT；服务接受 name/arguments，返回 content 字符串；不由 31 自动部署 |
| 固定语料基准 | 56/57 | 推荐接下一节，真实离线搜索，不需要在线 API |
| 在线 BrowseComp | 47 native | 见第十步，需要配置真实工具和裁判 API |

31 的 LIMIT 必须是正整数（默认 100），与 47/56/57 的 -1=全量约定不同。31 默认 MAX_CONTEXT/MAX_NEW_TOKENS/MAX_TOOL_TOKENS/MAX_TURNS 为 16384/8192/16384/32；测试更长历史时显式提高对应预算，并对比较模型保持相同设置。

## 32.11 第九步 A：固定语料离线评测

详细准备细节、下载失败回退和 corpus 来源必须阅读 43/44。这里给出阶段顺序。两个基准各用独立 ROOT_DIR；数据下载与索引通常只做一次，同一索引可比较多个模型。

### 32.11.1 HotpotQA：先跑不需要裁判模型的流程

```bash
# 联网准备机执行；后续阶段不联网。
ACTION=download bash scripts/sh/benchmark/57_hotpotqa_offline.sh

# 目标机，原始数据已就位后：
ACTION=prepare bash scripts/sh/benchmark/57_hotpotqa_offline.sh
ACTION=index bash scripts/sh/benchmark/57_hotpotqa_offline.sh

# 先检查少量题的完整生成与评分；不用另部署 API 或摘要模型。
ACTION=all RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/benchmark/57_hotpotqa_offline.sh

# 小样本流程和输出确认后，全量重新建结果目录：
# ACTION=all RUN_DIR="$RUN_DIR" LIMIT=-1 bash scripts/sh/benchmark/57_hotpotqa_offline.sh
```

Hotpot 使用官方有标签 dev、独立 2017 Wikipedia 摘要库，不能把题目附带 context 当作全库。索引不是网页全文，也不是实时 Wikipedia。其 EM/F1、支持事实和 Joint 指标与在线 BrowseComp 裁判准确率不可互换。

### 32.11.2 BrowseComp-Plus：生成与裁判评分分开

```bash
# 联网下载；全量 corpus 可能较大，提前按 43/44 准备空间。
ACTION=download bash scripts/sh/benchmark/56_browsecomp_plus.sh
ACTION=prepare bash scripts/sh/benchmark/56_browsecomp_plus.sh
ACTION=index bash scripts/sh/benchmark/56_browsecomp_plus.sh

# 只生成，暂不加载本地裁判：
ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/benchmark/56_browsecomp_plus.sh
```

记录屏幕打印的本次输出目录，设置 GEN_DIR 后单独评分，避免待测模型与大裁判同时占用 GPU：

```bash
GEN_DIR=/absolute/path/to/the/generated/offline_result
ACTION=score PREDICTIONS="$GEN_DIR/predictions.jsonl" \
JUDGE_MODEL=/models/Qwen3-32B LIMIT=8 \
  bash scripts/sh/benchmark/56_browsecomp_plus.sh
```

评分的 ROOT_DIR、题目顺序和 LIMIT 必须与生成一致。score 写新结果目录，不能把 OUTPUT_DIR 指回已存在的 GEN_DIR；评分时不必设置 RUN_DIR。完整答案评分需要本地裁判；RETRIEVAL_ONLY=1 只输出检索诊断，不能报告为答案准确率。

默认 BCP 是 search-only（ALLOW_VISIT=0），Hotpot 允许 local visit（ALLOW_VISIT=1）。提示词来自原训练 system，离线只调整实际能力、本地语料约束和最终输出格式。数据标注、qrels 和参考答案只用于评分，不进入模型问题。

### 32.11.3 离线参数修改与多卡

| 变量 | 默认 | 修改提示 |
|---|---|---|
| ACTION | all | download/prepare/index/run/score/all；all 不负责下载 |
| RUN_DIR | 空 | 正常训练完成后自动定位 final 和原训练提示 |
| MODEL_PATH / KIND | 空 / 自动 | 手动合并或中间模型时显式填目录和 beacon |
| PROMPT_DATA | 空 | 优先指定原训练 JSONL；否则从 RUN_DIR 定位；仅取 system/schema |
| TOPK / SNIPPET_TOKENS / READ_TOKENS | 5 / 512 / 4096 | 每查询返回数和每篇正文预算 |
| MAX_CONTEXT / MAX_NEW_TOKENS | 131072 / 8192 | 完整历史逻辑预算和每轮生成上限，不等于物理 KV 长度 |
| MAX_TOOL_TOKENS / MAX_TURNS / TOOL_COUNT | 32768 / 200 / 200 | 单条工具返回、生成循环和工具总次数限制 |
| LIMIT | -1 | 全量；先设 8 诊断，不把子集成绩报告为全量 |
| NUM_WORKERS | 1 | 每卡一个完整模型副本，按题并行；不是 tensor parallel |
| OUTPUT_DIR | 自动新目录 | 不覆盖；Plain/Beacon 每次各写新结果 |

```bash
# 每张卡必须能单独容纳完整 Beacon 模型；训练时用几卡与此独立。
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_WORKERS=4 ACTION=run \
RUN_DIR="$RUN_DIR" LIMIT=8 bash scripts/sh/benchmark/56_browsecomp_plus.sh
```

若 RUN_DIR 的原训练文件迁移后不可访问，增加 `PROMPT_DATA=/new/path/train.jsonl`；不要换成测试集。一个训练文件含多个不同 system 时，用只含目标 system/schema 的单 profile JSONL 显式指定，详见 45。

## 32.12 第九步 B：在线 BrowseComp 评测

此路线与离线基准并列，不是完成离线评测的前提。Beacon 使用 **47 的 native 模式**直接加载模型；不使用普通 SGLang 服务替代 Beacon 压缩。

### 32.12.1 准备官方代码、题目和私有 API 配置

```bash
# 下载固定版本 upstream 与 BrowseComp，未运行模型/裁判。
bash scripts/sh/benchmark/46_prepare_benchmark.sh

# 将配置复制到仓库外，再编辑占位符。
mkdir -p "$HOME/.config/searchagent"
cp scripts/sh/benchmark/49_benchmark_env.example.sh "$HOME/.config/searchagent/benchmark.env.sh"
```

填写私有文件里的 SEARCH_PROVIDER 与搜索 key、JINA_API_KEYS、SUMMARY_API_URL/KEY/MODEL、SCORER_URLS/KEY/MODEL。native 需要真实搜索、网页读取、摘要和裁判服务；不需要 OPENSEEKER 模型服务或 E2B。地址必须从运行 47 的节点可达，127.0.0.1 指当前节点。

首次还需按 42 安装在线专用依赖（requirements.eval-openseeker.txt 及 upstream 所需工具依赖）。不要把真实 key 写到 Git 或复制到实验说明。

### 32.12.2 少量题，再扩展到全量

```bash
RUN_DIR="$RUN_DIR" MODE=native LIMIT=2 DRY_RUN=1 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 实际访问在线服务，可能产生 API 费用：
RUN_DIR="$RUN_DIR" MODE=native LIMIT=2 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 确认模型轨迹、服务及裁判覆盖后再全量：
# RUN_DIR="$RUN_DIR" MODE=native LIMIT=-1 \
# ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
#   bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

RUN_DIR 自动选择模型和原训练提示。若使用手动合并的模型或迁移了文件：

```bash
MODEL_PATH="$RUN_DIR/models/model/final_recovered" KIND=beacon MODE=native \
PROMPT_DATA=/data/original_train.jsonl LIMIT=2 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

47 native 的 MAX_TOKENS 默认 32768（每轮）、MAX_CONTEXT=131072、MAX_TURNS=200、MAX_TOOL_TOKENS=32768。WORKERS 是官方服务模式的生成并发，不让 native 自动多卡按题并行；JUDGE_WORKERS 控制在线裁判并发。TOOL_COUNT 在 47 沿用官方模式参数，native 的循环上限由 MAX_TURNS 控制。

47 原样 official-v1/v2 与 48 Plain 服务入口仍保留，供另做官方实现对照；它们不是 Beacon 的推理路线。

## 32.13 第十步：查看轨迹、解释指标并归档

| 评测入口 | 先看什么 | 如何解释 |
|---|---|---|
| 31 | predictions_*.jsonl 与同名 .metrics.json | EM/F1、失败状态与使用的 backend；replay 仅作动作诊断 |
| 56/57 | predictions.jsonl、manifest.json、summary.metrics.json、per_question.json | 生成完成率、预算/工具失败、检索与任务指标；只有 run 时还没有完整评分文件 |
| 47 | result_tool*.jsonl、coverage.json、summary.metrics.json、judge.log | 在线轨迹与裁判覆盖；不能只报告成功题分母 |

编辑 41 的 INPUTS 指向本次真实预测路径，再运行：

```bash
bash scripts/sh/report/41_predictions_html.sh
```

41 的默认文件名不一定存在；先看实际输出路径。42_summarize_results.sh 同样要编辑 RESULT_PATHS/OUTPUT_PATH，适用于既有 EM/F1 预测汇总；BCP/Hotpot 的正式指标以各自官方评分输出为准。

检查顺序：

1. 是否产生非空最终回答，status 是否 completed？
2. search/visit 格式是否正确，有没有 invalid_call、tool_error 或 replay_miss？
3. 是否重复推理并耗尽 generation_limit？先读轨迹，不用盲目增大 token 预算。
4. 是否真正检索过，query 是否改变，返回了哪些证据？
5. 生成正常后，才分析召回率、答案准确率、引用或支持事实。

保存 RUN_DIR、数据/tokenizer/代码指纹、实际配置、模型路径、提示词来源/hash、语料与索引身份、预算、工具/裁判设置、题目数和失败分母。Plain/Beacon 对照应固定这些条件；改变数据长度、工具环境或压缩比时建立新结果目录。

## 32.14 按顺序执行的最短清单

| 顺序 | 必做/可选 | 执行与产物 |
|---|---|---|
| 1 | 必做 | 01 环境、模型权重/tokenizer；59 可先做 CPU 检查 |
| 2 | 必做判断 | 已有合适 prepared 则复用；否则配置并运行 10 |
| 3 | 可选 | 11 长度筛选 / 13 observation 过滤；40 抽查样本 |
| 4 | 必做 | 编辑 30 的 dataset、模型、长度、压缩比、步数 |
| 5 | 必做 | ACTION=plan 运行 30 或 38，记录实际 RUN_DIR |
| 6 | 可选 | 12+37 独立显存试验，必要时读 51/52 |
| 7 | 必做 | 用 51_experiment.py train --run-dir 启动审查好的计划 |
| 8 | 必做检查 | train_summary + final；仅自动合并失败/中间 checkpoint 才运行 35 |
| 9 | 可选诊断 | 31 少量生成；明确 replay/corpus/HTTP 的意义 |
| 10 | 选择评测路线 | 56 BCP、57 Hotpot 或 46+49+47 在线 BrowseComp；先少量题再全量 |
| 11 | 必做 | 查原始轨迹与官方指标，41 导出 HTML，记录可比实验条件 |

真正开始训练前，优先确认数据路径、tokenizer 身份、max_length、训练步数与模型资源；真正报告成绩前，优先确认生成是否完成、评测来源和失败覆盖率。
