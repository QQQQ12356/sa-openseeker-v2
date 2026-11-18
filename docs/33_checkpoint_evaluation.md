# 33 从训练检查点运行 Benchmark

Plain 和 Beacon 的 BrowseComp、BrowseComp-Plus、HotpotQA 测试入口均支持选择训练中的 `checkpoint-<step>`。OpenSeeker、Research 和混合数据实验使用同一套测试脚本。数据身份及基座由所属训练实验确定，混合实验通过 `SA_PROMPT_DATA` 显式选择评测提示词。

## 1. 选入口并填写参数

| Benchmark | Plain | Beacon | 准备条件 |
|---|---|---|---|
| BrowseComp | `scripts/plain/sh/40_test_browsecomp.sh` | `scripts/beacon/sh/40_test_browsecomp.sh` | 准备题目和固定版本的在线工具，配置搜索、读取与总结服务 |
| BrowseComp-Plus | `scripts/plain/sh/42_test_browsecomp_plus.sh` | `scripts/beacon/sh/42_test_browsecomp_plus.sh` | 准备题目、本地语料和索引 |
| HotpotQA | `scripts/plain/sh/44_test_hotpotqa.sh` | `scripts/beacon/sh/44_test_hotpotqa.sh` | 准备 fullwiki 题目、独立维基语料和索引 |

| SH 参数 | 默认值 | 作用与修改提示 |
|---|---|---|
| `SA_TRAINING_RUN` | 当前方法的 OpenSeeker 实验占位路径 | 必填。填训练实验根目录，目录内应有 `job.json`、`experiment.json` 和 `models/model/`。Research 实验通常位于 `artifacts/experiments/<方法>/research/`。 |
| `SA_CHECKPOINT` | 空字符串 | 留空时读取成功训练的 `models/model/train_summary.json` 中的 `final_model_path`。填 `checkpoint-200` 时使用本实验的 `models/model/checkpoint-200`。也可填指向同一目录的 Linux 绝对路径，或相对项目根目录的完整路径。数字是 optimizer step。 |
| `SA_MODEL_PATH` | 空字符串 | Plain 和 Beacon 的测试脚本中留空。模型来源由训练实验与 `SA_CHECKPOINT` 确定。 |
| `SA_PROMPT_DATA` | 空字符串 | OpenSeeker / Research 实验留空，由训练记录定位并核对训练 JSONL。mixed 实验必填，指向所有行渲染后具有一致 system 和工具定义的规范化 JSONL。只读取 system 和工具定义。 |
| `SA_BENCHMARK_DIR` | 对应 benchmark 的 `v1` 目录 | 填已经准备好的版本。离线测试需要同版本的索引，`SA_GENERATION__TITLE_WEIGHT` 应与建索引时一致。 |
| `SA_RUN_NAME` | `full` | 给测试记录起名，可用 `ckpt200_smoke` 或 `ckpt200_full`。 |
| `SA_GENERATION__LIMIT` | `-1` | `-1` 测全部题目。新检查点建议先跑前 5 至 10 题验证加载和工具调用，再切换到全量。 |
| `CUDA_VISIBLE_DEVICES` / `SA_GENERATION__NUM_WORKERS` | `0` / `1` | 在线 BrowseComp 按题顺序运行。两个离线入口可用多个 worker，每个 worker 在一张可见 GPU 上加载完整模型，worker 数应与计划使用的 GPU 数匹配。 |
| `SA_GENERATION__DECODING` | `greedy` | Plain 支持 `greedy` 和 `sample`，Beacon 使用 `greedy`。比较不同训练步数时沿用同一组生成与工具预算。 |

例如，编辑 `scripts/beacon/sh/44_test_hotpotqa.sh` 中对应的参数行：

```bash
export SA_TRAINING_RUN=artifacts/experiments/beacon/research/REPLACE_WITH_RUN
export SA_CHECKPOINT=checkpoint-200
export SA_MODEL_PATH=''
export SA_PROMPT_DATA=''
export SA_BENCHMARK_DIR=artifacts/benchmarks/hotpotqa/v1
export SA_RUN_NAME=ckpt200_smoke
export SA_GENERATION__LIMIT=5
```

填入真实实验目录后，在 Linux 项目根目录运行：

```bash
bash scripts/beacon/sh/44_test_hotpotqa.sh
```

要比较最终模型，在同一个测试入口把 `SA_CHECKPOINT` 改为 `''`，并修改 `SA_RUN_NAME`。要换到 Plain，改用 Plain 入口和 Plain 训练实验。其他测试参数见 [60 参数手册](60_parameters.md)。

## 2. 检查点格式与加载方式

| 训练产物 | 所需文件与配置 | 加载方式 |
|---|---|---|
| Plain 全量模型 | `config.json`、模型权重或权重索引及全部分片、tokenizer | 直接加载目录中的权重。 |
| Plain LoRA | `adapter_config.json`、`adapter_model.safetensors` 或 `adapter_model.bin`、原训练基座 | 加载原训练基座并挂载 LoRA。训练配置优先提供基座路径。 |
| Beacon 全量模型 | Beacon 模型的 `config.json`、模型权重或权重索引及全部分片、tokenizer | 直接恢复 Beacon 权重与模型配置。 |
| Beacon LoRA | LoRA 配置和权重、`beacon_state.pt`、原训练基座及 Beacon 配置 | 使用实验归档的配置构建 Beacon 模型，恢复 `beacon_state.pt`，再挂载 LoRA。 |

LoRA 测试优先使用检查点目录中的 tokenizer，缺少 tokenizer 时使用原训练基座的 tokenizer。Plain 和 Beacon 的 LoRA 检查点可以直接推理，无需先合并。优化器和调度器状态不参与 benchmark 推理。

显式指定周期检查点时，只要目录已经写完且文件齐全，训练处于 `running` 或 `failed` 也能测试。计划运行（`job.json` 的 `operation` 为 `plan`）没有检查点。训练仍在运行时，选择已保存完毕且在评测加载期间不会被 `SAVE_TOTAL_LIMIT` 清理的检查点。离线多 worker 会分别加载同一个选定目录。

## 3. 核对结果并评分

1. 测试仍写入 `<实验>/evaluation/<benchmark>/<测试记录>/`。查看 `evaluation.json` 的 `checkpoint` 和 `model_path`，核对选择值与解析后的周期检查点路径。最终模型模式下可在 `results/manifest.json` 的 `model` 查看实际权重目录。
2. 查看 `console.log` 中的加载信息。找不到检查点时，列出实验的 `models/model/`，确认步数实际保存且还在保留范围内。缺少 adapter 权重、Beacon 状态或权重分片时，请恢复完整的检查点目录。
3. 找不到基座或提示词数据时，核对 `models/model/resolved_config.json` 和 `experiment.json` 中记录的路径，确保原始基座与 prepared 训练 JSONL 在当前 Linux 环境可读。
4. 评分仍使用同方法、同 benchmark 的 `41/43/45_score_*.sh`。把**测试记录根目录**填进 `SA_EVALUATION_DIR`，评分会沿用该次测试的预测和题目范围。对同一 benchmark 比较不同检查点时，各次测试使用独立记录。

完成评分后运行 `scripts/tools/sh/81_prediction_report.sh` 可查看逐题 LLM Judge 结果。`SA_SCORE_DIR` 留空时选择该测试最新完成、有逐题 Judge 文件且预测文件指纹一致的评分记录；填写时必须指向某次评分根目录（含 `job.json`、`results/`），程序校验方法、benchmark、测试目录与预测指纹，不会合并其他测试的评分。没有匹配评分时仍生成轨迹报告，并显示“未评分”。报告展示判定、判定来源、Judge 自报置信度（如有）、理由（如有）、原始输出及重试记录，支持按正确、错误、判定失败、未评分筛选。顶部 Judge 正确率只统计有效 0/1 判定，同时单列失败和未评分数；这不是评分脚本中包含所有题目的官方准确率分母。原预测文件及其 EM/F1 不会被改写。

需要排查 Judge 错误时，`SA_SCORE_DIR` 也可显式指定已经结束但状态为 `failed` 的评分任务，前提是已写出逐题 Judge 文件且来源校验通过。运行中的任务不可选择；失败任务中尚未写出的题目显示“未评分”。自动选择仍只使用成功完成的评分任务。

## 4. Plain 多轮推理缓存

Plain 测试自动复用同一题内的推理缓存，无需设置 SH 参数。工具返回后只编码新增消息。上一轮缓存对应的 token 与本轮输入前缀完全一致时，模型只对未缓存的 token 执行 prefill。若输出文本重新编码后改变了前缀，当前轮会重新 prefill。切换题目时清空缓存。

在线 BrowseComp 的 `console.log` 中，`input_tokens` 是该轮完整输入长度，`cached_input_tokens` 是复用的前缀长度。首轮或前缀不匹配时为 `0`。三个 benchmark 的预测记录都保存每道题累计复用的 token 数，可用它判断缓存是否命中。

Plain 和 Beacon 的生成停止检查使用 Transformers `StopStringCriteria`，按词表预计算的匹配关系检查生成 token 的有限后缀，避免每生成一个 token 都把本轮完整输出复制到 CPU 并解码。结束标记跨 token、与相邻文本合并时仍能匹配；Plain 只检查本轮新增输出，提示词内的标记不会触发停止。EOS、生成预算和最终输出提取规则保持不变。仅对没有 Hugging Face 词表接口的简化测试 tokenizer 保留原有解码检查。

## 5. 查看测试进度

三个 benchmark 在终端和测试目录的 `console.log` 使用相同的完成进度格式，例如 `[benchmark browsecomp-plus generation] done=12/50 status=completed sample=8.4s elapsed=103.2s`。

- `done` 是已写入的预测数和本次选中的总题数，分母受 `SA_GENERATION__LIMIT` 影响。遇到工具失败或达到预算的题也会写入预测并计入完成数，具体结果看 `status`。
- `sample` 是该题的耗时，`elapsed` 是生成进度开始以来的累计耗时。离线索引校验会先显示 `checking local index`，长题运行期间完成数会停在上一条。
- 离线多卡运行时，终端的 `done` 汇总所有 worker。各卡的日志保存在 `results/workers/worker_<rank>.log`。
