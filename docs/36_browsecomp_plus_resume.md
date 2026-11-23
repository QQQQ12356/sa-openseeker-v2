# 36 测试续测

所有数据评测入口都可把一次已完成测试扩展到更多题目。例如原测试完成前 40 道题，目标改为 60 后，新任务复用前 40 条预测，只生成新增的 20 道题。新目录包含完整的 60 题预测。

| 数据范围 | 测试入口 | 合并后评分 |
|---|---|---|
| BrowseComp | 各方法的 `40_test_browsecomp.sh` | 同方法的 `41_score_browsecomp.sh` |
| BrowseComp-Plus | 各方法的 `42_test_browsecomp_plus.sh` | 同方法的 `43_score_browsecomp_plus.sh` |
| HotpotQA | 各方法的 `44_test_hotpotqa.sh` | 同方法的 `45_score_hotpotqa.sh` |
| 训练数据或同来源保留划分 | `scripts/tools/sh/84_test_training_data.sh`，支持 train、validation、test | 生成后自动汇总合并结果的 EM/F1 |

前三项的「各方法」指 `scripts/plain/sh/`、`scripts/beacon/sh/` 和 `scripts/baseline/sh/`。续测用原来的同一个入口。

BrowseComp-Plus 的 42 若在生成期间中断，可用 [85 恢复评分](#85-从中断的-browsecomp-plus-测试恢复评分) 汇总已写入的预测并计算临时指标。

## Linux 上从 40 题补到 60 题

1. 找到原测试根目录。正式 benchmark 的目录包含 `job.json`、`evaluation.json` 和 `results/`。84 脚本的预测文件与 `summary.metrics.json` 在根目录。原测试须已完成生成并完成归档，有无既有评分均可。
2. 打开原测试所用的 SH。保留模型、题集、提示词、检索器、工具和解码配置。若刚切换 Git 版本，可对照原测试归档中的 `source_launcher.sh` 或 `evaluation.json`。
3. 修改三项变量：

```bash
export SA_RESUME_FROM='/path/to/evaluation/browsecomp-plus/original-run'
export SA_GENERATION__LIMIT=60
export SA_RUN_NAME=extend_40_to_60
```

4. 运行原测试所用的脚本。例如 Plain 的 BrowseComp-Plus：

```bash
bash scripts/plain/sh/42_test_browsecomp_plus.sh
```

5. 日志应显示 `reuse 40 predictions; generate 20 additional questions`。正式 benchmark 测试成功后，将**新测试根目录**填进对应评分脚本的 `SA_EVALUATION_DIR`。84 脚本直接查看新目录中的 `summary.metrics.json`。

## 变量说明

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_RESUME_FROM` | 空 | 续测来源。填写一个已完成的测试根目录，可用项目相对路径或 Linux 绝对路径。需要从第 1 题新跑时留空。 |
| `SA_GENERATION__LIMIT` | 正式 benchmark 为 `-1`，84 为 `10` | 续测后的累计题数上限。正式 benchmark 按选题顺序，84 按划分文件顺序。原测 40、目标 60 时填 `60`。设 `-1` 会扩到全部可用题。若无新增题则停止。 |
| `SA_RUN_NAME` | 正式 benchmark 为 `full`，84 为 `smoke` | 新运行标签，可体现来源题数与累计目标，程序追加时间和随机后缀。 |
| `SA_GENERATION__NUM_WORKERS` | `1` | BrowseComp-Plus 和 HotpotQA 可按本次资源调整，只分配新增题目。其他生成预算沿用原测试。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可按资源调整，worker 数不能超过已分配的可见 GPU。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 离线流程可调整本次是否重新扫描索引 SHA256。索引清单与原测试一致性的核对始终执行。 |
| `SA_EVALUATION_DIR` | 评分脚本中的占位目录 | 正式 benchmark 评分时填写新测试根目录，自动按累计题数和原筛选规则重建题集。 |

## 哪些设置要保持一致

- **模型和提示词。** 选择同一个方法及训练实验或基座，解析到同一个本地检查点路径，并保持该目录内容原样。程序通过路径核对模型，旧测试没有权重内容指纹。最终渲染的 system prompt 会与原逐题预测记录的 SHA256 对比。
- **题目和重合筛选。** 使用同一份题目文件或训练数据划分。BrowseComp 和 BrowseComp-Plus 启用重合筛选时，程序复制并复用原测试的 `overlap.json`。原题目序列必须恰好是新选择的有序前缀。HotpotQA 和未启用重合筛选的测试沿用题目源顺序。
- **检索与推理。** 离线评测使用同一检索器及索引。BrowseComp-Plus 的 Qwen3-Embedding-8B 模式还要沿用原向量索引、模型路径和设备。在线 BrowseComp 沿用相同搜索服务及网页摘要服务配置。在线结果会随时间变化，合并结果包含两次运行各自获得的网页观测。TopK、工具预算、解码和随机种子等沿用原配置。
- **训练数据评测。** `SA_RESUME_FROM` 须指向同一训练实验下的 `evaluation/training-data/<任务>`，并保持 `SA_DATASET_DIR`、`SA_SPLIT`、`SA_INDEX_DIR` 不变。续测按样本逐条核对系统提示词。
- **原始预测。** 原预测文件和运行清单的 SHA256 须匹配归档。程序逐行核对题目 ID、问题、比较答案、顺序与提示词。本功能面向已完成的测试扩容。原测试中已经归档的超限、工具错误或空回答也会沿用。

## 新目录与评分

| 文件 | 续测时的含义 |
|---|---|
| `evaluation.json` | 新测试配置、原测试路径及合并后预测文件的 SHA256。 |
| `results/manifest.json` | 完整题目列表、模型和索引配置。其 `resume` 字段记录原测试目录、复用题数、新生成题数及原结果指纹。 |
| `results/predictions.jsonl` | BrowseComp-Plus 和 HotpotQA 合并结果。BrowseComp 使用 `results/result_tool<N>.jsonl`，84 使用根目录的 `predictions.jsonl`。示例中共 60 行，`sample_index` 从 0 连续到 59。 |
| `results/extension/` | 仅本次新增题目的预测和运行记录。84 脚本使用根目录下的 `extension/`。 |
| `results/generation.metrics.json` | 离线 benchmark 的合并状态计数和本次续测耗时。BrowseComp 可查看 `results/coverage.json`，84 可查看根目录的 `summary.metrics.json`。 |
| `overlap.json` | 启用重合筛选时，从原测试复制的报告。 |

评分读取合并后的预测，按原评分流程处理全部目标题目。旧的 40 题评分记录仍归原目录。可从已成功完成的 60 题续测目录再扩到 100 题。若本次续测失败，可查看 `console.log` 和 `results/extension/`，修正后仍从上一个成功完成的测试目录开始。

## 85 从中断的 BrowseComp-Plus 测试恢复评分

`scripts/tools/sh/85_recover_browsecomp_plus.sh` 适用于 Plain、Beacon 和 Baseline 的 **42 BrowseComp-Plus** 测试。它读取已写完的预测，合并后用与 43 相同的评分逻辑计算临时指标。原测试因异常退出而标为 `failed`，或被强制终止后仍显示 `running`，都可以恢复评分。开始前确认 42 的主进程和所有 worker 已退出，避免读取正在写入的文件。已成功完成的测试继续使用 43。

### 运行方式

1. 从 42 的终端日志找到测试记录根目录。该目录应含 `job.json`、`results/manifest.json`，以及至少一个写完一题的预测文件。启用训练重合筛选时保留根目录下的 `overlap.json`。`evaluation.json` 通常要等测试成功才写出。
2. 在 HPC 上打开 `scripts/tools/sh/85_recover_browsecomp_plus.sh`，设置 `SA_EVALUATION_DIR`。只需要 `evidence_recall`、`gold_recall` 等检索指标时，保留 `SA_RETRIEVAL_ONLY=true`。

   ```bash
   export SA_EVALUATION_DIR='artifacts/experiments/plain/openseeker/<训练记录>/evaluation/browsecomp-plus/<测试记录>'
   export SA_RUN_NAME=partial_score
   export SA_RETRIEVAL_ONLY=true
   ```

3. 如需计算答案准确率，把 `SA_RETRIEVAL_ONLY` 改为 `false`，填写 `SA_JUDGE__JUDGE_MODEL`，并将 `CUDA_VISIBLE_DEVICES` 设置为获分配的 GPU。若要与 43 的结果比较，使用相同的裁判快照、预算、采样参数和种子。

   ```bash
   export SA_RETRIEVAL_ONLY=false
   export SA_JUDGE__JUDGE_MODEL='/path/to/local/judge-model'
   export CUDA_VISIBLE_DEVICES=0
   ```

4. 激活原评测的 Linux Python 环境，在项目根目录执行：

   ```bash
   bash scripts/tools/sh/85_recover_browsecomp_plus.sh
   ```

### 恢复评分参数表

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `PYTHON_BIN` | `python` | 使用已激活的 Linux 环境。需要指定解释器时，可填写其绝对路径。 |
| `HF_HUB_OFFLINE` | `1` | 恢复评分使用本地资产和模型。流程会启用离线模式。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 答案评分选择获分配的 GPU。仅计算检索指标时无需 GPU。 |
| `SA_EVALUATION_DIR` | 空 | 必填。填中断的 42 测试根目录，支持项目相对路径和绝对路径。 |
| `SA_RUN_NAME` | `partial_score` | 输出标签，支持字母、数字、下划线和连字符。程序会追加 UTC 时间和随机后缀。 |
| `SA_RETRIEVAL_ONLY` | `true` | `true` 只算检索指标。`false` 同时计算答案评分指标，调用本地裁判。 |
| `SA_JUDGE__JUDGE_MODEL` | 空 | 答案评分必填，指向已下载的本地模型目录。 |
| `SA_JUDGE__JUDGE_ATTEMPTS` | `3` | 每条有效答案的最大裁判尝试次数。 |
| `SA_JUDGE__JUDGE_MAX_TOKENS` | `4096` | 裁判单次输出的 token 上限。 |
| `SA_JUDGE__JUDGE_MAX_CONTEXT` | `32768` | 裁判总上下文 token 上限，需给裁判输出留出空间。 |
| `SA_JUDGE__JUDGE_ENABLE_THINKING` | `false` | 是否启用裁判模型的 thinking 模式，按模型与对照实验设置。 |
| `SA_JUDGE__JUDGE_TEMPERATURE` | `0.0` | 裁判采样温度，填非负有限数。 |
| `SA_JUDGE__JUDGE_TOP_P` | `0.8` | 裁判 nucleus 采样阈值，取值为 `(0, 1]`。 |
| `SA_JUDGE__JUDGE_TOP_K` | `20` | 裁判采样候选数，填正整数。 |
| `SA_JUDGE__SEED` | `42` | 裁判随机种子。 |

`SCRIPT_DIR`、`PROJECT_ROOT`、`SA_ENTRY`、`SA_LAUNCHER` 由启动脚本管理。原测试的 benchmark 目录、`LIMIT`、重合筛选和题目范围从归档读取，运行时不用再次填写。模型目录等裁判选项只在答案评分时生效。

### 预测来源、指标分母和输出

- 如果 `results/predictions.jsonl` 已生成，直接使用它。否则读取 `results/workers/worker_*/predictions.jsonl`。若中断发生在续测中，复用经 SHA256 校验的原测试预测，再读取 `results/extension/` 下已落盘的新预测。
- 汇总时跳过文件末尾缺少换行的记录。遇到重复 ID、原题集之外的 ID、题目文本不一致，或完整行的 JSON 无法解析时停止，并报告文件和行号。确定原 benchmark 和筛选报告保持可读时，才能复用原测试的题集。
- **分母是原计划题数。** 例如原计划 830 题，恢复 600 条记录，`coverage` 为 `600/830`，剩余 230 题的检索召回按 0 计。答案评分只调用裁判处理已有且有效的答案，缺失题仍计入准确率分母。`completed_predictions` 表示有有效答案的题数，可小于已恢复的记录数。评分摘要的 `complete` 表示评分过程成功完成，生成覆盖率见 `recovery.coverage`。
- 85 只读取预测和原 benchmark 资产，重新计算指标时不会启动生成模型或检索模型，也不会补跑缺失题目。每次评分都创建新目录。原测试记录、预测分片和正式评分目录保留原样。

| 文件 | 内容 |
|---|---|
| `recovery/<运行标签>/job.json` | 恢复评分配置、原任务状态和本次运行状态。 |
| `recovery/<运行标签>/console.log` | 各预测文件恢复条数、末行跳过情况、汇总进度及评分日志。 |
| `recovery/<运行标签>/predictions.jsonl` | 汇总后的逐题预测。评分按原题目顺序输出。 |
| `recovery/<运行标签>/recovery.json` | 计划题数、恢复题数、覆盖率、缺失 ID、各文件统计、生成状态计数及合并文件 SHA256。 |
| `recovery/<运行标签>/results/summary.metrics.json` | 检索或答案评分摘要，包含 `prediction_count`、`missing_predictions` 和 `recovery` 字段。 |
| `recovery/<运行标签>/results/judge.jsonl` | 逐题指标、缺失状态及适用时的裁判结果。 |

### 常见情况

| 现象 | 处理 |
|---|---|
| 找不到 `results/manifest.json` | 确认目录指向 42 的测试根目录。如果任务在创建结果清单前就退出，没有可供恢复评分的题集记录。 |
| `job.json` 仍显示 `running` | 在 Linux 上确认主进程和 worker 都已退出。强制终止可能来不及写回任务状态。 |
| 当前题集或语料与原测试不一致 | 恢复原测试使用的 benchmark 版本及其 `prepared` 产物和 `overlap.json`。 |
| 裁判报错或解析失败 | 查看本次 `results/judge.jsonl` 和 `console.log`，修改裁判设置后重跑 85。每次运行会保留独立记录。 |
