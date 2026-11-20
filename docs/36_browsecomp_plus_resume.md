# 36 测试续测

所有数据评测入口都可把一次已完成测试扩展到更多题目。例如原测试完成前 40 道题，目标改为 60 后，新任务复用前 40 条预测，只生成新增的 20 道题。新目录包含完整的 60 题预测。

| 数据范围 | 测试入口 | 合并后评分 |
|---|---|---|
| BrowseComp | 各方法的 `40_test_browsecomp.sh` | 同方法的 `41_score_browsecomp.sh` |
| BrowseComp-Plus | 各方法的 `42_test_browsecomp_plus.sh` | 同方法的 `43_score_browsecomp_plus.sh` |
| HotpotQA | 各方法的 `44_test_hotpotqa.sh` | 同方法的 `45_score_hotpotqa.sh` |
| 训练数据或同来源保留划分 | `scripts/tools/sh/84_test_training_data.sh`，支持 train、validation、test | 生成后自动汇总合并结果的 EM/F1 |

前三项的「各方法」指 `scripts/plain/sh/`、`scripts/beacon/sh/` 和 `scripts/baseline/sh/`。续测用原来的同一个入口。

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
