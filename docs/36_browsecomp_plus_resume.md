# 36 BrowseComp-Plus 续测

Plain、Beacon、Baseline 的 `42_test_browsecomp_plus.sh` 可以把一次已完成测试的结果扩展到更大的题目范围。比如原测试完成了筛选后的前 40 道题，目标改为 60 后，新任务直接复用前 40 条预测，只对新增的 20 道题加载模型并生成答案。新目录会包含完整的 60 题预测，可直接交给原评分脚本。

## Linux 上从 40 题补到 60 题

1. 在项目根目录找到原测试根目录。它包含 `job.json`、`evaluation.json` 和 `results/`。原测试须已完成生成并完成归档，有无既有评分均可。
2. 打开同一方法的 `42_test_browsecomp_plus.sh`。保留原测试的模型、题集、提示词、检索器、工具和解码配置。若刚切换 Git 版本，可对照原测试归档中的 `source_launcher.sh` 或 `evaluation.json`。
3. 修改三项变量：

```bash
export SA_RESUME_FROM='/path/to/evaluation/browsecomp-plus/original-run'
export SA_GENERATION__LIMIT=60
export SA_RUN_NAME=extend_40_to_60
```

4. 运行同一方法的测试脚本。例如 Plain：

```bash
bash scripts/plain/sh/42_test_browsecomp_plus.sh
```

5. 日志应显示 `reuse 40 predictions; generate 20 additional questions`。新测试成功后，将**新测试根目录**填进同方法 `43_score_browsecomp_plus.sh` 的 `SA_EVALUATION_DIR`，运行评分。

## 变量说明

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_RESUME_FROM` | 空 | 续测来源。填写一个已完成的测试根目录，可用项目相对路径或 Linux 绝对路径。目录内应有 `job.json`、`evaluation.json` 和 `results/manifest.json`。需要从第 1 题新跑时留空。 |
| `SA_GENERATION__LIMIT` | `-1` | 续测后的累计题数上限，按重合筛选后的源顺序选题。原测 40、目标 60 时填 `60`。设 `-1` 会扩到所有剩余题。若筛选后可用题数不足，则取可用题数。没有新增题会停止。 |
| `SA_RUN_NAME` | `full` | 新运行标签。建议体现来源题数与累计目标，程序追加时间和随机后缀。 |
| `SA_GENERATION__NUM_WORKERS` | `1` | 可按本次资源调整，只分配新增题目。其他生成预算应沿用原测试。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可按资源调整，worker 数不能超过已分配的可见 GPU。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 可调整本次是否重新扫描索引 SHA256。索引清单与原测试一致性的核对始终执行。 |
| `SA_EVALUATION_DIR` | 评分脚本中的占位目录 | 评分时填写新测试根目录，自动按累计题数和原重合筛选报告重建题集。 |

## 哪些设置要保持一致

- **模型和提示词。** 选择同一个方法及训练实验或基座，解析到同一个本地检查点路径，并保持该目录内容原样。程序通过路径核对模型，旧测试没有权重内容指纹。最终渲染的 system prompt 会与原逐题预测记录的 SHA256 对比。
- **题目和重合筛选。** 使用同一份 prepared 题目文件。Plain 和 Beacon 的 `SA_OVERLAP`、匹配阈值及训练数据指纹应相同，程序复制并复用原测试的 `overlap.json`。原题目 ID 序列必须恰好是新选择的有序前缀。Baseline 按同一 benchmark 版本的源顺序选题。
- **检索与推理。** 使用同一检索器及索引。Qwen3-Embedding-8B 模式还应保留向量索引、模型路径和查询编码设备。TopK、观测预算、访问开关、调用上限、解码与随机种子等沿用原配置。BM25 与 Qwen 的结果应分别续测。
- **原始预测。** 原预测文件和运行清单的 SHA256 须匹配归档。程序逐行核对题目 ID、问题、比较答案、顺序与提示词。本功能面向已完成的测试扩容。原测试中已经归档的超限、工具错误或空回答也会沿用。

## 新目录与评分

| 文件 | 续测时的含义 |
|---|---|
| `evaluation.json` | 新测试配置、原测试路径及合并后预测文件的 SHA256。 |
| `results/manifest.json` | 完整题目列表、模型和索引配置。其 `resume` 字段记录原测试目录、复用题数、新生成题数及原结果指纹。 |
| `results/predictions.jsonl` | 按源顺序合并的完整预测。示例中共 60 行，`sample_index` 从 0 连续到 59。 |
| `results/extension/` | 仅本次新增题目的预测和运行记录。诊断续测时可以查看该目录。 |
| `results/generation.metrics.json` | 合并后的状态计数和本次续测耗时。该耗时只计算当前任务。 |
| `overlap.json` | 启用重合筛选时，从原测试复制的报告。 |

评分读取合并后的预测，按原评分流程处理全部目标题目。旧的 40 题评分记录仍归原目录。可从已成功完成的 60 题续测目录再扩到 100 题。若本次续测失败，可查看 `console.log` 和 `results/extension/`，修正后仍从上一个成功完成的测试目录开始。
