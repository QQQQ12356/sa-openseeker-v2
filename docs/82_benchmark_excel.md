# 82 Benchmark 评分结果导出 Excel

## 1. 用途与入口

将已有的 BrowseComp、BrowseComp-Plus、HotpotQA 评分结果汇总成 `.xlsx`，用于比较 Plain、Beacon、Baseline 的实验结果。支持一次汇总多个训练实验、测试批次和评分批次。

| 文件 | 职责 | 修改位置 |
|---|---|---|
| [82_benchmark_excel.sh](../scripts/tools/sh/82_benchmark_excel.sh) | Linux 运行入口，保存所有用户参数 | 日常只修改这里的可编辑参数 |
| [82_benchmark_excel.py](../scripts/tools/py/82_benchmark_excel.py) | 读取 SH 参数、校验入口身份、调用共享实现 | 新增参数时同步修改 SCHEMA |
| [export_benchmark_excel.py](../search_comp/workflows/export_benchmark_excel.py) | 查找摘要、读取身份、组织工作表、写出 Excel | 新增汇总列或适配评分文件时修改 |
| [requirements.tools-export.txt](../requirements.tools-export.txt) | CPU 导出依赖 `openpyxl>=3.1.5,<4` | 依赖变更后在 Linux 重新安装并验证 |

导出只读取已有 JSON/JSONL 文件，不调用评分器、不加载模型，不需要 GPU、搜索服务或裁判服务。评分结果的数值与分母沿用源文件，不重新计算、不按多个实验取平均。当前 Windows 环境只编辑文件及做静态检查；实际导出在 Linux 执行。

## 2. Linux 使用步骤

1. 完成 benchmark 测试，再运行对应方法的 `41_score_browsecomp.sh`、`43_score_browsecomp_plus.sh` 或 `45_score_hotpotqa.sh`。
2. 确认评分目录中存在 `results/summary.metrics.json`。正常完成的评分，其 `job.json` 为 `status=complete`，摘要为 `complete=true`。
3. 编辑导出 SH 的 `SA_INPUTS`。默认扫描 `artifacts/experiments`；需要限定实验时改为具体评分、测试或训练目录。
4. 激活 Linux Python 环境并执行导出。每次运行创建新目录，终端输出 Excel 的完整路径。

```bash
# 已安装统一 requirements.txt 的环境已包含导出依赖。
# 也可在独立 CPU 环境只安装下面这一份，不需要安装 CUDA/PyTorch。
python -m pip install -r requirements.tools-export.txt
bash scripts/tools/sh/82_benchmark_excel.sh
```

项目根目录以外启动时，使用 SH 的绝对路径；脚本自动切换到项目根目录。入口不接受附加命令行参数，不直接运行 PY。输入是 Linux 路径，不使用 Windows 盘符。

## 3. 完整参数表

| SH 实际变量 | PY 逻辑字段 / 类型 | 默认值 | 作用与修改提示 |
|---|---|---|---|
| `PYTHON_BIN` | Shell 解释器路径 / 字符串 | `${PYTHON_BIN:-python}` | 使用已激活环境的 python；可指定 `/opt/venv/bin/python`，不含额外命令参数。唯一允许从服务器继承的用户运行参数。 |
| `SA_INPUTS` | `INPUTS` / list | `["artifacts/experiments"]` | 非空 JSON 字符串列表；每项是目录或名为 `summary.metrics.json` 的文件。每个输入都必须存在且至少找到一份摘要。目录递归扫描；输入重叠按解析后的绝对路径去重。 |
| `SA_OUTPUT_DIR` | `OUTPUT_DIR` / path | `artifacts/reports/benchmarks` | 导出操作的父目录，必须解析到本项目 `artifacts/` 内。每次在其下创建带时间戳的新子目录，不是 `.xlsx` 文件路径。 |
| `SA_RUN_NAME` | `RUN_NAME` / str | `benchmark_excel` | 子目录标签；1–96 字符，以字母或数字开头，其余可用字母、数字、下划线和连字符。程序自动追加 UTC 时间与随机后缀。 |
| `SA_INCLUDE_DETAILS` | `INCLUDE_DETAILS` / bool | `true` | 导出存在的逐题指标和裁判记录；`false` 仅生成 `summary`、`metrics`。大量实验或长文本时建议关闭。 |
| `SA_INCLUDE_INCOMPLETE` | `INCLUDE_INCOMPLETE` / bool | `false` | 默认跳过摘要未声明 `complete=true`，或标准评分任务状态不是 complete 的记录。`true` 可导出已有的不完整摘要用于排错，表中保留原状态。不能补出尚未生成的摘要。 |

普通 `export SA_*=` 会覆盖服务器同名变量，修改参数应编辑当前 SH。布尔值写 `true/false`；列表使用严格 JSON；Shell 外层单引号、JSON 元素双引号。路径含空格也放在 JSON 字符串中。不支持通配符；需要批量输入时填写共同父目录。

| 内部变量 | 作用 | 修改提示 |
|---|---|---|
| `SCRIPT_DIR` | 当前 SH 所在目录 | 自动计算，不修改 |
| `PROJECT_ROOT` | 从 SH 向上三级定位项目根目录 | 自动计算，不修改 |
| `SA_ENTRY` | `tools/82_benchmark_excel`；配对入口身份 | 必须匹配 PY 所在目录和文件名，不修改 |
| `SA_LAUNCHER` | 当前 SH 的绝对路径 | 用于归档实际运行参数，不修改 |

此工具不设置 HF、CUDA 或裁判环境变量；这些变量不参与 Excel 导出。SH/PY 的通用约定见 [05 脚本结构](05_script_layout.md)。

## 4. 输入范围与配置示例

以下均为替换 SH 参数的示例，路径占位符需改成 Linux 上已有目录。

```bash
# 默认：所有方法、所有 benchmark、所有找到的评分批次。
export SA_INPUTS='["artifacts/experiments"]'

# 一个测试批次：汇总该测试的所有评分重试，保留每次记录。
export SA_INPUTS='["artifacts/experiments/plain/openseeker/RUN/evaluation/hotpotqa/EVAL"]'

# 精确对比两次评分，避免将历史重试混入比较。
export SA_INPUTS='["artifacts/experiments/plain/openseeker/RUN/evaluation/hotpotqa/EVAL/scores/SCORE", "artifacts/experiments/baseline/untrained/evaluation/hotpotqa/EVAL/scores/SCORE"]'

# 导入独立复制的评分摘要；身份可能不完整，见下文。
export SA_INPUTS='["/data/export/summary.metrics.json"]'

# 只需要指标表时关闭逐题导出。
export SA_INCLUDE_DETAILS=false
```

扫描目标仅为精确文件名 `summary.metrics.json`，不把预测文件、`generation.metrics.json`、任意 `metrics.json` 当成评分摘要。不同路径的相同内容视为不同记录；同一路径仅导出一次。按绝对文件路径排序生成 `score_000001` 等关联编号；编号仅在本次工作簿内有效。所有评分重试分别保留，不自动选择最新一次、不合并同一模型结果。

标准输入布局：

```text
<evaluation>/
├── evaluation.json                 方法、benchmark、训练来源、推理范围
├── results/manifest.json           实际模型路径等测试来源
└── scores/<score>/
    ├── job.json                    operation=score、状态、时间、裁判配置
    └── results/
        ├── summary.metrics.json    必需：整体指标
        ├── per_question.json       可选：HotpotQA / BrowseComp-Plus 逐题指标
        ├── judge.jsonl             可选：上述离线 benchmark 裁判记录
        └── result_for_judge_eval.jsonl  可选：在线 BrowseComp 裁判记录
```

标准结构中，方法和 benchmark 从评分任务及测试身份读取，并检查两者一致；摘要中的 benchmark 比较时忽略大小写。身份冲突直接报错。独立摘要若没有相邻的标准评分 `job.json`，`job_status` 标为 `untracked`，仅凭摘要的 `complete=true` 决定默认是否纳入；方法、数据集、训练来源等无法确认的字段保留空白，不从目录名猜测。需要完整身份时保留上述目录结构及 JSON 清单。

只读导出不重新验证原始模型、预测文件或语料；它不能替代测试和评分阶段的来源校验。复制后的历史绝对路径可作为来源文本导出，不要求对应权重仍可访问。应在评分进程结束后导出，避免读取正在变化的文件。

## 5. Excel 工作表

| 工作表 | 一行表示 | 内容与用途 |
|---|---|---|
| `summary` | 一份评分摘要 | 主要比较表；固定列，含来源、完成状态、关键指标和 SHA256 |
| `metrics` | 一份摘要中的一个字段 | 保留所有摘要字段；嵌套对象展开为 `avg_tool_stats.search` 等点分名称，数组及空对象保存为 JSON 文本 |
| `details` | 一个逐题源文件中的一条记录 | 仅 `SA_INCLUDE_DETAILS=true` 创建；提取答案、分数、错误、检索指标和裁判理由等关键内容 |

工作表均冻结首行、启用筛选、固定列宽。小数显示四位，文件保留底层数值精度；比例数值不自动乘 100。空白表示源文件缺失/null/不适用，数值 0 保持为 0。布尔值保持布尔类型。没有任何 Excel 公式，模型答案中以 `=` 开头的文本也按普通字符串保存。

### 5.1 summary 字段与口径

| 列 | 含义与注意事项 |
|---|---|
| `record` | 本次工作簿的记录编号；关联 metrics/details |
| `method` / `benchmark` | 方法和测试集身份；独立摘要可能缺少 method |
| `dataset_name` / `training_run` | 训练数据类别和训练实验路径；baseline 通常为 untrained、空训练路径 |
| `model_path` | 优先用 evaluation.json 的模型路径，否则取测试 manifest 的 model；不加载模型 |
| `benchmark_dir` / `limit` | benchmark 资产版本路径、生成时的题目上限；limit=-1 表示该任务使用全部题目 |
| `score_run` / `evaluation_dir` | 评分批次名称、测试目录；用于区分评分重试 |
| `job_status` / `complete` | 任务状态、摘要完成标记；必须同时查看，不完整结果不应作为正式对比 |
| `created_at` / `finished_at` | 评分任务记录的原始时间字符串；通常为 UTC |
| `count` | 评分题目总数；在线 BrowseComp 映射 expected |
| `prediction_count` | 预测条数；在线映射 unique_predictions，不代表成功回答条数 |
| `missing_predictions` | 缺失预测数量；直接沿用评分摘要 |
| `judge_errors` | 裁判错误数；在线映射 judge_unknown_or_error，未提供时空白 |
| `em` / `f1` | 0–1；HotpotQA 沿用官方答案指标，其他两项沿用项目补充答案指标 |
| `llm_judge_accuracy` | 0–1；语义裁判准确率。未开启 HotpotQA LLM 裁判时通常为空白 |
| `accuracy` | 0–1；源摘要提供的 accuracy，如完整 BrowseComp-Plus 评分 |
| `official_accuracy` | 在线 BrowseComp 的已获得有效标签题目的准确率；裁判不完整时分母与全题数不同 |
| `accuracy_all_questions` | 在线 BrowseComp 全题分母准确率；不能与 official_accuracy 混淆 |
| `evidence_recall` / `gold_recall` | BrowseComp-Plus 检索召回，0–1；不代表答案准确率 |
| `sp_em` / `sp_f1` / `joint_em` / `joint_f1` | HotpotQA 支持事实及联合指标，0–1；依赖模型实际生成的 supporting facts |
| `citation_precision` / `citation_recall` / `citation_coverage` | BrowseComp-Plus 引用指标，0–1；沿用评分器统计口径 |
| `subset` / `corpus_scope` | 是否子集、语料范围；比较前确认数据范围一致 |
| `score_mode` / `metric` | 评分模式及原始指标说明；retrieval-only 的答案准确率留空 |
| `judge_model` / `judge_attempts` | 本地裁判路径和尝试次数；在线裁判服务模型名不在现有评分清单中时不推断 |
| `summary_file` / `summary_sha256` | 摘要绝对路径及导出时文件摘要，支持回查来源 |

未放入固定列的指标仍保留在 metrics，例如 `Accuracy (%)`、`Recall (%)`、`Calibration Error (%)`、`calibration_available`、precision/recall、裁判策略和逐工具统计。名称带 `(%)` 的值保持原始 0–100 口径。校准误差为 0 时，应结合 calibration_available 判断是否有足够有效样本。

### 5.2 details 字段与对齐规则

| 列组 | 说明 |
|---|---|
| `record` / `source_file` / `source_kind` / `source_row` | 关联汇总记录及原始文件；JSON 数组按 1 起算元素序号，JSONL 使用真实行号 |
| `id` / `item_index` | 原始题目 ID 或在线裁判 item_index；不通过文件读取顺序推断题目对应关系 |
| `query` / `reference_answer` / `prediction` | 沿用该条记录的 query、answer；预测优先取 prediction，再取 candidate_answer，再取 final_response |
| `completed` / `score` / `correct` / `em` / `f1` | 源文件已有的完成标记及分数；不在导出时重新打分 |
| `evidence_recall` / `gold_recall` | 源记录已有的逐题检索召回 |
| `score_source` / `judge_error` / `judge_reasoning` | 裁判来源、错误及 judge_result.reasoning；用于排错 |
| `tool_calls` | 该条记录已有的工具调用数，缺失时空白 |

HotpotQA / BrowseComp-Plus 同一道题可能分别有 per_question 和 judge 两条记录，使用 source_kind 区分；不把两类记录当成两道题计数，不以 details 行数重新计算准确率。在线文件仅导出 `type=item` 的条目，忽略汇总行。缺少某个可选文件不会报错，也不会补造题目、参考答案、轨迹或分数。

本工具不读取完整 Agent 对话、检索正文或逐次裁判重试历史。需要完整预测轨迹时使用 `81_prediction_report.sh`。逐题参考答案和问题仅在相应评分记录包含时可见；缺失时从 source_file 或测试目录回查。

## 6. 输出与边界

```text
artifacts/reports/benchmarks/<RUN_NAME>_<UTC时间>_<随机后缀>/
├── benchmark_scores.xlsx          成功写入后才从 .building.xlsx 改名
├── export_manifest.json           输入、纳入/跳过记录、源文件与工作簿摘要、各表行数
├── job.json                       benchmark_excel 操作状态与解析后参数
├── source_launcher.sh             本次 SH 副本
├── source_script.py               本次 PY 适配器副本
└── console.log                    导出日志及错误堆栈
```

每次创建新目录，不向已有工作簿追加，也不覆盖历史导出。失败任务保留日志；修正参数后重新运行。没有任何可导出记录时失败退出，避免生成看似成功的空表。

| 情况 | 行为与处理提示 |
|---|---|
| 未安装 openpyxl | 提示安装 requirements.tools-export.txt；在 Linux 安装后重试 |
| 输入路径不存在或没有摘要 | 报错；先完成评分，或将 INPUTS 改为包含真实摘要的目录 |
| 默认模式遇到 failed/running/complete=false | 跳过并写入 export_manifest.json；排错可显式打开 INCLUDE_INCOMPLETE |
| JSON 损坏、结构不符、NaN/Infinity、身份冲突 | 报错，不静默丢弃损坏数据；检查日志中的文件路径 |
| OUTPUT_DIR 在 artifacts 外 | 报错；修改为本项目 artifacts 下的父目录 |
| 字符串超过 Excel 单元格 32767 个 UTF-16 单元 | 截断并附原文件回查提示；在导出清单 clipped_cells 计数，源文件保持完整 |
| XML 不支持的控制字符/孤立代理字符 | 从单元格文本移除；在 cleaned_cells 计数，源文件保持完整 |
| 整数绝对值达到 10^15 | 按字符串保存，避免 Excel 数字精度损失 |
| 表数据超过 1048575 行 | 自动创建 details_2、metrics_2 等续表，重复表头；不丢行 |

工作簿采用流式写入，适合批量结果；逐题 JSON 数组仍按一个文件读入内存，JSONL 按行读取。关闭 INCLUDE_DETAILS 可明显减小文件体积。本文描述的是代码约定，真实 Linux 端到端导出仍需按 [99 验证记录](99_validation.md) 完成验证。

## 7. 后续修改提示

| 修改需求 | 修改位置及同步要求 |
|---|---|
| 增加常用汇总指标 | 在共享实现 SUMMARY_COLUMNS 增加原始摘要字段名；不存在的字段自动空白；同步本文字段表 |
| 增加别名转换 | 在 load_record 中明确映射；只映射口径相同的字段，不用不同分母的指标互相兜底 |
| 增加其他逐题文件 | 修改 DETAIL_FILES、detail_rows 与导出字段映射；确认题目 ID 和汇总行规则 |
| 新增可调参数 | 同时修改 SH 的 export/中文注释、PY 的 SCHEMA、函数入参及本文和 61 清单 |
| 改变格式、列宽或拆表限制 | 修改 SheetWriter；保留纯文本写入和 Excel 容量约束 |

Linux 无模型验证：`python -m unittest discover -s tests -p 'test_export_benchmark_excel.py' -v`。测试使用临时合成评分文件，不启动真实 benchmark。Windows 本次不执行该测试或导出入口。
