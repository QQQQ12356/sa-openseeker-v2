# 05 SH 与 PY 分离规范

## 1. 目录与职责

所有运行任务使用一对同名文件。`sh/` 保存用户参数及运行环境，`py/` 保存类型定义和执行适配代码。Plain、Beacon、Baseline 各自有独立目录，数据与公共工具遵循相同约定。

```text
scripts/
  plain/
    sh/10_train_openseeker.sh
    sh/11_train_research.sh
    sh/40_test_browsecomp.sh
    sh/41_score_browsecomp.sh
    sh/42_test_browsecomp_plus.sh
    sh/43_score_browsecomp_plus.sh
    sh/44_test_hotpotqa.sh
    sh/45_score_hotpotqa.sh
    py/  与上述 SH 一一对应的 .py 文件
  beacon/sh/、beacon/py/     同样八对文件
  baseline/sh/、baseline/py/ 六对文件，仅测试和评分
  data/sh/、data/py/         十对模型/数据/benchmark 准备文件
  tools/sh/、tools/py/       六对安装、报告、Excel 导出、验证文件
```

调用链只有 `SH → 同名 PY → 共享实现`。各 SH 不 source 其他公共配置，不调用另一个方法的 SH，不拼接训练/评测命令行选项。训练与离线 GPU worker 保留必要的进程隔离。

## 2. 编辑与执行

1. 打开对应 `sh/` 文件，修改有说明的 export 行。
2. `PYTHON_BIN` 默认使用当前激活环境的 python；环境安装入口默认 python3.11。可在 SH 中填解释器绝对路径，路径中含空格也可。
3. 在 Linux 使用 `bash /项目路径/scripts/plain/sh/10_train_openseeker.sh`；无需 chmod，也无需添加额外选项。
4. SH 自动定位并切换到项目根目录，因此可从任意工作目录启动。
5. PY 不接受独立运行，缺少配对 SH 身份时会提示正确入口。不要修改 PY 的 SCHEMA 来设置参数。

## 3. 命名映射

| SH 中实际名称 | PY 中逻辑名称 | 说明 |
|---|---|---|
| `HF_ENDPOINT` | ENVIRONMENT.HF_ENDPOINT | 真正的环境变量直接 export |
| `SA_MODEL_PATH` | MODEL_PATH | 普通任务参数统一加 SA_ |
| `SA_TRAINING__MAX_LENGTH` | TRAINING.max_length | 双下划线表示嵌套键 |
| `SA_GENERATION__LIMIT` | GENERATION.limit | 各测试文件独立预算 |
| `SA_JUDGE__JUDGE_MODEL` | JUDGE.judge_model | 裁判模型本地路径 |
| `SA_DISTRIBUTED__ZERO2__ZERO_OPTIMIZATION__STAGE` | DISTRIBUTED.zero2.zero_optimization.stage | 多卡配置完整展开 |
| `SA_BEACON__LOSS_SEGMENTS__ANSWER_CONTENT` | BEACON.loss_segments.answer_content | Beacon 分段损失 |

60 手册讲逻辑含义与联动约束；61 清单逐文件列出实际名称、类型和默认值。参数全部只在 SH 中保存默认值，PY 类型定义没有另一套默认值。

## 4. 值的写法

| 类型 | SH 示例 | 规则 |
|---|---|---|
| 字符串 | `export SA_MODEL_PATH='/models/Qwen3.5-4B'` | 含空格仍是一个值；修改路径时保留引号 |
| 整数 | `export SA_GENERATION__LIMIT=-1` | JSON 整数；不使用 01 形式 |
| 小数 | `export SA_TRAINING__LEARNING_RATE=5e-05` | JSON 数值，可使用科学计数法 |
| 布尔值 | `export SA_PLAN_ONLY=true` | true/false 或 1/0，建议前者 |
| 可选空值 | `export SA_TRAINING_RUN=''` | 空字符串表示 None；仅适用于 optional 类型 |
| 可选数值 | `export SA_TRAINING__MAX_TRAIN_STEPS=100` | optional 可传数值、字符串或 null |
| 字符串列表 | `export SA_MODELS='["/models/Qwen3.5-4B"]'` | 严格 JSON，用双引号表示元素 |
| 多文件列表 | `export SA_INPUTS='["/data/a.jsonl", "/data/b.jsonl"]'` | 不使用 Python 单引号列表写法 |
| 空列表 | `export SA_INPUTS='[]'` | 与 SNAPSHOT_DIR 的互斥规则仍适用 |

SH 变量值不要在等号两侧加空格。字符串中如果需要单引号，可改用双引号并注意 Shell 的变量展开规则。参数加载不使用 eval，不将字符串作为 Python 代码执行；类型错误在调用业务操作前报告。

## 5. 环境、镜像与凭证

所有 SH 保留自身需要的 HF_ENDPOINT、离线开关、GPU 参数，以及在线工具或裁判参数。HF_ENDPOINT 可改为 https://hf-mirror.com；下载阶段使用 HF_HUB_OFFLINE=0，模型已准备的训练/测试/评分使用 1。

普通 export 行是本文件的明确值，优先于服务器同名环境变量。PYTHON_BIN 和在线凭证使用 `${NAME:-默认值}` 的继承形式。服务地址、模型名和密钥由服务器环境传入，不要把真实密钥写入 SH；运行归档会复制该文件。

## 6. 归档

实验与报告操作保留 `source_launcher.sh`、`source_script.py`、解析后的设置和日志。代码摘要仍覆盖共享实现。SH 改动会随新任务保存，不覆盖旧实验；安装和检查工具保持各自说明中的输出约定。

Windows 不运行 SH 或真实实验；静态检查只读取 SH 中的字面默认值并校验类型，既不执行 Bash，也不加载模型。

## 7. Benchmark Excel 导出

`tools/sh/82_benchmark_excel.sh → tools/py/82_benchmark_excel.py → workflows/export_benchmark_excel.py` 遵循同样的配对规则。可编辑参数为 INPUTS、OUTPUT_DIR、RUN_NAME、INCLUDE_DETAILS、INCLUDE_INCOMPLETE；依赖及完整说明见 [82 Excel 导出](82_benchmark_excel.md)。导出只读取已有评分结果，在 Linux 使用 CPU，不调用模型或重新评分。
