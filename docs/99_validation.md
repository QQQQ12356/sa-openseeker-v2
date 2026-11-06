# 99 工程验证记录

第 1–4 节保留此前 SH/PY 分离阶段的检查结果；本次 Excel 导出功能的检查范围与待验证项见第 5 节。历史通过数量不代表本次新增功能已完成运行验证。

## 1. 本次范围

Windows 仅编辑代码并执行静态与无模型单元检查；没有运行 SH 业务入口，也没有启动下载、真实数据预处理、训练、模型推理或裁判请求。

## 2. 检查结果

| 检查项 | 结果 | 范围 |
|---|---|---|
| Python 语法与安全导入 | 143 个文件通过 | 共享实现、37 个 PY 适配器、测试和固定参考 |
| SH/PY 配对 | 37 对 | Plain 8、Beacon 8、Baseline 6、数据 10、工具 5 |
| Bash 语法 | 37 个文件通过 bash -n | 只解析语法，不执行任务正文 |
| SH 默认参数类型 | 全部通过 | 无需 Bash 执行即可读取字面默认值，并按对应 PY 类型定义校验 |
| 无模型测试 | 51 项通过，0 失败、0 错误 | 保留流程/协议/评分测试，加 SH 解析与归档测试 |
| 实验矩阵 | 15 个组合通过接线检查 | Plain/Beacon × 两种数据 × 三项 benchmark，加 baseline × 三项 |
| 文档链接与编号 | 12 份文档通过 | 总览、目录规范、三种方法指南、完整变量清单等 |
| 官方参考 | SHA256 一致 | HotpotQA 和 BrowseComp-Plus 固定官方资产 |
| 换行 | SH 全部 LF | .gitattributes 显式规定 Shell 换行 |

新增检查覆盖嵌套参数名称还原、环境变量保留、布尔值和 JSON 列表、可选空值/路径/数值、类型错误、遗漏 export、配对入口身份、静态读取不执行 Shell 内容，以及实际 SH 参数文件归档。

15 个组合使用微型替代后端，只验证参数传递、模型来源和目录归属，不能代替真实 Linux 推理验证。

## 3. Linux 复现

安装环境后运行：

```bash
bash scripts/tools/sh/90_validate.sh
```

基础检查只用 Python 标准库。可修改该 SH 的 SA_RUN_UNIT_TESTS 或 SA_TEST_PATTERNS；本次追加了 Excel 导出测试，缺少 openpyxl 时其工作簿测试自动跳过，输入解析测试仍可运行。完整模型算法回归另运行 `scripts/tools/sh/91_test_algorithms.sh`；它需要 Linux 训练依赖和相应测试条件，本次未执行。

## 4. 运行边界

Linux 环境安装、CUDA/驱动、DeepSpeed 与多卡、完整预处理、真实训练与三项 benchmark 仍待目标服务器验证。本次保留既有算法和产物路径，只调整入口结构、参数传递、SH 归档与相关说明。

## 5. 2026-11-06 Benchmark Excel 导出

本次在 Windows 修改代码，只执行静态检查，不运行 SH 入口、单元测试、真实数据处理、benchmark、训练或 Excel 导出，也未安装依赖。

| 检查项 | 本次结果 | 边界 |
|---|---|---|
| Python 语法 | 146 个文件通过 compile 检查 | 只编译检查，不执行编译结果或导入业务模块 |
| SH/PY 配对及身份 | 38 对通过；工具目录 6 对 | 静态读取 SCHEMA 和 export，检查 SA_ENTRY 一致 |
| SH 默认值类型与 LF 换行 | 38 份通过 | 使用 shlex/JSON 解析字面值，不运行 Bash |
| 文档编号与本地链接 | 13 份通过 | 包含新增 82 指南 |
| 补丁空白检查 | git diff --check 通过 | 不代表功能测试通过 |
| 新增导出测试 | 11 项已编写，尚未执行 | 覆盖输入去重、来源身份、状态、指标尺度、三项 benchmark 文件适配、Excel 文本与容量边界等 |
| Linux 端到端导出 | 待执行 | 安装依赖后使用真实评分目录确认工作簿内容及可打开性 |

Linux 验证步骤：

```bash
python -m pip install -r requirements.tools-export.txt
python -m unittest discover -s tests -p 'test_export_benchmark_excel.py' -v
# 完整实验环境中还可运行已有的统一检查入口。
bash scripts/tools/sh/90_validate.sh
# 编辑 SA_INPUTS 后，验证真实评分结果。
bash scripts/tools/sh/82_benchmark_excel.sh
```

单独执行新增测试只需标准库与 openpyxl，不需要 CUDA 或模型；统一检查入口包含其他实验模块检查，仍需相应环境。新增功能的完整使用及修改说明见 [82 Excel 导出](82_benchmark_excel.md)。

## 6. 2026-11-06 Research 正确性筛选

移除 Research 预处理与二次筛选入口仅允许 any 的限制，统一使用 Correct/Incorrect/any 共用筛选规则。Research 默认仍为 any；不新增标签推断或答案打分。已同步四份数据 SH 注释、20 数据指南、60 参数手册与 61 参数清单。

本次 Windows 静态检查通过：146 个 Python 文件语法、38 对 SH/PY 配对及参数默认值/LF、13 份编号文档及本地链接、git diff --check。已新增两项 Research 测试，并扩展既有二次筛选测试，覆盖三种模式、缺失/未知标签和公开 validation 划分保留；本次未执行这些测试或实际预处理。

Linux 可运行 `python -m unittest discover -s tests -p 'test_research_sft.py' -v` 和 `python -m unittest discover -s tests -p 'test_experiment_contracts.py' -v`，或使用完整环境中的 `bash scripts/tools/sh/90_validate.sh`。实际筛选时修改 SA_CORRECTNESS，并使用新的 SA_PROFILE。
