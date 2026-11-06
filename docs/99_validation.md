# 99 SH/PY 分离验证记录

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

默认只用 Python 标准库。可修改该 SH 的 SA_RUN_UNIT_TESTS 或 SA_TEST_PATTERNS。完整模型算法回归另运行 `scripts/tools/sh/91_test_algorithms.sh`；它需要 Linux 训练依赖和相应测试条件，本次未执行。

## 4. 运行边界

Linux 环境安装、CUDA/驱动、DeepSpeed 与多卡、完整预处理、真实训练与三项 benchmark 仍待目标服务器验证。本次保留既有算法和产物路径，只调整入口结构、参数传递、SH 归档与相关说明。
