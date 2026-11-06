# 90 重构决策与迁移说明

## 来源

- 源目录：`sa-openseeker-v2`；源分支：`tmp`。
- 基准提交：`b61922897ec0eeef54b043a2826501735077184c`。
- 目标目录：`sa-reconstruction`，独立文件树，不依赖源目录，不复制数据、模型、运行结果或 `.git`。
- Windows 只编辑代码与做无模型检查；真实下载、预处理、训练、推理和评分均在 Linux 执行。

## 决策

| 项目 | 最终约定 | 原因 |
|---|---|---|
| baseline | 原始、未训练的 Qwen3.5 基座；仅三项测试和评分 | 统一方法含义，移除 baseline 训练路线 |
| plain / beacon | 各支持 OpenSeeker 与 Research 训练 | 数据身份与轨迹编码协议分别记录 |
| 公共入口 | SH 文件编辑参数，直接启动同名 PY 适配器，再调用共享库 | 移除 Shell → Python 转发 → CLI workflow 的调用链 |
| 共享实现 | experiments 管理产物和流程，workflows 保留数据/评测实现，trainer/models 保留算法 | 集中跨方法不变量，方法脚本独立配置 |
| 子进程 | 仅训练/合并隔离模型显存及离线多卡 worker 使用 | torchrun 需要独立进程；用户无需拼接参数 |
| 归档 | 数据按来源与版本，训练按方法/数据集/实验，评分挂在测试下 | 避免覆盖、跨方法串目录和误用预测 |
| 方法参数 | 沿用 tmp 中 Plain/Beacon 的训练、压缩与损失默认值 | 将本次范围限定在框架、入口、归档和文档重构 |
| 文档 | 00 总览、10 环境、20 数据、30–32 方法、60 参数、70 产物、90 迁移、99 验证 | 以实验顺序定位操作和修改点 |

## 编码规范来源

应用 [mattpocock/skills 的 codebase-design](https://github.com/mattpocock/skills/blob/main/skills/engineering/codebase-design/SKILL.md)：以少量稳定操作封装共享行为，删除只转发的模块，集中校验与归档逻辑，围绕调用接口验证行为。未安装全局插件、未引入工单流程。

## 迁移步骤

1. 将整个目标目录复制到 Linux，安装依赖并激活环境。
2. 原始数据可以通过数据脚本重新下载，也可在 `INPUTS` 显式列出已有本地文件；不要指向不存在的旧 Windows 绝对路径。
3. 使用同一 Linux 本地 `MODEL_PATH` 完成预处理和训练。迁移旧 prepared 数据时，如 tokenizer 身份或 manifest 路径改变，重新预处理。
4. 旧模型如需复现，应在旧项目使用旧入口；新项目的训练方法测试要求新的已完成训练实验清单，不自动猜测旧目录身份。
5. 旧 `outputs/` 和历史文档不复制到目标。新产物全部进入 `artifacts/`；源文件保持原样。

## 接口修正

- Research 重新筛选恢复公开划分时，使用 Research adapter 读取原始消息，保留来源身份。
- 基座模型下载解析并记录实际 commit；预处理与训练严格检查 tokenizer 身份、协议、长度上限和数据来源。
- 评分验证测试完成状态、方法、benchmark、预测摘要及测试 manifest 摘要，沿用测试题目版本与 LIMIT。
- BrowseComp 推理与评分要求提前准备官方工具和题目；评测时不隐式下载。

- BrowseComp 原生在线 Registry 现在执行脚本声明的 tool_count 上限，防止参数仅命名输出而未约束实际调用。

## SH/PY 分离增量调整

所有原 scripts/<group>/*.py 已迁移到 scripts/<group>/py/，并新增同名 sh/ 入口。参数默认值只保存在 SH，PY 仅保留参数类型和执行逻辑。不要继续使用旧路径；使用 bash 启动新入口。业务算法及已有数据/实验目录约定不变，运行记录新增 source_launcher.sh。
