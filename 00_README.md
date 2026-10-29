# sa-openseeker-v2

以 `../sa-qwen-openseeker` 为骨架，保留 Qwen3 / Qwen3.5、Plain / Beacon、Search-R1、多工具轨迹处理、LoRA、蒸馏算法和原有测试。OpenSeeker 训练保留原始回答，不追加参考答案；原有评测使用模型最后一轮回答计算 EM/F1；新增固定官方版本的 BrowseComp 在线生成与 LLM 裁判评测，见 [42 Benchmark](docs/42_openseeker_benchmark.md)。

只迁移 Polar 原始快照；本目录中的规范化数据由新代码重新生成，没有复制其他数据集、模型权重、旧训练结果或实验日志。本次没有启动真实模型训练。

Beacon 完整实验优先阅读 [32 从数据到评测](docs/32_beacon_experiment_guide.md)，再按阶段查阅 [文档索引](docs/00_index.md) 和 [45 统一训练与测试](docs/45_unified_agent_flow.md)。训练保留每条轨迹的原始 system；Plain/Beacon 的 native 在线测试复用对应训练提示，离线仅调整必要的工具能力和格式要求。当前使用原始 v3 数据协议，已有原始 v3 数据可直接复用。常用入口如下，参数直接修改各 shell 脚本顶部，然后运行，无需传命令行参数：

```bash
bash scripts/sh/01_test.sh                    # CPU 回归测试
bash scripts/sh/59_check_agent_alignment.sh   # 共享协议与 tiny 模型数值一致性检查，不训练
bash scripts/sh/10_prepare.sh                 # 全量规范化和分词；输出存在时拒绝覆盖
bash scripts/sh/11_select.sh                  # 长度区间/精确长度/正确性/观测筛选
bash scripts/sh/12_select_memory_probe.sh     # 生成显存测试训练集
bash scripts/sh/40_dataset_html.sh            # 离线查看数据
bash scripts/sh/46_prepare_benchmark.sh       # 官方代码和 BrowseComp 数据（联网，不请求裁判）
DRY_RUN=1 bash scripts/sh/47_openseeker_benchmark.sh  # 审查官方 Benchmark 运行配置
# 以下由使用者之后在训练服务器执行，本次没有执行：
# bash scripts/sh/20_plain_train.sh
# bash scripts/sh/30_beacon_train.sh
# bash scripts/sh/26_plain_memory_probe.sh
# bash scripts/sh/37_beacon_memory_probe.sh
# bash scripts/sh/21_plain_eval.sh
# bash scripts/sh/31_beacon_eval.sh
```

训练脚本支持将 `ACTION="train"` 改成 `ACTION="plan"`，只创建可审查的配置和启动命令。多卡参数集中在训练脚本顶部：`CUDA_VISIBLE_DEVICES`、`DISTRIBUTED_BACKEND`、`NPROC_PER_NODE`。配置脚本后直接运行即可。

目录分工：`search_comp/` 为核心实现，`search_comp/workflows/` 为可导入的流程实现，`scripts/sh/` 为编号的一键入口，`scripts/py/` 为编号的 Python 薄入口，`scripts/templates/` 为编号的 HTML 模板，`tests/` 为回归测试，`configs/` 为训练配置，`docs/` 为编号文档。

现有原始 v3 数据（train/validation/test=644/9/11）、43 条约 64k 的显存测试数据和数据 HTML 保留，可用于当前流程，详情见 [数据说明](docs/10_data.md) 与 [验证记录](docs/99_validation.md)。当前保留原始 system 的统一流程全量回归为 **564 通过、2 项跳过**，详细命令和边界见验证记录。

Beacon 训练支持固定/随机压缩比：修改 `30_beacon_train.sh` 的 `COMPRESSION_MODE`、`BEACON_RATIO` 和 `COMPRESSION_RATIOS` 即可，详见 [31 随机压缩说明](docs/31_random_compression.md)。默认保持固定压缩。

训练后独立 Benchmark：Plain/Beacon 均默认用 `MODE=native RUN_DIR=...` 运行 47；原样官方复现仍可用 48 部署 Plain，再显式选择 `MODE=official-v1` 或 `MODE=official-v2`。native 保留压缩、采用官方 search/visit 和裁判。先配置私有 API 环境文件；实际在线测试会调用这些服务。原有 21/31 的 EM/F1 测试继续保留。完整依赖、参数、失败分母和对齐边界见 [42 说明](docs/42_openseeker_benchmark.md)。

固定语料离线评测现已接入：`56_browsecomp_plus.sh` 使用独立 BCP corpus、持久化 BM25 与本地裁判；`57_hotpotqa_offline.sh` 使用有标注 dev、完整官方 Wiki 摘要库和官方 EM/F1。先 `ACTION=download` 一次性准备，之后 prepare/index/run/score/all 均严格离线。两者都支持从 RUN_DIR 读取 Plain/Beacon 训练产物。详细参数、逐步运行与修改提示见 [43 离线基准手册](docs/43_offline_benchmarks.md)，官方版本及许可证见 [44 来源说明](docs/44_offline_sources.md)。

四卡 A800 的 Beacon 长轨迹训练：先修改 `scripts/sh/30_beacon_train.sh` 中模型和数据路径，再用 `ACTION=plan bash scripts/sh/38_beacon_4xa800.sh` 只生成配置；实际训练运行 `bash scripts/sh/38_beacon_4xa800.sh`。显存原因、重计算修复和参数修改提示见 [52 四卡 A800 长轨迹 OOM](docs/52_beacon_oom.md)。

Plain 四卡长轨迹训练：`ACTION=plan bash scripts/sh/28_plain_4xa800.sh` 审查配置，实际训练去掉 `ACTION=plan`。参数统一在 `20_plain_train.sh`；分块 loss、激活卸载及限制见 [20 Plain](docs/20_plain.md)。
