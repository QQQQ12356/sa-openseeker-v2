# sa-openseeker-v2

以 `../sa-qwen-openseeker` 为骨架，保留 Qwen3 / Qwen3.5、Plain / Beacon、Search-R1、多工具轨迹处理、LoRA、蒸馏算法和原有测试。OpenSeeker 训练保留原始回答，不追加参考答案；原有评测使用模型最后一轮回答计算 EM/F1；新增固定官方版本的 BrowseComp 在线生成与 LLM 裁判评测，见 [42 Benchmark](docs/benchmark/42_openseeker_benchmark.md)。

只迁移 Polar 原始快照；本目录中的规范化数据由新代码重新生成，没有复制其他数据集、模型权重、旧训练结果或实验日志。本次没有启动真实模型训练。

Plain 完整实验优先阅读 [21 从数据到评测](docs/plain/21_plain_experiment_guide.md)，Beacon 参照 [32 从数据到评测](docs/beacon/32_beacon_experiment_guide.md)，再按阶段查阅 [文档索引](docs/00_index.md) 和 [45 统一训练与测试](docs/common/45_unified_agent_flow.md)。训练保留每条轨迹的原始 system；Plain/Beacon 的 native 在线测试复用对应训练提示，离线仅调整必要的工具能力和格式要求。当前使用原始 v3 数据协议，已有原始 v3 数据可直接复用。常用入口如下，参数直接修改各 shell 脚本顶部，然后运行，无需传命令行参数：

```bash
bash scripts/sh/tools/01_test.sh                    # CPU 回归测试
bash scripts/sh/tools/59_check_agent_alignment.sh   # 共享协议与 tiny 模型数值一致性检查，不训练
bash scripts/sh/data/10_prepare.sh                 # 全量规范化和分词；输出存在时拒绝覆盖
bash scripts/sh/data/11_select.sh                  # 长度区间/精确长度/正确性/观测筛选
bash scripts/sh/data/12_select_memory_probe.sh     # 生成显存测试训练集
bash scripts/sh/report/40_dataset_html.sh            # 离线查看数据
bash scripts/sh/benchmark/46_prepare_benchmark.sh       # 官方代码和 BrowseComp 数据（联网，不请求裁判）
DRY_RUN=1 bash scripts/sh/plain/47_plain_browsecomp.sh  # 审查官方 Benchmark 运行配置
# 以下由使用者之后在训练服务器执行，本次没有执行：
# bash scripts/sh/plain/20_plain_train.sh
# bash scripts/sh/beacon/30_beacon_train.sh
# bash scripts/sh/plain/26_plain_memory_probe.sh
# bash scripts/sh/beacon/37_beacon_memory_probe.sh
# bash scripts/sh/plain/21_plain_eval.sh
# bash scripts/sh/beacon/31_beacon_eval.sh
```

训练脚本支持将 `ACTION="train"` 改成 `ACTION="plan"`，只创建可审查的配置和启动命令。多卡参数集中在训练脚本顶部：`CUDA_VISIBLE_DEVICES`、`DISTRIBUTED_BACKEND`、`NPROC_PER_NODE`。配置脚本后直接运行即可。

目录分工：`search_comp/` 为核心实现，`search_comp/workflows/` 为可导入的流程实现，`scripts/sh/` 按功能分类存放编号 shell 入口，`scripts/py/` 为编号的 Python 薄入口，`scripts/templates/` 为编号的 HTML 模板，`tests/` 为回归测试，`configs/` 为训练配置。`docs/` 按 common、data、plain、baseline、beacon、benchmark、report、tools、architecture 和 project 分类；各模块的入口见 [文档索引](docs/00_index.md)。

OpenSeeker 的 21 入口支持直接评测 Plain LoRA adapter；原始 Qwen3.5-4B 在 OpenSeeker、在线 BrowseComp、离线 BrowseComp-Plus 和 HotpotQA 的基线入口及对照方法见 [22](docs/baseline/22_qwen35_baseline_and_adapter.md)。

原始 Qwen3.5 基线有独立加载器及 70–73 脚本，不调用 Plain 训练模型的加载脚本；详细操作和参数见 [24 独立基线实验指南](docs/baseline/24_qwen35_base_benchmark.md)。

Plain / Beacon native 的贪心与重复惩罚、Plain 采样和生成种子配置见 [23](docs/common/23_decoding.md)，默认保留原有贪心且不惩罚重复。

现有原始 v3 数据（train/validation/test=644/9/11）、43 条约 64k 的显存测试数据和数据 HTML 保留，可用于当前流程，详情见 [数据说明](docs/data/10_data.md) 与 [验证记录](docs/project/99_validation.md)。当前保留原始 system 的统一流程全量回归为 **564 通过、2 项跳过**，详细命令和边界见验证记录。

Beacon 训练支持固定/随机压缩比：修改 `30_beacon_train.sh` 内 `beacon` 段的 `beacon_compression_mode`、`beacon_ratio` 和 `beacon_compression_ratios`，详见 [31 随机压缩说明](docs/beacon/31_random_compression.md)。默认保持固定压缩。

训练与多卡 DeepSpeed 参数已放入各训练 shell；修改和计划审查方法见 [53 自包含训练配置](docs/tools/53_self_contained_training.md)。

训练后独立 Benchmark：Plain/Beacon 均默认用 `MODE=native RUN_DIR=...` 运行 47；原样官方复现仍可用 48 部署 Plain，再显式选择 `MODE=official-v1` 或 `MODE=official-v2`。native 保留压缩、采用官方 search/visit 和裁判。先配置私有 API 环境文件；实际在线测试会调用这些服务。原有 21/31 的 EM/F1 测试继续保留。完整依赖、参数、失败分母和对齐边界见 [42 说明](docs/benchmark/42_openseeker_benchmark.md)。

固定语料离线评测现已接入：各方案目录下的 `56_*_browsecomp_plus.sh`（Baseline 为 74） 使用独立 BCP corpus、持久化 BM25 与本地裁判；`57_*_hotpotqa.sh`（Baseline 为 75） 使用有标注 dev、完整官方 Wiki 摘要库和官方 EM/F1。先 `ACTION=download` 一次性准备，之后 prepare/index/run/score/all 均严格离线。两者都支持从 RUN_DIR 读取 Plain/Beacon 训练产物。详细参数、逐步运行与修改提示见 [43 离线基准手册](docs/benchmark/43_offline_benchmarks.md)，官方版本及许可证见 [44 来源说明](docs/benchmark/44_offline_sources.md)。

四卡 A800 的 Beacon 长轨迹训练：先修改 `scripts/sh/beacon/30_beacon_train.sh` 中模型和数据路径，再用 `ACTION=plan bash scripts/sh/beacon/38_beacon_4xa800.sh` 只生成配置；实际训练运行 `bash scripts/sh/beacon/38_beacon_4xa800.sh`。显存原因、重计算修复和参数修改提示见 [52 四卡 A800 长轨迹 OOM](docs/beacon/52_beacon_oom.md)。

Plain 四卡长轨迹训练：`ACTION=plan bash scripts/sh/plain/28_plain_4xa800.sh` 审查配置，实际训练去掉 `ACTION=plan`。参数统一在 `20_plain_train.sh`；分块 loss、激活卸载及限制见 [20 Plain](docs/plain/20_plain.md)。

各方案独立 Benchmark 入口、完整变量表、配置优先级与路径迁移见 [46 脚本手册](docs/benchmark/46_benchmark_scripts.md)。
