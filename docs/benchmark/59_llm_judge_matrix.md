# 59 三种方法、三个 Benchmark 的独立评分矩阵

当前运行顺序与所有变量见 [68 操作指南](../project/68_training_benchmark_guide.md)，训练参数见 [69](../project/69_training_parameters.md)。训练数据为 OpenSeeker / Research；每种数据训练出的模型均运行三个测试。Baseline 新入口使用全参数 SFT；旧 baseline/70–75 是未训练基座的历史对照。

## 59.1 独立入口

| 目录 | BrowseComp 测试 / 评分 | BrowseComp-Plus 测试 / 评分 | HotpotQA 测试 / 评分 |
|---|---|---|---|
| scripts/sh/plain | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |
| scripts/sh/beacon | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |
| scripts/sh/baseline | 40_test_browsecomp.sh / 41_score_browsecomp.sh | 42_test_browsecomp_plus.sh / 43_score_browsecomp_plus.sh | 44_test_hotpotqa.sh / 45_score_hotpotqa.sh |

每个文件有自己的参数，直接运行所选文件；没有统一方法分发入口。测试使用训练实验的 RUN_DIR，或 MODEL_PATH+规范化 PROMPT_DATA；评分使用 PREDICTIONS 和独立的新 OUTPUT_DIR，不加载待测模型。

## 59.2 评分内容

| Benchmark | EM/F1 | LLM judge | 接入 |
|---|---|---|---|
| BrowseComp | 补充答案 em/f1 | llm_judge_accuracy，等于 accuracy_all_questions | SCORER_URLS / SCORER_API_KEY / SCORER_MODEL_NAME |
| BrowseComp-Plus | 补充答案 em/f1，移除方括号引用 | llm_judge_accuracy，等于 accuracy；仍输出 Accuracy (%) | 本地 JUDGE_MODEL + 固定官方 prompt/parser |
| HotpotQA | 官方 em/f1，保留 yes/no 特例 | llm_judge_accuracy；答案语义等价补充，非官方分数 | 本地 JUDGE_MODEL + hotpot-answer-equivalence-v1 |

所有数值字段 em/f1/llm_judge_accuracy 均为 0–1。BCP 原检索召回、引用和校准仍输出；Hotpot 原支持事实及 Joint 指标仍输出。三项基准使用不同语料/工具环境，其成绩不可互换。

## 59.3 一键独立评分

```bash
# 在线 BrowseComp：ENV_FILE 中配置裁判服务，生成工具服务不在评分阶段使用。
ENV_FILE=/private/benchmark.env PREDICTIONS=/results/bc/result_tool200.jsonl \
  bash scripts/sh/plain/41_score_browsecomp.sh

# 离线两项：本地裁判目录必须存在，建议评分阶段独占 GPU。
PREDICTIONS=/results/bcp/predictions.jsonl JUDGE_MODEL=/models/judge \
  bash scripts/sh/plain/43_score_browsecomp_plus.sh
PREDICTIONS=/results/hotpot/predictions.jsonl JUDGE_MODEL=/models/judge \
  bash scripts/sh/plain/45_score_hotpotqa.sh
```

新正式评分入口必须计算三项指标，不接受跳过 LLM judge 的开关。底层 Python 及旧入口保留仅检索诊断、Hotpot 仅官方分数等历史能力。新生成入口不要求裁判，切勿把旧 ACTION=all 示例用作新测试入口的评分配置。

## 59.4 失败口径与裁判控制

所有选定问题保留在分母，缺失、失败状态和空答案计零；在线对齐阶段会将显式失败状态的回答置空后评分。裁判的传输、格式错误可有限重试，有效负判不重试；重试耗尽保留未知状态，complete=false 且非零退出。默认正负例校准，SKIP_JUDGE_PREFLIGHT=1 跳过需另行记录原因。

JUDGE_ATTEMPTS 默认 3；JUDGE_ENABLE_THINKING 默认 0；在线 JUDGE_MAX_TOKENS=1024、JUDGE_TIMEOUT=120、JUDGE_WORKERS=1；离线 JUDGE_MAX_TOKENS=4096、JUDGE_MAX_CONTEXT=32768、JUDGE_TEMPERATURE=0、JUDGE_TOP_P=0.8、JUDGE_TOP_K=20、SEED=42。采样相关参数只在温度为正时生效。完整含义和修改位置见 68.6；已有稳定裁判实现细节见 [54](54_stable_judging.md)。
