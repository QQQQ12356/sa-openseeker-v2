# 11 长度统计口径

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

用户提供的源文件位于 `../sa-qwen-openseeker/tmp/openseeker_v1_length_distribution.tokens.json`。其 identity 中的 method 是 `sum_content_tokens;add_special_tokens=False;no_truncation`，即逐消息正文分词后相加，未包含 ChatML role 边界和消息结束符。此文件的身份信息、总数、分位数和阈值累计数量已整理在 [11_reference_length_summary.json](11_reference_length_summary.json)，没有将旧长度缓存当作当前数据的精确筛选依据。

当前预处理通过训练共用的 `search_comp.data.openseeker.encode()` 计算 `lengths[模型标识].total`：包括每条消息的 ChatML 边界、system 提示词、问题、所有原始 assistant 和完整工具返回。`observations` 是允许压缩的正文 token 数；`assistant` 是监督 token 数。这一口径对应训练的数据输入，适合选择显存探测样本。

长度是压缩前的序列长度。Beacon 压缩比、observation 占比、工具轮次、思考段长度、模型、LoRA 和辅助蒸馏配置都会影响显存，因此相同 total 不保证相同峰值。更换 tokenizer、模板或轨迹正文后应重新运行 10；只改变长度筛选或损失权重可以运行 11。

以新生成的 `measurements.jsonl` 为准，运行 `44_plot.sh` 生成长度图，运行 `45_report.sh` 输出按正确性分组的长度与质量报告。区间内可能没有样本，精确长度尤其如此；先检查新 manifest 的 counts 再运行训练。
