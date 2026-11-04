# 11 长度统计口径


当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

用户提供的源文件位于 `../sa-qwen-openseeker/tmp/openseeker_v1_length_distribution.tokens.json`。其 identity 中的 method 是 `sum_content_tokens;add_special_tokens=False;no_truncation`，即逐消息正文分词后相加，未包含 ChatML role 边界和消息结束符。此文件的身份信息、总数、分位数和阈值累计数量已整理在 [11_reference_length_summary.json](11_reference_length_summary.json)，没有将旧长度缓存当作当前数据的精确筛选依据。

当前预处理通过训练共用的 `search_comp.data.openseeker.encode()` 计算 `lengths[模型标识].total`：包括每条消息的 ChatML 边界、system 提示词、问题、所有原始 assistant 和完整工具返回。`observations` 是允许压缩的正文 token 数；`assistant` 是监督 token 数。这一口径对应训练的数据输入，适合选择显存探测样本。

长度是压缩前的序列长度。Beacon 压缩比、observation 占比、工具轮次、思考段长度、模型、LoRA 和辅助蒸馏配置都会影响显存，因此相同 total 不保证相同峰值。更换 tokenizer、模板或轨迹正文后应重新运行 10；只改变长度筛选或损失权重可以运行 11。

以新生成的 `measurements.jsonl` 为准，运行 `44_plot.sh` 生成长度图，运行 `45_report.sh` 输出按正确性分组的长度与质量报告。区间内可能没有样本，精确长度尤其如此；先检查新 manifest 的 counts 再运行训练。

RESEARCH 数据同样按转换后的完整 encode 序列计长，保留筛选前分布与逐条 measurements；具体产物和修改 MAX_LENGTH 的步骤见 [09](09_research_sft.md)。

## 11.1 当前整体分布报告

`45_report.sh` 支持 PREPARED、OUTPUT、MODEL。MODEL 必须与 manifest.tokenizers 键相同；多 tokenizer 时不再取最大值混算。quality_length.json 的 populations.before_filter 为规范化成功的筛选前总体，retained 为实际 train+validation+test，train/validation/test 分别统计；input_counts 与 rejections 解释 invalid 和筛除原因。token 分布来自原 prepare 的真实训练编码长度，不加载 tokenizer 重新估算。

`44_plot.sh` 支持 MEASUREMENTS、MODEL、OUTPUT、POPULATION。all/retained/rejected 按 measurements 的原 prepare rejection 选择；后续重新筛选的数据，以最终 split 的 45 报告为准。字符统计、旧参考长度、字节 tokenizer 单元测试都不能替代训练 tokenizer token 分布。完整命令和完成条件见 [70](70_openseeker_experiment_guide.md#706-筛选前总体保留数据与各-split-的整体分布) 或 [71](71_research_experiment_guide.md#716-筛选前总体保留数据与各-split-的整体分布)。
