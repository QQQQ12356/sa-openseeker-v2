# 41 数据和预测 HTML

按实验顺序使用 HTML，见 [70 OpenSeeker](../data/70_openseeker_experiment_guide.md) 或 [71 Research](../data/71_research_experiment_guide.md)：先查原始记录，再查规范化训练行，最后查模型预测。HTML 抽样不能替代全量分布报告。

`40_dataset_html.sh`：修改 INPUT、OUTPUT、SPLIT、LIMIT、OFFSET，一键将原始/规范化数据转换为单文件离线 HTML。可查看问题、独立参考答案、原始 assistant 回答、工具 observation、正确性、长度字段和完整原始记录。LIMIT=0 导出全部，较长数据可按 OFFSET/LIMIT 分批。

`41_predictions_html.sh`：修改 INPUTS、OUTPUT_DIR、TITLE，将多个预测 JSONL 转为各自的 HTML，展示 EM/F1、状态、生成轨迹与原始字段。训练/评测没有生成预测之前应先填写有效文件。

模板位于 `scripts/templates/40_dataset.html` 和 `41_predictions.html`，通过文本节点显示轨迹内容。system 示例不会被识别为真实动作，工具正文中的标签按原文保留。数据和预测正文嵌入时做 JSON/HTML 转义；文件可下载后离线浏览，不需要网络服务。模板及脚本有 Python 和 Node 客户端回归覆盖。

RESEARCH 原始 JSON 字符串 messages 支持 reasoning/tool_call/tool_output/answer 的角色筛选与展示。使用 ACTION=html LIMIT=20 OFFSET=0 的一键入口，参数和原始/规范化视图区别见 [09](../data/09_research_sft.md)。
