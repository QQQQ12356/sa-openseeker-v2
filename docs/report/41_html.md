# 41 数据和预测 HTML

`40_dataset_html.sh`：修改 INPUT、OUTPUT、SPLIT、LIMIT、OFFSET，一键将原始/规范化数据转换为单文件离线 HTML。可查看问题、独立参考答案、原始 assistant 回答、工具 observation、正确性、长度字段和完整原始记录。LIMIT=0 导出全部，较长数据可按 OFFSET/LIMIT 分批。

`41_predictions_html.sh`：修改 INPUTS、OUTPUT_DIR、TITLE，将多个预测 JSONL 转为各自的 HTML，展示 EM/F1、状态、生成轨迹与原始字段。训练/评测没有生成预测之前应先填写有效文件。

模板位于 `scripts/templates/40_dataset.html` 和 `41_predictions.html`，通过文本节点显示轨迹内容。system 示例不会被识别为真实动作，工具正文中的标签按原文保留。数据和预测正文嵌入时做 JSON/HTML 转义；文件可下载后离线浏览，不需要网络服务。模板及脚本有 Python 和 Node 客户端回归覆盖。
