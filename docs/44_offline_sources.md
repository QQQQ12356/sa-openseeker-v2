# 44 离线基准来源、固定版本与许可证

## 44.1 官方参考文件

`search_comp/offline/reference/00_provenance.json` 记录每个参考文件的 SHA256。参考源文件原样保留，包括上游命名与许可证；它们不是本项目编写的生成器，也不是本项目新的可执行入口。首次 prepare/download 会将它们复制到 ROOT_DIR/official，之后对工作副本和随代码附带的副本均校验，不覆盖被修改的官方文件。

| 项目 | 仓库与固定提交 | 保留文件 |
|---|---|---|
| BrowseComp-Plus | [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402)，`046949032b0328319cc9a02663a759ec601d9402` | evaluate_run.py、decrypt_dataset.py、qrel_evidence.txt、qrel_golds.txt、MIT LICENSE |
| HotpotQA | [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot/tree/3635853403a8735609ee997664e1528f4480762a)，`3635853403a8735609ee997664e1528f4480762a` | hotpot_evaluate_v1.py、Apache-2.0 LICENSE.txt |

BrowseComp-Plus 的参考 evaluator 含 vLLM 依赖；本项目不会直接 import 整个模块，而是用 AST 仅加载官方纯函数与 prompt 常量。解密、判决解析、引用和校准函数不重写。本地生成裁判改用 Transformers，引擎差异记录到文档和结果配置。

Hotpot 官方 evaluator 源文件完整保存；执行时只在 AST 中把 ujson 导入替换成标准库 json，避免为 JSON 加速器安装额外包，所有指标函数不改变。默认指标结果是比例，不是百分数。

官方 BCP prompt 字符串包含 `\%` 转义，部分 Python 版本会报告 DeprecationWarning / SyntaxWarning；这来自固定上游源文件，不修改源文件来隐藏该差异。

升级官方版本时：修改 assets.py 的固定 revision，从新提交逐字节更新 reference 文件及许可证，重新生成 00_provenance.json，核对函数和 qrels 格式，运行全部回归，用全新 ROOT_DIR 准备实验。不能只修改 assets.json 使旧代码看起来像新版本。

## 44.2 数据来源与划分

- [BrowseComp-Plus 项目](https://github.com/texttron/BrowseComp-Plus)：固定语料推理搜索基准；问题来自 BrowseComp 的适配集合，使用该项目的 qrels 与 corpus，不与在线 BrowseComp 结果混用。
- [BCP 问题数据](https://huggingface.co/datasets/Tevatron/browsecomp-plus)：test，固定 `144cff8e35b5eaef7e526346aa60774a9deb941f`。原始问题/答案加密，用官方 canary 解密，解密产物仅放 outputs，不把题目明文提交仓库。
- [BCP 独立 corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)：提供 docid/text/url。首次下载解析精确 revision，记录在 assets.json；需要更强的事先版本约束时用 --corpus-revision 指定。
- [HotpotQA 官方主页](https://hotpotqa.github.io/)：使用带答案和 supporting_facts 的 dev。test 无公开答案，本流程不将其用于本地 EM/F1。
- [Hotpot 官方 Wiki 说明](https://hotpotqa.github.io/wiki-readme.html)：选择 2017-10-01 Wiki 的 introductory paragraphs 压缩包；文件 `enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`，官方 MD5 `01edf64cd120ecc03a2745352779514c`。程序验证该 MD5，同时记录 SHA256。
- [Hotpot 官方 HF 数据](https://huggingface.co/datasets/hotpotqa/hotpot_qa)：历史 CMU dev 主机不可达时使用 fullwiki validation 作为备选。保留下载的 Parquet 与 revision；不能用 HF 问题集的 context 代替独立 Wiki 全库。

本项目的 prepared_questions 不包含 dev 的候选 context，也不包含 BCP 参考文档正文。参考答案、qrels 和 supporting_facts 只用于评分，不进入模型 system/question，不作为 search/visit 的返回依据。

## 44.3 授权与使用记录

BCP 仓库与对应数据卡标为 MIT；随代码保留官方 LICENSE。Hotpot 源码采用 Apache-2.0；HotpotQA 数据及该项目发布的 processed Wikipedia 使用 CC BY-SA 4.0，详见官方主页和 Wiki 说明。原始网页/Wikipedia 内容的来源信息保存在 source_url 中。分发数据或发表结果时应遵循各官方许可并引用原论文。

本次仅附带小体积评测参考代码、许可证与 qrels；不提交 corpus、题目明文、模型权重、索引或评测日志。它们仍在 .gitignore 已覆盖的 outputs 下，由下载/准备脚本构建。没有新注册任何 API，也没有向外部服务发送测试题。
