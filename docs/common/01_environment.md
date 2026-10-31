# 01 运行环境

训练环境定义保留在根目录 `environment.train-cu126.yml`、`requirements.train-cu126.txt` 和 `requirements.txt`。默认环境名为 `search-comp-qwen35-cu126`，需先在目标服务器创建或安装。`scripts/sh/common/00_common.sh` 优先使用脚本里的 `PYTHON`，其次通过 conda 或 `~/miniconda3/envs/<CONDA_ENV>/bin/python` 查找，最后使用当前 Python。迁移服务器时可设置 `PYTHON` / `CONDA_ENV`。

预处理仅加载 tokenizer，不占模型训练显存；`OFFLINE=1` 只使用本地缓存。默认 `Qwen/Qwen3.5-4B`，切换模型或 tokenizer 后必须重新测量训练数据。不要将 Qwen3.5 长度直接当成 Qwen3 长度。

测试使用 tiny 随机模型、CPU 和离线模拟器，不下载或训练真实模型。`01_test.sh` 默认不开启 `FULL_MODEL_TEST`；显式开启才会加载真实模型执行额外验证。HTML 与 observation 单独过滤只需 Python 标准库。图表需要 matplotlib，Excel 需要 openpyxl。

`03_check_environment.sh` 是供目标训练服务器执行的 CUDA/CPUAdam 环境探针，会做小张量反向传播与优化器检查，本次没有执行。更完整环境细节见 [67 环境细节](../tools/67_environment_details.md)。

固定语料离线流程使用 SQLite FTS5、pyarrow 和既有模型环境，不要求 Java/Pyserini/vLLM/OpenAI SDK/ujson。56/57 的 prepare/index/run/score/all 禁止网络连接，依赖、数据和模型必须预先准备；只有显式 download 动作可联网。本地裁判仅 BCP 完整答案评分需要，Hotpot 官方 EM/F1 不需要辅助模型。详见 [43](../benchmark/43_offline_benchmarks.md)。

Plain HTTP 服务使用独立环境 `search-comp-sglang`，配置为根目录 `environment.serve-sglang.yml` 和 `requirements.serve-sglang.txt`。48 默认查找该服务环境；不要与训练依赖混装。服务 YAML 已包含 CUDA Toolkit 13.0 与 Ninja；激活后须将 CUDA_HOME/CUDA_PATH/FLASHINFER_NVCC 指向该环境，避免 PyTorch cu130 使用系统 CUDA 11.8 编译。驱动字段区分、模型架构核验及完整命令见 [48 服务部署](../plain/48_sglang_deployment.md)。
