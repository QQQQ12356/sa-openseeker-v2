# 10 Linux 环境准备

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [61 脚本参数清单](61_script_inventory.md)。

## 1. 运行条件

- 使用 Linux x86_64、Python 3.11、可见的 NVIDIA GPU、兼容 CUDA 12.6 的驱动。
- 安装 `git`；使用多卡 ZeRO 时还需可用的 C/C++ 编译器和构建工具。
- 单卡与 DDP 不需要 DeepSpeed；ZeRO-2/3 需要 DeepSpeed。Beacon 不支持 ZeRO-3。
- 预处理只加载 tokenizer，使用 CPU；训练和模型测试使用 GPU。默认合并 LoRA 使用 CPU，需要相应主机内存。
- benchmark Excel 导出只需 Linux CPU 与 `requirements.tools-export.txt`；统一依赖已包含该文件，也可独立安装，见 [82 Excel 导出](82_benchmark_excel.md)。

依赖版本沿用源分支的固定基线，本次没有在 Linux 安装或验证依赖组合。`environment.train-cu126.yml` 提供可选 Conda 系统工具环境描述；统一 Python 安装脚本见下一节。

## 2. 安装实验环境

编辑 `scripts/tools/sh/00_setup_environment.sh`：

| 变量 | 作用 | 修改提示 |
|---|---|---|
| `SA_ENVIRONMENT_DIR` | 独立 Python 环境目录 | 默认 `artifacts/environments/train` |
| `SA_TORCH_SPEC` / `SA_TORCH_INDEX` | CUDA PyTorch wheel 与来源 | 两者应与 CUDA 基线一致，变更后重新验证 |
| `SA_PIP_INDEX` | PyPI 来源 | 可替换为服务器镜像 |
| `HF_ENDPOINT` | Hugging Face 来源 | 可改为 `https://hf-mirror.com` |
| `SA_INSTALL_DEEPSPEED` | 是否安装多卡 ZeRO 依赖 | ZeRO 训练改为 `true` |
| `SA_DEEPSPEED_SPEC` | DeepSpeed 版本约束 | Linux 验证后固定实际版本 |

使用系统 Python 3.11 运行：

```bash
bash scripts/tools/sh/00_setup_environment.sh
source artifacts/environments/train/bin/activate
bash scripts/tools/sh/90_validate.sh
```

脚本先安装指定 CUDA PyTorch，再安装统一依赖，按需安装 DeepSpeed，并将实际依赖写入环境内 `installed_requirements.txt`。安装失败时按终端提示修复系统条件，再重新执行。后续 SH 默认通过 PYTHON_BIN 使用已激活环境中的 python。

## 3. 每个脚本的环境参数

SH 先 export 环境变量；PY 在导入模型、tokenizer 或 Hub 客户端前读取它们。修改待运行文件即可，不依赖其他方法的环境文件。

| 参数 | 作用 | 规则 |
|---|---|---|
| `HF_ENDPOINT` | Hub API/模型下载地址 | 下载时可选官方或镜像 |
| `HF_HUB_OFFLINE` | Hub 离线开关 | 下载脚本 `0`，本地训练/测试/评分 `1` |
| `CUDA_VISIBLE_DEVICES` | 可见物理卡编号 | 单卡如 `0`，多卡如 `0,1,2,3` |
| `NPROC_PER_NODE` | 训练进程数 | 与参与训练的可见卡数一致，每进程一张卡 |
| `DISTRIBUTED_BACKEND` | 多卡训练方式 | `ddp` / `zero2`；Plain 另支持 `zero3` |

共享运行层默认将 Hugging Face、PyTorch、XDG 缓存归档到 `artifacts/cache/`；如服务器已配置同名缓存环境变量，保留已有值。`HF_TOKEN` 等私密凭证在服务器环境中配置，脚本不打印或归档其值。

## 4. BrowseComp 在线依赖

BrowseComp 通过固定版本的 OpenSeeker 搜索/访问工具连接外网。测试脚本读取以下服务器环境变量：

| 变量 | 作用 |
|---|---|
| `SEARCH_PROVIDER` | 脚本内选择 `serper` 或 `tavily` |
| `SERPER_API_KEY` / `TAVILY_API_KEY` | 对应搜索服务密钥 |
| `JINA_API_KEYS` | 页面读取服务密钥 |
| `SUMMARY_API_URL` | 页面总结模型基础地址，如 `http://host:8000/v1`，程序追加 `/chat/completions` |
| `SUMMARY_API_KEY` | 总结服务密钥 |
| `SUMMARY_MODEL_NAME` | 总结模型名称 |

BrowseComp 独立评分读取：

| 变量 | 作用 |
|---|---|
| `SCORER_URLS` | 兼容 Chat Completions 的裁判服务基础地址，如 `http://host:8000/v1`；逗号分隔多个地址，程序追加 `/chat/completions` |
| `SCORER_API_KEY` | 裁判服务密钥 |
| `SCORER_MODEL_NAME` | 裁判模型名称 |

在服务器环境配置真实值，脚本使用 `os.environ.get()` 读取。环境值不会写入 `job.json`；运行脚本会被复制归档，因此不要把密钥字面量写入脚本。

## 5. 离线 benchmark

BrowseComp-Plus 和 HotpotQA 先显式下载，再处理、建索引。测试与评分阶段禁止联网，不能使用在线裁判地址替代 `SA_JUDGE__JUDGE_MODEL` 的本地模型目录。GPU 多进程仅用于测试模型副本；裁判在独立评分进程加载。

## 6. 在线工具的资源限制

各 BrowseComp 测试脚本可直接修改 SUMMARY_ATTEMPTS（1–10，默认 3）、SUMMARY_MAX_INPUT_CHARS（默认 40000）、SUMMARY_MAX_TOKENS（默认 2048）、SUMMARY_TIMEOUT（默认 120 秒）与 JINA_TIMEOUT（默认 50 秒）。这些值约束页面读取和总结，独立于待测模型的 GENERATION 预算。
