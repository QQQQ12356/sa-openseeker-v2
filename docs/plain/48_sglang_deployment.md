# 48 Plain 模型 SGLang 部署：独立 Conda 环境、启动与联调

入口：`scripts/sh/plain/48_serve_checkpoint.sh`。它读取本地已合并 Plain checkpoint，通过 `python -m sglang.launch_server` 提供 HTTP completion 服务。训练后的 Qwen3 与 Qwen3.5 必须分别核对架构支持；Beacon 使用 native 推理。

本文给出可执行的环境创建、启动、检查和评测连接命令。本地验证限于配置与文档。用户提供的远端日志已显示 Qwen3.5 权重加载成功，但在 FlashInfer JIT 阶段失败；新的 CUDA Toolkit 配置仍需远端重启验证，不能视作服务已完成联调。

## 48.1 先确认模型是否适合这个服务入口

| 模型产物 | 处理方式 |
|---|---|
| Plain Qwen3 完整模型，架构为 `Qwen3ForCausalLM` | 在支持该模型的 SGLang 环境执行下文步骤 |
| 原始 Qwen3.5 多模态模型，架构为 `Qwen3_5ForConditionalGeneration` | SGLang 有对应原生入口；它是原始基座，不能代替已训练 Plain 权重 |
| 本项目 Qwen3.5 Plain 纯文本训练产物，常见为 `qwen3_5_text` / `Qwen3_5ForCausalLM` | 用户提供的远端日志已显示该架构完成权重加载；仍需修复 JIT 工具链并验证实际请求 |
| 只有 `adapter_config.json` 的 LoRA 目录 | 先在训练环境合并；48 会拒绝此目录 |
| Beacon 模型或带 Beacon 配置的产物 | 48 会拒绝，使用 `beacon/47_beacon_browsecomp.sh` 的 native 模式 |

**以实际日志更新兼容性结论。** 本项目保存的是纯文本 `Qwen3_5ForCausalLM`。此前根据上游 `EntryClass` 对直接加载提出兼容性疑问；用户提供的远端日志已显示 `Load weight end ... type=Qwen3_5ForCausalLM ... mem usage=7.96 GB`，并完成 Mamba 状态缓存和 KV Cache 分配。因此这次失败的直接原因是 CUDA 编译工具链选择，不能归因于模型架构不支持。加载成功仍不代表权重数值、所有执行路径及 completion 输出已通过验证。

48 的 dry-run 仅检查路径和配置。其他版本或产物若确实出现架构不支持、权重名不匹配、缺失 `text_config` 等问题，仍需保留日志核对，不要只改架构名称或更换原始基座来代替训练权重。当前远端环境应先按 48.3 修复 CUDA 13.0 Toolkit 选择，再检查 health 与真实请求。
依据：[本项目文本主干加载器](../../search_comp/milestones/qwen35_text.py)、[SGLang v0.5.23 Qwen3.5 实现](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/qwen3_5.py)、[模型注册表](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/registry.py)。

## 48.2 独立环境与硬件前提

建议环境名 **`search-comp-sglang`**，与训练环境 `search-comp-qwen35-cu126` 分开。两者只共享磁盘上的模型目录。服务端不需要安装本项目的训练依赖或官方评测依赖；48 的路径解析仅使用 Python 标准库，随后交由 SGLang 加载模型。

本仓库新增配置：

- 根目录 `environment.serve-sglang.yml`：创建 Python 3.12、pip、CUDA Toolkit 13.0 和 Ninja 的 Conda 环境；Toolkit 提供 JIT 所需 nvcc 与开发文件。
- 根目录 `requirements.serve-sglang.txt`：固定 `sglang==0.5.23`；其余依赖由该发行版解析，安装成功后再冻结。

核对的 PyPI 元数据中，0.5.23 依赖包括 `torch==2.14.1`、`transformers==5.19.0`、`flashinfer_python[cu13]==0.7.2` 和 `sglang-kernel==0.4.10`。不要在这个环境先装训练所用的 torch 2.7/cu126，再强行安装 SGLang；不要混装 `requirements.txt`、`requirements.train-cu126.txt` 或 `requirements.eval-openseeker.txt`。

用户已确认远端使用 `torch 2.14.1+cu130`、`torch.version.cuda=13.0`、`sglang 0.5.23` 和 `flashinfer-python 0.7.2`。因此本配置将编译工具链固定为 **CUDA Toolkit 13.0**，与 runtime 对齐。CUDA 13.x 的驱动兼容基线是 R580 或更高；具体 CUDA 小版本、JIT/PTX、GPU 架构及内核还可能有更高要求。`nvidia-smi` 显示的 CUDA Version 是驱动能力，不是当前 Conda 环境的 CUDA runtime 版本。Conda 无法替换宿主机 NVIDIA 驱动。如果目标服务器仍是 CUDA 12.6 时代的旧驱动，不要直接套用这份环境；需由管理员安排驱动兼容方案，或重新选择并验证较旧的 SGLang/PyTorch/内核组合。

Toolkit 安装方法参考 [NVIDIA Conda 安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation)。

来源：[SGLang 安装说明](https://docs.sglang.io/docs/get-started/install)、[0.5.23 包元数据](https://pypi.org/pypi/sglang/0.5.23/json)、[NVIDIA CUDA 驱动兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。这里没有将“最后支持 CUDA 12 的镜像标签”当成同版本 PyPI wheel 支持 CUDA 12 的保证。

用户口述的 `13.3` 通常对应 `nvidia-smi` 的 **CUDA Version**，不是 Driver Version。驱动版本一般形如 `580.xx` 或 `590.xx`，用下方 `--query-gpu=...driver_version...` 明确查询。若页面实际显示 CUDA Version 13.3，表示驱动宣告的 CUDA 能力；无需据此把本环境 Toolkit 从 13.0 升为 13.3。编译器版本由 `nvcc --version` 确认。

在实际运行模型服务的计算节点检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
command -v conda
```

A100/A800 等 NVIDIA GPU 的可用内核取决于安装版本和模型；首轮用 BF16/FP16 原权重，避免把 FP8 或量化兼容问题叠加进来。4B 模型仅按 2 字节/参数估算，权重约 8 GB；还需要 KV/线性状态缓存、激活、CUDA graph、并发和运行时空间，该估算不能作为整机显存下限。先以 8192 上下文做部署检查，再逐步增加至正式预算；本脚本默认 131072，并不保证单卡能承载。

## 48.3 创建和安装独立 Conda 环境

在仓库根目录执行。下列步骤是供目标服务器执行的安装指令，本次修改没有执行安装。

```bash
cd /home/huangzj/proj/searchagent/sa-openseeker-v2
conda env create -f environment.serve-sglang.yml
conda activate search-comp-sglang

# 显式绑定解释器，防止继承之前终端设置的训练环境 PYTHON
export PYTHON="$CONDA_PREFIX/bin/python"
export CONDA_ENV=search-comp-sglang

# 显式选择环境内的 Toolkit，避免 /usr/local/cuda -> cuda-11.8
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
hash -r

# 确认这里显示 CUDA 13.0，并能查到 compress-mode，再安装服务依赖
test -x "$FLASHINFER_NVCC"
"$FLASHINFER_NVCC" --version
"$FLASHINFER_NVCC" --help | grep compress-mode
"$PYTHON" -m pip install --upgrade pip uv
"$PYTHON" -m uv pip install --python "$PYTHON" --prerelease=allow \
  -r requirements.serve-sglang.txt
"$PYTHON" -m pip check
```

已有 `search-comp-sglang` 环境用以下命令补齐依赖，无需删除或重建环境，也不需要更改系统 CUDA 链接：

```bash
conda env update -n search-comp-sglang -f environment.serve-sglang.yml
conda activate search-comp-sglang
export PYTHON="$CONDA_PREFIX/bin/python"
export CONDA_ENV=search-comp-sglang
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
hash -r
"$FLASHINFER_NVCC" --version
"$FLASHINFER_NVCC" --help | grep compress-mode
"$PYTHON" -m pip check
```

上面的更新会按 YAML 安装 CUDA 13.0 Toolkit 和 Ninja；已确认版本正确的 SGLang/PyTorch 不需要为这次编译器错误单独重装。更新后再次核对包版本。若更改环境名，用 `conda env create -n 自定义名称 -f environment.serve-sglang.yml`，并同步 `CONDA_ENV` 或显式 `PYTHON`。

安装时保留解析器选择的依赖，不使用 `--no-deps` 绕过冲突。目标节点无外网时，应先在匹配 Linux/CPU 架构/Python/CUDA 的联网环境准备 wheelhouse；有源码构建/JIT 依赖时仅复制 wheel 文件未必足够，需同时准备编译工具链和缓存。不要复制另一个服务器的 Conda 目录就认定 CUDA 内核兼容。

检查包版本与 GPU runtime：

```bash
"$PYTHON" - <<'PY'
import sys
from importlib.metadata import version
import torch
print('python:', sys.executable)
for name in ('sglang', 'torch', 'transformers', 'flashinfer-python'):
    print(name, version(name))
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
"$PYTHON" -m sglang.launch_server --help
```

### 48.3.1 固定 CUDA 路径并持久化

FlashInfer 编译器选择优先检查 `FLASHINFER_NVCC`；Toolkit 根目录通过 `CUDA_HOME`/`CUDA_PATH` 指定，缺省还可能从 PATH 或 `/usr/local/cuda` 查找。仅让 `command -v nvcc` 指向新版本不足以覆盖旧的 CUDA_HOME 或 FLASHINFER_NVCC；请同步设置三项。48 脚本只继承这些变量，不会自动激活环境或替换工具链。

确认 `$CONDA_PREFIX/bin/nvcc` 存在并输出 13.0 后，可把绝对路径保存在当前 Conda 环境配置中，每次激活自动恢复：

```bash
conda env config vars set -n search-comp-sglang \
  CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  FLASHINFER_NVCC="$CONDA_PREFIX/bin/nvcc"
conda deactivate
conda activate search-comp-sglang
export PYTHON="$CONDA_PREFIX/bin/python"
"$FLASHINFER_NVCC" --version
```

环境路径搬迁后需要重新设置这些绝对路径；新终端或作业必须重新 activate，或在作业脚本中显式 export。不要向 YAML 的 variables 字段写入字面量 `$CONDA_PREFIX` 并假设它一定会展开。如果根目录没有 nvcc，先执行 `find "$CONDA_PREFIX" -type f -name nvcc` 核对 Conda 包布局与安装结果；不要把缺失路径持久化。

也可使用管理员已安装的系统 CUDA 13.0 Toolkit，但 CUDA_HOME/CUDA_PATH/FLASHINFER_NVCC 必须共同指向该 Toolkit。避免继承 CUDA 11.8 的 LD_LIBRARY_PATH；不需要为了选编译器给 LD_LIBRARY_PATH 全局添加 CUDA 的所有库，更不要添加 stubs 目录。宿主机仍必须提供兼容驱动。

## 48.4 准备模型目录

有两种选择方式：

1. `RUN_DIR=/absolute/path/to/experiment`：读取 `experiment.json` 和 `models/model/train_summary.json` 的 `final_model_path`。
2. `MODEL_PATH=/absolute/path/to/merged-model`：直接指定完整模型目录。两者同时设置时 MODEL_PATH 决定权重，但仍读取 RUN_DIR 的实验记录；移动实验后建议只设置 MODEL_PATH，避免旧记录和相对路径干扰。

本项目 48 要求本地目录，不直接接受 `Qwen/...` Hub ID。目录应有正确 `config.json`、tokenizer 文件及完整权重分片/索引，且没有 `adapter_config.json`。48 只做有限路径/类型检查，dry-run 不检查所有权重或 tokenizer 文件是否完整。

```bash
export MODEL_PATH=/absolute/path/to/merged-plain-model
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['MODEL_PATH']).expanduser().resolve()
c = json.loads((p / 'config.json').read_text())
print('model:', p)
print('model_type:', c.get('model_type'))
print('architectures:', c.get('architectures'))
print('adapter:', (p / 'adapter_config.json').exists())
print('safetensors files:', len(list(p.glob('*.safetensors'))))
print('tokenizer_config:', (p / 'tokenizer_config.json').exists())
PY
```

如果训练保存的是 adapter，回到**训练环境**合并到新目录，部署仍使用服务环境：

```bash
# 单独的训练环境终端；adapter 路径替换为实际值
conda activate search-comp-qwen35-cu126
export PYTHON="$CONDA_PREFIX/bin/python"
DEVICE_MAP=cpu bash scripts/sh/plain/29_plain_merge_adapter.sh \
  /absolute/path/to/adapter --output_path /absolute/path/to/new-merged-model
```

合并需要原基座权重、PEFT 和足够主机内存；不要为了合并给服务环境覆盖安装训练依赖。正常训练已自动合并成功时直接使用最终模型目录。路径解析以仓库根目录为基准；部署和跨节点实验尽量使用绝对路径。

## 48.5 48 脚本全部变量及 CLI 边界

| 变量 | 默认 | 用途与修改提示 |
|---|---|---|
| `PYTHON` | 空 | 可执行解释器路径；优先级最高，建议设为服务环境的绝对路径 |
| `CONDA_ENV` | `search-comp-sglang` | 自动查找的环境名；脚本不会创建/激活环境 |
| `RUN_DIR` | 空 | 训练实验根目录，用于自动定位最终权重 |
| `MODEL_PATH` | 空 | 本地完整模型目录，优先于实验记录；不是 adapter 目录 |
| `CUDA_HOME` / `CUDA_PATH` | 启动脚本继承环境 | 本方案设置为服务环境的绝对路径，选择 CUDA 13.0 Toolkit |
| `FLASHINFER_NVCC` | 启动脚本继承环境 | 本方案设置为 `$CONDA_PREFIX/bin/nvcc`；显式覆盖系统 CUDA 11.8 编译器 |
| `CUDA_VISIBLE_DEVICES` | `0` | 对服务可见的物理卡列表，例如 `2,3`；进程内部重新编号为 0,1 |
| `HOST` | `127.0.0.1` | 监听地址；跨节点用 `0.0.0.0`，客户端用真实节点名/IP |
| `PORT` | `30018` | HTTP 端口，1—65535，启动前确认未被占用 |
| `TP_SIZE` | `1` | 张量并行卡数；不大于可见卡数，且须符合模型 head/分片约束；不是 NUM_WORKERS |
| `CONTEXT_LENGTH` | `131072` | 最大总上下文 token 预算；包含 prompt 和后续生成，不是单次回答长度 |
| `DRY_RUN` | `0` | 1 检查本地路径及 Beacon 标志并打印命令；不导入 SGLang、不占 GPU、不验证 runtime |

`PYTHON` 必须是可执行路径；若留空，公共脚本尝试 CONDA_ENV，找不到时会回退当前 Python。所以仅设置 CONDA_ENV 不保证环境确实存在；生产部署显式指定 PYTHON。脚本已将默认环境改为服务专用名称，旧服务作业若依赖原默认值，应显式设置原 CONDA_ENV/PYTHON。

末尾参数只传给本项目 `48_serve_checkpoint.py`，支持 `--run-dir`、`--model`、`--host`、`--port`、`--tp-size`、`--context-length`、`--dry-run`。当前**不会透传任意 SGLang 参数**；在 48 后直接加 `--mem-fraction-static` 或 `--served-model-name` 会被 argparse 拒绝。需要高级参数时使用 48.8 的直接启动方式。命令末尾重复的本项目标量参数通常后者生效，日常建议统一使用环境变量配置。

## 48.6 单卡、多卡与跨节点启动

先在服务环境的终端检查实际命令：

```bash
conda activate search-comp-sglang
export PYTHON="$CONDA_PREFIX/bin/python"
export MODEL_PATH=/absolute/path/to/merged-plain-model

DRY_RUN=1 CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 CONTEXT_LENGTH=8192 \
  bash scripts/sh/plain/48_serve_checkpoint.sh

# 真实启动：保持该终端运行；另开终端做健康检查
CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 CONTEXT_LENGTH=8192 \
  bash scripts/sh/plain/48_serve_checkpoint.sh
```

启动器打印 `[serve]` 只说明即将执行，不表示 HTTP 服务就绪。等模型加载和内核初始化完成后再访问 health。首次启动可能需要编译/下载内核；时间取决于环境与缓存。

已有训练实验可替换为 RUN_DIR：

```bash
# 清除前面示例导出的 MODEL_PATH，使 RUN_DIR 自动选择权重
unset MODEL_PATH
RUN_DIR=/absolute/path/to/plain-run CONTEXT_LENGTH=8192 \
  bash scripts/sh/plain/48_serve_checkpoint.sh
```

双卡张量并行及跨节点访问示例（仍为单节点部署模型）：

```bash
MODEL_PATH=/absolute/path/to/merged-plain-model \
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 CONTEXT_LENGTH=32768 \
HOST=0.0.0.0 PORT=30018 \
  bash scripts/sh/plain/48_serve_checkpoint.sh
```

`HOST=0.0.0.0` 表示在所有本地接口监听。启动器打印的 `http://0.0.0.0:...` 不能直接作为客户端地址；同机使用 127.0.0.1，其他节点使用如 `http://gpu-node-01:30018`。确保调度器分配的节点/卡与实际命令一致，节点之间能连通该端口。默认入口不设置 API key，适用于受控本地/集群网络；如需认证，在直接启动方式中配置服务认证并同步客户端。

## 48.7 健康检查、model ID 与第一次请求

另开终端，在服务所在节点运行：

```bash
conda activate search-comp-sglang
export PYTHON="$CONDA_PREFIX/bin/python"
export SERVICE_URL=http://127.0.0.1:30018
curl --fail --show-error "$SERVICE_URL/health"
curl --fail --show-error "$SERVICE_URL/v1/models"
```

health 成功后，从 `/v1/models` 的 `data[].id` 取得实际模型 ID。默认应对应启动器打印的绝对模型路径；以服务实际响应为准，不要随意起别名。

下面只检查 completion 链路，不作为模型质量或 benchmark 结果：

```bash
export OPENSEEKER_BASE_URL="$SERVICE_URL/v1/completions"
export OPENSEEKER_MODEL=/absolute/path/to/merged-plain-model
"$PYTHON" - <<'PY'
import json, os, urllib.request
payload = dict(model=os.environ['OPENSEEKER_MODEL'],
               prompt='Reply with OK.', max_tokens=16, temperature=0, stream=False)
req = urllib.request.Request(os.environ['OPENSEEKER_BASE_URL'],
    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as r:
    result = json.load(r)
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result.get('choices'), 'No completion choices in response'
PY
```

该请求是原始文本 completion；不会自动应用训练 chat template，所以输出未必只包含 OK。质量检查需使用与训练一致的序列化提示。确认响应中 choices、finish_reason、usage 等字段合理后，再以少量正式 benchmark 题测试。端点能响应不能证明权重正确加载或解码/提示与训练完全对齐；需同时检查启动日志是否缺权重及预测轨迹。

## 48.8 高级 SGLang 参数、显存与日志

如果默认配置发生 OOM，可先缩短上下文或降低并发。需要显存比例、prefill、served model alias 等参数时直接调用同一服务环境的 SGLang。以下命令与 48 使用相同完整模型目录，但不会执行本项目的 Beacon/adapter 检查，请先按前文确认模型：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sglang.launch_server \
  --model-path /absolute/path/to/merged-plain-model \
  --host 127.0.0.1 --port 30018 --tp-size 1 --context-length 8192 \
  --mem-fraction-static 0.80 --max-running-requests 1 \
  --chunked-prefill-size 2048
```

`mem-fraction-static` 控制静态内存池比例，降低它可为运行时留下空间，但也可能使缓存容量不足；不保证解决模型权重本身放不下的问题。`max-running-requests` 限制并发；`chunked-prefill-size` 降低长输入 prefill 峰值。不要把低上下文检查配置的结果称为原 131072 预算的正式结果。高级参数随版本变化，以当前环境的 `--help` 为准。参考 [SGLang 服务参数](https://docs.sglang.io/docs/advanced_features/server_arguments)。

前台运行并保存日志（在 tmux 或已有持久终端内执行）：

```bash
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/absolute/path/to/merged-plain-model \
CONTEXT_LENGTH=8192 bash scripts/sh/plain/48_serve_checkpoint.sh \
  2>&1 | tee "outputs/logs/sglang_plain_$(date +%Y%m%d_%H%M%S).log"
```

脚本使用 exec 启动服务，前台 Ctrl+C 可停止；退出后用 health 与 nvidia-smi 确认服务/worker 已退出。集群作业结束也会回收服务，不要在 benchmark 仍运行时释放作业。多实例使用不同端口和明确的 GPU 分配。

验证成功后记录完整环境，文件名加时间避免混淆：

```bash
"$PYTHON" -m pip freeze > outputs/logs/sglang_pip_freeze.txt
conda env export -n search-comp-sglang > outputs/logs/sglang_conda_export.yml
```

同时记录模型目录/版本、GPU、驱动、torch.version.cuda、完整启动参数和日志。freeze 文件是在目标机器实测后产生的快照，比仅固定顶层 sglang 版本更适合复现。

## 48.9 连接本项目 BrowseComp 评测

服务终端保持运行；另开**评测环境**终端。official-v1/v2 通过 HTTP 请求模型，评测环境不必安装 SGLang；native 则直接加载模型，不经过这个服务。

```bash
# 换成已安装 requirements.eval-openseeker.txt 的评测环境解释器
export EVAL_PYTHON=/absolute/path/to/eval-env/bin/python
unset RUN_DIR MODEL_PATH PROMPT_DATA

# 私有文件按 benchmark/49 示例配置搜索、Jina、摘要及裁判服务
# 并设置 OPENSEEKER_BASE_URL=http://gpu-node-01:30018/v1/completions
# 以及 OPENSEEKER_MODEL=/absolute/path/to/merged-plain-model
# 先 dry-run 检查 official 模式配置；此检查不调用 API
PYTHON="$EVAL_PYTHON" MODE=official-v1 DRY_RUN=1 LIMIT=2 \
ENV_FILE=/private/path/benchmark.env.sh \
  bash scripts/sh/plain/47_plain_browsecomp.sh

# 确认服务与外部工具可用后实际运行 2 题
PYTHON="$EVAL_PYTHON" MODE=official-v1 LIMIT=2 \
ENV_FILE=/private/path/benchmark.env.sh \
  bash scripts/sh/plain/47_plain_browsecomp.sh
```

official 模式不要继承 `RUN_DIR`/`MODEL_PATH` 或 `PROMPT_DATA`；这些属于 native 或服务部署参数。可在评测终端先 `unset RUN_DIR MODEL_PATH PROMPT_DATA`。official-v2 还需要 E2B 沙箱配置；48 服务只部署待测模型，不提供搜索、摘要、裁判或 E2B。原样官方模式的提示、停止策略与 native 不完全相同，见 [42 在线评测说明](../benchmark/42_openseeker_benchmark.md)。

如果本项目 Qwen3.5 纯文本产物无法由选定 SGLang 加载，可在已配置训练+评测依赖的 native 环境保留原权重完成评测：

```bash
PYTHON=/absolute/path/to/native-env/bin/python \
RUN_DIR=/absolute/path/to/plain-run MODE=native LIMIT=2 \
ENV_FILE=/private/path/benchmark.env.sh \
  bash scripts/sh/plain/47_plain_browsecomp.sh
```

此路径没有启动 HTTP 服务，也不宣称解决了 SGLang 架构兼容问题；它使用项目原有 Plain 加载器，并保留实际训练权重。离线 BCP/HotpotQA 同样无需 48 服务，入口见 [46 脚本手册](../benchmark/46_benchmark_scripts.md)。

## 48.10 常见故障与定位

| 现象 | 检查与处理 |
|---|---|
| `No module named sglang` | 检查 PYTHON 是否仍指训练环境；用该解释器执行 pip show sglang |
| `torch.cuda.is_available()` 为 False、driver insufficient | 在实际 GPU 节点检查卡分配、驱动与 CUDA 13 runtime；仅创建 Conda 环境不会升级驱动 |
| `Unsupported architecture` / `Qwen3_5ForCausalLM` 回退失败 | 核对 48.1 的纯文本限制，保留原 checkpoint；当前可改用 native，不能通过改架构名称证明兼容 |
| `text_config`、缺失权重、意外 vision 参数 | checkpoint 格式与后端模型类不匹配；停止并核对实际架构/权重键，不忽略大量 missing keys |
| `merge the LoRA adapter first` | 使用训练环境的 29 合并后指定新输出目录 |
| `Beacon needs --mode native` | 使用 beacon/47 native，不能作为普通 Plain 模型部署 |
| `nvcc fatal: Unknown option --compress-mode=size` | 远端已确认 runtime 13.0 配上系统 nvcc 11.8；按 48.3 安装 Toolkit 13.0 并同时设置 CUDA_HOME/CUDA_PATH/FLASHINFER_NVCC，重启进程；缩短上下文和关闭 CUDA graph 不修复此工具链问题 |
| `CUDA_HOME` / nvcc / PTX 错误 | 核对匹配的 CUDA Toolkit、编译器及驱动；不复用训练环境的旧 CUDA_HOME/LD_LIBRARY_PATH |
| 显存不足 | 缩短上下文、降低并发和 prefill，或使用符合模型约束的 TP；先检查是否还有其他进程占卡 |
| NCCL/TP hang | GPU 数量与 TP、节点分配及互联是否正确；先单卡小上下文复现 |
| connection refused / timeout | 等待加载完成；检查端口、进程日志、服务节点、HOST 与客户端地址 |
| 404 / model not found | completion 路径是 `/v1/completions`，model ID 从 `/v1/models` 读取 |
| 48 报 unrecognized arguments | 48 不透传高级 SGLang 参数，按 48.8 直接启动 |
| 服务可用但回答异常 | 用训练一致提示检查；核对 tokenizer、权重完整性、停止 token、上下文与解码参数 |

## 48.11 本次修改与验证边界

新增独立 Conda 与 SGLang 依赖配置；48 默认环境改为 `search-comp-sglang`，新增 HOST 环境变量和 DRY_RUN 的 0/1 校验，其余启动流程保持不变。实际默认值与旧 benchmark 迁移后的脚本保持一致（上下文仍为 131072）。旧调用可显式设定原 CONDA_ENV/PYTHON。

本次没有创建环境、安装 GPU 包、加载真实模型、部署服务或启动训练。静态检查和替代 checkpoint 的 dry-run 不覆盖真实 CUDA 内核或 completion 输出。用户提供的远端日志证明模型完成加载，但修正编译工具链后的服务启动和请求仍待验证。


### 48.11.1 远端失败案例与此次环境修订

远端日志显示 FlashInfer 0.7.2 在 prefill CUDA graph 初始化时调用 `/usr/local/cuda/bin/nvcc`，报 `Unknown option '--compress-mode=size'`。用户确认 `/usr/local/cuda` 指向 `cuda-11.8`，而 PyTorch 为 `2.14.1+cu130`。此次在 Conda YAML 中补齐 CUDA Toolkit 13.0 与 Ninja，并新增环境更新、路径显式选择和持久化说明。没有修改服务启动参数或训练依赖。

日志中的 `Load weight end ... Qwen3_5ForCausalLM` 已完成；Mamba/KV 缓存分配后进入内核编译，当前错误不是已证实的显存不足或架构不支持。Ninja build failed 和后续 sigquit 为编译失败的连锁结果。修复后首先确认新的编译命令不再调用 `/usr/local/cuda/bin/nvcc`，再执行 health 和 completion 检查。尚无证据要求删除整个缓存、重装权重或降级 SGLang；也不建议直接从 FlashInfer 源码中删掉该编译参数。
