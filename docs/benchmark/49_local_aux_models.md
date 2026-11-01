# 49 本地摘要与裁判：Qwen3 / Qwen3.5 + SGLang

## 49.1 作用与最小修改范围

在线 BrowseComp 的 visit 摘要使用 `SUMMARY_*`，答案评分使用 `SCORER_*`。本文中 SUMMARY_MODEL / SCORE_MODEL 指这两种角色；代码实际变量是 `SUMMARY_MODEL_NAME` / `SCORER_MODEL_NAME`，不要改成不存在的变量名。

新增入口 `scripts/sh/benchmark/49_serve_aux_models.sh`，默认一次加载 `Qwen/Qwen3-32B`，以 `qwen-aux` 名称提供 `/v1/chat/completions`，摘要和裁判共用服务。更新 `49_benchmark_env.example.sh` 的两组默认值；保留所有 Python 评测调用、官方提示、训练代码及 48 待测模型部署方式。源码中没有受跟踪的 `49_benchmark_env.sh`，实际入口为带 `.example` 的模板；私有副本可命名为 `49_benchmark_env.sh`。

默认选 Qwen3-32B；也可指定 Qwen3.5 系列的完整原始模型，例如 `Qwen/Qwen3.5-4B` 或其本地目录。模型 ID、模型大小和对应架构须与实际下载的权重一致，不能把 Qwen3-32B 改名当作 Qwen3.5。辅助模型无需训练、无需本项目 adapter，不能用它代替待测 Plain/Beacon。

上游参考：[Qwen3 SGLang 部署](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/sglang.md)、[SGLang Qwen3.5 指南](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx)、[SGLang 服务参数](https://docs.sglang.io/docs/advanced_features/server_arguments)。上游指南会变化，应以安装版本的 `--help` 和实际请求为准。

## 49.2 环境和资源准备

复用 [48.2—48.3 独立服务环境](../plain/48_sglang_deployment.md)：`environment.serve-sglang.yml`、`requirements.serve-sglang.txt`。按该文创建 `search-comp-sglang`、安装依赖并配置 CUDA Toolkit。不要在训练环境覆盖安装 SGLang。

```bash
conda activate search-comp-sglang
export PYTHON="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
"$FLASHINFER_NVCC" --version
"$PYTHON" -m pip check
"$PYTHON" -m sglang.launch_server --help
```

服务脚本继承 CUDA 工具链设置，不自动安装或激活环境。`PYTHON` 优先于 `CONDA_ENV`；公共解析器找不到指定环境时可能回退当前 Python，正式部署应明确指定可执行的绝对路径。

Qwen3-32B 的 BF16 权重仅按参数估算约 64 GB（十进制），还需要缓存、激活和运行时显存。单张 80 GB 卡不保证默认上下文与并发可以运行；可先设并发 1、上下文 8192，或用两卡 TP。部署卡与 native 被测模型用卡必须规划，避免抢占。同一服务共用摘要和裁判可减少一份权重，但并发能力也共用。

默认 Hub ID 首次启动可能联网下载权重。无网计算节点先在联网环境下载完整模型到共享目录，再指定 `AUX_MODEL_PATH=/data/models/Qwen3-32B`。dry-run 不检查文件、下载权重或验证架构。

## 49.3 推荐：一键共用服务

在仓库根目录准备私有配置，仅首次复制，已有文件请直接编辑，避免覆盖密钥：

```bash
mkdir -p "$HOME/.config/openseeker"
# 仅首次执行：
cp -n scripts/sh/benchmark/49_benchmark_env.example.sh \
  "$HOME/.config/openseeker/49_benchmark_env.sh"
chmod 600 "$HOME/.config/openseeker/49_benchmark_env.sh"
export ENV_FILE="$HOME/.config/openseeker/49_benchmark_env.sh"
```

默认辅助 API 已与启动器匹配，无需填写第三方模型 key。搜索/Jina 仍需在私有文件填真实 key；仅启动本地模型不需要这些搜索 key。

```bash
# 仅审查命令：不导入 SGLang、不占 GPU、不访问网络
DRY_RUN=1 bash scripts/sh/benchmark/49_serve_aux_models.sh

# 使用者在 GPU 节点执行，前台持续提供服务；不会启动训练或 benchmark
CUDA_VISIBLE_DEVICES=0,1 AUX_TP_SIZE=2 \
  bash scripts/sh/benchmark/49_serve_aux_models.sh
```

用本地权重、较小上下文进行首次联调：

```bash
AUX_MODEL_PATH=/data/models/Qwen3-32B \
CUDA_VISIBLE_DEVICES=0,1 AUX_TP_SIZE=2 \
AUX_CONTEXT_LENGTH=8192 AUX_MAX_RUNNING_REQUESTS=1 \
  bash scripts/sh/benchmark/49_serve_aux_models.sh
```

启动日志不是就绪证明。保持终端运行，另开终端按 49.6 检查。前台 Ctrl+C 停止，集群作业结束会回收服务；之后确认 health 不再可用、GPU worker 退出。需要日志时在持久终端用 `set -o pipefail` 和 `2>&1 | tee outputs/logs/49_aux.log`，先创建日志目录；不要分享含认证信息的上游异常日志。

## 49.4 脚本全部变量及修改提示

启动顺序：公共脚本切到仓库根目录 → source 可选 ENV_FILE → source 49 模板补齐空值 → 构建服务命令。ENV_FILE 是可信 shell 文件，会执行其内容；相对路径以仓库根目录为准。模板 `${VAR:-默认值}` 保留已有非空变量；私有文件直接 `export VAR='值'` 则覆盖已有值。

| 变量 | 默认 | 说明与修改提示 |
|---|---|---|
| `PYTHON` | 公共解析器选择 | 服务解释器的可执行路径，建议明确指定；不要继承训练环境路径 |
| `CONDA_ENV` | `search-comp-sglang` | 仅帮助查找 Python，不安装/激活环境 |
| `ENV_FILE` | 空 | 私有 shell 配置；服务和评测使用同一文件 |
| `AUX_ROLE` | `shared` | `shared` / `summary` / `scorer`；shared 校验两组 URL、model、key 完全相同 |
| `AUX_MODEL_PATH` | `Qwen/Qwen3-32B` | Hub ID 或完整本地模型目录；换 Qwen3.5 时改此项，不是只改服务别名 |
| `CUDA_VISIBLE_DEVICES` | `0` | 物理 GPU 列表；两卡如 `0,1`，服务进程内部会重新编号 |
| `AUX_TP_SIZE` | `1` | 单节点张量并行数；需匹配可见卡数和模型分片约束 |
| `AUX_HOST` | `127.0.0.1` | 监听地址；跨节点用 `0.0.0.0`，客户端用真实节点名/IP |
| `AUX_PORT` | `8890` | 1—65535；改端口必须同步两组客户端 URL |
| `AUX_CONTEXT_LENGTH` | `32768` | prompt 与生成总 token 预算，不是单次输出长度；OOM 可先减小 |
| `AUX_MEM_FRACTION` | `0.80` | SGLang 静态内存池比例，取值有效性由 SGLang 校验；降低会减少缓存容量 |
| `AUX_MAX_RUNNING_REQUESTS` | `4` | 服务并发上限，独立于评测 WORKERS/JUDGE_WORKERS；首次建议 1 |
| `DRY_RUN` | `0` | 仅允许 0/1；1 打印非敏感命令，不启动进程，不验证依赖/权重/显存 |
| `CUDA_HOME` / `CUDA_PATH` / `FLASHINFER_NVCC` | 继承环境 | 指向与服务 runtime 匹配的 Toolkit，见 48；不是模型参数 |

客户端配置表：

| 变量 | 模板默认 | 说明 |
|---|---|---|
| `SUMMARY_API_URL` | `http://127.0.0.1:8890/v1` | chat base URL，不追加 `/chat/completions`；启动器不从 URL 解析监听端口 |
| `SUMMARY_MODEL_NAME` | `qwen-aux` | summary/shared 的 `--served-model-name`，不是权重路径 |
| `SUMMARY_API_KEY` | `EMPTY` | 默认无认证；真实值会配置为服务 `--api-key` |
| `SCORER_URLS` | `$SUMMARY_API_URL` | scorer chat base URL；共用模式必须与摘要 URL 字符串相同 |
| `SCORER_MODEL_NAME` | `$SUMMARY_MODEL_NAME` | scorer 的服务别名；多个端点必须接受相同名称 |
| `SCORER_API_KEY` | `$SUMMARY_API_KEY` | 多个裁判端点必须接受相同 key |

`EMPTY` 仅为客户端占位符，不会设置为服务认证 key。其他值会启用 SGLang 认证；启动器不打印 key，但命令行可能被同机进程查看。这里的认证只保护本地模型服务，不代替搜索/Jina 的 key。

脚本末尾参数通过 Bash 数组原样传给 SGLang，例如 `--chunked-prefill-size 2048`；不使用 eval，不把字符串当命令。额外参数只打印数量，以免泄露 key；自行记录非敏感参数以复现。避免通过末尾重复覆盖 model/port/api-key，统一修改上述变量，否则客户端配置可能失配。服务脚本不写入私有配置，也不自动修改 URL。shell 校验角色、正整数、端口上限及 DRY_RUN；其他值由 SGLang 校验。

## 49.5 两个独立服务与 Qwen3.5 替换

在同一个私有文件明确设置两组 URL 和名称：

```bash
export SUMMARY_API_URL='http://127.0.0.1:8890/v1'
export SUMMARY_MODEL_NAME='qwen-summary'
export SUMMARY_API_KEY='EMPTY'
export SCORER_URLS='http://127.0.0.1:8891/v1'
export SCORER_MODEL_NAME='qwen-scorer'
export SCORER_API_KEY='EMPTY'
```

在两个服务终端分别启动（都设置相同 ENV_FILE 和服务环境 PYTHON）：

```bash
# 终端 A：摘要服务
AUX_ROLE=summary AUX_MODEL_PATH=/data/models/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0 AUX_PORT=8890 \
  bash scripts/sh/benchmark/49_serve_aux_models.sh

# 终端 B：裁判服务
AUX_ROLE=scorer AUX_MODEL_PATH=/data/models/Qwen3-32B \
CUDA_VISIBLE_DEVICES=1,2 AUX_TP_SIZE=2 AUX_PORT=8891 \
  bash scripts/sh/benchmark/49_serve_aux_models.sh
```

想让两种角色共用 Qwen3.5，只需保留默认 shared 配置并修改 AUX_MODEL_PATH。本入口默认 `--reasoning-parser qwen3`，将思考内容与最终 `content` 分离；它不关闭 thinking，也不保证短输出。官方客户端主要读取 content，因此必须检查实际最终答案非空，尤其是生成预算耗尽时。需要关闭思考时遵循所选模型与安装版本支持的 chat template 配置；本次不修改官方请求或替换 tokenizer 模板。Qwen3 与 Qwen3.5 的模型支持随版本变化，不能仅凭脚本成功打印证明可用。

跨节点示例：服务 `AUX_HOST=0.0.0.0`，私有文件两组 URL 使用 `http://gpu-node-01:8890/v1`。不要把 `0.0.0.0` 写入客户端 URL。该服务可与 48 的待测 Plain 服务同时运行，但需要不同端口和明确 GPU 分配。

## 49.6 健康检查和真实请求

以下由使用者在服务就绪后执行，本次修改没有发出实际模型请求。先加载同一私有配置：

```bash
source "$HOME/.config/openseeker/49_benchmark_env.sh"
curl --fail --show-error -H "Authorization: Bearer $SUMMARY_API_KEY" \
  "${SUMMARY_API_URL%/v1}/health"
curl --fail --show-error -H "Authorization: Bearer $SUMMARY_API_KEY" \
  "${SUMMARY_API_URL%/}/models"
```

检查 `/models` 返回 ID 与 SUMMARY_MODEL_NAME 一致；分别部署时也检查裁判。用标准库对两个角色各发一个最小 chat 请求，包含官方裁判使用的扩展参数：

```bash
"$PYTHON" - <<'PY'
import json, os, urllib.request
for url, model, key in (
    (os.environ['SUMMARY_API_URL'], os.environ['SUMMARY_MODEL_NAME'], os.environ['SUMMARY_API_KEY']),
    (os.environ['SCORER_URLS'].split(',')[0], os.environ['SCORER_MODEL_NAME'], os.environ['SCORER_API_KEY']),
):
    payload = dict(model=model, messages=[dict(role='user', content='Reply with OK.')],
                   max_tokens=1024, temperature=0, stream=False, skip_special_tokens=False)
    request = urllib.request.Request(url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    choice = result['choices'][0]
    print(model, choice['finish_reason'], choice['message'].get('content'))
    assert choice['message'].get('content'), '最终 content 为空，检查 thinking 与生成预算'
PY
```

失败时先看服务日志：连接失败检查节点/端口/启动进度；401 检查 key；model not found 检查服务别名；上下文错误检查官方输入和生成预算；OOM 缩短上下文、降低并发或增加 TP；nvcc 错误按 48 的工具链说明处理。真实请求成功仍不能证明摘要/裁判质量等同原论文。

## 49.7 连接评测与验证边界

服务终端继续运行；评测终端使用评测环境 Python，不必安装 SGLang：

```bash
PYTHON=/absolute/path/to/eval-env/bin/python \
ENV_FILE="$HOME/.config/openseeker/49_benchmark_env.sh" \
MODE=native RUN_DIR=/absolute/path/to/beacon-run LIMIT=2 \
WORKERS=1 JUDGE_WORKERS=1 DRY_RUN=1 \
  bash scripts/sh/beacon/47_beacon_browsecomp.sh
```

确认配置和全部服务后，由使用者去掉 DRY_RUN 实际评测。Plain 可换成 `plain/47_plain_browsecomp.sh`；official 模式还需独立的 48 待测服务，见 [42](42_openseeker_benchmark.md)。49 不提供搜索、Jina 或 E2B；搜索/Jina 仍需外网。固定语料 56/57 与 baseline 74/75 不读取这组配置。

正式实验记录辅助权重版本/目录、服务版本、GPU、上下文/并发、thinking 行为和全部非敏感参数。更换裁判会改变分数，应固定裁判后对比结果。

本次仅执行 Bash 语法、默认/分角色 dry-run、变量继承和错误参数检查；未创建环境、下载权重、启动 SGLang、调用外部 API 或启动训练。dry-run 不构成真实 GPU 部署验证。
