#!/usr/bin/env bash
# 49 Benchmark API 配置示例；完整说明：docs/42_openseeker_benchmark.md 的 42.3。
# 复制到仓库外的私有路径，填写后：ENV_FILE=/private/path/env.sh bash scripts/sh/benchmark/47_openseeker_benchmark.sh
# ${VAR:-默认值} 表示已有非空环境变量优先；也可在私有文件改用 export VAR='实际值'。
# YOUR_... 都是占位符；只需替换当前模式使用的变量组。
# native：搜索 + Jina + SUMMARY + SCORER；official-v1 另需 OPENSEEKER；v2 再需 E2B。
# 仅 PREDICTIONS 评分：只需 SCORER 三项。

# ===== 1. 待测模型服务（仅 official-v1 / official-v2） =====
# 获取方式：先用 48_serve_checkpoint.sh 部署训练后的 Plain 模型，再复制其打印值。
# BASE_URL 是 completion 完整地址；MODEL 是服务接受的模型 ID，通常是部署的模型路径。
# 示例：http://127.0.0.1:30018/v1/completions 与 /data/models/plain-run/final。
# native 直接加载模型并保留 Beacon 压缩，不使用这两项。
export OPENSEEKER_BASE_URL="${OPENSEEKER_BASE_URL:-http://127.0.0.1:30018/v1/completions}"
export OPENSEEKER_MODEL="${OPENSEEKER_MODEL:-YOUR_SERVED_MODEL_PATH}"

# ===== 2. 搜索服务（二选一） =====
# Serper：登录 https://serper.dev/，从账号控制台复制 API Key。
# SERPER_KEY_ID 填实际 key，不是账号 ID；若已设置 SERPER_API_KEY，上游优先使用后者。
# Tavily：登录 https://app.tavily.com/，在 API Keys 获取/创建 key。
# 选 tavily 时改 SEARCH_PROVIDER=tavily 并填 TAVILY_API_KEY，Serper key 可保留占位符。
export SEARCH_PROVIDER="${SEARCH_PROVIDER:-serper}"
export SERPER_KEY_ID="${SERPER_KEY_ID:-YOUR_SERPER_KEY}"
export TAVILY_API_KEY="${TAVILY_API_KEY:-YOUR_TAVILY_KEY}"

# ===== 3. 网页读取：Jina =====
# 获取：https://jina.ai/api-dashboard/key-manager；文档：https://docs.jina.ai/
# 当前官方实现把 JINA_API_KEYS 当作一条 Bearer token，不要写多条 key 的逗号列表。
# 即使用 Tavily 做搜索，visit 仍使用此 Jina 配置。
export JINA_API_KEYS="${JINA_API_KEYS:-YOUR_JINA_KEY}"

# ===== 4. visit 的摘要模型 =====
# 自部署：向服务管理员获取 base URL、模型 ID、认证 key；/v1/models 可用于确认模型 ID。
# 第三方：从供应商 API 文档/模型列表/密钥控制台获取以上三项。
# URL 填 chat API 的 base URL，例如 http://127.0.0.1:8890/v1，不加 /chat/completions。
# MODEL 填服务实际接受的名称，例如你注册的 judge-summary；没有部署就不能直接用这个名字。
# KEY：只有服务确实无需认证时才填 EMPTY；外部服务填实际 key。
export SUMMARY_API_URL="${SUMMARY_API_URL:-YOUR_SUMMARY_URL}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-YOUR_SUMMARY_MODEL}"
export SUMMARY_API_KEY="${SUMMARY_API_KEY:-YOUR_SUMMARY_KEY}"

# ===== 5. 答案裁判模型（所有真实评分都需要） =====
# 获取方式同 SUMMARY；URL 仍是 base URL，不加 /chat/completions。
# 可以与摘要共用服务，此时在私有文件中设置：
# export SCORER_URLS="$SUMMARY_API_URL"
# export SCORER_MODEL_NAME="$SUMMARY_MODEL_NAME"
# export SCORER_API_KEY="$SUMMARY_API_KEY"
# 多 URL 可用逗号分隔，但所有端点须接受同一个 model 和 key。
# 固定官方裁判还发送 skip_special_tokens=false；请确认目标 chat 服务接受此扩展参数。
export SCORER_URLS="${SCORER_URLS:-YOUR_SCORER_URL}"
export SCORER_MODEL_NAME="${SCORER_MODEL_NAME:-YOUR_SCORER_MODEL}"
export SCORER_API_KEY="${SCORER_API_KEY:-YOUR_SCORER_KEY}"

# ===== 6. E2B 云沙箱（仅 official-v2） =====
# 获取：https://console.e2b.dev/?tab=keys，注册/登录后复制 API Key。
# 官方说明：https://docs.e2b.dev/quickstart；不是集群 SSH 密钥。
# v1/native 不需要，可保留占位符。
export E2B_API_KEY="${E2B_API_KEY:-YOUR_E2B_KEY}"

# ===== HPC 地址填写 =====
# 127.0.0.1 指运行 47 的当前节点；服务在别的节点时填其可达名称/IP。
# 例如：http://gpu-node-01:8890/v1；0.0.0.0 是服务监听地址，不是客户端请求地址。
# 若 48 服务需被其他节点访问，可在 48 命令末尾加 --host 0.0.0.0。
# 不要把真实密钥提交到 Git；示例中的所有 URL/模型名都需要对应真实运行的服务。

# 完全离线的固定语料评测请用 56/57；它们不读取本文件，也不需要上述在线 API。
# BrowseComp-Plus 使用本地 JUDGE_MODEL；HotpotQA 使用官方 EM/F1，不需要裁判模型。
