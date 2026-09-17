#!/usr/bin/env bash
# E4: 原版 vLLM 最小启动（无任何 DA 改动，仅监听 127.0.0.1）。
#
#   source /root/attnview/env.sh
#   bash /root/attnview/tools/serve-vanilla.sh [--tag NAME] [--port N] [--max-model-len N] [...]
#
# 交付物：本文件即 serve-command.sh 的可复跑形式；日志写 $ATTNVIEW_LOGS_DIR/serve-<tag>.log。
set -uo pipefail

source /root/attnview/env.sh

TAG="$(date -u +%Y%m%dT%H%M%SZ)"
PORT=8000
MAX_MODEL_LEN=8192
GMEM=0.90
# 冻结 revision（E1/E2 已核对的 sha）。离线（HF_HUB_OFFLINE=1）时必须显式给出：
# 用 commit sha 下载时缓存里没有 refs/main，revision=None 会解析失败。
REV="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# 混合模型的 mamba 状态块池是 max_num_seqs 的硬约束：默认 1024 会超过实测可用块数（635），
# 引擎在 CUDA Graph 捕获前的 resolve_cudagraph_mode_and_sizes 直接报错退出（v0.29.0）。
# 本阶段只要短输入，取保守值 256（余量 2.5×，不放大图捕获规模）。
MAX_NUM_SEQS=256
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=${2:?}; shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=${2:?}; shift 2 ;;
    --gpu-memory-utilization) GMEM=${2:?}; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

LOG="$ATTNVIEW_LOGS_DIR/serve-$TAG.log"
mkdir -p "$ATTNVIEW_LOGS_DIR"

export HF_HUB_OFFLINE=1

# 【已修复，2026-09-18】曾有一处已标注偏离：VLLM_USE_FLASHINFER_SAMPLER=0。
#   原因：项目 CUDA 前缀里 nvcc 13.4 与 cuda.h 13.0 不一致，触发 flashinfer 内置 CCCL 的
#   cuda/std/__cccl/cuda_toolkit.h:41 编译期 #error；且该前缀缺少 libcudart.so 开发链接与
#   lib64/stubs/libcuda.so，JIT 链接同样不可能成功（见 logs/serve-e4-attempt3-*.log）。
#   经用户授权在项目 venv 内统一 CUDA 版本（nvidia-cuda-runtime 13.0.96 → 13.4.92）并补齐
#   前缀的 dev 链接与驱动 stub 后，FlashInfer JIT 路径恢复可用，故该规避已移除。
#   修复前后对照：logs/serve-e4.log（偏离版本）vs logs/serve-e4b.log（修复后，默认启用 FlashInfer 采样）。

{
  echo "=== E4 vanilla vLLM serve ==="
  echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CMD: HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3.8-27B --revision $REV --served-model-name qwen3.8-27b --host 127.0.0.1 --port $PORT --dtype bfloat16 --tensor-parallel-size 1 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GMEM ${EXTRA[*]-}"
  echo "LOG=$LOG"
  echo "HOSTNAME=$(hostname)"
  echo "HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "VLLM=$(command -v vllm)"
  echo "--- serve output ---"
} | tee "$LOG"

execcmd=(vllm serve Qwen/Qwen3.8-27B
  --revision "$REV"
  --served-model-name qwen3.8-27b
  --host 127.0.0.1 --port "$PORT"
  --dtype bfloat16 --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GMEM")
[ ${#EXTRA[@]} -gt 0 ] && execcmd+=("${EXTRA[@]}")

set -o pipefail
"${execcmd[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
{
  echo "--- serve end ---"
  echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EXIT=$rc"
} | tee -a "$LOG"
exit "$rc"
