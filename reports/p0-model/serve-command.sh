#!/usr/bin/env bash
# 阶段 02 交付：原版 vLLM 最小启动命令（实测可复跑）。
#
# 实测环境：AutoDL 容器 container-host，1× RTX PRO 6000 Blackwell
# Server Edition（cc 12.0，97887 MiB），驱动 580.142；vLLM 0.29.0（轮子）+ 源码 checkout
# 98dff2a8（tag v0.29.0）。详细环境事实见 results/p0-env/environment-report.md。
#
# 用法：bash serve-command.sh          # 前台运行；Ctrl-C 退出
#       bash serve-command.sh &        # 后台
#
# 除下面两处说明外，全部为 vLLM 默认值，未做容量或性能调参。

set -uo pipefail
source /root/attnview/env.sh          # 唯一环境入口：venv、项目自带 CUDA、HF_HOME/HF_ENDPOINT

export HF_HUB_OFFLINE=1               # 只用已下载的本地快照，不做任何网络访问

# 说明 1（离线必需）：必须显式给 revision。用 commit sha 下载时 HF 缓存里没有
#   refs/main，revision=None 会抛 LocalEntryNotFoundError 直接退出。
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# 说明 2（容量约束，非调参）：混合模型的 mamba 状态块池决定 max_num_seqs 上限。
#   默认 1024 超过实测可用块数（652 @ gmu 0.90 / 8K），引擎在
#   resolve_cudagraph_mode_and_sizes 直接报错退出。本阶段只做短输入，取 256（余量 >2×）。
#   下一阶段做容量/并发扫描时必须重新确定该值。

# 前置（一次性）：项目自带 CUDA 前缀需完整，否则 FlashInfer 等 JIT 路径不可用。
#   已在本机执行并写进阶段 01 脚本的第 4 步：
#     bash /root/attnview/setup-local-cuda.sh
#   它做三件事：① 校验 nvcc 与 cuda.h 的 minor 版本一致（不一致直接报错并给出修法）；
#   ② 补 libcudart.so 开发链接；③ 安装官方同版本驱动 stub（lib64/stubs/libcuda.so）。
#   本机修复记录：nvidia-cuda-runtime 13.0.96 → 13.4.92（与 nvcc 13.4.92 对齐）。

exec vllm serve Qwen/Qwen3.8-27B \
  --revision "$REV" \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90

# 实测启动结果：
#   修复前（logs/serve-e4.log，VLLM_USE_FLASHINFER_SAMPLER=0 规避）：
#     init engine 69.77 s（compilation 21.24 s）；Available KV 31.24 GiB；314,187 tokens
#   修复后（logs/serve-e4b.log，命令与本文件一致，无任何规避）：
#     Using FlashInfer for top-p & top-k sampling.
#     init engine 44.87 s（compilation 0.95 s）；Available KV 31.24 GiB；314,187 tokens（逐项相同）
#   `Application startup complete.` → `GET /health` 200
# 编译缓存在 /root/.cache/vllm（系统盘）——env.sh 未设 VLLM_CACHE_ROOT，属已记录的待修正项。
