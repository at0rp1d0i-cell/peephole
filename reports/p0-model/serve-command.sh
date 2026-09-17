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
# 本命令的相对项（除下面注释标明的两处）全部为 vLLM 默认值，未做容量或性能调参。

set -uo pipefail
source /root/attnview/env.sh          # 唯一环境入口：venv、项目自带 CUDA、HF_HOME/HF_ENDPOINT

export HF_HUB_OFFLINE=1               # 只用已下载的本地快照，不做任何网络访问

# 偏离 1（必需）：离线模式下必须显式给 revision。用 commit sha 下载时 HF 缓存里没有
#   refs/main，revision=None 会抛 LocalEntryNotFoundError 直接退出。
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# 偏离 2（缺陷规避）：项目合并 CUDA 前缀里 nvcc 13.4 与 cuda.h(13.0) 版本不一致，
#   flashinfer 内置 CCCL 的编译期兼容检查（cuda/std/__cccl/cuda_toolkit.h:41）会 #error，
#   导致 sampling 算子 JIT 失败、引擎启动中止。此开关按 vLLM 官方文档化行为
#   （envs.py:853-860）改用 PyTorch 原生 top-k/top-p 路径；不改模型、精度、attention
#   backend、块大小、图模式与 KV 预算。真正修法见 model-report.md 的阻塞项。
export VLLM_USE_FLASHINFER_SAMPLER=0

# 偏离 3（容量约束，非调参）：混合模型的 mamba 状态块池决定 max_num_seqs 上限。
#   默认 1024 超过实测可用块数（635/652），引擎在 resolve_cudagraph_mode_and_sizes
#   直接报错退出。本阶段只做短输入，取 256（余量 >2×）。
exec vllm serve Qwen/Qwen3.8-27B \
  --revision "$REV" \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90

# 实测启动结果（E4，2026-09-17T19:03Z 起）：
#   init engine (profile, create kv cache, warmup model) took 69.77 s (compilation: 21.24 s)
#   Available KV cache memory: 31.24 GiB
#   GPU KV cache size: 314,187 tokens, Maximum concurrency for 8,192 tokens per request: 38.35x
#   Application startup complete.（health=200）
# 首次冷启动另需 torch.compile 28 s + GDN Triton 内核 JIT（缓存目录见下）；编译缓存在
#   /root/.cache/vllm（系统盘）——env.sh 未设 VLLM_CACHE_ROOT，属已记录的待修正项。
