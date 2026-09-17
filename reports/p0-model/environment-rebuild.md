# 环境重建命令（阶段 02 结束时的权威版本）

用途：在**空容器**上按本文顺序可重建出与本阶段结束时一致的环境。全部步骤只写 `/root/attnview`
（数据盘），不装 apt 包、不动系统 CUDA/驱动、不写 `/etc`。

## 0. 前置

- 数据盘 `/root/autodl-tmp` 挂载正常；`/root/attnview` → `/root/autodl-tmp/attnview`（实例重建后若链接丢失：
  `ln -s /root/autodl-tmp/attnview /root/attnview`）。
- 素材快照已就位（`material/attnview/`），环境脚本取自其 `remote/` 或项目根的副本。

## 1. 解释器与依赖（阶段 01 脚本）

```bash
bash /root/attnview/install-runtime.sh     # uv + 托管 CPython 3.12.13 + venv + vllm 0.29.0 + 源码 checkout
```

结果：torch 2.13.0+cu130、vllm 0.29.0（轮子）、triton 3.7.1、transformers 5.17.0、flashinfer-python 0.6.18。

## 2. CUDA 项目前缀（含阶段 02 的第 4 步修复）

```bash
bash /root/attnview/setup-local-cuda.sh
```

该脚本现在做 4 步：暴露前缀 → sm_120 cubin 自检 → **版本一致性硬校验** → **补 `libcudart.so` 开发链接 +
官方 13.4.92 驱动 stub + 复刻 JIT 链接形态的自检**。任一环节不满足都会以非零码退出并给出修法。

## 3. CUDA 组件版本对齐（阶段 02 的已授权变更，见 `dependency-delta.md`）

```bash
source /root/attnview/env.sh
"$ATTNVIEW_PYTHON" -m pip install --upgrade \
  "nvidia-cuda-runtime==13.4.92" "nvidia-cuda-nvrtc==13.4.92" "nvidia-cuda-cupti==13.4.92"
"$ATTNVIEW_PYTHON" -m pip check          # 期望：No broken requirements found.
bash /root/attnview/setup-local-cuda.sh  # 再跑一次，让第 4 步校验/补齐
```

> 顺序说明：`install-runtime.sh` 装出来的 `nvidia-cuda-runtime` 可能是与 nvcc 不同的 minor
> （本机曾是 13.0.96 对 13.4.92）；第 3 步把它对齐。若上游依赖集将来已自带对齐版本，此步为幂等空操作。

## 4. 环境层验证

```bash
bash /root/attnview/verify-runtime.sh      # 期望全项 exit 0（含 pip check / import / 小张量 CUDA / nvcc）
```

## 5. 模型权重（阶段 02 实测，可断点续传）

```bash
source /root/attnview/env.sh               # 固定 HF_ENDPOINT=https://hf-mirror.com、HF_HOME=models/hf-home
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

实测：55,586,114,863 B / 6135 s（≈9.06 MB/s）。校验方式见 `model-identity.md`。

## 6. 启动服务与自检

```bash
bash /root/attnview/reports/p0-model/serve-command.sh      # 前台；文档内注明两项说明
curl -s http://127.0.0.1:8000/health                       # 期望 200
"$ATTNVIEW_PYTHON" /root/attnview/tools/e5-request.py --out /tmp/e5 --tag e5
```

环境变量（`env.sh` 已固定，含阶段 02 新增两项）：
`VLLM_CACHE_ROOT=$ATTNVIEW_HOME/caches/vllm`、`FLASHINFER_WORKSPACE_BASE=$ATTNVIEW_HOME/caches`
—— 使 vLLM 的 torch.compile/AOT/autotune 缓存与 FlashInfer 的 JIT 缓存都落数据盘
（默认 `~/.cache` 在系统盘，一次运行即 270 MB）。

## 7. 与本阶段结束状态的核对点

| 核对项 | 期望值 |
| --- | --- |
| `nvcc --version` / `cuda.h` 的 `CUDA_VERSION` | 13.4（V13.4.92）/ 13040 |
| `pip check` | 无冲突；freeze 197 行，与 `dependency-delta.md` 的最终态一致 |
| 全注意力 backend / FA 版本 | `FLASH_ATTN` / FlashAttention version 2 |
| `block_size` / `mamba_block_size` / `mamba_cache_mode` | 784 / 784 / align |
| `kernel_block_sizes`（探针实测，见 `evidence/p0-model/e4-kernel-block-probe.json`） | 见 model-report §4.3 第 3 项 |
| `Available KV cache memory` / `num_gpu_blocks` / `GPU KV cache size` | 31.24 GiB / 652 / 314,187 tokens (38.35×@8192) |
| `cudagraph_mode` | FULL_AND_PIECEWISE（PIECEWISE 51 + FULL 35） |
| 采样实现 | FlashInfer top-k/top-p（JIT 可用） |
