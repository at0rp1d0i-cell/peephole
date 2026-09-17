# E4 预期基线（开机前预置，不是结论）

日期：2026-09-17（UTC）。性质：**等待权重下载期间**依据本地 `config.json` 与 vLLM v0.29.0 源码
checkout（`98dff2a8`）做的算术与默认值预测。用途只有一个：E4 启动日志一出来，逐项比对，
把"预期值命中"与"异常/降级"分开。**不得**把本文件当作实测结果或验收证据。

## A. 日志里会出现的默认值及其代码出处

| 项 | 预期 | 出处 |
| --- | --- | --- |
| `max_num_batched_tokens` | 8192（设备 ≥70 GiB 且名字不含 a100、`OPENAI_API_SERVER`） | `vllm/engine/arg_utils.py:2710-2722` |
| `max_num_seqs` | 默认 1024，随后会被 KV 容量修正 | 同上 |
| chunked prefill | 默认开（生成式模型 `is_chunked_prefill_supported=True`） | `vllm/config/model.py:2043`、`arg_utils.py:2762` |
| prefix caching | 默认开（生成式模型 `is_prefix_caching_supported=True`） | `vllm/config/model.py:2090`、`vllm/config/cache.py:138` |
| `mamba_cache_mode` | 默认 `none`；prefix caching 打开时是否被改写成 `align` 待日志确认（`align` 是 Qwen3.5 允许的模式，`all` 会直接 `NotImplementedError`） | `vllm/config/cache.py:188`、`vllm/model_executor/models/qwen3_5.py:332` |
| `cudagraph_mode` | 默认 `FULL_AND_PIECEWISE`；混合（GDN+mamba）模型走降级路径 | `vllm/config/compilation.py:615`、`:1375 resolve_cudagraph_mode_and_sizes` |
| cc 12.0 非 MLA backend 优先级 | `FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT`（逐个过 `validate_configuration`，可能因 head_size/block_size 被否） | `vllm/platforms/cuda.py:157-163`、`vllm/v1/attention/backend.py:117 supports_block_size` |
| 框架块大小与 kernel 块大小 | 两者分离；框架 `block_size` 只需是 kernel 要求的整数倍（hybrid_blocks） | `vllm/v1/attention/backend.py:128-133` |
| MTP | 权重映射把 `mtp.` 前缀映射为 `None` → 不加载，符合方案"关闭 MTP" | `vllm/model_executor/models/qwen3_5.py:322,478` |

## B. 显存算术（用于交叉核对日志里的 `Available KV cache memory`）

全注意力 KV（config 实测，与方案 §3.2 一致）：

```
2 (K/V) × 16 层 × 4 KV heads × 256 head_dim × 2 B = 65,536 B/token = 64 KiB/token
```

GDN 每请求状态（TP=1；`mamba_cache_dtype=auto` 与 `mamba_ssm_cache_dtype=auto` → 都取模型 dtype bf16；
注意 config 里的 `mamba_ssm_dtype: float32` **不是** vLLM 的 cache dtype）：

| 分量 | 形状 | 每层字节 | ×48 层 |
| --- | ---: | ---: | ---: |
| conv state | (3, 10240) = 30,720 el | 61,440 B (bf16) | **2.81 MiB** |
| ssm/temporal state | (48, 128, 128) = 786,432 el | 1,572,864 B (bf16) | **72.0 MiB** |
| 合计 | — | — | **≈74.8 MiB / 请求** |

- 形状：`conv_dim = head_k_dim×num_k_heads×2 + head_v_dim×num_v_heads = 128×16×2 + 128×48 = 10240`，
  `conv_kernel_size-1 = 3`（`gated_delta_net_state_shape`，`mamba_utils.py:258`）；
  布局默认 `SD = (state_len, dim)`（`_orient_conv_shape`，`mamba_utils.py:162`）。
- dtype：`_mamba_state_dtype`（`mamba_utils.py:97`）在 `mamba_ssm_cache_dtype="auto"` 时
  **temporal dtype = conv dtype = 模型 dtype**；若显式 `--mamba-ssm-cache-dtype float32`，
  ssm 部分翻倍 → 合计 ≈218.8 MiB/请求。
- 8K 上下文单请求粗算：全注意力 KV 512 MiB + GDN 状态 74.8 MiB ≈ **0.57 GiB**
  （不含激活、workspace、CUDA context、图捕获、碎片）。日志里的 `Available KV cache memory`
  扣掉 `gpu_memory_utilization=0.90` 下的峰值基线之后应落在同一量级。

**E4/`--max-model-len 8192`/`--gpu-memory-utilization 0.90` 下的数值目标**（用于机械化比对，
不是结论；混淆项是"激活峰值"与"图捕获"两项，日志各给一行）：

| 量 | 预期 | 依据 |
| --- | ---: | --- |
| 设备总量 | 97887 MiB = 95.59 GiB | `nvidia-smi` 实测 |
| `gpu_memory_utilization` 预算 | 86.03 GiB | 95.59 × 0.90 |
| 权重加载（日志 `Model loading took %s GiB`） | ≈51.8 GiB | checkpoint 张量 51.747 GiB；`mtp.` 被映射为不加载 |
| 预算 − 权重 | **≈34.2 GiB** | 留给激活 + KV 池 + 图捕获 + workspace |
| `Available KV cache memory`（日志） | 预期 25–32 GiB 区间 | 上者扣掉激活峰值/图捕获后的余量；两项日志分别给出 |
| `GPU KV cache size`（token 数） | 若 KV 池 ≈30 GiB → ≈**49 万 token**（即 8K 下约 60 个请求位） | 64 KiB/token 的算术；上界，实际受 mamba 组块大小与对齐削减 |

日志核对顺序固定为：`Model loading took` → `Available KV cache memory` → `GPU KV cache size`，
三者与上表逐项对照；任何一项偏离都要先在日志里找原因（例如 vision tower 未加载、图捕获额外占用、
注意力组用不同的块大小），**不得**直接改口径对齐。

## C. E4 失败时的分类特征（按此归因，不自行绕过）

| 类型 | 日志特征 |
| --- | --- |
| (a) 模型/模板不兼容 | registry 找不到 arch；processor/tokenizer 报错；权重名不匹配；chat template 渲染失败 |
| (b) backend/内核不支持 SM120 或 head_size=256 | `validate_configuration` 的 `invalid_reasons`（如 "block_size not supported"、head size 不支持）；FlashAttention/FlashInfer import 或 cubin 失败；静默降级到 `TRITON_ATTN` |
| (c) 显存不足 | profiling/KV cache 分配阶段 OOM（`gpu_memory_utilization=0.90` 下权重 51.8 GiB + 激活 + 图捕获） |
| (d) 其他 | 依赖缺失（如 `causal_conv1d`/GDN 内核）、端口占用、离线加载路径问题 |

## D. head_size=256 与 kernel 块大小的判定链（R02 的 P0 待测项，先给可证伪的预期）

> **事后更正（2026-09-18）**：本节第 3 步的预期"kernel 块大小 = 16"**是错的**，已由运行时探针推翻。
> `MultipleOf(16)` 的语义是"16 的任意倍数都可接受"，而 `select_common_block_size`（`v1/worker/utils.py:310`）
> 的 **Case 1** 在 manager 块被所有 backend 支持时**直接返回 manager 块本身**；784 % 16 == 0 → 取 784、
> 不做 hybrid 拆分。实测值见 `evidence/p0-model/e4-kernel-block-probe.json`
> （`kernel_block_sizes=[784,784,784,784]`、`hybrid_splitting_used=false`）与 model-report §4.3 第 3 项。
> 下文保留原始推断文本，作为"推断≠证据"的对照记录。

`R02` 的 scouter 曾推断"FA4 的 hd256 内核要求 major ∈ (10,11)，SM120 不适用 → 预期内核块大小 16"。
本地源码把这条链写成了可核对的三步：

1. `uses_fa4_hd256_kernel(head_size, head_size_v)` → 本模型 `head_dim=256` → **True**（`v1/attention/backends/fa_utils.py:224-233`）。
2. `get_flash_attn_version(..., supports_fa4_hd256=True)` 只有在 `fa_version==4 and is_device_capability_family(100)` 时才保留 FA4（`fa_utils.py:306-308`）；本机 capability 为 **(12,0)**，不属于 family 100 → 预期**退回 FA2 路径**或直接不被选中。（这一步与实测一致。）
3. ~~`FlashAttentionBackend.get_supported_kernel_block_sizes()` 返回 `MultipleOf(16)` → kernel 块 = 16。~~
   **（该步错误，见上方更正）**

FlashInfer 侧：`get_supported_kernel_block_sizes()` 只在与 trtllm-gen 动态内核可用时广告 ≥128 的页大小，
且断言 `page_size <= 64 or (is_device_capability_family(100) and ...)`（`v1/attention/backends/flashinfer.py:422,802-806`）。

另需注意一个**已装包事实**：venv 内既没有 `flash-attn` 也没有 `vllm-flash-attn`，只有 `flashinfer-python 0.6.18`
（`pip list` 实测）。因此 `FLASH_ATTN` 即使排在优先级首位（`platforms/cuda.py:157-163`），
也很可能在 `validate_configuration` 阶段因 ImportError 被判无效而下沉——**日志必须给出 WARNING 或
"invalid_reasons" 才能定论，不得据本段推断写成实测结论**。

## E. 协议标签在目标 tokenizer 下的切分（下一阶段输入；本阶段不做解析）

来源：`evidence/p0-model/e3-protocol-tokens.json`（`e3-probe-protocol-tokens.py`，纯 CPU）。

- 8 个控制标签 `<global> </global> <focus> </focus> <local> </local> <answer> </answer>`
  **都不在词表**（`vocab_size=248,077`），一律按普通文本切分。
- `<global>` → `['<','global','>']`（3 token）；`</global>` → `['</','global','>']`；
  `<focus magic_chunks="8">` → `['<','focus','Ġmagic','_chunks','="','8','">']`（7 token）；
  `<answer>42</answer>` → `['<','answer','>','4','2','</','answer','>']`（8 token）。
- 直接含义（供下一阶段设计，不在本阶段结论内）：标签必然跨 token，增量解析必须做跨步缓冲；
  `<`、`</`、`>` 与普通文本共用，`focus`/`answer` 是普通词——对应方案 §4.1 已登记的
  "保留标签无法与正文完全无歧义隔离"的限制。
