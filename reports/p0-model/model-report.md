# 阶段 02 模型报告（原版 vLLM 最小运行，远端实机）

日期：2026-09-17（UTC，会话窗口 17:09Z–19:40Z；含 19:12Z 之后的一段环境缺陷修复，见 §6.1）。执行者：agent（远端 `lab` 容器内命令执行）。
工作单：[`docs/stage-02-model-serve.md`](../../docs/stage-02-model-serve.md)（编写基线 `4addfcf`）。
素材快照：会话开始 `SNAPSHOT-COMMIT.txt = 463586c2…`，会话中途被重新同步为 `c49ddd63…`；
已复核关键不变量（两候选 revision、字节数、交付物清单）**未变**，故按同一工作单推进。
环境事实沿用 [`results/p0-env/environment-report.md`](../p0-env/environment-report.md)。

**结论：支持** —— 目标模型在原版 vLLM 0.29.0 上可下载、可校验、可加载、可服务、可生成，且干净会话可复跑；
工作单 §3 E4 的七项取证全部拿到。会话中途发现并**修复**了一处环境缺陷（CUDA 套件 minor 版本不一致
导致 FlashInfer JIT 路径不可用），修复经用户授权、只动项目 venv 与项目 CUDA 前缀，并已写成可复跑脚本步骤；
修复后服务以**无任何规避 flag** 的原始命令启动，FlashInfer 采样路径可用（§6.1）。
**不属于本阶段、一律未验证**：长上下文可用性、容量/并发、性能与净收益、任务质量、DA 机制行为、
`kernel_block_size` 冻结、CUDA Graph 下读取视图可用性（§8）。

## 1. 机器与版本（复核阶段 01，无漂移）

| 项 | 实测 |
| --- | --- |
| 主机 | `container-host`（AutoDL 容器，SSH 别名 `lab`） |
| GPU | 1× RTX PRO 6000 Blackwell Server Edition，**cc 12.0**，97887 MiB，驱动 580.142；会话开始时 0 MiB / 0% |
| 磁盘 | `/root/autodl-tmp` 200 GiB（下载前已用 8.4 GiB → 结束后 61 GiB）；`/` 30 GiB（剩 18 GiB） |
| vLLM | 0.29.0（运行轮子）；源码 checkout `98dff2a8`（tag v0.29.0），工作区干净 |
| 环境入口 | `source /root/attnview/env.sh`；`verify-runtime.sh` 全项 exit 0（E0 复核） |

## 2. 模型 identity 与下载（E1/E2）

- **主候选 `Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`**，32 文件 / **55,586,114,863 B**；
  与工作单 §3 E1 表格**逐项零差异**（files −0，bytes −0）。public、非 gated、apache-2.0。
- 第二候选 `Qwen/Qwen3.6-27B`（`6a9e13bd…`，29 文件 / 55,586,107,940 B）**只做了 identity 复核，未下载**（工作单要求）。
- 下载：`hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2…`，`HF_ENDPOINT=https://hf-mirror.com`，
  `HF_HOME=/root/attnview/models/hf-home`（数据盘）。**17:10:35Z → 18:52:50Z，实际 6135 s（1 h 42 m）**，
  均速 **≈9.06 MB/s**（55.586 GB / 6135 s）。工作单预期 3.5–4.2 h（按 3.49 MB/s 外推），实测比该基线快约 2.6×。
- 空间：`df` 前后 9,009,283,072 → 64,574,763,008 B（+55.57 GB），落盘位置在数据盘（无系统盘写入）。
- **校验**：逐文件大小 32/32 通过；19 个 LFS 文件 sha256 与 Hub 元数据**全部一致**；总字节与 Hub 精确相等；
  另对 13 个非 LFS 文件做 git blob id 校验（13/13 通过）。重跑同一 `hf download` 命令 0.32 s 命中缓存（"已校验并跳过"）。
- 观测到的上游异常：仓库自带 `crc32.txt` 对 `chat_template.jinja` / `tokenizer_config.json` / `generation_config.json`
  三项 CRC 与文件不符，而三者的 git blob id 与 Hub 完全一致 → **上游 CRC 清单陈旧**，非下载问题。

## 3. 配置、tokenizer 与模板核对（E3）

**与工作单/方案对照（差异如实列出，未向方案数字对齐）**

| 项 | 实测 | 与方案/工作单 |
| --- | --- | --- |
| 层结构 | 64 层：**16 全注意力**（层号 3,7,…,63）+ **48 GDN 线性注意力**，`full_attention_interval=4` | 与 §3.1 一致 |
| 全注意力 KV 形状 | 24 Q heads / **4 KV heads** / **head_dim 256** | 与 §3.1 一致 |
| KV 逻辑大小 | `2×16×4×256×2 = 65,536 B = 64 KiB/token` | 与 §3.2 一致 |
| checkpoint 张量总量 | `model.safetensors.index.json` → 55,562,855,904 B | 与 §3.2 引用的数字**精确一致** |
| 原生上下文 | `max_position_embeddings = 262,144` | 与 §3.1 一致 |
| dtype | `torch_dtype=bfloat16`，无 `quantization_config` | 与 §3.3 一致 |
| MTP | config 含 `mtp_num_hidden_layers=1`；vLLM 权重映射把 `mtp.` 置 None → **不加载** | 与 §3.3"关闭 MTP"一致 |
| **架构** | `Qwen3_5ForConditionalGeneration`、`language_model_only=false`、含 `vision_config` 与 image/video token → **原生视觉语言模型** | **方案 §3 未记录**：vision tower 会被加载与 profiling（见 §4），§3.2"纯文本加载有机会少加载部分模块"在本栈**未发生** |
| eos | `tokenizer_config.eos_token=<|im_end|>`；`text_config.eos_token_id=248044`；`generation_config.eos_token_id=[248046,248044]` | 方案未记；三处口径不同，后续采样契约需注意 |
| 模板 | `enable_thinking` 默认**开**（并注入 `reasoning_effort=xhigh` 系统指令）；`preserve_thinking` 默认 on | §3.3 要求关闭内置 thinking → 调用时显式 `enable_thinking=false` |

- 模板哈希：`chat_template.jinja` sha256 `c3cf9e34…`，与 `tokenizer_config.json` 内嵌副本逐字节相同；
  `tokenizer.json` sha256 `0997f410…`；`tokenizer_class=Qwen2Tokenizer`，`model_max_length=262144`。
- 渲染样例（消息：system "You are a precise assistant. Answer with a single short sentence." + user "Name the capital of France."）：
  - `enable_thinking=false` → **36 token**，文本 `…<|im_start|>assistant\n<think>\n\n</think>\n\n`（空 think 块收尾）
  - 默认（thinking 开）→ **72 token**，含 reasoning 指令与未闭合 `<think>\n`
  - 完整文本见 `model-identity.md`；服务端 `/tokenize` 渲染与本地渲染 **token id 序列完全一致**（§5）。
- **协议标签切分**（下一阶段输入，本阶段只记录）：`<global> </global> <focus> </focus> <local> </local> <answer> </answer>`
  **均不在词表**（vocab 248,077）；`<global>`=3 token、`</global>`=3、`<focus magic_chunks="8">`=7、`<answer>42</answer>`=8。
  含义：标签必然跨 token，增量解析必须跨步缓冲；`<`、`</`、`>` 与普通文本共用，`focus`/`answer` 是普通词。

## 4. 原版最小启动（E4）与逐项取证

### 4.1 实际启动命令

```bash
source /root/attnview/env.sh
export HF_HUB_OFFLINE=1
vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.90
```

可复跑形式另见 `serve-command.sh`。除下列两项外全部为 vLLM 默认值，未做容量或性能调参：

1. **`--revision`（离线必需）**：`HF_HUB_OFFLINE=1` 下 `revision=None` 会因缓存无 `refs/main` 抛
   `LocalEntryNotFoundError` 直接退出（用 sha 下载时不会写 `refs/main`）。
2. **`--max-num-seqs 256`（容量约束）**：默认 1024 超过实测可用 mamba 状态块数（652），引擎在
   `resolve_cudagraph_mode_and_sizes` 直接报错退出（见 4.2 失败链）。

> 会话中途还曾有过第三项（`VLLM_USE_FLASHINFER_SAMPLER=0`）用于规避环境缺陷；该缺陷已按 §6.1 修复，
> 规避已从命令中移除。修复前后的两次运行证据都保留：`logs/serve-e4.log`（有规避）与 `logs/serve-e4b.log`（无规避）。

### 4.2 失败链与修法（原始日志全部保留）

| # | 日志 | 失败分类 | 根因与修法 |
| --- | --- | --- | --- |
| 1 | `logs/serve-e4-attempt1-offline-resolve-fail.log` | (d) 其他／离线加载路径 | `snapshot_download(revision=None, local_files_only=True)` → `LocalEntryNotFoundError`；修法：显式 `--revision` |
| 2 | `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | (d) 其他／配置与容量不匹配 | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (635)`（`vllm/config/compilation.py:1513`）；修法：`--max-num-seqs 256` |
| 3 | `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | (b) 内核/工具链 | FlashInfer sampling 算子 JIT：`ninja` 构建失败，根因 `CCCL` 报 `CUDA compiler and CUDA toolkit headers are incompatible`——**nvcc 13.4（`nvidia-cuda-nvcc 13.4.92`）对 `cuda.h` 13.0（`nvidia-cuda-runtime 13.0.96`，`CUDA_VERSION=13000`）** |
| 3b | 同上日志（清理产物后重放 `ninja -C …/cached_ops/sampling -f build.ninja`） | (b) 工具链／前缀不完整 | 版本对齐后**编译通过、链接失败**：`ld: cannot find -lcudart`。核对轮子发现 **13.0.96 与 13.4.92 的 `nvidia-cuda-runtime` 布局相同**——都只有带 SONAME 的 `libcudart.so.13`，既无开发用 `libcudart.so`，也无 `lib64/stubs/libcuda.so`；即该前缀**从来不可能完成 JIT 链接**，此前被编译期错误掩盖 |

**修法（经用户授权，只动项目 venv 与项目 CUDA 前缀；系统 CUDA/驱动/apt 均未触碰）**：
① `nvidia-cuda-runtime` 13.0.96 → **13.4.92**（与 `nvidia-cuda-nvcc`/`nvidia-cuda-crt` 对齐，`pip check` 仍干净，
freeze 仅此一行变化）；② 补开发链接 `libcudart.so → libcudart.so.13`；
③ 从 NVIDIA 官方 redist 取**同版本**驱动 stub：`cuda_cudart-linux-x86_64-13.4.92-archive.tar.xz`
（sha256 `0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5`）中的 `lib/stubs/libcuda.so`。
三步已写成 `setup-local-cuda.sh` 的**第 4 步**（含版本一致性硬校验与"复刻 JIT 链接形态"的自检），
脚本幂等、重跑结果一致，`verify-runtime.sh` 仍全项通过。修复后 `sampling.so` 成功产出（2.3 MB），
服务以无规避命令启动且日志出现 `Using FlashInfer for top-p & top-k sampling.`。

**不属于**模型不兼容或 SM120/head_size=256 不支持：模型加载、KV 分配、图捕获在全部尝试中**均成功**。

### 4.3 启动日志逐项取证（工作单 §3 E4 的 1–7）

1. **dtype 与权重加载**：`dtype=torch.bfloat16`；`Loading weights took 10.84 s`；
   `Model loading took 51.1 GiB memory and 12.582087 seconds`（干净会话复跑 10.64 s / 12.355386 s）。
2. **全注意力层实际 backend**：`Using FLASH_ATTN attention backend out of potential backends:
   ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']`，且 `Using FlashAttention version 2`
   ——即候选表首位的 **FLASH_ATTN 被选中**（与"venv 无 flash-attn 包会下沉"的事前推断相反，以日志为准）；
   **FA4 的 hd256 内核不适用**（`fa_utils.py:306-308` 要求 capability family 100，本机 12.0），故落到 FA2。
   GDN 层：`Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` + `GDN decode kernel: cuda`。
   ViT 与 MMEncoder 注意力：`FLASH_ATTN`。采样：**FlashInfer top-k/top-p**
   （修复后运行日志出现 `Using FlashInfer for top-p & top-k sampling.`；修复前那次规避运行见 §4.1 注）。
   另有 `Using V2 Model Runner`、`JIT kernel warmup`、`Warming up Qwen Triton kernels (model_type=qwen3_5_text)`、
   `Running FlashInfer autotune with 8192 tokens`（FlashInfer 的预编译与 JIT 两条路径现均可用，见 §6.1）。
3. **块大小**：
   - **manager（框架）块大小 = 784 token**：日志 `platforms/interface.py:918`（为让注意力页 ≥ mamba 页而抬升）；
     metrics `block_size="784"`、`mamba_block_size="784"`、`mamba_cache_mode="align"`、`user_specified_block_size="False"`、
     `_block_size_resolved="True"`；mamba 页 +0.13% 填充至与注意力页**恰好相等**（784 × 64 KiB = 49 MiB）。
   - **kernel 块大小 = 784（运行时实测，与 manager 块相同、不做拆分）**：
     `evidence/p0-model/e4-kernel-block-probe.json` → `kernel_block_sizes = [784, 784, 784, 784]`、
     `hybrid_splitting_used = false`、`num_blocks = 652`。
   - **判据（源码）**：`prepare_kernel_block_sizes`（`vllm/v1/worker/utils.py:442`）对注意力组调用
     `select_common_block_size(manager_block_size, backends)`（同文件 `:310`），其 **Case 1** 是
     "manager 块大小被所有 backend 支持则**直接返回它**"；`FlashAttentionBackend` 声明
     `[MultipleOf(16)]`（`v1/attention/backends/flash_attn.py:112-116`），784 % 16 == 0 → 命中 Case 1；
     mamba 组按 `kernel_block_sizes.append(kv_cache_spec.block_size)` 直接取同值。
     V2 runner 把它存为 `self.kernel_block_sizes`（`v1/worker/gpu/model_runner.py:587`）。
   - **更正**：本报告早期版本曾由 `[MultipleOf(16)]` 推出"kernel 块 = 16、784/16=49 走 hybrid 拆分"，
     该推断**错误**（`MultipleOf(16)` 表示"16 的任意倍数都可接受"，不是"最小值 16"）。以本次实测为准。
   - **kv 组结构（顺带实测）**：4 个组 = 3 × `MambaSpec`（backend `GDNAttentionBackend`，声明 `[MultipleOf(1)]`）
     + 1 × `FullAttentionSpec`（`FlashAttentionBackend`，`[MultipleOf(16)]`）；四组的 manager 块均为 784，
     `supports_manager_block_size=true`。这也解释了 0.13% 的 mamba 页填充（3 个 mamba 组各 16 层的状态页
     略小于 49 MiB，被补齐到与注意力页相等）。
   - **探针配置（可复现）**：`PYTHONPATH=tools/kbs-probe`（内含 `sitecustomize.py`）+ `KBS_PROBE_OUT=<输出>`，
     在**与基线完全相同的 serve 参数**下运行一次（`tools/e4-kernel-block-probe.sh`，日志 `logs/serve-e4c.log`）。
     钩子只读地包装 `GPUModelRunner.initialize_kv_cache`，调用后抄出 `kernel_block_sizes` 等字段；
     必须用 sitecustomize 是因为 **EngineCore 跑在子进程**，父进程的 monkeypatch 不生效。
     **未修改 vLLM 源码**，钩子不改变任何返回值。
4. **head_size=256 是否被接受**：**接受**（走 FA2 路径；未触发 backend 校验失败，无 `invalid_reasons`）。
5. **CUDA Graph / 编译模式**：`cudagraph_mode = FULL_AND_PIECEWISE`、`mode = VLLM_COMPILE`（`backend=inductor`）；
   实捕获 **PIECEWISE 51 个 + FULL 35 个**，`Graph capturing finished in 14 s (0.62 GiB)` 与 `16 s (0.30 GiB)`；
   `CUDA graph pool memory: 0.3 GiB (actual) / 0.69 GiB (estimated)`。
   **注意**：方案 §3.1 预期"GDN 会把引擎级图模式压到 `UNIFORM_BATCH`、首版可能只能 eager/piecewise"，
   本版本**完整捕获了 FULL 图**；mamba 的约束以另一种方式出现（`max_num_seqs` ≤ mamba 块数，见 4.2）。
6. **KV cache**：`Available KV cache memory: 31.24 GiB`；`GPU KV cache size: 314,187 tokens,
   Maximum concurrency for 8,192 tokens per request: 38.35x`；metrics：`num_gpu_blocks="652"`、
   `kv_cache_size_tokens="314187"`、`kv_cache_max_concurrency="38.3529…"`、`kv_cache_layout(日志)=LBNHC`、
   `cache_dtype=auto`（KV 为 bf16）、`mamba_cache_dtype=auto`、**`mamba_ssm_cache_dtype=float32`**。
7. **max_model_len 与 chunked prefill**：`Using max model len 8192`；`Chunked prefill is enabled with
   max_num_batched_tokens=8192`（与事前推断的默认值一致）；`enable_prefix_caching=True`（默认开），
   `Mamba cache mode is set to 'align' … when prefix caching is enabled`。
8. **启动耗时与 warning**：`init engine (profile, create kv cache, warmup model) took 69.77 s
   (compilation: 21.24 s)`（首次运行；干净会话复跑 44.06 s / compilation 1.11 s）；
   `Application startup complete.` → `GET /health` 200。warning 两条：① transformers 关于
   Qwen3VL 视频预处理与参考实现差异的说明（文本路径不适用）；②
   `Default vLLM sampling parameters have been overridden by the model's generation_config.json:
   {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}`（本次请求显式覆盖为贪心）。**无 backend 降级或 OOM 类 warning。**

### 4.4 显存账（与启动前预置基线的对照）

| 量 | 实测 | 启动前预置基线 | 结论 |
| --- | ---: | ---: | --- |
| 权重加载 | 51.1 GiB | ≈51.8 GiB | 命中 |
| 峰值激活 | 2.58 GiB | 未预估 | 新增 |
| CUDA graph 池 | 0.30 GiB（估 0.69） | 未预估 | 新增 |
| 预算（gmu=0.90） | 85.47 GiB（设备 94.43/94.97 GiB 可用） | 86.03 GiB | 命中 |
| KV 池 | **31.24 GiB** | 25–32 GiB | 命中 |
| KV token 容量 | 314,187（8K 下 38.35×） | ≈49 万（按 64 KiB/token 直算） | **未命中**：manager 块为 784 token 且混合池由 mamba 组分账，token 数与"KV 字节 ÷ 64 KiB"不等价 |
| GDN 每请求状态 | conv bf16 ≈2.81 MiB + ssm **fp32** ≈144 MiB ⇒ **≈146.8 MiB/请求** | 预置写 74.8 MiB（假设 ssm=bf16） | **基线需更正**：实测 `mamba_ssm_cache_dtype=float32` |

`num_gpu_blocks=652` 与 `max_concurrency=38.35` 在 8192 token 下自洽（652/38.35≈17 块/请求，含多组混合块）；
**每请求的块构成未从日志解析出**，标记为未解项，不留推测结论。

## 5. 短生成（E5）与 thinking 关闭的带内证据

请求：`/v1/chat/completions`，`temperature=0`、`seed=0`、`max_tokens=64`、
`chat_template_kwargs={"enable_thinking": false}`，prompt = "Reply with exactly one sentence: what is the capital of France?"。

| 项 | 实测（E4 运行） | 干净会话复跑（E6） |
| --- | --- | --- |
| HTTP | 200 | 200 |
| 端到端耗时 | 0.397 s | 0.404 s |
| usage | prompt 25 / completion 8 / total 33 | **完全相同** |
| finish_reason | `stop` | `stop` |
| 输出文本 | "The capital of France is Paris." | **完全相同** |
| `reasoning` 内容 | 无（字段存在但为空） | 无 |

**thinking 关闭的证明（非推断）**：同一 messages 走服务端 `/tokenize`——
`enable_thinking=false` → **36 token**；`enable_thinking=true` → **72 token**（含 reasoning 指令）；
且服务端渲染的 prompt 文本与 token id 序列与本机 transformers 渲染**完全一致**
（`server_matches_local_off_ids=true`、`server_prompt_equals_local_off_text=true`）。
原始请求/响应体：`evidence/p0-model/e5-{request,response-raw,record}.json`。

**环境修复后的复验（同一命令，无任何规避；`logs/serve-e4b.log`）**

| 项 | 修复前（`e5`） | 修复后（`e5b`） |
| --- | --- | --- |
| HTTP / finish_reason | 200 / `stop` | 200 / `stop` |
| usage | 25 / 8 / 33 | 25 / 8 / 33 |
| 输出文本 | "The capital of France is Paris." | 完全相同 |
| 端到端耗时 | 0.397 s | 0.394 s |
| 采样实现 | PyTorch 原生（规避态） | **FlashInfer top-k/top-p** |
| KV 池 / 块数 / token 容量 | 31.24 GiB / 652 / 314,187 | **逐项相同** |

**最终配置的干净会话复跑（`E6b`，2026-09-18；取代修复前的 `E6`）**

| 项 | 实测 |
| --- | --- |
| 会话 | `env -i PATH=<最小 PATH> HOME=/root bash --noprofile --norc tools/serve-vanilla.sh --tag e6b` |
| 配置 | 与 `serve-command.sh` 完全一致（含全部已授权 CUDA 修复与缓存重定向） |
| 启动 | `Application startup complete.`，`/health` 200；`init engine` 44.40 s（compilation 1.13 s） |
| FlashInfer 采样 | `Using FlashInfer for top-p & top-k sampling.` |
| KV 池 / 块数 / 容量 | 31.24 GiB / 652 blocks / 314,187 tokens（38.35×@8192） |
| 缓存落点 | `/root/attnview/caches/vllm/{torch_compile_cache,flashinfer_autotune_cache}`（新 `VLLM_CACHE_ROOT` 生效） |
| 短请求 | HTTP 200 / `stop` / usage 25-8-33 / **0.371 s**，输出与前述各次一致 |
| 四次运行一致性 | `e5`(0.397s) → `e5b`(0.394s) → `e6`(0.404s) → `e6b`(0.371s)，usage 与输出文本逐次相同 |

另做一次**非贪心冒烟**（`temperature=0.7, top_p=0.95, top_k=20`，`--tag e5b-sampling`）：HTTP 200、
`finish_reason=stop`、0.422 s、输出非空——用于证明 FlashInfer 采样路径在**运行时**确实被走到，
而不只是编译通过。**该冒烟不产生任何质量或性能结论。**

## 6. 环境缺陷修复记录与剩余建议

### 6.1 已修复：CUDA 套件 minor 版本不一致 + 前缀缺少 JIT 链接件（2026-09-18，经用户授权）

**最终状态（全部已授权修复完成后）**

| 组件 | 修复前 | 最终 | 说明 |
| --- | --- | --- | --- |
| `nvidia-cuda-runtime` | 13.0.96 | **13.4.92** | 与 nvcc 对齐（头文件 `CUDA_VERSION=13040`） |
| `nvidia-cuda-nvrtc` | 13.0.88 | **13.4.92** | 同批对齐 |
| `nvidia-cuda-cupti` | 13.0.85 | **13.4.92** | 同批对齐 |
| `nvidia-cuda-nvcc` / `crt` | 13.4.92 | 13.4.92 | 本来就一致 |
| `nvidia-cuda-cccl` | 13.3.4.3.1 | 13.3.4.3.1 | **不动**：上游 redist 无 13.4.x（最高 13.3.4.3） |

- 依赖清单：197 行未变，仅上述 3 行版本变化；`pip check` 干净。三段快照与差异见 `dependency-delta.md`
  与 `requirements.freeze.txt`（修复前的 `cuda-upgrade-freeze-before.txt` 原样保留）。
- 缓存位置：`env.sh` 新增 `VLLM_CACHE_ROOT=$ATTNVIEW_HOME/caches/vllm` 与
  `FLASHINFER_WORKSPACE_BASE=$ATTNVIEW_HOME/caches`；已把系统盘既有的 243 MB(vLLM) + 7 MB(FlashInfer)
  迁到数据盘（现 270 MB + 14 MB），系统盘不再被这两类缓存增长挤占。
- 重建顺序见 `environment-rebuild.md`。

- **现象**：FlashInfer 的 JIT 路径整体不可用。先表现为 sampling 算子编译期失败
  （CCCL `cuda/std/__cccl/cuda_toolkit.h:41` 的 `#error`，因为 `nvcc 13.4` 对 `cuda.h 13.0`），
  版本对齐后又暴露链接期失败（`ld: cannot find -lcudart`，因为轮子不含 `libcudart.so` 与驱动 stub）。
- **改动范围**（只动项目 venv 与项目 CUDA 前缀；系统 CUDA、驱动、apt、`/etc` 均未触碰）：
  1. `nvidia-cuda-runtime` 13.0.96 → **13.4.92**（与 `nvidia-cuda-nvcc` / `nvidia-cuda-crt` 同版本；
     `pip check` 仍 `No broken requirements found`；`pip freeze` 仅此一行变化，见
     `evidence/p0-model/cuda-upgrade-freeze-{before,after}.txt`）。
  2. 补开发链接：`$ATTNVIEW_CUDA/lib64/libcudart.so → libcudart.so.13`。
  3. 安装**官方同版本**驱动 stub：`cuda_cudart-linux-x86_64-13.4.92-archive.tar.xz`
     （`https://developer.download.nvidia.com/compute/cuda/redist/`，sha256
     `0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5`）中的 `lib/stubs/libcuda.so`。
- **可复跑形式**：`setup-local-cuda.sh` 第 4 步（版本一致性硬校验 + dev 链接 + stub + 链接自检），
  幂等、重跑结果一致；`verify-runtime.sh` 仍全项通过。
- **验证**：`sampling.so` JIT 构建成功（2.3 MB）；服务以**无任何规避 flag** 的命令启动并打印
  `Using FlashInfer for top-p & top-k sampling.`；KV/块数/容量与修复前逐项相同；非贪心采样冒烟通过（§5）。
- **剩余同类风险（已收敛）**：`nvidia-cuda-nvrtc` / `nvidia-cuda-cupti` 已同批升到 13.4.92；
  仅 `nvidia-cuda-cccl 13.3.4.3.1` 保持在 13.3.4.3.x，因为**上游不存在 13.4.x 的 cccl**
  （redist 中 cccl 的最高版本即 13.3.4.3）。当前无阻塞；若日后出现某算子 JIT 行为异常，
  应先确认是否与 cccl 的版本上限有关。

### 6.2 已处理：`VLLM_CACHE_ROOT` / FlashInfer 缓存改到数据盘

`env.sh` 现有两行（见 §6.1 最终状态表）：`VLLM_CACHE_ROOT="$ATTNVIEW_HOME/caches/vllm"` 与
`FLASHINFER_WORKSPACE_BASE="$ATTNVIEW_HOME/caches"`（后者使 FlashInfer 的 JIT 缓存落在
`$ATTNVIEW_HOME/caches/.cache/flashinfer`，沿用上游布局）。既有缓存已迁移，并在最终复跑中验证新路径生效。

### 6.3 其余需带入下一阶段的约束

1. **`max_num_seqs` 受 mamba 块池硬约束**：本配置下上限 652（实测值，随 gmu/长度变化）。
   下一阶段做容量/并发扫描（P0 C/D）时必须显式设置，否则默认 1024 会直接启动失败。
2. **`kernel_block_size` 已实测为 784（等于 manager 块，无拆分）**，但仍**未冻结为契约**：
   v0.29.0 不打印该值，需要 `tools/e4-kernel-block-probe.sh` 这类注入式探针才能观测（见 §4.3 第 3 项）。
   下一阶段若要依赖块对齐，应把该探针纳入常规取证，并在每次改动 `--block-size`/模型后重测。
3. 容器可用 RAM 报 **63.22 GiB**（vLLM 视角；阶段 01 的 `MemTotal` 为 ≈1007 GiB），加载 51.75 GiB 权重可行但余量不大，
   与后续是否开启 prefetch/多进程有关，记录备查。

## 7. 与工作单/方案的差异小结（如实记录，未对齐）

1. 模型是**原生 VL**（vision tower 参与加载与 profiling），方案 §3 未记录；§3.2 的"纯文本可少加载模块"未发生。
2. CUDA Graph 实际为 **FULL_AND_PIECEWISE 并成功捕获 FULL 图**，与方案 §3.1 的 `UNIFORM_BATCH` 预期不符。
3. 框架/manager 块大小是 **784 token**（非 16）；**kernel 块大小实测同为 784、不做 hybrid 拆分**
   （见 §4.3 第 3 项）。论文时代的 16/32 假设与本栈无关。
4. `enable_prefix_caching` **默认开启**，因此 `mamba_cache_mode` 自动取 `align`——方案首版列为"明确不支持 prefix caching"，
   下一阶段需决定是否 `--no-enable-prefix-caching` 并记录其对照影响。
5. 上游 `crc32.txt` 对 3 个文件陈旧。
6. 事前预置基线中的两处需更正：`mamba_ssm_cache_dtype` 实测为 float32（非 auto→bf16）；KV token 容量不能按 64 KiB/token 直算。

## 8. 未验证项（不得据此宣称）

长上下文（32K/64K/128K/256K）可用性；容量与并发的稳定承载点；prefill/decode 耗时组成与注意力占比；
吞吐/延迟与任何性能或净收益结论；任务质量；DA 机制的任何行为；`kernel_block_size` 已冻结；
CUDA Graph 下读取视图可用性；prefix caching 与 DA 的交互；第二候选模型（3.6-27B）的任何实机行为；
本阶段**未触及**工作单 §6 禁止的 C（容量与并发曲线）、D（耗时组成）、E（协议筛查）。

## 9. 检查结论标签

| 检查 | 结论标签 | 依据 |
| --- | --- | --- |
| 素材快照可用于本阶段 | 支持（内容级；快照无 `.git`，无法用 git 复核同步） | E0 记录 + 不变量复核 |
| 模型 revision 与文件清单复现 | 支持 | E1 JSON、E2 逐文件校验 |
| 权重落数据盘且校验通过 | 支持 | 32/32 大小、19/19 LFS sha256、总字节精确一致 |
| config / tokenizer / 模板核对 | 支持（含 6 处与方案的差异） | E3 JSON + 渲染样例 |
| 原版服务可启动并留完整日志 | 支持（修复后**无任何规避 flag**） | E4/E4b 日志 + `/health` 200 |
| backend / 块大小 / 图模式 / dtype / KV 预算取证 | 支持（kernel 块大小已有运行时实测值 784） | E4 日志 + `/metrics` + `e4-kernel-block-probe.json` |
| 一次短生成成功且原始请求响应留档 | 支持 | E5 JSON |
| 干净会话可复跑（最终配置） | 支持 | `serve-e6b.log` + `e6b-record.json`（修复前的 E6 已被其取代） |
| FlashInfer JIT 路径可用 | **支持（修复后）** | 修复前失败日志 attempt3 + 修复后 `sampling.so` 与 `serve-e4b.log` |
| 长上下文、容量、性能、质量、DA | 本阶段未验证 | — |

## 10. 原始产物位置（远端）

- 日志：`/root/attnview/logs/serve-e4.log`（修复前，含规避）、`serve-e4b.log`（修复后，无规避）、
  `serve-e4c.log`（kernel 块大小探针）、`serve-e6.log`（修复前的旧复跑，已被取代）、`serve-e6b.log`（最终配置复跑）、
  `serve-e4-attempt{1,2,3}-*.log`（三次失败原始日志）
- 证据：`/root/attnview/evidence/p0-model/`（E1/E2/E3/E5/E6 的 JSON 与控制台输出、`e4-metrics.txt`、`e4b-metrics.txt`、
  `e4c`/`e6b` 相关记录、`e4-extract.json`、`e6-extract.json`、`e5b-*` 与 `e5b-sampling-*`、
  `e4-kernel-block-probe.json`（kernel 块大小实测）、
  `cuda-upgrade-freeze-{before,after,final}.txt` 与 `requirements.freeze.txt`）
- 启动前预置基线（含事前推断与事中更正）：`/root/attnview/evidence/p0-model/e4-expected-baseline.md`
- 环境探针：`/root/attnview/evidence/after-model/env-report-20260917-1910.md`
- 可复跑命令：`/root/attnview/tools/serve-vanilla.sh`（= 本目录 `serve-command.sh` 的实现）；
  kernel 块大小探针：`/root/attnview/tools/e4-kernel-block-probe.sh` + `tools/kbs-probe/sitecustomize.py`
- CUDA 前缀修复：`/root/attnview/setup-local-cuda.sh` 第 4 步（版本一致性校验 + dev 链接 + 官方驱动 stub + 链接自检）
- 依赖差异与重建顺序：本目录 `dependency-delta.md`、`environment-rebuild.md`
