# 阶段 02 模型报告（原版 vLLM 最小运行，远端实机）

日期：2026-09-17（UTC，会话窗口 17:09Z–19:12Z）。执行者：agent（远端 `lab` 容器内命令执行）。
工作单：[`docs/stage-02-model-serve.md`](../../docs/stage-02-model-serve.md)（编写基线 `4addfcf`）。
素材快照：会话开始 `SNAPSHOT-COMMIT.txt = 463586c2…`，会话中途被重新同步为 `c49ddd63…`；
已复核关键不变量（两候选 revision、字节数、交付物清单）**未变**，故按同一工作单推进。
环境事实沿用 [`results/p0-env/environment-report.md`](../p0-env/environment-report.md)。

**结论：支持（带 1 处已标注的能力规避 + 1 项环境阻塞）** —— 目标模型在原版 vLLM 0.29.0 上可下载、
可校验、可加载、可服务、可生成，且干净会话可复跑；工作单 §3 E4 的七项取证全部拿到。
**不支持**：FlashInfer 的 JIT 编译路径（sampling 算子）在本环境不可用，属工具链版本不一致的环境缺陷，
本阶段以官方文档化开关规避并单列上报（§6 阻塞项 1）。
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
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.90
```

可复跑形式另见 `serve-command.sh`。除下列三项外全部为 vLLM 默认值，未做容量或性能调参：

1. **`--revision`（离线必需）**：`HF_HUB_OFFLINE=1` 下 `revision=None` 会因缓存无 `refs/main` 抛
   `LocalEntryNotFoundError` 直接退出（用 sha 下载时不会写 `refs/main`）。
2. **`VLLM_USE_FLASHINFER_SAMPLER=0`（缺陷规避，见 §6 阻塞项 1）**：官方文档化 opt-out
   （`vllm/envs.py:853-860`），把 top-k/top-p 换成 PyTorch 原生路径。
3. **`--max-num-seqs 256`（容量约束）**：默认 1024 超过实测可用 mamba 状态块数（652），引擎在
   `resolve_cudagraph_mode_and_sizes` 直接报错退出（见 4.2 失败链）。

### 4.2 三次失败与修法（原始日志全部保留）

| # | 日志 | 失败分类 | 根因与修法 |
| --- | --- | --- | --- |
| 1 | `logs/serve-e4-attempt1-offline-resolve-fail.log` | (d) 其他／离线加载路径 | `snapshot_download(revision=None, local_files_only=True)` → `LocalEntryNotFoundError`；修法：显式 `--revision` |
| 2 | `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | (d) 其他／配置与容量不匹配 | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (635)`（`vllm/config/compilation.py:1513`）；修法：`--max-num-seqs 256` |
| 3 | `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | (b) 内核/工具链 | FlashInfer sampling 算子 JIT：`ninja` 构建失败，根因 `CCCL` 报 `CUDA compiler and CUDA toolkit headers are incompatible`——**nvcc 13.4（`nvidia-cuda-nvcc 13.4.92`）对 `cuda.h` 13.0（`nvidia-cuda-runtime 13.0.96`，`CUDA_VERSION=13000`）**；修法：本阶段用官方开关规避，真正修法见 §6 |

**不属于**模型不兼容或 SM120/head_size=256 不支持：模型加载、KV 分配、图捕获在三次尝试中**全部成功**。

### 4.3 启动日志逐项取证（工作单 §3 E4 的 1–7）

1. **dtype 与权重加载**：`dtype=torch.bfloat16`；`Loading weights took 10.84 s`；
   `Model loading took 51.1 GiB memory and 12.582087 seconds`（干净会话复跑 10.64 s / 12.355386 s）。
2. **全注意力层实际 backend**：`Using FLASH_ATTN attention backend out of potential backends:
   ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']`，且 `Using FlashAttention version 2`
   ——即候选表首位的 **FLASH_ATTN 被选中**（与"venv 无 flash-attn 包会下沉"的事前推断相反，以日志为准）；
   **FA4 的 hd256 内核不适用**（`fa_utils.py:306-308` 要求 capability family 100，本机 12.0），故落到 FA2。
   GDN 层：`Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` + `GDN decode kernel: cuda`。
   ViT 与 MMEncoder 注意力：`FLASH_ATTN`。采样：本运行被显式关闭 FlashInfer（§4.1 偏离 2）。
   另有 `Using V2 Model Runner`、`JIT kernel warmup`、`Warming up Qwen Triton kernels (model_type=qwen3_5_text)`、
   `Running FlashInfer autotune with 8192 tokens`（预编译 FlashInfer 路径可用，见 §6 阻塞项 1 的范围界定）。
3. **块大小**：框架/manager 块 **`block_size = 784` token**（日志 `interface.py:918`：为让注意力页 ≥ mamba 页而抬升；
   metrics `block_size="784"`、`user_specified_block_size="False"`、`_block_size_resolved="True"`）；
   `mamba_block_size = 784`（`align`）、mamba 页 +0.13% 填充至与注意力页**恰好相等**（=784×64 KiB=49 MiB）。
   **kernel 块大小 = 16（代码推导，非日志观测）**：FLASH_ATTN 的 `get_supported_kernel_block_sizes()` 在非 FA4-hd256 路径返回
   `[MultipleOf(16)]`（`v1/attention/backends/flash_attn.py:112-116`），框架块 784 是它的整数倍（784/16=49），
   由 `prepare_kernel_block_sizes`（`v1/worker/gpu_model_runner.py:7514`）拆分为 **hybrid blocks**。
   **v0.29.0 的日志与 `/metrics` 都不直接打印 kernel 块大小**——本条属代码路径推导，未冻结。
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

## 6. 阻塞项与建议（需用户决策，本阶段未自行处理）

1. **CUDA 工具链版本不一致（最高优先）**：项目合并前缀里 `nvcc 13.4`（`nvidia-cuda-nvcc 13.4.92`）与
   `cuda.h 13.0`（`nvidia-cuda-runtime 13.0.96`）组合被 flashinfer 内置 CCCL 的编译期检查判为不兼容，
   使 **FlashInfer 的 JIT 编译路径整体不可用**（本次只暴露在 sampling 算子）。预编译路径不受影响
   （启动时 `Running FlashInfer autotune with 8192 tokens` 正常）。建议：把 `nvidia-cuda-runtime-cu13`
   升到与 nvcc 同 minor（13.4.x），或改造 `setup-local-cuda.sh` 只合并同一版本族的轮子。
   属依赖 pin 变更，**本阶段按工作单未改**；在修正前，任何依赖 FlashInfer JIT 的功能（含部分注意力后端、
   autotune 之外的自定义算子）都会复现同一失败。
2. **`torch.compile` 缓存落系统盘**：vLLM 用 `VLLM_CACHE_ROOT`（默认 `~/.cache/vllm`），`env.sh` 未设置；
   本次已写 182 MB 到 `/root/.cache/vllm`（含 flashinfer autotune 缓存），系统盘仅剩 18 GiB。
   建议在 `env.sh` 增加 `VLLM_CACHE_ROOT="$ATTNVIEW_HOME/caches/vllm"`（属阶段 01 脚本范围，未擅自改）。
3. **`max_num_seqs` 受 mamba 块池硬约束**：本配置下上限 652（实测值，随 gmu/长度变化）。
   下一阶段做容量/并发扫描（P0 C/D）时必须显式设置，否则默认 1024 会直接启动失败。
4. **`kernel_block_size` 未冻结**：仅代码推导为 16，日志与 `/metrics` 均不输出。下一阶段若要在
   P2 依赖块对齐，建议先用一行探针（读 model runner 的 `_kernel_block_sizes`）把它变成实测值。
5. 容器可用 RAM 报 **63.22 GiB**（vLLM 视角；阶段 01 的 `MemTotal` 为 ≈1007 GiB），加载 51.75 GiB 权重可行但余量不大，
   与后续是否开启 prefetch/多进程有关，记录备查。

## 7. 与工作单/方案的差异小结（如实记录，未对齐）

1. 模型是**原生 VL**（vision tower 参与加载与 profiling），方案 §3 未记录；§3.2 的"纯文本可少加载模块"未发生。
2. CUDA Graph 实际为 **FULL_AND_PIECEWISE 并成功捕获 FULL 图**，与方案 §3.1 的 `UNIFORM_BATCH` 预期不符。
3. 框架块大小是 **784 token**（非 16），与 kernel 块 16 通过 hybrid blocks 桥接；论文时代的 16/32 假设与本栈无关。
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
| 原版服务可启动并留完整日志 | 支持（带 1 处已标注规避） | E4 日志 + `/health` 200 |
| backend / 块大小 / 图模式 / dtype / KV 预算取证 | 支持（`kernel_block_size` 仅为代码推导，未冻结） | E4 日志 + `/metrics` |
| 一次短生成成功且原始请求响应留档 | 支持 | E5 JSON |
| 干净会话可复跑 | 支持 | E6 日志（逐项一致） |
| FlashInfer JIT 路径可用 | **不支持** | attempt3 日志 + 复现命令 |
| 长上下文、容量、性能、质量、DA | 本阶段未验证 | — |

## 10. 原始产物位置（远端）

- 日志：`/root/attnview/logs/serve-e4.log`、`serve-e6.log`、`serve-e4-attempt{1,2,3}-*.log`
- 证据：`/root/attnview/evidence/p0-model/`（E1/E2/E3/E5/E6 的 JSON 与控制台输出、`e4-metrics.txt`、`e6-metrics.txt`、`e4-extract.json`、`e6-extract.json`）
- 启动前预置基线（含事前推断与事中更正）：`/root/attnview/evidence/p0-model/e4-expected-baseline.md`
- 环境探针：`/root/attnview/evidence/after-model/env-report-20260917-1910.md`
- 可复跑命令：`/root/attnview/tools/serve-vanilla.sh`（= 本目录 `serve-command.sh` 的实现）
