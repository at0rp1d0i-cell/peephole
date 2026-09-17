# M1 接入设计：调用链、连接点与 API 子集建议

日期：2026-09-18。执行者：远端执行 Agent。阶段：03（CPU/设计，未接 GPU、未改 vLLM）。
证据级别见每节标注：**【自读】**= 本阶段自己打开源码/实机日志读到；**【侦察核对】**= 只读侦察 agent 给出的
`文件:符号:行号`，我未逐行复读（复核时按此清单抽检）；**【实测】**= 本阶段 CPU 复跑产物；**【待核对】**= 尚无证据。

版本身份（**【自读】** `venvs/attnview/lib/python3.12/site-packages/vllm/_version.py`）：安装包 `0.29.0` /
`g98dff2a81`，与源码 checkout `98dff2a81` 同源（**侦察核对**：行号抽检一致、运行栈回溯指向 site-packages）。
运行期 runner = V2 `vllm/v1/worker/gpu/model_runner.py`（**实测** `logs/serve-e6b.log` 有 `Using V2 Model Runner`）。

## 1. 请求状态归属与生命周期

| 对象 | 位置 | 生命周期 |
| --- | --- | --- |
| `RequestState`（槽位化 per-request：`req_id_to_index` / `index_to_req_id` / `free_indices` / `all_token_ids` / `num_computed_tokens`） | `v1/worker/gpu/states.py:26-96` **【侦察核对】** | 请求级；`add_request:96` 建、`remove_request:135` 释放，槽位复用 |
| 调度器侧请求对象（`Request`、块分配、`num_computed_tokens`） | `v1/core/sched/scheduler.py:189` / `v1/core/kv_cache_manager.py:118` **【侦察核对】** | 请求级；`finish_requests:2417` / `_free_request:2480` / `_free_blocks:2509` / `_drain_deferred_frees:2548` |
| EngineCore 与 API server **跨进程**（msgspec 序列化） | `v1/engine/__init__.py:107-158`（`EngineCoreRequest`）/`:196-238`（`EngineCoreOutput`） **【侦察核对】** | 请求级 |
| Worker↔EngineCore | TP=1 走 `UniProcExecutor`（同进程不同线程风格的 future），**不跨进程** | `v1/executor/abstract.py:74-77` / `v1/executor/uniproc_executor.py:26-45` **【自读】** |
| 三者索引对应 | `gather_batch_req_state:1106`（`idx_mapping:1136-1137`）把 `req_id` ↔ 行号对上；`sort_batch_req_ids:2124` 保证顺序一致 **【侦察核对】** | 每步重建 |

**对 DA 的含义**：读取视图的运行状态应挂在 **worker 的 `RequestState` 同级**（请求级、随 `finish_requests`/`free_states` 释放），
而不是 API server；API server 只负责 §4.1 的输出适配。混批重排由 `idx_mapping` 管辖，**不能用列表位置当 request 标识**
（合同 C6.4 的"批次重排不得串状态"在这里落地）。CPU 侧已用 `ProtocolRegistry` 按 `request_id` 隔离并写单测，
但**只证明 CPU 对象隔离**，不代表 vLLM 混批/取消已支持（R07 保持打开）。

## 2. token 回到 host 的时机与 t/t+1 生效（本阶段最关键结论）

事实链（**【自读】**）：

1. `EngineCore.step`（`v1/engine/core.py:597-630`）：`schedule()` → `execute_model(non_block=True)` → `future.result()`
   → 若为 `None` 则 `sample_tokens(grammar_output)` → `_process_aborts_queue()` → `update_from_output()`。
2. `UniProcExecutor.AsyncOutputFuture.result`（`uniproc_executor.py:26-45`）才真正触发
   `AsyncModelRunnerOutput.get_output()`；host 侧 token ids 的唯一出处是
   `v1/worker/gpu/async_utils.py:166-177`（`copy_event.synchronize()` + `tolist()`）**【侦察核对】**。
3. V2 runner 在自己的 `sample_tokens` 里完成 forward→sample→**`AsyncOutput` 构造（发起异步 D2H）**→
   `postprocess_sampled(...)`（`v1/worker/gpu/model_runner.py:1894` 与 `:1921`，**自读**）。也就是说：
   **采样结果在 worker 内先可用（GPU/np），host 侧字符串/ids 要等 D2H 同步**。
4. 异步调度默认开启（`v1/config/vllm.py:1231-1311`）；`max_concurrent_batches:558-568`；
   `AsyncScheduler._update_after_schedule`（`v1/core/sched/async_scheduler.py:19-49`）会**乐观推进** `num_computed_tokens`
   并用占位 token 调度下一步。**【侦察核对】**

推论（**结论**）：

- 若把增量解析放在 **API server 或 `update_from_output`**（host 侧），则第 t 步采样出的 token 要到**第 t+1 步的
  读回点**才在 host 可用；而第 t+1 步的 metadata 在 worker 内构建。若二者顺序恰好错开，声明只能影响第 t+2 步 forward
  ——**那就是"延迟一拍"，合同 C3.5 不允许**（不能靠延迟生效冒充满足）。
- 正确做法是把"解析 + 视图更新"放在 **worker 内、采样之后、下一次 `build_attn_metadata` 之前**：
  即 V2 runner 的 `postprocess_sampled`（`model_runner.py:1921`）处或其后的新增钩子。此时第 t 步的 token 已在 worker
  可用，第 t+1 步 forward 消费该 token 时用的正是新视图 → **恰好满足"第 t 步解析、第 t+1 步生效"，且不需要改动
  调度器、不需要关闭异步调度**。
- 代价与要求：worker 侧需要"token id → 文本片段"的能力。V2 worker 未必持有 tokenizer，建议**在初始化时从 tokenizer
  词表构建一份只含 ASCII 控制字符（`< > / = " ,` 与字母数字）的 `{token_id: bytes}` 小表**（一次性，避免每步同步）；
  CPU 侧已验证：逐 token `decode([id])` 拼接与整段文本完全一致（**实测** `tools/p1cpu-demo.py` 断言
  `"".join(fed) == GENERATION_SCRIPT`），因此 token 级喂入是可行的。
- **不得**用"每步额外 host 同步"来换取正确性而不计量：那会在 decode 热路径加同步点（R10 要测）。

若最终选择 host 侧解析（例如为了与 API 层共用代码），则必须列出并**测**其一：
(a) 对带 DA 的请求关闭异步调度（`Scheduler` 而非 `AsyncScheduler`）——损失 overlap；
(b) 把 D2H 读回提前到本步末尾并同步等待——损失等于每步一次同步。
两种都属"显式代价"，不得默认静默采用。

## 3. 写入侧：块表、slot、位置（读视图绝不碰这些）

| 项 | 位置 | 更新时机 |
| --- | --- | --- |
| `BlockTables.block_tables`（kernel 侧块表）/ `input_block_tables` / `slot_mappings` | `v1/worker/gpu/block_table.py:44/66/71` **【侦察核对】** | `append_block_ids:118`（manager→kernel 展开 `:126-129`）、`apply_staged_writes:141`、`compute_slot_mappings:196` |
| slot 计算 | `_compute_slot_mappings_kernel:290+`（`block_indices/offsets/slot_ids:316-322`，`PAD` 填充 `:300-307`）**【侦察核对】** | 每步前 |
| host→GPU 写入通道 | `StagedWriteTensor`（`v1/worker/gpu/buffer_utils.py:114-174`）/ `FusedStagedWriter:210` **【侦察核对】** | 每步 |
| 块分配/释放 | `KVCacheManager.allocate_slots:343` / `free:566` / `get_block_ids:696` / `take_new_block_ids:799` **【侦察核对】** | 调度期 |

**"采样步"与"已写 KV 长度"的对应（本项目约定，必须固定下来）**

- 第 `s`（1 基）个 decode step 消费生成流第 `s-1` 个 token，把它写入位置 `prompt_len + s - 1`；
  该步 attention 的 KV 上界（含刚写入的自身）= `prompt_len + s`。
- 生成流第 `t` 个 token 内闭合的声明 → `effect_step = t + 1`（**就是消费该 token 的那次 forward**）。
- **不得**把"已生成 token 数"直接当"已写 KV 长度"或当"可见上界"：三者差 1 是本项目最容易出错的点。
  本阶段把它写成 `readview.py` 顶部约定 + `state.py`/`trace.py` 的字段（`kv_len_after`、`next_write_position`、
  `write_position`），并在轨迹里逐行显示（**实测** `evidence/p1-cpu/fixed-trace.md`：写位置严格 +1、
  读清单在同一 kv_len 下随模式变化）。

**读视图与写入分离（I3）**：读取视图只改"本步给 attention 看的块清单"，`slot_mapping`、canonical 块表、
引用计数、`num_computed_tokens` 全部不动；`global` 恢复时重新覆盖全部有效块。CPU 侧已用**独立逐位置参考**
逐 step 对照（**实测**：115 个生成步全部一致，`reference_mismatch_steps = []`），并在轨迹中给出"读清单变化 6 次、
写位置严格 +1"的证据。

## 4. attention metadata：按 KV 组建一次、组内各层共享

- hybrid 模型的 metadata 构造：`v1/worker/gpu/attn_utils.py:247 build_attn_metadata`（按 KV 组循环 `:277`、
  每组构造 `CommonAttentionMetadata:295`、`builder.build:330`、**层共享 metadata:335-336**）**【侦察核对】**。
- 实例中的 4 组（**实测** `evidence/p0-model/e4-kernel-block-probe.json`）：3×`MambaSpec`(GDN) + 1×`FullAttentionSpec`(FA2)。
- 全注意力组内的**每个请求占块表的一行**，因此"每请求不同的可见块集合"**可以**用"每步重写该请求那一行的块表"表达；
  同一 KV 组内**逐层不同**的视图则无法表达（对 DA 无影响：论文的掩码对全注意力层一致，附录 B）。
- 若某天需要逐层不同，可选路径是新增 attn group 或改 `update_block_table`/`get_attention_context`（**不建议**，改动面大）。

FA2 契约（**【侦察核对】** + 论文附录 B 对照）：

| 能力 | 结论 |
| --- | --- |
| 每请求一行 int32 page table，条目 = **kernel 块号** | 支持任意非连续块号 → **块级非连续子集可表达** |
| `seqused_k=seq_lens` 界定该行有效 K 长度 | 尾块/部分块由它界定；超出部分不读 |
| `window_size` | 整次调用单一标量对（来自 layer 属性）→ **不能**用于 per-request 视图 |
| 因果掩码 | bottom-right 对齐到该请求自己的 K 区间；**源码不传 token positions** → 推断按"可见序列序数"，**必须用 dense masked reference 在 GPU 上验证**（R03 保持打开） |
| `mask_mod` / `aux_tensors` / per-request causal | **FA4 专用，本机 cc 12.0 走 FA2 → 不可用** |
| kernel 块大小 | 只要求 16 的倍数（784 = 16×49），无 32/64/128 要求 |

**`-1` 的边界（规范 §8.3 的落地）**：`read-view-spec.md` 的 `-1` 是**本项目内部**约定，不能推断 FA2 会跳过它。
vLLM 自身的填充约定是 `NULL_BLOCK_ID=0`（`v1/attention/backends/utils.py:46`）与 `PAD_SLOT_ID=-1`（`:45`）
**【侦察核对】**，且 `BlockTables.gather_block_tables:157` 会"padding 行清零"。因此适配层必须：
把内部 `-1` **翻译成该后端接受的填充值**（建议 `NULL_BLOCK_ID=0`），并且只依赖 `seqused_k` 界定有效前缀
（尾随条目不得被解引用）。CPU 侧 `to_kernel_args()` 已实现行/列双向越界拦截与"有效前缀不得含 -1"（I4/I8），
**但它不宣称目标后端可直接消费 `-1`**。

## 5. GDN（线性注意力）路径不参与块筛选

- GDN metadata：`v1/attention/backends/gdn_attn.py:84 builder.build:214`，用 `mamba_get_block_table_tensor:222`，
  取 `non_spec_state_indices = block_table[:, 0]`（state 索引来自 **MambaSpec 组**的块表）**【侦察核对】**；
  混合逻辑在 `v1/worker/gpu/model_states/mamba_hybrid.py:166 prepare_attn` / `:317 build_attn_metadata`
  / `:341 postprocess_state`（align 融合 kernel `:168-176`）**【侦察核对】**。
- 结论：**全注意力的读取视图与 GDN 的 state 续写互不影响**（不同 KV 组、不同 metadata 字段）。
  GDN 的 state 不是分页 KV，按附录 B 与合同 C4 只对全注意力层做掩码。
- 但**共享块池记账**：DA 不释放/跳过任何块（`skip→null block`、`remove_skipped_blocks` 路径**不得**被 DA 触发），
  所以第一阶段**不声称显存下降**（与方案 §6.3 一致）。

## 6. 推荐连接点（按改动面从小到大）

**A. worker 内 per-step 钩子（推荐）**

1. 新增 `DARequestState`（解析器 + 模式 + 声明片段 + trace），与 `RequestState` 同生命周期：
   在 `add_requests:1049` 建、`finish_requests`/`free_states`（`model_runner.py:1525-1533`）释放；
   键一律用 `req_id`（**不**用行号）。
2. 在 `postprocess_sampled`（`model_runner.py:1921`）之后：逐请求喂入本步采样 token（token→片段用一次性构建的
   小表），更新模式与"下一步视图输入"。
3. 在 `build_attn_metadata`（`attn_utils.py:247`，或紧随 `FlashAttentionMetadataBuilder.build`
   `flash_attn.py:545` 之后）**只对全注意力组**：把带 DA 请求的那几行改成"本步可见块表"。
   写入目标应是**每步复用的一块 read-only 块表缓冲**（形状一次生成内不变，满足 I5 与图捕获前提），
   而不是 canonical `block_tables`（否则违反 I3）。
4. 需要的新增数据：每请求 `visible_blocks → physical ids` 的行构造（CPU 侧已有纯函数，可直接搬运）；
   一次小规模 H2D 拷贝（每 DA 请求一行），或等价的设备端 scatter。

所需改动文件（预估）：`v1/worker/gpu/model_runner.py`、`v1/worker/gpu/attn_utils.py`（或 `flash_attn.py` 的 builder）、
新增 `v1/worker/gpu/da/*`（协议层，纯 CPU）。**不触碰 kernel**。

**B. host 侧解析 + 关闭 DA 请求的异步调度**：改动更像"服务层"，但损失 overlap 且要单独测；作为 A 的退路。

**C. 通过 `CommonAttentionMetadata` 的 per-request 扩展字段传递（如 `rswa_prefix_lens:435` / `mm_req_doc_ranges:428`
   这类既有先例）**：**【侦察核对】** 表明该版本已有 per-request 数组字段的先例，可作为"不经块表改写"的备选；
   但 FA2 是否接受"按请求的前缀长度"语义**未被本阶段证实**（`rswa_prefix_lens` 属 SWA 路径）→ 列为待核对。

## 7. 后续 GPU 验证项（本阶段不做）

1. `global` 路径与原版逐元素一致（无缝退化）。
2. `focus/local` 对照 dense masked reference 的 attention/logits 在**预先定好**的容差内（R03）。
3. 因果语义：压缩块表下 FA2 的掩码是"可见序列序数"还是"全局位置"——用强制轨迹 + dense 参考判定。
4. 图捕获：FULL/PIECEWISE 下每步改写块表是否破坏捕获（R08）。
5. 热路径开销：解析 + 行改写 + 拷贝的每步成本（R10），以及是否需要在 DA 请求上关闭异步调度。
6. prefix caching / 抢占场景下的视图语义（R07）。

## 8. 本设计的限制与待决（交给本地复核）

- 上述行号来自 pin `98dff2a81`；升级版本需重新核对（**版本规则**同协议合同 §7）。
- **【待核对】**：`sample_tokens` 与 `execute_model` 在同一 step 内的先后（我自读了 `model_runner.py:1855-1935`
  与 `core.py:597-630`，但"batch queue 深度 2 时 future 对应哪一步"未逐行确认）→ 直接决定 A 是否天然满足 t+1；
  建议接 GPU 后在 worker 内打时间戳验证（**不靠推断**）。
- **【待核对】**：worker 是否已有可用的 detokenize 能力（若有，可省掉自建小表）。
- `-1` 的最终填充约定（建议 `NULL_BLOCK_ID=0`）需在真实 backend 上确认不被解引用。

## 9. M1 Chat Completions 子集建议（E4 要求的接口边界草案）

原则（方案 §4.1）：**公共响应只承载用户结果与常规元数据**；控制标签、内部分析、原始流、trace 一律不透传。
下表只列**建议**，不冻结公网字段语义；未定义分支列为待决。

| 面 | 建议（M1 非流式） | 依据/备注 |
| --- | --- | --- |
| 端点 | `POST /v1/chat/completions`，非流式（`stream=false`） | 方案 §4.1 已确认 M1 仅非流式 |
| 必要请求字段 | `model`、`messages`（取最后一条 user 内容作为 question）、`max_tokens`、`temperature`/`top_p` | 其余选项**不支持即报错**，不做静默忽略 |
| 上下文注入 | 素材由服务端按方案组织（DA 臂把文档切成 magic chunk），**不从 `messages` 里读长文档** | 避免把协议机制暴露给客户端 |
| 不支持选项 | `tools`、`tool_choice`、`response_format`、`logprobs`、`n>1`、`stop` 之外的自定义、`stream=true` | 首版明确返回 400 + 错误码，**不静默降级** |
| 响应内容 | `choices[0].message.content` = `extract_public_output()` 的答案；`finish_reason` 由引擎原因映射 | 答案区外分析与标签**不进响应** |
| `usage` | `prompt_tokens`（含协议包装，**公开口径**）、`completion_tokens`（**内部实际生成**）、`visible_completion_tokens`（公开内容量） | 隐藏输出不免除耗时与计费口径：内部/公开分开记账（方案 §4.1） |
| `max_tokens` 语义 | 限制**内部生成** token 数（含协议开销） | 若按公开口径限制，会出现"截断答案"难解释 |
| 错误映射 | 答案缺失/截断 → 200 + `finish_reason="length"`/`"stop"` 时内容为空**或** 502（两种都可，需定一种）；协议异常不单独暴露 | 错误消息**不得**拼入内部文本；映射待决 |
| TTFT | 同时记录"内部首 token"与"公开首结果" | 用户可见 TTFT 用后者 |
| 并发/隔离 | 按 request_id 隔离；结束/取消/断连释放缓冲 | 方案 §4.1 生命周期要求 |
| 待决 | `stop` 支持范围、`finish_reason` 具体取值、多答案区选择、截断答案的 HTTP 语义、是否提供 `/v1/usage` 之外的口径 | 均未冻结，不得由实现默认值替代 |
