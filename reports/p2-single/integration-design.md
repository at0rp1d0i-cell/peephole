# 阶段 05 精简接入设计（单请求、真实 Qwen3.8-27B、版本锁定窄 patch）

日期：2026-09-18。执行者：远端执行 Agent。范围：SUP-004 检查点 1（设计 + 源码核对 + CPU 探针）。
证据级别：**【自读】**＝本轮我自己读到并复核；**【实测】**＝本仓 CPU 探针产物；**【侦察】**＝只读侦察 agent 的 `文件:行号`（未逐行复读）；**【待 GPU】**＝需检查点 2 实测。

## 0. 设计决定（先说结论）

1. **接入路线**：**引擎侧解析 + 每步视图下推 + 只覆写全注意力组的 read metadata**。
   解析发生在 `EngineCore.step()` 拿到 host 侧 token 之后；视图随 `SchedulerOutput` 下发；worker 只在 `build_attn_metadata` 的**单个 KV 组**上替换 `block_table`/`seq_lens`/`max_seq_len`。
2. **本阶段要求 `async_scheduling=False`（`--no-async-scheduling`）**：默认解析为 `True`（`config/vllm.py:1311`「Enable async scheduling unless there is an incompatible option」），此时 engine core 走 `step_with_batch_queue()`，第 t+1 步的 metadata 会在第 t 步 token 回到 host 之前构建、下一步输入 id 在 device 侧拼装 → **未经额外依赖处理的默认异步流水不能保证 C3.5「t 解析、t+1 生效」**（不是对所有异步设计的绝对断言，而是本阶段拒绝该配置并在启用时 fail-fast）。关闭后走 `step()`，读回在**同一步内**、且**早于**下一次 metadata 构建（证据见 §2）。
3. **不改 kernel、不改 canonical 写入/位置/GDN**：FA2 路径不读 `attn_metadata.slot_mapping`（**【侦察】** `flash_attn.py:1038-1186` 无该字段），读写天然解耦；GDN 走 MambaSpec 组、字段不同。
4. **不硬编码块大小**：FA2 不接收 `block_size` 参数（由 4-D page 张量形状决定）；本项目只从运行期配置取 manager/kernel 块大小。
5. **版本锁定的可撤销 patch**：运行时 import 的是 site-packages 副本且与源码树**逐字节一致**（**【自读】** `diff -rq` 差异 0；`vllm.__file__` 指向 site-packages），源码树改动不会生效 → 部署脚本必须**同时**覆盖两处并留 manifest 与回滚。

## 1. 身份（复核过的版本事实）

| 项 | 值 | 证据 |
| --- | --- | --- |
| vLLM 源码 pin | `98dff2a81d747d1dba01a47f939f48c3526d4206` | 【自读】`git -C /root/attnview/vllm rev-parse HEAD` |
| 运行时副本 | `venvs/attnview/lib/python3.12/site-packages/vllm`，`0.29.0`，`commit_id='g98dff2a81'` | 【自读】`vllm.__file__`；【侦察】`vllm/_version.py` |
| 两副本关系 | **逐字节一致**（0 个差异文件），非 editable（无 `direct_url.json`/`__editable__*`） | 【自读】`diff -rq`；【侦察】RECORD 无 direct_url |
| 运行 runner | V2 `vllm/v1/worker/gpu/model_runner.py`（`use_v2_model_runner` 路径） | 【侦察】`gpu_worker.py:457-465`；【自读】`logs/serve-e6b.log` 有 `Using V2 Model Runner` |
| 模型 | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，权重在位 | 【自读】`hub/blobs` 52G、18/18 分片可解析、`e2-verify.json` 32/32 |
| 缓存/层结构 | 4 组：3×MambaSpec(GDN) + 1×FullAttentionSpec(FA2)，manager/kernel 块大小**运行期读取** | 【实测】`evidence/p0-model/e4-kernel-block-probe.json`（该配置为 784，不作常量） |

## 2. 时序：D2H-ready → 解析 → 下一 forward（本设计的关键裁决）

**代码事实（【自读】）**

```text
EngineCore.step()                                   v1/engine/core.py:600-625
  scheduler_output = scheduler.schedule()           # 本步调度账本（含每请求 canonical block_ids 来源）
  future = model_executor.execute_model(so, non_block=True)   # :609
         └─ worker: 本步请求态更新 → prepare_inputs/prepare_attn → 建 metadata(:1657) → 前向
            → 暂存 ExecuteModelState，**返回 None**（execute_model 本身不采样）
  model_output = future.result()                    # :615 —— 此时通常为 None
  if model_output is None:                          # 本版本采样是**另一次**调用
      model_output = model_executor.sample_tokens(grammar_output)   # 同步分支内部解包
         └─ worker: sample → AsyncOutput（旁路流发起 D2H、record copy_event）
         └─ executor 解包：AsyncOutput.get_output() → copy_event.synchronize()
              async_utils.py:167-168   ← 到这里的 host 才可读 token（只解包一次，不重复 get_output）
  scheduler.update_from_output(so, model_output)    # :621 → 之后本适配层解析（§4）
  # 循环下一次 → 才重新 schedule()/execute_model() → 才重新构建 metadata
```

- 【自读】worker 侧 `execute_model` 内构建 metadata：`model_runner.py:1657`（`self.model_state.prepare_attn(...)`）→ `attn_utils.build_attn_metadata`（`attn_utils.py:247`）；前向在 `:1776`，`ExecuteModelState` 暂存在 `:1799`。
- 【侦察】`sample_tokens`（`model_runner.py:1818-1974`）在 `:1894` 构造 `AsyncOutput`，D2H 发在旁路流（`async_utils.py:165`）；`get_output` 只能调用一次（`v1/outputs.py:428-437`）。
- 【自读】`max_concurrent_batches`（`config/vllm.py:562-568`）：`async_scheduling=False` 且 PP=1 → **1** → `EngineCore.batch_queue is None`（`core.py:210-216`）→ 走 `step()`（`core.py:236`）。
- 【侦察】`async_scheduling=True` 时：`step_with_batch_queue`（`core.py:638-750`）先发起下一步 `execute_model` 再等待上一步结果；且下一步 `input_ids` 由 device 侧 `last_sampled_tokens` 拼装（`model_runner.py:1293-1301`），调度器用 `num_output_placeholders` 占位（`v1/core/sched/async_scheduler.py:22-44`）。

**决定**：DA 运行**固定** `--no-async-scheduling`，并在 worker 与本适配层各加一条**启动断言**（`vllm_config.scheduler_config.async_scheduling is False` 且 `vllm_config.max_concurrent_batches == 1`），否则**拒绝服务并报错**，不做「延迟一拍」的降级。

**解析插入点（唯一）**：`EngineCore.step()` 内 `model_output` 到手之后、`scheduler.update_from_output` 之前/之后均可，但**必须**在同一次 `step()` 返回前完成（下一次 `schedule()` 会读取协议状态生成视图）。实现为一次显式调用：

```text
da_adapter.on_step_outputs(model_output, scheduler_output)   # 解析 + 更新请求协议状态 + 记录 trace
```

**代价（只作定性，不作收益判断）**：token 解析复用**所选同步基线**既有的 D2H 完成点（`future.result()` / `sample_tokens` 内的 `get_output`），本设计在 `execute_model`/`sample_tokens` 内**不插入**新的 GPU↔CPU 同步，因此不触发稳态同步检查（`vllm/utils/gpu_sync_debug.py:217-232`）。新增成本是**元数据侧**的：每步每 DA 请求一次小块 H2D + 一次设备侧 `index_select`/填充（见 §5 计量入口），以及**禁用异步调度本身**的流水损失；两者都**待实测**，本阶段不作任何「影响有限」之类判断。

## 3. 请求参数通道（SUP-004 §2.1）

- 【自读】`SamplingParams.extra_args: dict[str, Any] | None`（`sampling_params.py:345`），`SamplingParams` 是 `msgspec.Struct`；既有的同类先例：`kv_transfer_params` / `ec_transfer_params`（`entrypoints/openai/**/protocol.py`）。
- 【自读】全树检索：**没有任何消费方**因未知键报错；只有两个 KV connector 读特定键。→ 可用作内部键（设计用 `extra_args["attnview"]`）。
- 【实测】CPU 往返探针 `tools/p2-probe-params-channel.py` → `evidence/p2-single/params-channel-probe.json`：`SamplingParams` 与 `EngineCoreRequest` 经 `MsgpackEncoder/MsgpackDecoder` 往返后 `extra_args` **逐字段相等、嵌套类型保真**；构造时 `__post_init__` 不改写。附带发现：`MsgpackDecoder(NewRequestData)` 在本环境抛 `NameError: name 'torch' is not defined`（该模块把 `torch.Tensor` 写成字符串前向引用且未在全局导入）；TP=1 的 `UniProcExecutor` 是**同进程直调**（`serial_utils.run_method`），不经过该解码器，故不影响本设计，仅如实记录。
- **安全边界**：该键由**内部驱动器**设置；公共 API 的 `vllm_xargs` 同样落到 `extra_args`，因此适配层必须**忽略/拒绝**来自 API 层的 `attnview` 键（检查点 2 加一条单测），内部扩展不对外开放。
- **载荷**（每请求一次）：`{"attnview": {"protocol": "v1.0", "prompt_len": N, "segment_spans": [[s,e),...], "local_window_span": [s,e), "sink_span": [0,16), "enforce_global": false}}`；其中 span 来自阶段 03 的最终 token-span 映射（**不得**用原文字符偏移当 token 下标）。载荷**不得**含任何几何字段（`kernel_block_size` 等一律拒绝，见 `FORBIDDEN_PAYLOAD_KEYS`）。

## 4. 状态所有权与生命周期（SUP-004 §2.2）

| 状态 | 所有者 | 创建/释放 | 证据 |
| --- | --- | --- | --- |
| 协议状态（模式、声明引用、增量解析缓冲、字节级 detokenizer、trace） | **EngineCore 侧适配层**，按 `req_id` 键（`ProtocolRegistry`） | 请求进入时创建；请求完成/取消/异常时释放（`update_from_output` 的 finish 路径 + abort 队列） | 【自读】`core.py:612`（aborts）、`:621`（update_from_output）；CPU 侧已有 `ProtocolRegistry`（`src/attnview/state.py:223`） |
| 每步下推的视图（可见 logical 块、有效读长度、模式、effect_step） | engine → worker 的单步载荷，**不常驻** | 随 `SchedulerOutput` 生成/消费 | 本设计 |
| 读 metadata 落位（FA 组行缓冲/`seq_lens` 覆写张量） | **worker**，与既有容器同级（不新建平行容器） | 首次需要时按 `req_state_idx` 分配；稳态复用 | 【侦察】建议按 `req_state_idx` 放进 `RequestState`、每步张量放进 `InputBatch`/`InputBuffers`（`gpu/states.py:9-29`、`gpu/input_batch.py:17-113`） |
| 请求槽位/块表 | `RequestState`（`req_id_to_index`）/`BlockTables` | 槽位 LIFO 复用、块表无请求级 clear（重新 add 时 `overwrite=True`） | 【侦察】`states.py:91/126`、`block_table.py:113-133` |
| 清理入口（必须挂接） | worker 侧 `finish_requests`(985)→`_remove_request`(967)（含 preempted 合并）；`add_requests`(1012)/`apply_staged_writes`；`shutdown`(2031) | 【侦察】同上 | 适配层在 `_remove_request` 对应位置清 DA 状态；engine 侧在请求终结时释放协议状态 |
| 行号对应 | **不使用列表位置当请求标识** | 每步 `req_id → batch 行`：`gather_batch_req_state`(1106) + `InputBatch.req_ids/idx_mapping` | 【自读/侦察】`model_runner.py:1126-1131`、`input_batch.py:43-50` |

- 不使用全局单例保存「当前模式」；模式只存在于 per-request 协议状态与单步载荷中。
- 【自读】`_remove_request` 的级联顺序是 `model_state.remove_request` → `req_states.remove_request` → …（`model_runner.py:968-982`，注释说明顺序原因）；DA 清理挂在 `model_state.remove_request` 同级或之后，语义为「按 req_id 清 DA 槽位」。

## 5. FA 读 metadata：成套字段与落位（SUP-004 §2.5）

**成套关系（【自读/侦察】`attn_utils.py:247-336` + `flash_attn.py:1038-1186`）**

| 字段 | 来源 | 本设计动作 |
| --- | --- | --- |
| `block_table`（**二维整数物理块 ID 表**，per KV 组；四维的是 K/V page 缓存，勿混用） | `block_tables[i]`（`BlockTables.gather_block_tables` → `input_block_tables[i]` 持久副本） | **只对全注意力组**替换为 DA 行缓冲（每请求一行；行内容由 worker 用**运行期**几何从 canonical kernel 表做设备侧 gather，宽度一次生成内常量） |
| `seq_lens` → kernel 的 `seqused_k` | `input_buffers.seq_lens`（**跨组共享 buffer**，亦被 sampler/rejection 复用） | **不原地改写共享张量**；为 FA 组提供**独立** `seq_lens` 张量：每行 = `seqused_k` = 各可见块被可见 span 覆盖的位置数之和（最大可见块取实际尾长），**复用 `gpukv.read_table_from_read_view` 的结果**，不在适配层重算 |
| `max_query_len` → `max_seqlen_q` | `input_batch.num_scheduled_tokens.max()` | 不动（decode 步恒为 1） |
| `max_seq_len` → `max_seqlen_k` | `prepare_attn` 传入的共享 int（`seq_lens_cpu_upper_bound` 上界） | FA 组用可见读长度的上界（DA 专用值）；其余组不动 |
| `query_start_loc` | `input_batch.query_start_loc` | **不动**（`seqused_k` 的配套是 `query_start_loc`；改它会影响 Mamba/GDN 与 indexer） |
| `slot_mapping` | 每层 dict（写路径） | **不动**；且 FA2 路径不读 `attn_metadata.slot_mapping`（【侦察】） |
| `scheduler_metadata` / `max_num_splits` | builder 依 `seq_lens`/`max_seq_len` 生成（FA3/split 路径） | 复查是否需随 DA 长度重建；FA2 本机路径下若为 None 则无影响（【待 GPU】确认日志/断言） |
| `block_size` | **不传给 kernel**，由 page 张量形状决定 | 从运行期 `kv_cache_config` 读取 kernel 块大小；**不写 784 常量** |

**无效槽与宽度（规范 I4/I5/§8.3）**

- 本项目内部 `-1` 约定**不得**直接进 kernel；vLLM 自身填充约定为 `NULL_BLOCK_ID = 0` / `PAD_SLOT_ID = -1`（【自读】`v1/attention/backends/utils.py:45-46`）→ 翻译为 `0`，并以 `seqused_k` 界定有效前缀（尾随条目不得被解引用）。
- 宽度在一次生成内**常量**：取该请求最大长度对应的块数（`ceil(max_model_len / kernel_block_size)` 量级），与 canonical 行同宽 → 图捕获/形状稳定前提成立。
- **地址稳定（必须）**：`input_block_tables` 是**跨步持久** buffer，其 `data_ptr` 被 FULL 图捕获与 Mamba `align` ctx 缓存（【侦察】`cudagraph_utils.py:650-655`、`mamba_hybrid.py:172-183`）→ DA 读表**不得**在步间重新分配或改变形状/地址；实现为一次性分配、稳态复用（本阶段跑 eager，仍按此实现以免后续图模式返工）。
- 非块对齐的 `seqused_k` 由 FA kernel 既有测试覆盖（【侦察】`tests/kernels/attention/test_flash_attn.py`：block_size=16、kv_lens=18/37/463/2011）→ 尾块语义可表达；但**因果掩码按可见序列序数还是全局位置仍未定**【待 GPU】（规范 §5 保持打开）。
- 【侦察】**per-group 覆写已有先例**：`model_states/encoder_decoder.py` 覆写 `encoder_seq_lens` → 说明「按组替换 `seq_lens`」是该版本既有模式，不属新机制。

## 6. 只换读视图：写入侧/位置/GDN 保持（SUP-004 §2.4）

| 关注点 | 结论 | 证据 |
| --- | --- | --- |
| canonical 写入 slot | 由 `BlockTables.compute_slot_mappings`（positions + canonical 分块表）算出，经 `build_slot_mappings_by_layer` → `forward_context.slot_mapping` → `unified_kv_cache_update → impl.do_kv_cache_update`（**写入唯一入口**）；**与读表无关** | 【自读】`model_runner.py:1368-1381`、`:1645-1647`、`:1761`；【侦察】`block_table.py:191`、`layers/attention/attention.py:677-724` |
| 读表副本 | `gather_block_tables` 把 canonical 表**复制**进持久 `input_block_tables[i]`，再进 `CommonAttentionMetadata.block_table_tensor` | 【侦察】`block_table.py:149`、`cudagraph_utils.py:650-655` |
| 位置/RoPE | 由 `req_states.num_computed_tokens + query_start_loc` 生成、经 `model_inputs['positions']` 直达 rotary_emb；**不由块表推导** | 【自读】`model_runner.py:1320-1359`、`:1703`；【侦察】`input_batch.py:331-385` |
| GDN/线性注意力 | state 索引来自 **MambaSpec 组自己的** 块表（`mamba_get_block_table_tensor`），读写 `conv_state`/`ssm_state`；`has_initial_state` 依赖 `seq_lens`/`query_start_loc` → **再次说明共享 `seq_lens` 不可改写** | 【侦察】`gdn_attn.py:401-402`、`qwen_gdn_linear_attn.py:1308-1316,1354-1378`、`mamba_attn.py`/`utils.py:1131-1175` |
| 共享张量 | `seq_lens`/`query_start_loc`/`block_table` 的消费者**远超 FA**（其它后端、MLA、sparse indexer、Mamba、图捕获、spec decode） → 一律**不原地改写** | 【侦察】HookPoints §4 全表 |
| 会越界的改动（禁止） | 改 `CommonAttentionMetadata` 字段语义、改 `for_cudagraph_capture`、改 `seq_lens` 语义、改 `block_table` 形状/地址、改 `idx_mapping`/行序、要求异常回滚 | 【侦察】HookPoints §7 |

## 7. patch 清单与部署/撤销（版本锁定、可撤销）

**改动面（最小）**

| # | 文件 | 函数/行 | 改动 | 波及 |
| --- | --- | --- | --- | --- |
| 1 | `vllm/v1/worker/gpu/attn_utils.py` | `build_attn_metadata`（247，组循环 276-336） | 新增可选入参 `da_fa_override`（默认 `None`）；**仅**对全注意力组换用 `(block_table, seq_lens, seq_lens_cpu_upper_bound, max_seq_len)` 的覆写值 | 6 个调用方 + 图捕获调用方（【侦察】）。默认 `None` 时**不改变原行为**（同一批入参走原分支）；此结论**待**无 DA 分支实际执行后才算核实，本轮不称「逐字节等价」 |
| 2 | `vllm/v1/worker/gpu/model_runner.py` | `execute_model`（1657 调用点）、`prepare_attn`/`prepare_dummy_attn`（1368/1389）、`sample_tokens`（1826） | 从 `scheduler_output` 取出本步 DA 载荷；查 `InputBatch` 行→`req_id` 映射；构造/复用 DA 行缓冲；`dummy_run=True` 时**强制忽略** | 唯一调用链（worker/`_dummy_run`/warmup）【侦察】 |
| 3 | 新模块 `vllm/v1/worker/gpu/attnview_adapter.py`（+ engine 侧同模块） | — | 视图落位与协议状态逻辑集中在此，避免散落 | 新增文件，便于整体撤销 |
| 4 | `vllm/v1/engine/core.py` | `step()`（600-625） | 读回后调用适配层解析；构造本步 DA 载荷 | 单点；`step_with_batch_queue` 路径**显式拒绝**（含断言） |
| 5 | `vllm/v1/core/sched/output.py` | `SchedulerOutput` | 新增可选字段 `da_step_views: dict[str, tuple[...]] | None = None` | TP=1 走同进程直调；多进程时该结构需可序列化（【实测】`extra_args` 往返已证；本字段为原生类型） |
| 6 | `vllm/sampling_params.py` | 无强制改动 | 仅用既有 `extra_args` | — |

**部署/撤销**

- 运行时 import 的是 site-packages 副本（【自读】`vllm.__file__`），源码树改动**不生效** → 部署脚本 `tools/p2-apply-patch.sh`：
  1. 记录两处（源码树 + site-packages）目标文件的**前置 sha256** 与 `vllm/_version.py` 的 `commit_id`；
  2. 从 `vllm-patch/`（本仓版本化目录）复制替换/新增文件；
  3. 校验替换后 sha256 与 patch 清单一致；写 `evidence/p2-single/patch-manifest.json`（HEAD、文件、前后哈希）；
  4. `--revert` 从 `vllm-patch/orig/` 恢复并再次校验（回滚后 `diff -rq` 必须回到 0 差异）。
- 不直接无记录修改 site-packages（规范要求）；不改 `requirements.freeze.txt`、不装新包。
- 复跑对照：patch 后必须能用同一命令跑「无 DA 请求（原版路径）」并得到与阶段 02 基线一致的断言（检查点 2 的 global 退化前置）。

## 8. 配置清单（必须显式设置 + 生效核对）

| 项 | 期望值 | 依据（v0.29.0 实际字段/开关） | 生效核对点 |
| --- | --- | --- | --- |
| async scheduling | **False** | `--no-async-scheduling`（`arg_utils.py:1645`）；字段默认 `None`（`config/scheduler.py:179`）→ **自动解析为 True**（`config/vllm.py:1262-1311`）；`disable_async_output_proc` 在本版本**已删除**（全树 0 匹配），等价开关只有它 | ⚠️ **盲区**：启动日志回显 `VllmConfig.__str__`（`v1/engine/core.py:123-127`）**不含** `async_scheduling` → 必须用**断言**核对：`max_concurrent_batches == 1`（`config/vllm.py:562-568`）且调度器类为 `Scheduler` 而非 `AsyncScheduler`（`config/scheduler.py:202-207`），并由适配层打印一行自证 banner |
| 图捕获 | **eager** | `--enforce-eager`（`arg_utils.py:925`）；默认 False（`config/model.py:241`）；O2 默认 `cudagraph_mode=FULL_AND_PIECEWISE`（`config/vllm.py:301`）→ 必须显式关；`enforce_eager` 关编译关图（`config/vllm.py:1370-1376`、`:1584-1588`） | 运行时唯一守卫 `gpu_worker.py:786-788`（`if not enforce_eager: capture_model()`）→ 断言 `compilation_config.cudagraph_mode == NONE`，并确认日志无 `Graph capturing finished`（`model_runner.py:960-964`） |
| prefix caching | **关闭** | 默认 **True**（`config/cache.py:138`）→ 必须 `--no-enable-prefix-caching`（`arg_utils.py:1281-1287`；解析行为外部证据 `tests/engine/test_arg_utils.py:487-490`） | 断言 `cache_config.enable_prefix_caching is False` + 启动日志回显 `enable_prefix_caching` |
| FlashAttention 版本 | **必须显式为 2** | `--attention-config.flash_attn_version=2`（`config/attention.py:41`，`Literal[2,3,4] \| None`）；默认 `None` ⇒ 由平台能力决定：`major==9` 且支持则 FA3、`major==10` 且支持则 FA4、**其余回退 FA2**（`fa_utils.py:96-117`），FA4 仅在 9.x/10.x/11.x 可用（`flash_attn_interface.py:72-84`）⇒ **本机 SM120 默认即 FA2**。显式固定的目的是复跑与可审计，不是纠正错误默认值；同名后端 `FLASH_ATTN` 覆盖 FA2/3/4（`flash_attn.py:879-899`） | 适配层 `derive_geometry` 强制 `flash_attn_version == 2`（`None`/3/4 一律 `UnsupportedConfig`）；运行期再核对 worker 日志 `Using FlashAttention version 2`（`flash_attn.py:897`） |
| 投机/MTP | 关闭（默认即满足） | `speculative_config` 默认 `None`（`config/vllm.py:372`），仅显式提供或 HF config 含 `speculators_config` 时启用（`transformers_utils/config.py:660-663`） | 断言 `speculative_config is None` |
| 确定性执行（固定轨迹） | `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 默认 1（`envs.py:155`）；确定性执行断言见 `uniproc_executor.py:180-183` | 断言 + 记录实际取值 |
| chunked prefill | 允许（v1 默认 `enable_chunked_prefill=True`，`config/scheduler.py:109`） | **所有 prefill chunk 走原版**；只在最后 prefill 采出 g0 后进入协议解析 | trace 中首条 `mode=global`，`effect_step` 自生成流起算 |
| 采样确定性 | 固定 seed + temperature 0（固定轨迹用） | 本阶段固定轨迹要求 | 请求记录 + 两次重放比对 |
| 总序列预算 | ≤8192；TP=1；单活跃请求 | 工作单边界 | 启动配置 + 断言 |

## 9. 预定数值对照合同（草案，**待本地裁决**，不自行放宽）

分三层、各自独立定义，**不得**把阶段 04 的注意力输出容差搬到 logits：

1. **原版可复现性基线（先测）**：同配置、同请求、eager、固定 seed，跑两次原版 → 记录 `max_abs`（预期 0；若非 0，则它是一切比较的下界证据）。
2. **global 退化（结构等价）**：同最终 token 输入、同采样/同步/eager 配置，`global` 视图 vs 原版；判据 `max_abs ≤ 基线`（即结构未引入数值差）；插入层关闭做同向对照。
3. **masked 正确性**：候选 vs **dense masked reference**（同 prefix、同 token 序列、同块外扩集合；参考从原始位置与 canonical KV 独立取值），位置/ dtype 分开报告：
   - 逐层 attention 输出（bf16）：候选容差 `τ_att`，需由**误差来源**推导（bf16 累加 + 归约顺序差异），而非拍定；给出候选值与推导过程，交本地裁决。
   - 选定位置的 logits：`τ_logits` 同法推导（深度累积），单独报告，不借用 `τ_att`。
   - 阶段 04 的 `atol=0.015/rtol=0.01` 是**合成 K/V 的 kernel 级**容差，**不**用于此处。
4. **顺序重复请求**：A 结束后 B 重新初始化，B 的起始模式/片段映射不残留 A（trace 判据）；不宣称混批已支持。
5. **强制轨迹与自由生成分离**：teacher-forcing 入口仅测试启用，记录强制点、原始模型输出/采样与实际消费 token；普通路径不启用。

## 10. 待 GPU 验证（检查点 2 内）

1. 第 t 步解析 → 第 t+1 步生效的**逐 step 时间戳证据**（`copy_event`、metadata 构建、前向）。
2. 压缩块表下的因果语义（可见序列序数 vs 全局位置）——用 dense masked reference 判定（规范 §5）。
3. `scheduler_metadata`/`max_num_splits` 是否需要随 DA 长度重建（FA2 路径确认）。
4. DA 行缓冲的 H2D/device scatter 成本、每步额外拷贝（R10 输入，不宣称收益）。
5. 异常/取消路径：请求取消时 DA 状态释放；异常时**不**隐式回退 global（C6.2），并按合同记录。
6. 图模式：本阶段 eager；FULL/PIECEWISE 下的可行性**不在本阶段**（R08）。

## 11. 未定项（请本地裁决）

1. `τ_att`/`τ_logits` 的具体取值与推导口径（§9.3）——按工作单要求先交本地，不自行选宽容差。
2. 视图下推载体：`SchedulerOutput` 新增字段（本设计推荐）vs 复用 `extra_args` 之外的第二通道；若本地要求零改动调度结构，需退回 worker 侧自同步路线（代价：每步显式同步 + 放弃 async 流水，且会触发稳态同步检查）。
3. `global` 视图是否也走 DA 路径（统一代码、多一次行缓冲写入）还是按请求直接走原版路径（少一次改写、但两条路径需各自证明等价）——本设计**默认后者**（更少假设）。
4. 尾部有效长度的口径：以「已写 KV 长度」为准还是以「可见块数 × 块大小」为准（阶段 03 已在 CPU 侧固定为前者，需要 GPU 侧复述确认）。

## 12. 证据索引

| 证据 | 位置 |
| --- | --- |
| CPU 通道探针 | `tools/p2-probe-params-channel.py` → `evidence/p2-single/params-channel-probe.json` |
| 源码树 vs 安装副本一致性 | `diff -rq vllm/vllm venvs/.../site-packages/vllm` → 0 差异（本轮实测） |
| pin 与版本 | `git -C vllm rev-parse HEAD`、`vllm/_version.py` |
| 组结构与块大小 | `evidence/p0-model/e4-kernel-block-probe.json` |
| 阶段 03 语义基线 | `reports/p1-cpu/integration-design.md`、`src/attnview/{readview,state,parser,decode}.py` |
| 阶段 04 数值路径 | `reports/p1-gpu/read-view-report.md`、`evidence/p1-gpu-v6/` |

**限制**：本文件是设计，不含 GPU 实测；所有 `【待 GPU】` 项与 §11 未定项在检查点 2 前不作结论。设计若需改协议/后端/模型候选/量化/公开 API 合同，另提证据与本地方案对照。

---

## 13. R1 修订（2026-09-18，依 `inbox/SUP-004-R1.md`）

本节记录对初稿的**实质性更正**与随之落地的实现；与正文冲突处以本节为准。

### 13.1 读取长度与当前写入时刻（原 §11.4 错误，已更正）

| 量 | 正确定义 | 说明 |
| --- | --- | --- |
| `attention_kv_len` | **本次 attention 发生时**的 canonical 有效长度，**含本次 forward 正常写入的当前 token** | 不停止在上一采样时刻；阶段 03 的 `RequestProtocolState.attention_kv_len_next` 正是这个量 |
| `seqused_k` | **各可见块被可见 span 覆盖的位置数之和**；最大可见块取**实际尾长** | 既不是完整历史长度，也不是可见块数 × 块大小；由 `gpukv.read_table_from_read_view` 产出（`ReadTable.seqused_k`），适配层**不重算** |
| `tail_len` | 最大可见块的有效位置数 | `ReadTable.tail_len` |
| `needed_width` | `ceil(seqused_k / kernel_block_size)` | **必须**等于可见块数；不等即拒绝（前端按前缀语义索引该列数） |
| 表宽 `width` | `Geometry.max_width`，一次生成内常量（I5） | 后端只索引前 `needed_width` 列；填充列复用最后一块（合法块号，绝不放 `-1`） |

**真实语义夹具（b=784，实现内已自检）**：`prompt_len=6272`、`attention_kv_len=6273`（含当前 token）时，
`local` 视图可见 `[0,5,6,7,8]`、`counts=[784,784,784,784,1]`、`seqused_k=3137`、`tail_len=1`、
对齐后**确实排除已写块 1–4**；同一状态下 `global` 视图 `seqused_k=6273==attention_kv_len`。
CPU 探针里 `[0,5,6,7]`+`6272` 的组合在 b=784 下**不可能**，只是**传输夹具**（原证据保留）。

### 13.2 载荷与几何边界

- 载荷只含协议/布局事实：`protocol`、`prompt_len`、`segment_spans`、`local_window_span`、`sink_span`、`enforce_global`。
  **禁止**出现 `kernel_block_size`/`max_width`/`num_blocks` 等几何字段（实现内 `FORBIDDEN_PAYLOAD_KEYS` 硬拒）。
- 几何只在 worker 侧从**运行期 runner 的实际对象**读取并**互相核对**：
  `runner.block_tables.block_sizes` ↔ 各组 `kv_cache_spec.block_size`（manager 层）；
  `runner.kernel_block_sizes` ↔ `runner.block_tables.kernel_block_sizes`（kernel 层）；
  `runner.block_tables.blocks_per_kv_block` ↔ `block_sizes // kernel_block_sizes`。
  目标组必须是 `runner.attn_groups[fa]` 里的**真实 `FullAttentionSpec`**、后端在本阶段支持集内，
  且并行度为单卡 TP1。任一处不一致即拒绝；`blocks_per_kv_block != 1` 时显式拒绝（`UnsupportedConfig`）。
  **后端名 `FLASH_ATTN` 不等于 FA2**：pin 里同一实现按 `impl.vllm_flash_attn_version` 选 FA2/3/4
  （`flash_attn.py:879-899`）：`major==9`→FA3、`major==10`→FA4、其余→**FA2**（`fa_utils.py:96-117`），
  FA4 仅 9.x/10.x/11.x 可用（`flash_attn_interface.py:72-84`）⇒ **本机 SM120 默认即 FA2**。
  本阶段仍要求**显式** `attention_config.flash_attn_version == 2`（`None` 一律拒绝），
  目的是复跑与可审计，而**不是**"纠正本机默认 FA4"（此前表述有误，已更正）。
  检查点 2 仍须核对 worker 日志 `Using FlashAttention version 2`。
  （否则 `seqused_k` 的计量单位会变）。**不使用默认 784**。
- 稳定读数：`BlockTables.num_blocks.np` 是 host 镜像 → 越界检查（可见块 < 已分配块）**不需要**读 GPU 张量；
  稳态**不做**设备张量的 `.cpu()`/`.item()` 读；计量项只统计**本模块内的代码插桩次数**
  （拷贝/索引/填充/标量赋值/CPU 暂存读取），**不是**运行时同步或耗时的证据，计数为 0 也不得解释为"零成本"。

### 13.3 真实调用链与参数通道（初稿漏 `MambaHybridModelState`）

```text
EngineCore.step()                                    [engine 进程]
  ├─ （schedule 之后）attnview.attach_plans(scheduler_output)     ← 注入上一步解析出的计划
  └─ （update_from_output 之后）attnview.on_step_outputs(...)      ← 登记/清理/解析/产出下一步计划
GPUModelRunner.execute_model                         [worker 进程]
  ├─ attnview_adapter.fa_override_for_step(self, scheduler_output, input_batch, block_tables)
  └─ model_state.prepare_attn(..., da_fa_override=…)   ← 显式 kwarg（仅非 None 时传，其它 ModelState 不受影响）
       └─ MambaHybridModelState.prepare_attn (mamba_hybrid.py:232-330)
            └─ build_attn_metadata(..., da_fa_override=…)           ← 单点落位（组循环内，仅 FA 组）
```

- 落位**构造新对象**：只替换该组的 `block_table`/`seq_lens`/`seq_lens_cpu_upper_bound`/`max_seq_len`；
  `query_start_loc`、`slot_mapping`、`dcp_local_seq_lens`、Mamba/GDN 各字段保持 canonical。
- FA2 前置断言：AOT scheduler 应为 `None`（`flash_attn.py:445`：`aot_schedule = version == 3`，本机 FA2），
  运行期断言，不引入 FA3 路径。
- 写入侧不变：`build_slot_mappings_by_layer → forward_context.slot_mapping → do_kv_cache_update`；FA2 不读 `attn_metadata.slot_mapping`。

### 13.4 prefill / warmup / dummy / 终结（判据用真实状态）

| 场景 | 判据（实测字段，不用启发式） |
| --- | --- |
| prefill（含末尾单 token chunk） | 运行期 `is_prefilling`（`num_computed_prefill_tokens < prefill_len`）；**不得**用 `query_len==1` 代替 → 一律不落位 |
| decode | `is_prefilling=False` 且该行存在计划 → 落位 |
| dummy / profile / 图捕获 | `dummy_run=True` 一律不落位；`for_cudagraph_capture` 分支不落位；本阶段 eager |
| warmup | `warmup.py` 手工构造 `NewRequestData` 走真实 `execute_model` → 因**无载荷**（无 `extra_args["attnview"]`）而直通，不创建协议状态 |
| 普通请求 | 无载荷 → 全程不经过适配层 |
| 终结 / 取消 / 抢占 | 登记与计划在 `schedule()` **之后、`execute_model()` 之前**（`on_step_scheduled`）；解析仍在 `update_from_output` **之后**（`on_step_outputs`）。`SchedulerOutput.finished_req_ids` 是**上一步**终结集合的快照（`schedule()` 返回前已清空，`scheduler.py:1495`），所以终结判定以**本步 finish 项 + 账本核对**为准；清理**幂等**，不会被同一步输出重建 |
| 抢占（本阶段不支持） | `preempted_req_ids` 在**执行前**处理：释放 DA 状态 + 记不支持事件 + 调 `Scheduler.finish_requests(..., FINISHED_ABORTED)` **中止该请求**（不能只写 trace）；无法中止时抛 `UnsupportedConfig` |
| 异常 | 不隐式回退 global（C6.2）。fail-fast 时**先**中止本步涉及的内部请求再抛错（`schedule()` 已改过调度器状态） |

### 13.5 支持范围与拒绝方式

- 不支持配置（异步调度、`max_concurrent_batches != 1`、非 eager、前缀缓存、投机、块粒度不一致）
  在**首次登记 DA 请求**时 `raise UnsupportedConfig`（fail-fast、响亮失败、不静默降级）；
  **普通请求完全不受影响**（原版可复跑，async 调度照常）。
- 为此检查只放在「出现 DA 请求」的路径上，不放在引擎启动路径上（初稿曾计划放在启动，已改）。
- **批次队列路径的守卫（本轮补）**：`async_scheduling=True` 时 engine core 走 `step_with_batch_queue()`，
  本适配层的 `attach_plans`/`on_step_outputs` **根本不会被调用** → 若不额外守卫，DA 请求会**静默退化成原版读取**
  （既违反 C3.5 又不报错）。因此在 `step_with_batch_queue()` 的 `schedule()` 之后立即调用
  `AttnViewEngine.refuse_unsupported_step(scheduler_output)`：一旦本步新调度中出现带载荷的内部请求即
  `raise UnsupportedConfig`（响亮失败，不静默降级）。
- 数值阈值本阶段**不拍定**：先交付观测/参考与 CPU 门禁，本地复核后再放行原版/global 校准。

### 13.6 实现交付（本轮新增，均在版本锁定与可撤销前提下）

| 产物 | 说明 |
| --- | --- |
| `src/attnview/step_plan.py` | 每步计划纯逻辑：载荷校验、几何校验、复用 `readview`+`gpukv` 构造计划、门禁、trace |
| `vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py` | worker 侧：几何推导、覆写构造、计量（**不重算协议算术**） |
| `vllm-patch/files/vllm/v1/engine/attnview_engine.py` | EngineCore 侧：登记/解析/计划/清理/trace（纯 Python，不 import torch） |
| `tools/p2-gen-patch.py` | 从 pin 生成 patched 文件 + `manifest.json`（每个替换必须**恰好命中一次**） |
| `tools/p2-apply-patch.py` | `apply`/`verify`/`revert`：拒绝未知前置哈希；撤销恢复原文件**并删除**新增文件与包目录，恢复后校验指纹 |
| 部署面 | 8 个文件被改（`sched/output.py`、`engine/core.py`、`attn_utils.py`、`model_runner.py`、`model_states/{interface,default,mamba_hybrid}.py`、`gpu_worker.py`）+ 2 个新模块 + 7 个包文件；源码 checkout 与 site-packages **两处都部署** |
| 事实更正 | `SchedulerOutput` 是 **dataclass**（`@dataclass`，非 `msgspec.Struct`）→ 新增字段为普通 dataclass 字段并带默认值 |

### 13.7 证据口径更正

- `evidence/p2-single/params-channel-probe.json` 有 **8 条**检查（此前回执写 7 条，更正）。
- 该探针**只证明** `SamplingParams`/`EngineCoreRequest` 的 msgpack 往返保真；`NewRequestData` **类型化解码**
  在本环境仍失败（`NameError: torch`），因此**不宣称** worker 运行通道已验证；TP=1 走 `UniProcExecutor`
  同进程直调（不经该解码器）是**本轮自读**的源码事实，仍需检查点 2 实测确认。

### 13.8 第二轮更正（本地复核指出的三个 blocker，均已修并加回归测试）

**(a) 计划构造时机**：初稿把 `build_step_plan` 放在 `on_step_outputs`（输出解析之后、**下一次 `schedule()`
之前**），并用 `attention_kv_len_next` 索引当时的 canonical 块映射 —— 这**错**：块是按本次调度分配的，
例如 `prompt_len=6272`（8 块）时，消费 g0 的那次 forward 要写**块 8**，而块 8 直到本次 `schedule()` 才分配。

- 现在：`on_step_outputs` 只做登记/丢弃终结/解析/释放；**计划在 `attach_plans` 内构造**（紧跟 `schedule()` 之后），
  长度按本次调度算：`attention_kv_len = num_computed_tokens + num_scheduled_tokens`（decode 步即 prompt + 已生成，
  含本步写入的当前 token），模式/引用取解析后的当前状态（第 t 步解析 → 本步生效）。
- **prefill 按运行期进度跳过**：`num_computed_tokens < num_prompt_tokens` 即不出计划（末尾 prefill chunk 也可能
  只调度 1 个 token，不能用 token 数判断）；同时校验载荷 `prompt_len` 与调度器 `num_prompt_tokens` 一致，不一致拒绝。
- 回归测试：`test_plan_after_schedule_uses_allocated_block_and_current_token`（8→9 块轨迹）、
  `test_plan_before_block_allocation_would_overrun`（把计划挪回 schedule 之前必越界报错）、
  `test_no_plan_during_prefill_even_single_token_chunk`。

**(b) 真实 tokenizer 接线**：初稿 `core.py` 构造 `AttnViewEngine` 时**没传** `token_text_of`，而模块内
默认分支直接抛「tokenizer 未注入」→ 首个真实采样 token 必崩；而我的测试全都注入了假取词函数，把这个缺口盖住了。

- 现在：新增 `make_token_text_of(vllm_config)`，用 v0.29.0 的真实入口
  `vllm.tokenizers.get_tokenizer(name, tokenizer_mode=…, trust_remote_code=…, revision=…)`（函数内**惰性**导入，
  模块级保持零 vllm/torch 依赖），`core.py` 显式注入；普通请求永不触发加载，DA 请求首 token 即可解析。
- 回归测试：`test_default_construction_uses_real_tokenizer`（**不注入**取词函数，用本地快照的真实 tokenizer
  编码 `<local>` 并断言状态机进入 local）、`test_core_py_passes_tokenizer_factory`（AST 断言 `core.py` 传了工厂）、
  `test_missing_tokenizer_raises_clear_error`（缺 model_config 时抛 `TokenizerUnavailable` 而非含糊错误）。

**(c) 部署脚本事务化**：初稿「边检查边覆盖、最后才写记录」→ 后置目标冲突会留下无法常规 revert 的半部署；
`revert` 不核对现状就覆盖/删除、还无条件 `rmtree` 包目录。

- 现在：`apply` 先**全目标预检**（任一不符即整体拒绝、不碰任何文件）→ 备份 + **先落盘事务日志**（标 `applying`）
  → 复制并逐项校验 → 任一步失败**自动回滚**；`revert` 先**全量核对**（现状须等于部署后哈希、备份须等于前置哈希）
  再动文件，不符即整体拒绝（`--force` 才可越过，且仍校验备份）；只删本次新增文件，包目录用 `rmdir` 逐级清理、
  清理本次产生的字节码缓存，目录内非本次文件**保留并告警**（不算撤销失败）。
- 场景测试（隔离临时树）：`test_last_target_conflict_blocks_whole_deploy`、
  `test_revert_refuses_when_target_changed_after_apply`（含 `--force`）、
  `test_revert_keeps_unrelated_files_in_package_dir`、`test_apply_verify_revert_round_trip`；
  真实环境演练见 `evidence/p2-single/deploy-drill.log`。

**(d) 进度语义（第三轮 blocker）**：初稿在 `attach_plans` 里用 `request.num_computed_tokens + num_scheduled_tokens`
算长度 —— 错。pin 的 `Scheduler.schedule()` 在**返回之前**就调用 `_update_after_schedule`
（`vllm/v1/core/sched/scheduler.py:1383`），其中执行 `request.num_computed_tokens += num_scheduled_token`
（`:1461`），所以调度器账本里的值是 **post**。再 `+scheduled` 会把 6273 算成 6274，并让"末尾单 token prefill chunk"
（post == prompt_len）被误判成 decode。现在：

- `attention_kv_len = post_computed_tokens`（**不再相加**；decode 步即 prompt + 已生成，含本步写入的当前 token）；
- `pre_computed = post - scheduled` 只用于一致性校验（`< 0` 即账本异常，拒绝）；
- 门禁改为**只对非 global 模式出计划**：global（含全部 prefill 步）一律走原版读 metadata；
  若"尚未生成任何 token"（`pre < prompt_len`）却已是非 global 模式 → 协议/时序冲突，直接拒绝。
- 测试夹具按**真实推进**构造（`FakeScheduler.schedule()` 镜像 `:1461` 的自增），并用源码锚定测试
  （`test_progress_semantics_are_anchored_to_pin_source` 断言 pin 里确实存在该调用与自增语句）防止夹具与源码漂移。

**(e) 配置门禁接线（第三轮 blocker）**：初稿只**定义**了 `assert_supported_config`，**没有任何调用者** →
"同步调度但开启前缀缓存/图/投机"时带载荷的请求仍会进入适配器。现在两处都实际调用：

- 引擎侧：首次登记 DA 请求时调用纯函数 `check_supported_config`（`register_new_requests` 内，结果缓存；普通请求不触发）；
- worker 侧：`fa_override_for_step` 在**分配覆写缓冲之前**调用 `assert_supported_config(runner.vllm_config)`；
- 同时修掉门禁自身一个真 bug：`str(CUDAGraphMode.NONE)` 实际是 `'NONE'`（不是 `'CUDAGraphMode.NONE'`），
  原白名单会把**合法的 eager 配置误拒**；现按"最后一段名字"归一化（并兼容 0/None）。
- 测试：`tests/test_p2_adapter_override.py` 走**真实入口** `fa_override_for_step` + CPU 张量：
  合法配置下断言 gather/填充/`seq_lens`/`max_seq_len` 与非 DA 行保持；四类不支持配置（前缀缓存/异步/图/投机）
  各自抛 `UnsupportedConfig` 且**未分配任何缓冲**；引擎侧另有登记即拒绝与合法放行两例。

**(f) 交付可复现性（第三轮 blocker）**：`.gitignore` 的未锚定 `vllm/` 会把 `vllm-patch/files/vllm/...`
一并忽略 → 两个新适配模块只存在于工作区、**从未进入提交**，交付无法从 commit 复现。现在：

- 规则改为 `/vllm/`（只忽略仓库根目录的源码 checkout）；
- 新增 `tests/test_p2_committed_state.py`：用 `git archive HEAD` 导出**独立临时树**，
  断言 manifest 引用的每个来源都在提交里、且**内容哈希与 manifest 声明一致**、且不再被 `.gitignore` 忽略。
  该测试断言的是提交态，因此失败即意味着"有产物没入库/入库的是旧版本"。

**(g) 执行中取消/抢占的存活核对（第四轮 blocker）**：初稿的 `on_step_outputs` 先 `register_new_requests` 再
`parse_outputs`，且只看 `SchedulerOutput.finished_req_ids` —— 但那个集合是**上一步**终结的快照
（`Scheduler.schedule()` 在返回前已 `self.finished_req_ids = set()`，`scheduler.py:1495`）；
执行中被取消的请求由 `_process_aborts_queue` 处理，`update_from_output` 会**跳过**它（`:1880`）并**不产出 finish 项**，
`_free_request` 会 `del self.requests[...]`（`:2512`）。因此旧顺序会**重建/解析已取消请求**。

- 现在：每一步先取**调度器账本存活集**（`self._requests()`），顺序为
  ①丢弃 `finished_req_ids`（上一步快照）→ ②**按存活集过滤后**再登记新请求 → ③把"账本里消失且本步无正常终结项"
  的请求判为执行中取消/抢占：**不解析**、直接释放并写 trace（`cancelled_during_step` / `preempted_during_step`，
  后者标 `unsupported: true` —— 抢占本阶段不支持，驱动侧据此停该用例）→ ④只解析「仍存活」或「本步**正常终结**」
  的请求（正常终结的 stop token 仍须进入解析与 trace）→ ⑤释放本步正常终结者。
- 测试：`AbortDuringStepTest` 覆盖 CPU 轨迹"schedule → 执行中 abort → 输出回收"（含"被取消请求的采样 token
  不得进入解析"、"正常 stop token 仍解析后释放"、"登记前已取消则不登记"、"抢占标记 unsupported"），
  并用源码锚定测试断言 pin 里确实有 `self.finished_req_ids = set()`、`del self.requests[...]`、
  `if request is None or request.is_finished():` 三处语义，防止夹具与源码漂移。

### 13.9 R2 返工（2026-09-18，依 `inbox/SUP-004-R2.md`；纯 CPU，未加载模型/GPU）

**(h) 真实抢占必须在执行前拒绝并中止请求。** `_preempt_request`（`scheduler.py:1405-1446`）**不删除**
`scheduler.requests` —— 只释放块、置 `PREEMPTED`、清 `num_computed_tokens` 并把请求放回 waiting；
`SchedulerOutput(preempted_req_ids=...)` 在 `:1344` 构造输出时携带本步集合、`:1496` 每步清空。
因此"账本里还在"**不能**当作"未被抢占"：新增 `on_step_scheduled` 在 `execute_model` **之前**
调用 `_reject_preempted`，对被抢占的 DA 请求（已登记或本步新调度）释放内部状态、
写 `preempted_during_step`（`unsupported: true`）trace、并调 `Scheduler.finish_requests(req_id,
RequestStatus.FINISHED_ABORTED)` **真实中止该请求**，同时记入 `engine.unsupported`；
拿不到终止 API 时抛 `UnsupportedConfig`（绝不"只写 trace 让驱动猜"）。
**只中止账本不够**：`SchedulerOutput` 是 `schedule()` 的返回值快照，被抢占的请求可能**同时**出现在本步的
`scheduled_new_reqs`/`num_scheduled_tokens` 里（pin 存在 reset 后同一步 preempt+resume 的路径），
所以执行前路径在中止请求后**抛 `UnsupportedConfig` 禁止本步 `execute_model`**（执行后路径仅释放与记录）。
测试用**真实 `_preempt_request`** 调用（最小宿主提供 `_free_request_blocks`/`waiting`/`encoder_cache_manager`
等）断言"请求仍在账本里"这一关键事实，并用**真实 patched `EngineCore.step`**（捕获 executor）断言
"同一 id 同时 preempted 且 scheduled 时 `execute_model` 未被调用"；"删账本再设 preempted"的假组合已删除。

**(i) 登记前置到 schedule 之后、execute 之前。** 原实现只在 `update_from_output` 之后登记：
首步即终结（EOS / `max_tokens=1`）时请求已被 `_free_request` 删除 ⇒ `parsed={}`、`traces=[]`；
配置错误也要先白跑一次模型。现在 `on_step_scheduled` 先做门禁与登记（含载荷/布局校验），
再按本次调度构造计划；`on_step_outputs` 只负责解析与清理。首步终结的请求因此会**解析一次真实采样
token**并**释放一次**（`_finish_and_release` 会先 `flush()` 增量 UTF-8 解码器，残留字节写入 trace 的
`leftover_text`，不静默丢弃）。`EngineCore.shutdown()` 已显式接线 `attnview.shutdown()`（flush + 释放）。
成功终结与取消走**两条路径**：前者由本步 finish 项驱动，后者由账本核对驱动。

**(j) 运行期几何不再是推断。** `derive_geometry(runner)` 读 `block_tables.block_sizes`/
`kernel_block_sizes`/`blocks_per_kv_block` 与 `runner.kernel_block_sizes` 并互核（含"manager 必须能被
kernel 整除"）；目标组要求恰好一个**真实 `FullAttentionSpec`**，后端名取自 `runner.attn_groups[fa]`
里的 `AttentionGroup.backend`（**不是** `KVCacheGroupSpec`——它没有 backend 字段），并要求 TP1。
split 例（manager 784 / kernel 16 / ratio 49）由 `Geometry.validate()` 以 `UnsupportedConfig` 拒绝，
不再报告 784/1。`DefaultModelState`（dense 路径）不在本阶段支持范围：收到覆写即**显式拒绝**，
不留"接收但不消费"的静默分支；目标模型（3×GDN + 1×FA）走 `MambaHybridModelState`，
覆写照旧转传给 `build_attn_metadata`，GDN canonical 对象不动。消费边界在 worker 侧显式核对：批内单活跃请求、单 query(decode) 步、`seq_lens_cpu_upper_bound` 必须存在。
**`num_scheduled_tokens == 1` 不能证明是 decode**（最后一个 prefill chunk 也可能只有 1 个 token）⇒
还须读 `input_batch.is_prefilling_np[行]` 为假；缺少该字段一律拒绝。

**(k) 门禁按实例隔离。** 去掉 `assert_supported_config(..., _cache={})` 的模块级共享缓存，
改为把校验结果记在**具体 runner 实例**上（`runner._attnview_config_checked`）；测试顺序为
"合法 A → 非法 B"，**不重载模块**，直接暴露旧缺陷。

**(l) `enforce_global` 显式消费。** 载荷要求强制 global 时：协议解析与 trace **继续**（状态机照常演进），
但**执行视图**强制 global——本适配层不出受限计划（worker 走原版读 metadata），并在 `enforce_global_steps`
与 trace 里**单列标记**（`protocol_mode` / `applied_view: "global"`），与"协议模式恰为 global"区分。

**(m) 顺手项。** ① FA 组 `max_seq_len` 取**覆写后 CPU 长度向量**的最大值（非 DA 行已镜像 canonical，
DA 行是 `seqused_k`，无需再并入原 canonical 上界；单请求压缩视图下确实缩短，便于耗时归因）；
② metadata 对照改为**真实调用 + 捕获 builder 输入**，按对象身份断言"只有 FA 组换、其它组保持 canonical"
（`build_attn_metadata` 开头的 `seq_lens[:num_reqs]` 切片会产生新对象，测试按此真实语义断言）；
③ 计量计数明确为**代码插桩范围**：删除从未更新的 `host_reads_of_device_tensors`，
新增 `scalar_assignments`/`host_reads_of_cpu_staging`，并在模块文档写明"设备侧同步/耗时须由 GPU 观测"。

### 13.10 校准专用观测钩子（SUP-004 校准单，默认零影响）

三个钩子都只在**对应环境变量被设置**时生效，普通运行完全不介入（无行为差异）：

| 钩子 | 位置（pin 事实） | 作用 | 环境变量 |
| --- | --- | --- | --- |
| 强制轨迹 | worker `model_runner.py`：`self.sample(...)` **之后**、PP broadcast / `AsyncOutput` / `postprocess_sampled` **之前**（`:1861-1920`） | **原地**替换 `sampler_output.sampled_token_ids` ⇒ worker 历史与送往宿主的 token 是同一个值（不分叉）；同时记录**原始采样** | `ATTNVIEW_CALIB_FORCE` / `ATTNVIEW_CALIB_FORCE_LOG` |
| 完整 logits | worker `sample`：`compute_logits(...)` **之后**、grammar/sampler **就地改写之前**（`:1421-1432`） | 保存**全词表** logits（top-k 归一化后无法做全词表误差与尾部非有限值检查） | `ATTNVIEW_CALIB_ARM` 控制文件的 `logits_path`（只对 `target_req_id` 生效） |
| 覆写 trace | `fa_override_for_step` 返回前 | 逐步记录 `seqused_k`/可见块/`max_seq_len`/缓冲指针、**入参指纹**（`data_ptr`/`_version`，CPU 侧元数据、不读设备）与真实 stream/图捕获状态 | `ATTNVIEW_CALIB_TRACE` |

- 强制轨迹格式：`{"tokens": [step][row] = [token_ids...]}`（每步 × 每个请求行）；格式不符或轨迹用尽即**拒绝**，不静默降级。
- 前两个钩子含显式 D2H/H2D 同步，属于**校准专用**，不是稳态行为；报告必须分别标注。
- 不修改 prompt 合同：校准沿用阶段 03 的 `render_arm` 输出与其 token span。

