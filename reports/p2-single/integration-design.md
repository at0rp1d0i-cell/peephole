# 阶段 05 精简接入设计（单请求、真实 Qwen3.8-27B、版本锁定窄 patch）

日期：2026-09-18。执行者：远端执行 Agent。范围：SUP-004 检查点 1（设计 + 源码核对 + CPU 探针）。
证据级别：**【自读】**＝本轮我自己读到并复核；**【实测】**＝本仓 CPU 探针产物；**【侦察】**＝只读侦察 agent 的 `文件:行号`（未逐行复读）；**【待 GPU】**＝需检查点 2 实测。

## 0. 设计决定（先说结论）

1. **接入路线**：**引擎侧解析 + 每步视图下推 + 只覆写全注意力组的 read metadata**。
   解析发生在 `EngineCore.step()` 拿到 host 侧 token 之后；视图随 `SchedulerOutput` 下发；worker 只在 `build_attn_metadata` 的**单个 KV 组**上替换 `block_table`/`seq_lens`/`max_seq_len`。
2. **必须 `async_scheduling=False`（`--no-async-scheduling`）**：默认解析为 `True`（`config/vllm.py:1311`「Enable async scheduling unless there is an incompatible option」），此时 engine core 走 `step_with_batch_queue()`，**第 t+1 步的 metadata 会在第 t 步 token 回到 host 之前构建**，且下一步输入 id 在 device 侧拼装 → C3.5「t 解析、t+1 生效」**不可能满足**。关闭后走 `step()`，读回在**同一步内**、且**早于**下一次 metadata 构建（证据见 §2）。
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
  scheduler_output = scheduler.schedule()           # 本步调度账本（含每请求 logical block_ids）
  future = model_executor.execute_model(so, non_block=True)   # :609
  model_output = future.result()                    # :615 → AsyncOutputFuture.result()
     └─ UniProcExecutor.collective_rpc(non_block=False) → AsyncOutput.get_output()
          └─ copy_event.synchronize()               async_utils.py:167-168   ← 到这里的 host 才可读 token
  model_output = sample_tokens(...) if None         # 见下：本版本 sample_tokens 单独调用
  scheduler.update_from_output(so, model_output)    # :621
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

**代价**：不新增任何 GPU↔CPU 同步——该同步是引擎在本配置下**本来就要做**的（`future.result()`）。此外本设计在 `execute_model`/`sample_tokens` 内**不插入**任何同步，因此不会触发稳态同步检查（`vllm/utils/gpu_sync_debug.py:217-232`）；这条路线的取舍是**放弃 async scheduling 的流水重叠**（单请求、延迟受限场景影响有限；成本归 R10 后续测量）。

## 3. 请求参数通道（SUP-004 §2.1）

- 【自读】`SamplingParams.extra_args: dict[str, Any] | None`（`sampling_params.py:345`），`SamplingParams` 是 `msgspec.Struct`；既有的同类先例：`kv_transfer_params` / `ec_transfer_params`（`entrypoints/openai/**/protocol.py`）。
- 【自读】全树检索：**没有任何消费方**因未知键报错；只有两个 KV connector 读特定键。→ 可用作内部键（设计用 `extra_args["attnview"]`）。
- 【实测】CPU 往返探针 `tools/p2-probe-params-channel.py` → `evidence/p2-single/params-channel-probe.json`：`SamplingParams` 与 `EngineCoreRequest` 经 `MsgpackEncoder/MsgpackDecoder` 往返后 `extra_args` **逐字段相等、嵌套类型保真**；构造时 `__post_init__` 不改写。附带发现：`MsgpackDecoder(NewRequestData)` 在本环境抛 `NameError: name 'torch' is not defined`（该模块把 `torch.Tensor` 写成字符串前向引用且未在全局导入）；TP=1 的 `UniProcExecutor` 是**同进程直调**（`serial_utils.run_method`），不经过该解码器，故不影响本设计，仅如实记录。
- **安全边界**：该键由**内部驱动器**设置；公共 API 的 `vllm_xargs` 同样落到 `extra_args`，因此适配层必须**忽略/拒绝**来自 API 层的 `attnview` 键（检查点 2 加一条单测），内部扩展不对外开放。
- **载荷**（每请求一次）：`{"attnview": {"protocol": "v1.0", "segment_token_spans": [[s,e),...], "scaffold_spans": [[s,e),...], "sink_tokens": 16, "kernel_block_size": <运行期读取>, "enforce_global": false}}`；其中 span 来自阶段 03 的最终 token-span 映射（**不得**用原文字符偏移当 token 下标）。

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
| `block_table`（4-D page 张量，per KV 组） | `block_tables[i]`（`BlockTables.gather_block_tables`） | **只对全注意力组**替换为 DA 行缓冲（每请求一行 = 可见 logical 块→kernel 块号映射） |
| `seq_lens` → kernel 的 `seqused_k` | `input_buffers.seq_lens`（**跨组共享 buffer**，亦被 sampler/rejection 复用） | **不原地改写共享张量**；为 FA 组提供**独立** `seq_lens_da` 张量（每行 = 可见有效读长度） |
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
| 1 | `vllm/v1/worker/gpu/attn_utils.py` | `build_attn_metadata`（247，组循环 276-336） | 新增可选入参 `da_group_override`；**仅**对全注意力组用 `(block_table_da, seq_lens_da, max_seq_len_da)` | 6 个调用方 + 图捕获调用方（【侦察】）；默认 `None` 时行为与原版**逐字节等价** |
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
