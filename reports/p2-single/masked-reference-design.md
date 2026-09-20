# 测试专用独立 masked 参考执行（CPU）设计

日期：2026-09-20。状态：**CPU 实现 + 检查完成，实机接线未验证**。依据工作单 `inbox/SUP-004-masked-reference.md`（SHA256 `0f366593…a1e`）；前置 ACCEPT `inbox/SUP-004-masked-prep-accept.md`（`ac072b80…b29e`）。仍属阶段 05，未放行 GPU/阈值。

## 1. 改动点（不新建引擎、不改生产默认行为）
| 文件 | 作用 |
| --- | --- |
| `src/attnview/reference_dense.py` | **测试专用**独立 dense 参考：独立可见位置、物理块表 gather、FP32 显式 QK/softmax/V、原地写 `output`、默认关闭的开关与异常恢复 |
| `tools/p2-ref-cpu-checks.py` + `configs/p2-ref-cpu/checks.json` | 针对性 CPU 检查（22 项）与真实几何配置 |
- 生产路径**无调用点**：`TestOnlyReferenceSwitch` 默认 `enabled=False`，只对目标 `request_id` + 指定层/步生效；异常时自动恢复为关闭并记录。
- 提交指纹：脚本 `23e0738cd64d396e…`、模块 `9bff84056e26873c…`、配置 `51873822f5217b90…`；运行 HEAD `2c30381`。

## 2. 独立性说明
- **可见位置**只由**协议原始 span**（sink / local_window / `segment_spans[ref-1]` / response=[prompt_len, kv_len)）与**独立时间线（mode/refs）**算出；不调用候选的筛选或块表转换 helper，也不读压缩读表。
- **K/V 取值**用 `物理块表 + 逐位置 gather`：逐位置 `divmod` 定位逻辑块与块内偏移，拒绝越界物理块号；不填充、不读未分配/未写槽。
- **数值**在 FP32 显式计算，再按真实输出 dtype cast；**同时返回未 cast 的 FP32 与 cast 结果**，报告必须标明比较对象。
- **端到端 decode 参考**：与候选臂共用**原版 global prefill**，但参考臂使用**自己的**真实 Q/canonical K/V/独立 mask/dense 输出，并**自己的请求与 KV/GDN 状态**，消费同一固定 token 轨迹；同一时刻仅一个活跃请求；不允许把候选隐藏状态/KV 逐步喂给参考臂。

## 3. 接线依据与约束（pin `98dff2a8`，仅源码，无运行证明）
- `Attention.forward` 先 `unified_kv_cache_update` 再 `unified_attention_with_output`；KV 写发生在 `impl.forward` **之前**。
- `unified_attention_with_output` 调用 `impl.forward` 后**忽略其返回值**、继续使用传入的 `output` 缓冲 ⇒ **参考实现必须原地写入该 output 缓冲**；"只返回新张量"不会驱动后续层（本检查集中有专门反例检出该错误）。
- `flash_attn.py` 的 `forward_includes_kv_cache_update=False`；`do_kv_cache_update` 经真实 `slot_mapping` 写 KV。
- 不支持的量化/特殊 attention 特性**直接拒绝**（`ReferenceError`），不静默近似；GQA 头数不整除即拒绝。

## 3b. 端到端挂接路径（本单要求的实现，已落地）
| 文件 | 作用 |
| --- | --- |
| `src/attnview/reference_hook.py` | `TestOnlyReferenceAttachment`：沿用 harness 模式 `original = impl.forward` → 安装同签名 wrapper → `restore()` **恢复原方法本身并断言身份**；`finally` 上下文；**覆盖账本** `(layer, step, action)`；异常即 `restore()` + 关闭开关 |
| `reference_dense.perform_reference_attention()` | 参考路径入口：从 harness 风格 `attn_metadata` 取几何 → 独立 mask → 物理块表 gather → FP32 → **cast 到 `output.dtype`** → **原地写入 output**（返回 None）；返回真实指标 |
- **位置两类分开**（工作单 §2 要求）：`semantic_positions`（协议语义并集）与 **`read_positions`（向块边界外扩后截断到 KV 有效长度）**；dense 参考只用后者。实测 local、kv 7841：语义 **1087** → 读取 **2353**（块 (0, 8, 9, 10)，每块 [784,784,784,1]）。
- **生命周期**：参考臂与候选臂各自独立请求/KV/GDN 状态对象；同一时刻仅一个活跃请求；prefill 复用以"两次独立请求自然得到相同初态"为前提，不复用上一请求残留。
- **真实指标（非恒真）**：`output_dtype=torch.bfloat16`、`fp32_to_output_max_abs=0.0004484206438064575`、`rms=5.590420914813876e-05`、`rel_l2=0.0016386855859309435`、`ref_abs_max=0.1426146924495697`、`non_finite_count=0`（由挂接路径落盘，未设通过阈值）。

## 3c. 固定 pin 真实接口桥接（58121cb，已落地并 CPU 验证）
| 文件 | 作用 |
| --- | --- |
| `src/attnview/reference_bridge.py` | `unpack_native_kv`：按 pin 的 `kv_cache.transpose(1, 2).split(head_size, -1)` 解包原生 4 维 KV；`TimelineBridge`：独立时间线（config 声明 + 真实 tokenizer 累计，段末 token 触发、t+1 消费）；`resolve_geometry`：`block_table` **取自真实二维张量**、模式/几何取自时间线，并与 `metadata.seq_lens` **交叉核对不一致即拒绝**；`perform_reference_attention_native`：真实签名下的参考执行 |
| `src/attnview/reference_hook.py` | 同签名 wrapper（`layer, query, key, value, kv_cache, attn_metadata, output, ...`）：默认关闭、目标 request/层/步命中才接管、覆盖账本、异常即恢复关闭、`restore()` 恢复**原方法**并校验身份 |
- **CPU 门禁（工作单 §1/§5，已实现并测试）**：`scale` **取自真实 `impl.scale`**（缺失即拒绝，不得用 head_dim 推导），开关 `scale` 只作一致性核对；native KV 块长必须 == `bridge.kernel_block_size`；仅允许 **BF16 + causal + 单请求 + 单 token decode**（`num_decode_reqs`/`num_actual_tokens` 从真实 metadata 读取，非 1 即拒绝）；wrapper 收到任何额外参数（如 `output_scale`/`output_block_scale`）**直接拒绝**，量化/特殊 attention 特性不得被静默丢弃（不降级为 dense）。
- **形状兼容（按 `attention.py:524-525`）**：query 与传入 impl 的 output 都是 `[num_tokens, num_heads, head_dim]`；本参考**只接受单 token decode**（`num_tokens != 1` 即拒绝），把 `[heads, head_dim]` 结果**写回 output 的对应视图**（`output[0]`），并校验写回前后 `data_ptr` 不变（不得替换缓冲）。夹具已用真实三维 `[1, heads, head_dim]`，实测通过。
- **真实 metadata 字段仅用其真实拥有的**（`block_table` 二维张量、`seq_lens`、`query_start_loc` 等）；**不做**把 mode/refs/几何塞进 metadata 的自造接口。
- 指纹：`reference_bridge.py 7b2dfea3c38b5bc0…`、`reference_hook.py ea24618b0ddc427e…`、`reference_dense.py 32845d8f543ea485…`、`p2-ref-cpu-checks.py fd5269c9fba676d4…`；运行 HEAD `58121cb`。

## 4. CPU 覆盖（20 项，真实对象驱动，见 `evidence/p3-calib/masked-prep/ref-cpu-checks.json`）（20 项，见 `evidence/p3-calib/masked-prep/ref-cpu-checks.json`）
- 非顺序物理映射:gather 行 == 物理块表映射出的行
- 逻辑块号 ≠ 物理块号(映射确实非顺序)
- 可见位置含当前 token 且以 kv_len-1 结尾
- 可见位置全部 < kv_len(不读未写位置)
- FP32 参考与朴素逐位置实现一致(≤1e-5)
- cast 结果与未 cast FP32 的差异被记录(比较对象明确)
- 结构反例:替换一个可见位置后被独立真值检出(集合不同)
- 结构反例:尾长 −1(少读当前 token)被独立真值检出
- 结构反例:越界位置被拒绝
- 目标层/步启用
- 非目标 request id 一律拒绝
- 未指定层拒绝(缺层)
- 未指定步拒绝(缺步)
- 默认关闭(未 enable 时不接管)
- 开关未启用时进入参考路径即报错
- 参考实现原地写入传入 output(数据指针不变且内容非零)
- 参考写入值与 FP32 参考 cast 后逐元素一致
- 接线反例:只返回新张量、不写原缓冲 ⇒ 输出仍为初始值(被检出)
- 异常(形状不符)后恢复为关闭
- 两请求状态对象独立(互不影响)
- 两请求各自独立推导 mask(不共享残留状态)
- GQA 头数不整除时拒绝(不静默近似)

## 5. 尚缺的实机证据
- 真实引擎内的接线：`Attention.forward → unified_kv_cache_update → unified_attention_with_output` 的实际调用顺序、覆盖期间的写回与恢复、真实 `slot_mapping` 写 KV、真实 GQA/RoPE 观测点、图模式/FA2 后端、模型数值与 GDN 传播。**所有这些仍未验证**；本单完成的是"按真实签名/真实 metadata/原生 KV 形状可挂接"的 CPU 实现与检查，不是实机接线正确性。
- 本检查用**桩件 impl + 挂接驱动**模拟 harness 接线（`unified_attention_with_output` 忽略返回值、消费原 output 缓冲）；**未在真实引擎/GPU 上验证**，不以桩件通过宣称实机接线正确。表宽/后端/捕获状态/真实 slot_mapping 写 KV 等仍属实机范围。
- 数值合同（容差）**未冻结、未内置通过阈值**；输出仅给 max_abs/RMS/相对 L2/参考幅度/非有限计数与 FP32-vs-cast 差异。

## 6. 可复跑入口
```bash
source /root/attnview/env.sh
CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-ref-cpu-checks.py \
  --config configs/p2-ref-cpu/checks.json \
  --out evidence/p3-calib/masked-prep/ref-cpu-checks.json
```
