# masked reference GPU diagnostic

本报告记录在终版 commit 上、用入库入口跑出的两次真实 Qwen3.8-27B GPU 事务（独立 reference 臂 + masked 候选臂），以及二者在同一固定轨迹上的对照。它是 correctness（正确性）诊断证据，不是正式 benchmark（基准测试）、质量验收或数值阈值冻结。

## Run identity

- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；vLLM pin `98dff2a81d747d1dba01a47f939f48c3526d4206`
- 运行配置：单卡 BF16、TP1、eager/同步调度、显式 FlashAttention 2（`flash_attn_version=2`）、`max_model_len=8192`、`max_num_batched_tokens=8192`、单活跃请求、关闭 prefix caching / 投机 / CUDA Graph
- Source HEAD：`4d32a16376730e25d518241a82f902ee23e67700`；驱动 `tools/p2-calib-run.py` sha256 `a9da137f6350b79977e225c6592f69e7d140ea9353539571dad24fb961eb6635`（等于该 commit 的文件内容），源码快照随 manifest 落盘且结束时哈希未变
- 事务入口：`tools/p2-diagnostic-run.py`（`apply → verify → run → finally revert`）；配置 `configs/p2-masked-reference/reference.json`（sha256 `6a69cf307cc5738464351cea10ef83a64e59ced4c8ff2b693d8339aee3b25b83`）与 `configs/p2-masked-smoke/diagnostic.json`（sha256 `2c2d5cfac6d29d1b4db986bba9d4ecd47c3bf2986c2cc84ad2d36661b091207c`）
- 夹具：prompt 7834 token、29 次采样 / 28 次消费 decode forward；协议载荷 `protocol=v1.0, prompt_len=7834, segment_spans=[[362,2316],[2365,4255],[4304,6756]], local_window_span=[6765,7834], sink_span=[0,16], enforce_global=false`；输入与轨迹哈希见各 manifest 的 `sources.input_hashes`

| 角色 | 证据目录 |
| --- | --- |
| 独立 reference（decode 由 FP32 dense 参考接管） | `evidence/p3-masked-reference/reference-20260921-run5/` |
| masked 候选 | `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/` |
| 对照摘要（CPU 复算） | `evidence/p3-masked-reference/reference-20260921-run5/summary.json` |
| 候选侧局部 FP32 观察 | `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/numeric-audit.json` |

## 运行代码 ≠ 交付代码？三方核对

两条终版事务用的驱动 `tools/p2-calib-run.py` 是 sha256 `a9da137f6350…`：等于 commit `4d32a16` 的文件内容，也等于两个 run 目录里冻结的 `run/source/p2-calib-run.py`。`src/attnview/**` 在本轮 commit 上没有改动（`git diff 73364dc 4d32a16 -- src/` 为空），且 5 个 reference run 的 manifest 记录的 22 个模块哈希彼此相同。下表把"哪次 run 用的是哪份代码"钉死，不作追认：

| run | manifest `script_sha256` | 该驱动的可核来源 | 说明 |
| --- | --- | --- | --- |
| run1 | `f38197c8e6ac…` | commit `73364dc` 的 `tools/p2-calib-run.py`（同哈希） | 无 GDN state 探测的版本 |
| run2 | `03069799700c…` | `evidence/p3-masked-reference/reference-20260921-run2/run/source/p2-calib-run.py` | GDN 探测首版：要求 FA/GDN storage 独占，真机被该门禁拒绝（见该 run 的 `run/error.txt`） |
| run3 | `d1051431ec1b…` | `evidence/p3-masked-reference/reference-20260921-run3/run/source/p2-calib-run.py` | 同 run2，拒绝再次发生 |
| run4 | `65d231037d0c…` | `evidence/p3-masked-reference/reference-20260921-run4/run/source/p2-calib-run.py` | 首版 GDN 证据口径（逐层 `alias_pairs`=1536），上一轮交付所用 |
| run5 | `a9da137f6350…` | commit `4d32a16` + 同哈希快照 | 交付版本：去重 storage 重叠计数 |

run4 的驱动**不是**本轮代码。两份快照的差异只有 GDN 别名统计的呈现（先按 `(ptr, bytes)` 去重再计重叠对），读取视图、参考计算与写回路径未变；可直接核验：

```bash
diff evidence/p3-masked-reference/reference-20260921-run4/run/source/p2-calib-run.py \
     <(git show 4d32a16:tools/p2-calib-run.py)
```

本轮没有把该未上机的观测改动追认给 run4，也不用它解释 run4 的数字——本报告的全部数字来自 run5 / `4d32a16`。run4 的 manifest、结构化快照与捕获大件按"在盘未入库集合"登记（`reports/evidence-registry.json`），不随仓发行。

## 事务结果

两次事务的 `transaction.json` 四阶段退出码均为 0（apply / verify / gpu / revert），结束时 `vllm-patch/deployed.json` 不存在；两份 manifest 都是 `exit_code=0`、`failures=[]`、无 `error.txt`。上一轮（`reference-20260921-run4` + `diagnostic-20260921-current`）由未入库的临时外壳启动，外层记录了非零 shell 状态；其驱动 `65d231037d0c…` 既不是本轮代码也不是任何提交，快照随本轮入库（见上一节的核对表）。本轮因此用入库入口在同一 commit 上重跑，把证据绑到 commit 与可复跑命令；两轮的 capture 哈希与对照数字逐位相同（见"重复性"）。

## Reference run 结果

- 29/29 forward 完成，16/16 FullAttention 层覆盖；prefill 为 passthrough，28 个 decode forward 全部由独立 FP32 dense 参考接管：用本臂自己的真实 Q 与 canonical K/V、独立可见 mask（声明配置 + 真实 token 分段 + renderer 原始 spans，块外扩后截到当前有效长度）计算 FP32 结果，cast 到真实 BF16 后原地写入传入 output。
- 覆盖账本 464 条 = prefill passthrough 16 + decode overrode 448（16 层 × 28 步）；448 条 decode 记录全部 `wrote_in_place=true`、`buffer_ptr_preserved=true`、`non_finite_count=0`。
- 参考方法与捕获在 finally 恢复（`restored_layers` = 全部 16 层，结束时 `enabled=false`），恢复完成后才执行额外 cleanup 请求。
- 本次 request 16.12 s、startup 48.54 s。

## GDN / KV 证据

- capture 找到 48 个带 bound 两张量 `kv_cache` 的 linear-attention 层；29/29 步都从真实 metadata 取得 `non_spec_state_indices_tensor`，只对目标请求的 state 行做 SHA-256 摘要（不复制整份 recurrent cache），缺步 0、非有限 state 0；样本层在 29 步上摘要各不相同，说明状态确实在更新并被观测到。
- vLLM allocator 会把各 cache group overlay 到同一块 backing（`worker/utils.py:387-389`）：按 (ptr, bytes) 去重后的 FA/GDN storage 重叠对数 `alias_pair_count=1`、`shared_backing_expected=true`。本报告不把原始的指针别名当作缺陷，也不声称物理分配独立；独立性依据是真实层/组身份（`FullAttentionSpec` 组）、state-index 选择、逐步摘要与独立的 FA 层捕获。
- masked 候选臂在本次运行同样启用了该探针（`required=true`），48 层 × 29 步无缺步、无非有限值。

## Attention 与 logits 观察（candidate vs reference）

口径：`diff = candidate - reference`，`relative_l2 = ||diff|| / ||reference||`。没有设置任何通过线。

| 对象 | 数量 | max_abs | max_rms | max_relative_l2 | 非有限 |
| --- | --- | --- | --- | --- | --- |
| decode Q（16 层 × decode 6/7/20/25） | 64 | 0.125 | 0.0208161 | 0.0147758 | 0 |
| decode attention 输出 | 64 | 0.5625 | 0.0261061 | 0.0164634 | 0 |
| 逐步完整 logits | 29 | 0.17578125 | 0.0325839 | 0.0153838 | 0 |

29 步 logits 的 argmax 全部一致（`argmax_mismatch=0`），候选与参考的产出 token 轨迹逐 token 相同。

候选侧局部 FP32 块外扩观察（`tools/p2-masked-smoke-audit.py`，只用候选轨迹的 Q/K/V，64 点）：`candidate_vs_fp32` max_abs ≤ 0.12401580810546875、relative L2 ≤ 0.0018260934157297015；`candidate_vs_bf16_reference` max_abs ≤ 0.125、relative L2 ≤ 0.0014283563941717148。

## 重复性

本轮（run5 / 4d32a16）与上一轮（run4 / current）在相同固定输入上给出相同的 capture 哈希：reference `e92691924ec9c9a6a83361428f653a14b1d2cf2fb3980ffbb58b09040f3be9d6`、masked `936383f66fa22e3c081059bfcb998923aa68a657f178e607556b94dd97dac499`；attention Q/out 与 29 步 logits 的对照数字、以及候选侧局部 FP32 观察值逐位相同。即两条事务链在重新执行时复现了同一结果。差别只有 GDN 别名口径：上一轮记录逐层 alias 对 1536（层数乘积），本轮记录去重后的重叠对数 1。

## 复跑

下面命令在**当前 checkout 的 commit** 上跑出**新一轮**事务（同口径、新目录），不会重放 run4/run5 的字节；要核对历史轮次，用上一节表格里的 `run/source/p2-calib-run.py` 快照或对应 commit（run1 → `73364dc`，run5 → `4d32a16`）比对，不要拿当前工作树当"当时那份代码"。本报告数字对应 commit `4d32a16`（驱动 `a9da137f6350…`）。

```bash
source env.sh

# 1) 独立 reference 事务（新目录；驱动拒绝非空输出目录，launcher 拒绝已有 deployment journal）
CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-diagnostic-run.py \
  --config configs/p2-masked-reference/reference.json \
  --out evidence/p3-masked-reference/reference-<new-id>

# 2) masked 候选事务
CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-diagnostic-run.py \
  --config configs/p2-masked-smoke/diagnostic.json \
  --out evidence/p3-masked-smoke/diagnostic-<new-id>

# 3) CPU 对照摘要（拒绝覆盖已有 summary；arm/HEAD/revision/载荷/输入哈希/层名映射/轨迹不一致即失败）
CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-reference-summary.py \
  --reference evidence/p3-masked-reference/reference-<new-id>/run \
  --candidate evidence/p3-masked-smoke/diagnostic-<new-id>/run \
  --out evidence/p3-masked-reference/reference-<new-id>/summary.json

# 4) 候选侧局部 FP32 观察
CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-masked-smoke-audit.py \
  --run evidence/p3-masked-smoke/diagnostic-<new-id>/run \
  --out evidence/p3-masked-smoke/diagnostic-<new-id>/numeric-audit.json
```

命令、阶段退出码、PID、HEAD、配置与源码哈希由事务记录与 manifest 保存；原始日志、capture、cleanup 检查与 GDN/FA 快照都在上述两个证据目录内。历史 run（run1 无 GDN 探针、run2/run3 探针要求独占 storage 被拒、run4 首版探针）保留未删。

## 边界（未做 / 未放行）

- 不是 masked 数值验收：没有冻结阈值，上表差异只是观察；不声称任意 prompt、自由生成、任务级质量、perplexity 或 API 质量。
- 不是性能结论：reference 侧含同步 CPU copy 与 FP32 dense 计算，耗时含捕获与同步开销，不得用于性能宣传。
- GDN/KV 只做 state-index 选择、逐步摘要与 FA 捕获分离；没有逐字节验证 cache 写入内容，没有证明物理分配独立。
- 长上下文（>8192）、混批、量化、公共 HTTP/API 转发、正式 benchmark 仍未放行；本次只跑两次有界诊断事务。
