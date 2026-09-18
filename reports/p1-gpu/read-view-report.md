# 阶段 04 报告（返工后）：GPU 读取视图的独立数值验证

日期：2026-09-18（实际 `date`，克隆实例）。返工依据：`inbox/SUP-003-R1.md`。
冻结配置：`configs/p1-gpu/read-view-check-v2.json`（commit `1617164`，
sha256 `990b3510df3ce45824400beb6b8744aedccb632a1a9f6abe223f6c6e511fa5d8`，**masked 用例之前**提交）。
实现与证据提交：`36c9af7`。证据目录：`evidence/p1-gpu-v2/`（6 件 + SHA256 索引）。

**结论：支持（被测 GPU 张量路径 + 阶段 03→04 数据面连通）**——同一份**常驻 GPU KV 缓存**上，
由阶段 03 `ReadView` 产生的读取表与有效长度，使 focus/local 读取与"同可见集合、独立推导"的
FP32 参考在预冻结容差内一致；读取选择确实生效；canonical 内容与追加位置不被读视图改变。
**不足以判断**：真实模型 logits、RoPE/位置、GDN 状态、请求生命周期、图模式、质量与净收益。

## 0. 对上一版（commit `476b8ef`）的撤回

上一版报"36 case-runs 全通过"是**错误夹具下的参考比较**，其关于"GPU 常驻 KV / 协议 decode 完整覆盖"
的结论**撤回**。复核确认的缺陷（本地已用 CPU 独立复现）：

| 缺陷 | 影响 |
| --- | --- |
| `build_cache` 按 `i` 写入、读取走 `l2p[i]` | L=2587 时逻辑块 0、2 对应物理块 7、11 整段仍是哨兵 8.0；候选与参考读到同一份哨兵，"通过"对块映射没有证明力 |
| CPU 张量 + 每次 `.to('cuda')` | 未证明同一 GPU 缓存的常驻与内容不变；hash/追加检查都只在 CPU |
| focus/local 用例漏掉当前 token 所在块 | 不满足本阶段承诺的 decode 语义，不构成 focus 协议验证 |
| `global_recovery` 无真实模式切换、`preceded_by` 未执行 | 未验证"一次持续轨迹里恢复完整读取" |
| 参考与候选共用 `valid_count`/规范化块列表 | 有效长度错误会同时进入两边 |
| 若干判据只记录不参与 exit | 输出 False 仍可能返回 0 |

旧日志与失败证据**原样保留**在 `evidence/p1-gpu/`（含当时被标为 PASS 的记录），仅供追溯。

## 1. 夹具与真值（R1 §1）

- 先按**逻辑位置** `p ∈ [0, kv_len)` 生成 K_true/V_true（种子化，bf16），再按 canonical 映射散布：
  `cache[l2p[p // b], p % b] = truth[p]`，其余槽保持哨兵 8.0。
- **逐位置核对**（K 与 V 两者）：`physical[l2p[i], off] == logical[i*b+off]`，本次轨迹核对 **6272** 个位置；
  每个逻辑块有效尾之后的槽、以及未承载已写逻辑块的物理块，必须整段是哨兵。
- 分布分离：主用例 `N(0, 1.0)`（普通分布，**不再放大**，避免 one-hot 退化）；
  仅 `sensitive_counterexample` 对**被排除**块 1..6 放大 ×5。

## 2. 常驻 GPU 缓存与写入不变量（R1 §2）

一条轨迹只把 K/V 上传一次，入口断言 `device` 与 `data_ptr` 不变（`k=140364576980992`、`v=140363859755008` 全程一致）；
候选函数只接受已有 CUDA 缓存，**不 gather、不重传**。参考允许 CPU/FP32 gather。

轨迹 `trajectory_resident`（seed 0，同一缓存，9 步）：

| step | 模式 | kv_len | seqused_k | 可见块 | 尾长 | max_abs | 自检 | K/V 变化槽 | 写入 slot |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 0 | global | 6272 | 6272 | 0–7 | 784 | 0.00025 | 1.9e-07 | — | 2352 |
| 1 | focus(1) | 6273 | 2353 | 0,1,7,8 | 1 | 0.00035 | 2.4e-07 | [2352]/[2352] | 2353 |
| 2 | focus(1,3) | 6274 | 3922 | 0,1,5,6,7,8 | 2 | 0.00025 | 1.9e-07 | [2353]/[2353] | 2354 |
| 3 | local | 6275 | 1571 | 0,7,8 | 3 | 0.00042 | 2.1e-07 | [2354]/[2354] | 2355 |
| 4 | local | 6276 | 1572 | 0,7,8 | 4 | 0.00045 | 1.8e-07 | [2355]/[2355] | 2356 |
| 5 | local | 6277 | 1573 | 0,7,8 | 5 | 0.00051 | 1.8e-07 | [2356]/[2356] | 2357 |
| 6 | local | 6278 | 1574 | 0,7,8 | 6 | 0.00044 | 1.9e-07 | [2357]/[2357] | 2358 |
| 7 | local | 6279 | 1575 | 0,7,8 | 7 | 0.00048 | 1.9e-07 | [2358]/[2358] | 2359 |
| 8 | global | 6280 | 6280 | 0–8 | 8 | 0.00024 | 2.2e-07 | [2359]/[2359] | 2360 |

要点：**真实模式切换**（global→focus→focus→local×5→global）；**有效长度随追加逐步变化**（1571→1575）；
每次追加**只改一个 canonical slot、K 与 V 双侧一致**，写入位置由原逻辑位置决定（2352→2360，
第 1 步跨入逻辑块 8 = 物理块 3）；每步读数前后 K、V 内容未变；当前 token 所在块始终保留。

## 3. 数据面连通（R1 §3）

可见集合不再硬编码：由 `TokenLayout(prompt_len, segment_spans, local_window_span, sink_span)` +
`ViewInputs` → **阶段 03 `build_read_view`** → `ReadView.visible_blocks/physical_block_ids/visible_spans`
→ `gpukv.read_table_from_read_view` 转成**无 `-1`** 读取表（FA2 不得收到 `-1`；阶段 03 的 `-1` 属尾部填充约定）。

每个测量项都与 oracle 的**独立预期**逐项比对并参与门禁：块集合、物理表、`seqused_k`、下一写入位置、
当前块可见性。

## 4. 参考独立性与全判据门禁（R1 §4）

- `src/attnview/gpuoracle.py`：从语义输入（mode/refs/spans/prompt_len/kv_len）用**逐位置布尔 mask**
  独立推导可见集合，自行做块外扩与因果上界（query_len=1 → 上界 = kv_len−1），从**逻辑真值**按位置 gather，
  FP32 **逐 head** 计算（GQA：h → h//6）。**不 import 候选 gpukv，不由候选读表反推**。
- 第三实现自检：同一可见集合再算一遍 SDPA，与 oracle 的偏差全用例 ≤ 4.2e-07（门限 1e-5）。
- `src/attnview/gpucheck.py`：每次测量产生判据并**全部参与 PASS/exit**——必需字段齐全
  （`schema_complete`）、数值、oracle 自检、读取表无 `-1`、表宽、数据面 5 项、**K 与 V 两项内容判据
  （必须都为 `True`，`None`/缺字段即失败）**、常驻性、**追加声明 `append_declared`（必须显式 True/False；
  声明追加就必须给出 `expected_slots`）**、追加 slot 逐槽匹配（K 与 V 双侧，**附加**于内容判据而非替代），
  外加用例级行隔离与敏感度 → 本轮共 **852 条判据**。
  失败保留**完整**错误信息（含最差偏差位置与数值）。CPU 失败注入测试覆盖：每类判据的显式 False、
  `K=None`/`V=None`/两项都缺/整个 `kv_integrity` 缺失、缺少任一必需段、`appended` 非布尔、
  `appended=True` 无 `expected_slots`、追加槽只错一侧（`tests/test_gpukv.py`，106 项 CPU 全通过）。
  该轮收紧来自 advisor 复核：原实现用 `is not None` 守卫，导致 `k_unchanged=None` 时连 `v_unchanged=False`
  都不检查、缺失整个 `kv_integrity` 也能通过。

## 5. 结果汇总（12 条目 × 3 种子：**852 条判据 / 0 失败 / exit 0**）

| 用例 | 可见块（seed 0） | seqused_k | 尾长 | max_abs | rms |
| --- | --- | ---: | ---: | ---: | ---: |
| trajectory step0（global） | 0–7 | 6272 | 784 | 0.00025 | 6.1e-05 |
| trajectory step3（local+追加） | 0,7,8 | 1571 | 3 | 0.00042 | — |
| trajectory step8（global 恢复） | 0–8 | 6280 | 8 | 0.00024 | — |
| focus_current_block（focus refs=(3)） | 0,5,6,7 | 3136 | 784 | 0.00030 | 6.1e-05 |
| local_excludes_document | 0,7 | 1568 | 784 | 0.00050 | 8.7e-05 |
| tail_1 | 0,5,6 | 1569 | 1 | 0.00052 | 8.6e-05 |
| tail_b_minus_1 | 0,6 | 1567 | 783 | 0.00046 | 8.7e-05 |
| tail_b | 0,6 | 1568 | 784 | 0.00058 | 8.7e-05 |
| sensitive_counterexample | 0,7 | 1568 | 784 | 0.00050 | 8.7e-05 |
| b2_rows 行0（global） | 0–7 | 6272 | 784 | 0.00025 | — |
| b2_rows 行1（local） | 0,7 | 1568 | 784 | 0.00050 | — |
| block16_control（补充） | 0,2,3 | 37 | 5 | 0.00362 | 5.1e-04 |
| arbitrary_subset_supplement | 0,2,7 | 2352 | 784 | 0.00032 | 7.2e-05 |

- **敏感度**：`sensitive_counterexample` 的 masked 与 full 输出最大差 **15.04**（门限 0.15）→ 读取选择确实生效。
- **行隔离**：B=2 行0 逐行单独调用 vs 批量 = 4.9e-04，行1 = 0.0（均在容差内；行0 的 4.9e-04 是 bf16
  批量/单行归约差异，非串表）。
- **越读负对照**（诊断）：正确 `seqused_k`=1569 → max_abs 0.0005；人为 +100 → **8.12**（哨兵可靠抓越读，
  且反证正确调用不读 `seqused_k` 之后的槽）。
- 峰值显存 156 MiB（含验证用 clone），逐用例秒级；**非性能测量**。

## 6. 环境与占用（R1 §5）

`torch 2.13.0+cu130`、`vllm 0.29.0`（checkout `98dff2a81d747d1dba01a47f939f48c3526d4206`）、python 3.12.13、
RTX PRO 6000 Blackwell cc(12,0) 97887 MiB、驱动 580.142、入口 `vllm.vllm_flash_attn.flash_attn_varlen_func`
（`fa_version=2`）、eager、BF16。
`nvidia-smi`：**memory.used 0 MiB / utilization.gpu 100% / 无可见 compute 进程** → 不据此宣称独占，
未停止任何其他作业；本轮只做正确性验证。

## 7. 如实记录的问题与限制

1. **冻结配置里两个信息性 `expect_blocks` 我手算错了**：step2 写 `[0,1,6,7,8]`，实际为 `[0,1,5,6,7,8]`
   （`align_outward(4700)` 向下对齐到块 5，不是块 6）。该字段**不参与判据**，以 oracle 独立推导为准；
   保留错误原样以免事后改写冻结件，在此如实说明。
2. **主用例普通分布下误差量级 ~5e-4**：远小于容差（atol 1.5e-2）；相对偏差约 1e-3 量级。
   `block16_control` 误差 0.0036 略高（块更小、分布相同），仍在容差内。
3. **B=2 行0 的"行隔离"为 4.9e-04 而非 0**：同为 bf16 归约顺序差异，非读取表串用；已按容差判定并记录。
4. 本阶段只覆盖 **query_len=1 且保留当前 token** 的因果 decode；`seqused_k` 的前缀语义、块粒度读取
   与"有效长度 = 覆盖位置数"的关系**不外推**到多 query/chunked prefill。
5. 未涉及：真实权重 logits/自由生成、RoPE/位置变换、GNB/线性注意力层、prefix caching、抢占、图模式、
   HTTP/API、请求生命周期、并发与性能矩阵、质量与净收益。
6. `block16_control` 与 `arbitrary_subset_supplement` 是**非协议补充用例**（后者可见集合不来自 ReadView），
   不用于宣称协议 decode 覆盖。

## 8. 命令与复现

```bash
source /root/attnview/env.sh
python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check-v2.json  # exit 0，852 判据 / 0 失败
python3 tools/p1gpu-overread-control.py                                              # exit 0，越读负对照
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -t tests               # 106 项 CPU（含失败注入）
```

## 9. 推荐下一步

1. 单请求 runner 接入工作单：解析同步点（AsyncOutput D2H ready）、需禁用功能（图模式/prefix caching/抢占）、
   同步代价；在真实引擎上用本阶段用例回归。
2. 模型级验证：真实权重下 64K/128K 的 logits/短生成对照（含 RoPE 与位置处理），再谈质量与收益。
3. 把本阶段 784 主用例纳入回归入口；`block16` 与"显式块集合"用例只作补充。
