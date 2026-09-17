# 阶段 04 报告：GPU 读取视图的独立数值验证（BF16/FA2、784、query_len=1）

日期：2026-09-18（远端 GPU 会话）。工作单：`context/SUP-003-material/docs/stage-04-gpu-read-view.md`（素材 commit
`0283abce5552cfb6fa3655e540731eef86eef723`）。冻结配置：`configs/p1-gpu/read-view-check.json`
（commit `9b02471`，sha256 `00cec9135cb1036d90cddcafccb0f4919ac2c4d06b91e6830fd25561bc71466b`）。
证据：`evidence/p1-gpu/`（7 件 + 索引）。

**结论：支持（被测 GPU 张量路径）**——同一份完整 KV 缓存上，用"读取表 + 有效长度"表达的 focus/local 读取
与相同可见集合的独立 dense masked reference 在预冻结容差内一致；读取选择确实生效；canonical 内容与追加位置
不被读视图改变。**不足以判断**（真实模型 logits、RoPE/位置处理、GDN 状态、请求生命周期、图模式、质量与净收益）。
本阶段**不是**模型级接入、M1 API 完成或性能收益的证据。

## 1. 实际调用与字段

| 项 | 实际值 |
| --- | --- |
| 入口 | `vllm.vllm_flash_attn.flash_attn_varlen_func`（vLLM **vendored** FA2，同目录 `_vllm_fa2_C.abi3.so`） |
| `fa_version` | **2**（显式传入；阶段 02 服务日志 `flash_attn.py:897 Using FlashAttention version 2` 为同源依据） |
| 版本 | python 3.12.13 / torch 2.13.0+cu130 / vLLM 0.29.0（源码 checkout `98dff2a81d747d1dba01a47f939f48c3526d4206`，tag v0.29.0） |
| 设备 | RTX PRO 6000 Blackwell Server Edition，cc (12,0)，97887 MiB，驱动 580.142 |
| 执行模式 | eager，无 CUDA graph；无预热/吞吐测量 |
| dtype | bfloat16（无量化；`q/k/v` 均 bf16，参考 FP32） |
| 形状 | `q` = [B, 24, 256]；`k`/`v` 缓存 = [12, 784, 4, 256] 连续 bf16（各 19.3 MB）；`block_table` = [B, width] int32；`seqused_k` = [B] int32 |
| 其余实参 | `cu_seqlens_q` = arange(B+1)、`max_seqlen_q` = 1、`max_seqlen_k` = max(seqused_k)、`softmax_scale` = 256^−0.5 = 0.0625、`causal` = True、`window_size` = None、`softcap` = 0.0 |
| 顶点块大小 | **784**（阶段 02 实测 `kernel_block_sizes=[784,784,784,784]`，`evidence/p0-model/e4-kernel-block-probe.json`）；16 仅作可选对照 |
| 物理块映射 | 固定非连续置换 `[7,2,11,0,5,9,1,8,3,10,4,6]`（12 块 × 784 = 9408 token 容量） |
| GQA 映射 | q head `h` → kv head `h // (H/KVH)` = `h // 6`；**逐 head 反推实测确认**（逐 head 参考误差 0.0015~0.0024，其余 kv head 组合误差 0.77~1.27） |
| 容差 | `atol=1.5e-2 / rtol=1e-2`（上游 `tests/kernels/attention/test_flash_attn.py` BF16 默认，checkout 已核读） |

**候选路径合规性**：完整 KV 保持常驻，只构造单独的读取表与有效长度；**没有**把被选 KV 复制成另一份连续
缓存来冒充分页读取（gather 只出现在独立参考 oracle 里）。有效读取表中**不含 `-1`**；B=2 时较短行用其自身
最后一块补齐宽度，后端只按 `seqused_k` 读取。

## 2. 逐用例结论（12 用例 × 种子 0/1/2 = 36 case-runs，全部通过）

| 用例 | 关键字段（seed 0） | max_abs | rms | 相对 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| full_global | seqused_k 2587、尾 235、表 `[7,2,11,0]` | 0.0306 | 0.00701 | 0.194% | 支持（基准误差，含最大历史长度） |
| focus_multi_interval | 保留块 `[0,1,4,5]`→表 `[7,2,5,9]`、seqused 3136 | 0.0312 | 0.00589 | 0.224% | 支持（非相邻两段，跳过中间块） |
| focus_unordered_duplicate_cpu_norm | 输入 `[5,4,1,0,4,0]` → 与上同表同值 | 0.0312 | 0.00589 | 0.224% | 支持（CPU 规范化：去重 + 逻辑升序） |
| local_excludes_document | 保留 `[0,6,7]`→表 `[7,1,8]`、seqused 2352 | 0.0295 | 0.00636 | 0.217% | 支持；**与 full 输出 max_abs 17.38**（≫10×atol）→ 读取选择确实生效 |
| tail_1 | kv_len 1569、有效 `[784,784,1]`、尾 1 | 0.0308 | 0.00454 | 0.180% | 支持（尾块仅 1 token；其后未写槽为哨兵 8.0） |
| tail_b_minus_1 | 有效 `[784,784,783]`、尾 783 | 0.0608 | 0.00712 | 0.372% | 支持（最大绝对偏差；由 `rtol·\|ref\|` 通过，见 §4.4） |
| tail_b | 有效 `[784,784,784]`、尾 784 | 0.0305 | 0.00701 | 0.193% | 支持 |
| append_cross_block | 1568 → 追加 1 token 跨入块 2；**追加后**读取 seqused 1569、尾 1 | 0.0160 | — | — | 支持（只写 slot 8624 = `phys[2]*784+0`；旧 KV 未动） |
| original_write_position | local 视图下逐步追加 5 次；写入 slot 2348→2351→8624 全部等于原逻辑位置 | 0.0159 | — | — | 支持（每步只改 1 个指定 slot；追加后读取 785/尾 1 亦通过） |
| global_recovery | focus/local 后恢复完整表 | 0.0310 | 0.00596 | 0.234% | 支持（KV 内容哈希与 local 时相同） |
| batch2_rows | 行0 `[7,2,11,0]`/2587、行1 `[7,1,8]`/2352（不同模式/长度/物理 ID） | 0.0309 / 0.0296 | 0.00650 / 0.00591 | 0.195% / 0.205% | 支持（两行各自对单行参考一致；**行间隔离误差 0.0**，无串表） |
| block16_control | 块大小 16、retained `[0,2]`、seqused 32 | 0.0000 | 0.0000 | 0.000% | 支持但**退化**（见 §4.3），不替代 784 主用例 |

**跨种子**：33 条可量化记录全部落在预冻结容差内；最大相对偏差 0.372%（tail_b_minus_1），其余 ≤ 0.23%。
**参考自检**：批量参考与逐 head 参考逐用例一致（max_abs = 0.0）——这是发现并修掉参考自身 GQA 收缩 bug 的守卫（§4.1）。
**显存/耗时**：峰值 37.8~38.4 MiB；逐用例 0.37~0.73 s（仅执行诊断，非性能结论）。

**负对照（诊断，非通过条件）**：同一读取表把 `seqused_k` 人为 +100（读入尾块哨兵），
输出相对参考的 max_abs 从 0.031 升到 0.20~0.33 → 本套检查**能抓到越读**，且证明后端确实不读 `seqused_k`
之后的槽（正确调用下尾块与大段未写槽不影响结果）。

## 3. 必须解释的字段（逐用例已落盘）

`evidence/p1-gpu/cases-seed{0,1,2}.json` 与 `summary.json` 逐条含：

1. 原始有效 KV 长度（`kv_len`）与逻辑→物理映射（配置冻结 `logical_to_physical`）；
2. 模式/原始保留块集合、规范化后集合、块对齐后的物理读取表；
3. 读取表**有效块数/宽度**（`width`，无 `-1`、无虚填充）、**后端实际 `seqused_k`**、每块有效计数与**尾块有效 token 数**（`tail_len`）、被丢弃的空块（`dropped_empty_blocks`）；
4. **下一 token 的 canonical 写入位置与物理 slot**（`write_slot`），并实测"只改该 slot"；
5. 误差指标（`max_abs`/`rms`/`max_abs_ref`/`relative_max`）、峰值显存、耗时。

缓存内容指纹在只读用例前后一致（`cache_unchanged_by_read = True`）；追加用例按设计改变指定 slot，
以 `append_wrote_only_designated_slot` 与 `all_writes_used_original_positions` 记录。

## 4. 已知限制与如实记录的问题

1. **参考实现自身出过一个 bug（已修）**：批量参考用 `einsum("hsk,shd->hd")` 收缩时把 KV head 维隐式求和，
   导致与 kernel 差 2.12；用"逐 head 逐 kv head"独立写法与逐 head 反推定位并修正为显式 `[S,H,D]` 展开。
   现在批量参考与逐 head 参考逐用例一致（max_abs = 0.0），该自检保留在每次运行中。**这不影响任何用例结论**，
   但说明参考也必须被验证。
2. **反向也验证过**：修 bug 前，分页 FA2 与非分页 varlen FA2 的输出**逐元素完全相同**（0.0），
   说明偏差来自参考而非调用方式；这同时确认了 `block_table` + `seqused_k` 的读取与"按序拼接块"的语义一致。
3. **16 对照用例退化**：`DOCUMENT_BLOCKS` 被放大 ×5，短序列下 softmax 近似 one-hot，误差为 0
   （`max_abs_ref` = 8.0 = 哨兵量级，说明输出几乎等于某个 V 向量）。它只证明该路径能执行并一致，
   数值敏感度不足；主用例仍是 784。
4. **最大绝对偏差来自 `rtol` 而非 `atol`**：`tail_b_minus_1` 的 0.0608 > atol 0.015，靠 `rtol·|ref|`
   （被放大块 |ref| ~16）通过。这是预冻结容差对的正常行为，**未放宽**；若后续阶段要提高敏感度，
   应改用未放大的数值分布，而不是改容差。
5. **"保留块无有效 token 会被丢弃"是实现选择**：`dropped_empty_blocks` 如实上报，但"模式引用了尚未写入的块"
   时应报错还是忽略，属于协议/运行时契约问题，留给本地判断（本阶段不擅自定为契约）。
6. **只覆盖 query_len=1 的因果 decode**：本阶段的"有效长度 = 各保留块有效数之和、尾块取有效尾长"关系
   不外推到多 query / chunked prefill；RoPE 与位置编号未参与（原始逻辑位置保持不变，本阶段不做位置变换）。
7. 未涉及：真实模型 logits/自由生成、GDN/线性注意力层、prefix caching、抢占、图模式、HTTP/API、
   请求生命周期、并发与性能矩阵、质量与净收益。

## 5. 命令与复现

```bash
source /root/attnview/env.sh
python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check.json   # exit 0，36 case-runs / 0 failures
python3 tools/p1gpu-overread-control.py                                               # exit 0，越读负对照
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -t tests                # 106 项 CPU（含 18 项 gpukv）
```

## 6. 推荐下一步

1. 单请求 runner 接入（下一阶段工作单）：明确 AsyncOutput 的 D2H 同步点、需禁用的功能（图模式/prefix caching/抢占）
   与同步代价，用真实引擎复跑本阶段的最小用例集。
2. 模型级验证：在真实权重下对 64K/128K 做 logits/短生成对照（含 RoPE 与位置处理），再谈质量与收益。
3. 若要提高数值敏感度，改数值分布（不放大、加长序列）而不是动容差；并把本阶段 784 用例纳入回归入口。
