# 阶段 04 session closeout（GPU 读取视图数值验证）

日期：2026-09-18（实际 `date`）。用途：阶段切换/上下文压缩前的精简交接。
本文件**不是**验收，也不清掉未解决问题；compact 后首次继续必须重读本文件 + 最新 inbox/review + 新 SUP-004，回恢复 ACK 再动手。

## 1. 当前状态

| 项 | 值 |
| --- | --- |
| 验收 | **ACCEPT**：`inbox/SUP-003-review.md`（2026-09-18 10:12 CST），范围＝**被测 GPU 张量路径 + 阶段 03→04 数据面**；真实模型接入、位置/GDN、服务、质量与性能**未在范围内** |
| 当前 commit | `9fae77f`（报告数值口径更正）；实现基线 `38982cf`（＝交付运行 HEAD）；工作区除**未跟踪**的本地复跑目录外干净——该目录此后已移出仓内发行（`independent-review`），见 `reports/evidence-index.md` 的树外登记表 |
| 交付证据 | `evidence/p1-gpu-v6/`：`run-manifest.json`（HEAD `38982cf`、`code_clean=true`、起止 10:09:22→10:09:34、exit 0、972 判据/0 失败）、`control-manifest.json`、`summary.json`、`cases-seed{0,1,2}.json`、`run.log`、`negative-control.{json,log}`、`evidence-index.md`（9 件逐个 sha256） |
| 冻结配置 | `configs/p1-gpu/read-view-check-v3.json`（commit `b96d08a`）sha256 `3797326784d336030c8c5c59817b20d0010618a2904407668045dd36adbf96d1`；BF16/FA2/784/query_len=1/种子 0–2/atol 0.015、rtol 0.01 **未变** |
| 独立复跑（本地主代理） | `evidence/p1-gpu-local-review-20260918-1011/`——该目录已移出仓内发行（`independent-review`），原始字节归档在数据盘，路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表：`summary.json` sha256 `cc30f3de73c698e536f7e9855c30a7af44e8d2058abcffe914c1e9c482ec6301`、`run-manifest.json` `9f0264bdaf7124d74b76209ddfc0c6a9b2a0b80ad26b69db3c9c296712d35a7d`；CPU 111 项/4.435 s、GPU 972 判据/0 失败，数字与交付轮**逐位一致** |
| GPU | 已释放（0 MiB 已分配）；未加载任何模型 |

## 2. 本阶段实际完成范围

1. **夹具（真值）**：按逻辑位置生成 K/V 真值 → 按 canonical `l2p` 散布到物理缓存；**逐位置**核对 K 与 V（6272 位置），并核对有效尾之后、未承载已写逻辑块的物理块整段为哨兵；主用例普通分布，敏感负对照单独放大。
2. **常驻 GPU 轨迹**：一条轨迹只上传一次 K/V；入口断言 device+data_ptr 不变；每步读数前后核对 K 与 V 内容；每次追加**只改**指定 canonical slot（K 与 V 双侧一致）。
3. **数据面连通**：可见集合来自阶段 03 `TokenLayout/ViewInputs → build_read_view`，再由 `gpukv.read_table_from_read_view` 转成**无 `-1`** 读取表（`ReadTable` 带 `block_size`）。
4. **独立参考**：`gpuoracle.py` 用逐位置布尔 mask 独立推导可见集合、自行做块外扩与因果上界，从逻辑真值 gather 后 FP32 逐 head 计算；另用 SDPA 作第三实现自检。
5. **判据门禁**：`gpucheck.py` 每次测量产生判据并全部参与 PASS/exit——必需字段齐全、数值、oracle 自检、读取表无 `-1`/表宽、数据面 5 项、**K 与 V 两项内容判据（None/缺字段即失败）**、常驻实测、追加声明与逐槽（K/V 双侧，期望来自独立原位置公式）、**配置预期 == oracle == 候选三方一致**、行隔离、敏感度、**负对照必须被拒绝**。
6. **边界守卫**：任何 kernel 调用前校验 `表宽 ≥ ceil(seqused_k/b)`、物理 ID ∈ `[0, num_blocks)`、表块大小 == 缓存块维。

## 3. 已经运行的验证与原始产物

```bash
source /root/attnview/env.sh
bash tools/p1cpu-run-tests.sh                                                        # 111 项 CPU，exit 0
python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check-v3.json --evidence evidence/p1-gpu-v6   # exit 0
python3 tools/p1gpu-overread-control.py --config configs/p1-gpu/read-view-check-v3.json --evidence evidence/p1-gpu-v6  # exit 0
```

| 结果 | 值 |
| --- | --- |
| 计数口径（勿混用） | **30 case-runs**（10 配置 × 3 种子）、**57 条 step/row 数值测量**（784 表 54 + 16 对照 3）、**972 条判据**（判据数≠独立样本数） |
| 784 合并误差 | 最大 `max_abs = **0.0006069093942642212**`（trajectory seed2 step5）、最大 `rms = 8.738009922283544e-05`（seed2 step3） |
| 16 对照 | 最大 `max_abs = 0.0036176443099975586`（非 784，仅补充） |
| 敏感度 | masked vs full 最大差 15.04（门限 0.15） |
| 行隔离 | B=2 行0 4.9e-04、行1 0.0（容差内） |
| 负对照 | `tail_1` 越读 +100（1569→1669，仍在**已分配**尾块、表宽 3 足够）被门禁拒绝，max_abs 8.12；A/local 2 列表抬到需 3 列在 **kernel 调用前**被拒绝 |
| 资源 | 峰值 156 MiB（含验证用 clone），逐用例秒级——**非性能测量** |

## 4. 历史失败与本轮纠正（原始件全部保留、不删除、不改写；v1–v5 代次已移出仓内发行）

本节引用的 v1–v5 代次均已移出仓内发行（`superseded`）；原始字节归档在数据盘，路径/字节/sha256 见
`reports/evidence-index.md` 的树外登记表。

| 轮次/问题 | 事实 | 处置 |
| --- | --- | --- |
| v1（`476b8ef`） | 夹具按 `i` 写入、按 `l2p[i]` 读取 → 读到哨兵；非 GPU 常驻；参考共用候选算法；判据不进门禁；用例缺当前块；无真实模式切换 | 结论**撤回**；`evidence/p1-gpu/` 保留并加撤回横幅——已移出仓内发行（`superseded`），见 `reports/evidence-index.md` 的树外登记表 |
| 我的参考 bug | 批量参考 einsum 把 kv head 维隐式求和，与 kernel 差 2.12；用逐 head 反写与逐 head 反推定位 | 修复并保留"批量 vs 逐 head"自检 |
| v2（`1617164`） | `expect_blocks` 漏块 5（`align_outward(4700)`→块 5）且代码未读取该字段 | v3 修正并**纳入判据**（三方一致），未改 span |
| v3（首轮正确夹具） | **配置提交晚于运行**：`b96d08a` 创建 10:07:10，产物 10:06:49；我一度用"事后拆提交 + mtime"声称满足时序 | **承认错误**；该轮不作交付依据；改为运行自记 manifest 并重跑 |
| v4 / v5 | v4 运行时工作区含未提交改动（`code_clean=false`）；v5 主/负对照清单同名互相覆盖 | 均不作交付依据，保留为诊断——已移出仓内发行（`superseded`），见 `reports/evidence-index.md` 的树外登记表；清单分名后重跑 |
| 负对照设计 | 初版用 A/local（2 列表）越读 → 需第 3 列，属**越界注入** | 改为 `tail_1`（+100 仍在同一已分配尾块），另加边界拒绝检查 |
| `gpucheck` 提前 return | `append` 段缺失时吞掉后续判据（residency 等） | CPU 注入测试发现并修复 |
| 边界消息 | 曾用 dummy 张量块大小（16）而非表声明的（784） | `ReadTable` 带 `block_size` 并与缓存块维一致性断言 |
| 验收数值口径 | 报告文字曾把 seed 0 的 `tail_b=0.00058` 当成合并上界 | 已更正为 784 全种子合并 `0.0006069093942642212` 并写明三种计数口径 |

**已知标签问题（不改原始件）**：`evidence/p1-gpu-v6/control-manifest.json` 的 `command` 文本沿用了主脚本名，
其 `script` 字段才指向负对照；实际命令为 `python3 tools/p1gpu-overread-control.py --config configs/p1-gpu/read-view-check-v3.json --evidence evidence/p1-gpu-v6`。原始件不重写，本处如实注明。

## 5. 未验证 / 限制

- 真实模型接入：**未**加载 27B 权重、**未**接 runner/HTTP，模型 logits、自由生成、位置/RoPE、GDN/线性注意力层均未验证。
- 只覆盖 **query_len=1 且保留当前 token** 的因果 decode；前缀语义与"有效长度＝覆盖位置数"不外推到多 query / chunked prefill。
- 未涉及：图模式、prefix caching、抢占、混批生命周期、并发/吞吐/延迟矩阵、质量与净收益。
- `block16_control` 与 `arbitrary_subset_supplement` 是**非协议补充**（后者可见集合不来自 ReadView）。
- GPU 占用读数异常（0 MiB / util 100% / 无可见进程）未归因；未停他人作业，不宣称独占。
- 边界守卫只拦"表宽不足/物理 ID 越界/块大小不一致"；已分配块内部的哨兵越读由数值判据拦截。

## 6. 待决项（给本地/用户）

sink 16 token 字面解释的合同回填；多/嵌套/截断答案的 HTTP 语义与 `finish_reason` 映射；上下文/问题请求封装字段；
CJK 句末标点是否进 C1.2 层级；附录 F 表格基线；解析路线 A/B/C 与同步代价（R10）；`-1` 填充的最终约定；
设备侧 parser、图模式、prefix caching/抢占（均未放行）。

## 7. 下一步与授权边界

- **阶段 05（SUP-004）**：**先交精简接入设计**（解析同步点、需禁用功能、代价与回退），等本地 review **再**加载模型；
  未收到 SUP-004 前不自行开新阶段、不加载模型。
- **授权边界（保持）**：不改 vLLM/kernel、不升级依赖、不改计费/关机、不操作用户其他配置终端、控制链接与密钥不入交接、
  不对外发布；新实例无关机截止。
- **compact 后首次继续**：重读本 closeout + 最新 `inbox/`（review/新工作单）+ `protocol.md` → 回**恢复 ACK**
  （当前 commit、范围、第一步动作）→ 再按 SUP-004 执行。
- **恢复入口**：`source /root/attnview/env.sh` → `git -C /root/attnview log -1 --oneline` →
  `bash /root/attnview/tools/p1cpu-run-tests.sh`；GPU 入口与配置见 §3。
- 本阶段为同阶段返工，按指示**未** compact；由本地安排 compact。
