# 阶段 03 报告：M1 接入设计与 CPU 协议准备（含 R1 返工）

- 日期：2026-09-18（远端 CPU 会话；未启动 GPU、未接 HTTP、未加载权重）
- 执行者：远端执行 Agent；素材快照 commit `0a4c392a2bc989f80f98057853abed31e7044f1c`（38/38 文件哈希校验通过）
- 初交 commit：`cb7bce19de6f2b71bf0fcf03b9b4a432a88d035a`（被 R1 判定 REWORK）；返工 commit 见 `outbox/SUP-002-R1-result.md`
- 状态：**完成（含 R1 返工）**，等待本地复核；无阻塞项，语义待决项见 §5

## 0. R1 返工：问题 → 改动 → 反例 → 结果 → 剩余限制

| # | R1 问题（初交实测缺陷） | 改动 | 反例 / 复现 | 修复后结果 | 剩余限制 |
| --- | --- | --- | --- | --- | --- |
| 1 | `extract.py` 用 `cleaned.strip()`，吞掉答案首尾空白 | 不再 strip；只做标签机械切分；`answer_empty` 只判空串 | `<answer>  x\n</answer>` 初交返回 `'x'` | 返回 `'  x\n'`（`test_leading_and_trailing_whitespace_is_preserved`）；patch 夹具改为保留末尾换行 | 空白仅"不裁剪"；是否需要规范化留待接口冻结 |
| 2 | 多/嵌套/残缺答案被当成功：`<answer>old</answer><answer>new` → ok=true 返回 old；嵌套把 `<answer>` 带进结果 | 计数开/闭标签：恰一对且顺序正确才提取；否则 `answer_ambiguous`（`opens>closes` 且只有 1 个开标签 → `answer_unterminated`），失败不回传原文 | 5 组反例见 `test_multiple_or_nested_or_stray_answers_are_ambiguous` | 5 组全部内部失败、无泄漏；**未**冻结 last-complete 语义（列为待决） | 多个答案区的"正确"语义仍未定（待决 §5.4） |
| 3 | `focus` 接受额外/重复属性；`<local>` 后 `</focus>` 会回 global | 属性名集合、重复属性、闭标签与当前模式对应全部校验；不匹配 → `mismatched_close` 且保持模式 | `<focus magic_chunks="1" extra="x">`、`<focus magic_chunks="1" magic_chunks="2">`、`<local>…</focus>` | 前两者分别 `unexpected_attribute`/`duplicate_attribute`；不匹配闭标签保持模式（`test_mismatched_closing_tag_is_an_anomaly_and_keeps_mode`） | 嵌套策略仍未定义（合同未规定），按"不静默扩大语法"处理并列为待决 |
| 4 | 未闭合 `<focus magic_chunks="` 的 `focus_attempts=0`（flush 异常漏出统计） | `ParseEvent.is_focus_attempt`；flush 事件进入统计与 trace；闭标签（含不匹配）不计分母 | `feed('<focus magic_chunks="'); finish()` 初交 attempts=0 | 现在 attempts=1/successes=0，且 flush 事件在 `state.flush_events` 与 trace 里（`test_incomplete_focus_buffer_counts_as_attempt`） | 词表中途截断成"foc"这类无法归属的残片不计入（已声明规则） |
| 5 | `state.py` 把已采样数当已写 KV 长度（off-by-one 语义错） | 重做字段：`written_kv_len = P + t`、"下一步写入位置 = P + t"、`attention_kv_len_next = P + t + 1`；trace 拆四列；覆盖 g0 / 首个闭合声明 / 尾块跨界 / **stop token 不再 forward** | prefill 采出 g0 时 KV 仍 P（`test_step_record_follows_the_timing_table`）；末 token 少一行（`test_stop_token_is_sampled_but_not_forwarded`） | 轨迹 115 行 = 114 次 forward + prefill；`attention_kv_len_last 7394`、`written_kv_len_last 7393` 逐行可核 | 与真实引擎的步对应仍需 GPU 侧时间戳验证（§2.2 路线 B） |
| 6 | `local_window_span` 末尾 7273 < prompt_len 7280，漏掉模板生成后缀 | local window 覆盖到真实 prompt 末尾 | 两臂现在都是 `[6211,7280]` / `[5763,5960]` | 断言 `end == prompt_len`（`test_local_window_reaches_prompt_end`）；不依赖 784 对齐补回来 | — |
| 7 | 两侧都超 cap 时被当成"无边界原子串"（`'a'*12000 + ' ' + 'b'*12000` → 单个 6001 token 段） | 每层边界只要有候选就切：优先左≤cap、其次右≤cap、否则**仍然切**并各自递归 | `test_both_sides_over_cap_still_cuts_at_coarse_boundary`、`test_multi_level_recursion_with_mixed_units` | 该输入 → 2 段（每段 3000/3000 token，按 4 字符/token 夹具）；混合多级用例无损 | 见 7c |
| 7b | 跨切割位置的 token 计数语义未声明 | 新增 `OffsetsIndex.straddling()`；**包含语义 + 跨边界 token = 全部 token**（不丢不重） | `test_crossing_tokens_are_accounted_and_reported`（1+2+1=4） | 不丢不重的等式成立 | 包含语义会**低估**跨边界 token（这正是 7c 要解决的） |
| 7c | **硬上限用包含语义判定 → 漏算**（监督/advisor 反例：`'ab cd ef'`、offsets `[(0,1),(1,4),(4,5),(5,8)]`、cap=2 时右段实覆盖 3 token 却报 2） | 分段全程改用**覆盖口径**（`OffsetsIndex.count_overlapping()`，与 `prompt.py` 生成最终 span 同一口径）参与上限判定与切点选择；**保持 C1.2 的层级顺序**（不做"token 边界优先"的额外轮次）；**空白层同时评估空白前/后两个切点**（BPE 会把空格并进相邻 token，只看一侧会错过唯一可用切点）；`_assert_cap_respected` 的**唯一豁免**是 C1.3 的"该单元在任何层级都没有可用边界" | `test_cap_holds_by_covering_count_on_adversarial_fixture`（该反例现在切成 `['ab ', 'cd', ' ef']` = 2/2/1，**全部 ≤cap**、精确复原、无豁免）、`test_boundary_cut_is_preferred_and_cap_holds`、`test_cap_exceedance_only_for_units_without_boundary`（每字符一个 token、cap=1 → `[3000, 1, 3000]`，只有两个无边界长串被标为 C1.3 超限） | 反例不再以"3"或"豁免"收场，而是给出满足上限的切法；**没有把漏计改称边界开销，也没有扩大 C1.3 的例外范围**（`cut_mid_token` 仅作元信息，不参与放行） | 若输入在**任何层级**都无可切边界（无空白长串一类），仍会保留超限段——这是 C1.3 允许的，且被独立标记与计数 |
| 8 | 设计文档声称"worker 内 CPU parser 不需同步、不必关异步调度" | **撤回**该结论；重写 §2/§6：画出 `AsyncOutput` 发起 D2H → `copy_event.synchronize()` → host 可用 → 解析 → 下一次 metadata 构造的依赖链；给 A/B/C 三条路线与代价 | `async_utils.py:115-163`（发起）/`:166-177`（同步后 tolist） | 现有结论：**存在不可避免的同步点**；路线 B（复用既有读回点）的成立与否**未证实**，必须用时间戳验证 | 代价与最终路线未定（R10） |
| 9 | 保真核对用字符多重集+7 句抽查，不能说逐字；`readview` 放行 `-2` 物理 ID；输入伪标签测试用桩对象 | 保真改为"有序正文逐字 + 表格行列单元格逐格 + 词多重集合"并声明能/不能证明；`readview` 拒绝负 ID、`to_kernel_args` 校验 count/width/行范围；伪标签测试改用**真实 tokenizer + 真实构造路径** | 逐格比较暴露并解释唯一差异（`ex-`+`tracted` 版式断词）；`test_negative_physical_id_in_mapping_is_rejected` | 保真 ①②③ 全通过（DA 2 表 9 行 27 格、Vanilla 1 表 5 行 15 格，de-hyphenation 1 处）；伪标签测试真实渲染后断言正文标签进了 prompt 且控制状态不变 | "真实引擎接线不会把输入喂进 parser"仍属接入测试（未验证） |

**R1 之外的补充（同一轮）**

- focus 统计已按论文口径输出**分子/分母 + 无尝试请求数**：`focus_attempts=2`、`focus_successes=2`、
  `legal_rate=1.0`、`requests_without_attempt=0`（`fixed-trace.json:focus_stats`）；`legal_rate` 在 attempts=0 时为 `null`（不可计算）。
- **sink 证据扩充**：两臂 16-token sink 与 **context / question / local window 均无交集**
  （`sink_intersects_context=false`、`sink_intersects_question=false`）；用 3 组不同输入（ASCII / 中文 / emoji）
  验证**前 32 token 逐 id 相同**（`fixed_prefix_consistent=true`）→ sink 区是输入无关的固定脚手架。
  字面偏差（DA 臂落在 tool 声明脚手架、两臂都不完全落在 system 指令正文）保留为事实，**未改模板/协议**。
- **UTF-8 增量解码**：`tokenizer.decode([id])` 逐 token 拼接在一般情形不安全（953/248077 个 token 单独解码含 U+FFFD）；
  已实现字节级增量解码器 `src/attnview/decode.py`，用 **pin 的固定反例** `(64253, 121)` 做测试：
  naive `��` vs 增量 `딽`（`test_pinned_split_pair_naive_vs_incremental`，不扫全词表）。
  **第二轮修复**：初版手写 `split_complete_utf8` 不校验续字节，反例字节 `e2 41 e2 82 ac` 会给出 `"�A���"`
  （错误前缀提前吃掉后面的合法尾段）；已改用标准库 `codecs.getincrementaldecoder("utf-8")(errors="replace")`，
  现为 `"�A€"`，并补两支测试：非法前缀不吃后续合法序列、结尾未完成序列在 flush 时按 replace 吐出。
- **素材与交付口径**：本轮**不再写**旧 `material/attnview` 快照（我早前写入的 `results/p1-cpu/` 已删除并在此说明）；
  代码、报告与证据都在实现仓内，原始素材保持可追溯。
  素材仓静态门禁**如实状态**：`bash material/attnview/scripts/verify.sh` 在远端 **FAIL**，
  输出 `FAIL 文件清单为空（仓库未初始化或全部被忽略）`——远端 `material/attnview` 是纯目录树（无 `.git`），
  该脚本用 `git ls-files` 取清单；**它不是本地素材仓的等价替代**，只在本地素材仓里有意义。
  我实际运行的是它的**分项检查**（JSON 语法、>5 MiB、密钥模式、Markdown 相对链接、必需文件清单），
  结果：新增文件全部通过；并发现远端旧树缺少本地已有文件 `docs/stage-03-protocol-read-view.md`
  与 `results/p0-model/preflight-model-facts.md`（我未在旧树补写，避免与素材仓所有权冲突）。

## 1. 交付物与测试入口

| 产物 | 路径 |
| --- | --- |
| 实现（CPU 协议层） | `src/attnview/{segmenter,prompts,prompt,parser,state,readview,reference,extract,trace,decode}.py` |
| 单测（**86 项**） | `tests/test_{segmenter,parser,readview,extract,state,prompt_template,decode}.py` |
| 测试入口 | `bash tools/p1cpu-run-tests.sh`（内部 `CUDA_VISIBLE_DEVICES=''`，不联网、不加载权重） |
| 演示与固定轨迹 | `python3 tools/p1cpu-demo.py`（真实 tokenizer，本地快照 `local_files_only=True`） |
| prompt 逐字保真 | `python3 tools/p1cpu-check-prompt-fidelity.py` |
| 证据索引 | 生成索引 `reports/evidence-index.md`（`evidence/p1-cpu/` 逐个文件 sha256；本阶段原机械表 `evidence/p1-cpu/evidence-index.md` 已被其取代、移出仓内发行，status=`superseded`） |
| **输入夹具（E2 要求"保留原文、无损分段、token IDs/offsets"）** | `evidence/p1-cpu/demo-fixtures.json`（完整原文 19217 字符、各段文本与字符区间、生成脚本、tokenizer/模板哈希）+ `evidence/p1-cpu/prompt-ids-offsets-{da,da_no_mask,vanilla}.json`（三臂各自的 `token_ids` 与 `offsets` 全量、segment/scaffold span） |
| 接入设计 | `reports/p1-cpu/integration-design.md`（含 M1 API 子集建议表） |

退出码：测试 `Ran 86 tests ... OK`（exit 0）；保真核对 `①②③ 全部通过`（exit 0）；演示 exit 0。原始输出 `evidence/p1-cpu/run.log`。

## 2. 结论要点（返工后）

### 2.1 分段与模板
3 段真实资料（1955/1891/1899 token）经模板渲染后映射为半开 token 区间，拼接无损；prompt 7280 token + 生成 115
= 7395 ≤ 8192（余量 797）；`da` 与 `da_no_mask` 渲染**逐字相同**；prompt 与论文摘录件**逐格/逐段一致**（§0 #9）；
thinking 必须直接传 `enable_thinking=False`（`chat_template_kwargs=` 被 transformers 5.17 静默忽略，三变体对照见证据）。

### 2.2 时序（合同 C3.5 的落地）
固定轨迹逐行显示：`t` 解析（`closed_at_token_index=t`）→ `effect_step=t+1`（消费该 token 的那次 forward）；
读清单在同一条轨迹里变化 6 次，写位置严格 +1（7280 → 7393，114 次 forward）；stop token（idx 114）被采样但无 forward 消费，
轨迹里**没有**对应行。**同步点不可避免**（§0 #8）：CPU 解析需要 D2H 就绪，路线与代价待 GPU 侧验证。

### 2.3 视图与独立参考
115 行轨迹（114 forward + prefill）与**独立逐位置参考**全部一致（`reference_mismatch_steps=[]`）；
34 个 step 存在被排除的**已写入**块（例：step 33 focus(1) 已写 10 块、排除 `[3,4,5,6]`，可见 `[0,1,2,7,8,9]`，声明位置全可见）；
I1/I2/I4/I5/I6/I8 均有单测，包含 `-2` 物理 ID 拒绝与 count/width 范围校验。

### 2.4 状态与输出隔离
两个 request_id 交错、释放后复用无残留（**对象级**；vLLM 混批/取消仍属 R07）；
公开提取只给最终答案区，代码/缩进保真，失败不回传原始流；同名保留标签碰撞按已接受限制记录。

## 3. 接入设计要点（`integration-design.md`）
运行期 V2 runner + FA2（安装包=`g98dff2a81` 源码同源）；视图落位：`build_attn_metadata`/`FlashAttentionMetadataBuilder.build`
之后，只改全注意力组对应行、写进每步复用的 read-only 块表缓冲；`-1` 必须翻译为后端接受的填充（建议 `NULL_BLOCK_ID=0`）；
GDN 组独立；首版不声称显存下降。**解析时序未定**，见 §0 #8。

## 4. 与阶段 02 基线的边界
未改 vLLM/kernel、未接 HTTP、未加载权重、未启动 GPU 计算、未下载第二候选、未跑矩阵；未运行 `verify-runtime.sh`；
未升级依赖；tokenizer 只从已落盘快照加载。

## 5. 待决项（需用户/本地判断）
1. **sink 与 C4.2 字面偏差**：接受"固定脚手架、无上下文"的实质满足，还是改 system 指令长度/模板（语义变更）？
2. **表格版式**：附录 F 表格是 `-layout` 双栏展开，我按列锚点把碎片拼回单元格；是否需要用 PDF 列提取重做基线？
3. **CJK 句末标点**是否纳入分段层级（合同 §C1.2 只列 `. ! ?` 与 `; : ,`；当前不扩展）。
4. **多答案区语义**（本轮改为内部失败，未冻结）、**嵌套**策略、**截断答案**的 HTTP 语义。
5. **协议额外 token 口径**（当前给"包装后 prompt 增量 ≈1320"与"标签字符占比"两种可算口径）。
6. `</global>` 与属性空白容忍度（合同未逐字列举，当前为已记录的项目决定）。
7. `-1` 填充约定、`finish_reason` 映射、`M1` 上下文/问题的请求封装字段。

## 6. 未验证（不得据此宣称）
真实引擎接入与接线、数值等价（R03）、CUDA Graph 兼容（R08）、混批/取消/抢占（R07）、prefix caching 交互、
热路径开销与最终解析路线（R10）、质量/收益（R05/R09）、`kernel_block_size` 的普遍性（784 仍只是该配置观测）。
本阶段产物均为 **CPU 数学/索引/协议**证据，**不是**实机支持声明；"阶段 03 全部验收通过"的初交表述**已撤回**。
