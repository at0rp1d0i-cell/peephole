# 阶段 03 session closeout（CPU 范围，ACCEPT 后）

日期：2026-09-18（远端 CPU 会话）。用途：阶段切换前的精简交接；本地随后 compact。
本文件不是验收，也不清掉未解决问题；compact 后首次继续必须重读本文件 + 最新 inbox/review + 下一阶段工作单。

## 1. 当前状态

| 项 | 值 |
| --- | --- |
| 当前 commit | `afd9069`（R2 收尾：补两支点名测试）；工作区干净（`git status --porcelain` = 0） |
| 本阶段提交链 | `cb7bce1`（初交）→ `17f7d35`（R1 主体）→ `7bef99b`（解码器/API 表）→ `af6d03e`（文档计数）→ `ae150a0`（advisor ① + overflow 分母 + 输入夹具）→ `54933ff`（advisor blocker：分段上限重做）→ `afd9069`（R2 点名测试） |
| 本地验收 | **SUP-002-review：ACCEPT**，范围是 `54933ff` 的**已核实 CPU 交付**（实现/参考/夹具/接入设计）；`afd9069` 只新增两支测试，不改行为 |
| 不代表 | M1 / P0 / P1 / P2 整体完成；真实引擎接入、GPU 数值、质量、性能与净收益仍未验证 |
| GPU/服务 | 未启动（`nvidia-smi` 0 MiB、无 `vllm` 进程）；未接 HTTP、未加载权重 |
| 素材 | 不再写旧 `material/attnview` 快照（本轮早前镜像已删）；代码/报告/证据都在实现仓 |

## 2. 实际完成范围（工作单 E0–E4）

- **实现**（`src/attnview/`，10 个模块）：`segmenter`（C1）、`prompts`/`prompt`（C2 + 最终 token-span 映射）、
  `parser`/`state`（C3 增量状态机 + 请求级状态与 C6 trace 字段）、`readview`（C4 纯函数视图 + I1–I8 校验）、
  `reference`（**独立**逐位置可见集合参考）、`extract`（§4.1 公开提取）、`trace`（逐 step 轨迹）、`decode`（字节级增量解码）。
- **测试**：`tests/` 7 个模块，**88 项**（无 skip）。
- **入口**：`bash tools/p1cpu-run-tests.sh`、`python3 tools/p1cpu-demo.py`、`python3 tools/p1cpu-check-prompt-fidelity.py`。
- **报告**：`reports/p1-cpu/stage-03-report.md`（含 R1/R2 逐项返工）、`reports/p1-cpu/integration-design.md`
  （调用链、接入点、M1 API 子集建议）。
- **证据**：`evidence/p1-cpu/`（**15 件**；逐个 sha256 见生成索引 `reports/evidence-index.md`，本阶段原机械表 `evidence/p1-cpu/evidence-index.md` 已移出仓内发行，status=`superseded`），其中
  `demo-fixtures.json`（完整原文 19,217 字符 + 各段完整文本）、`prompt-ids-offsets-{da,da_no_mask,vanilla}.json`
  （三臂全量 token ids/offsets）、`fixed-trace.json|md`（115 行 = 114 forward + prefill）、`prompt-facts.json`、
  `extraction.json`、`run.log`、`prompt-fidelity.txt`、`template-kwargs-check.txt`。

## 3. 已经运行的验证与原始产物

| 验证 | 结果 | 产物 |
| --- | --- | --- |
| 单元/行为测试 | 本地最近一次 `Ran 88 tests ... OK`（4.2 s，exit 0）；本地主代理在 `54933ff` 独立跑过 **86 项 / 4.605 s / exit 0** | `evidence/p1-cpu/run.log` |
| prompt 逐字保真 | ①有序正文逐字（DA 16 段 / Vanilla 5 段）②表格**行列单元格逐格**（DA 2 表 9 行 27 格、Vanilla 1 表 5 行 15 格）③词多重集合：全部通过（de-hyphenation 1 处） | `prompt-fidelity.txt` |
| 演示与固定轨迹 | exit 0：3 段 1955/1891/1899 token、prompt 7280/5960、114 次 forward、与独立参考逐位置一致、34 步存在被排除的已写块、focus 2/2 | `fixed-trace.json|md`、`demo-input.json` |
| 本地独立复核（主代理） | 读夹具重组原文逐字相等、文档 SHA256 匹配；重新加载固定 tokenizer 独立 tokenize 三臂：**全部 IDs/offsets 与归档一致**；local window 末尾均 = prompt_len；DA/no-mask 渲染字节相同 | 见 SUP-002-review 指纹表 |
| R1/R2 合成反例 | 正文首尾空白保留、残缺/多/嵌套答案内部失败、UTF-8 `e2 41 e2 82 ac` → `�A€`、不匹配闭标签/额外属性不切换模式、未闭合与超长 focus 均进分母、负物理 ID 被拒、分段反例 `ab cd ef` → `['ab ', 'cd', ' ef']` 覆盖计数 2/2/1 | 同上 + `tests/` |

## 4. R1 / R2 修复对应关系

| 审查项 | 要求 | 实现 | commit | 测试/证据 |
| --- | --- | --- | --- | --- |
| R1-1 | 答案正文首尾空白保真 | 去掉 `strip()`，`answer_empty` 只判空串 | `17f7d35` | `test_leading_and_trailing_whitespace_is_preserved` |
| R1-2 | 多/嵌套/残缺答案 → 内部失败、不泄漏 | 开/闭标签计数分类，`answer_ambiguous`/`answer_unterminated` | `17f7d35` | `test_multiple_or_nested_or_stray_answers_are_ambiguous` |
| R1-3 | 非法控制声明保持模式 | 属性名集合/重复属性/闭标签对应校验，`mismatched_close` | `17f7d35` | parser 6 组用例 |
| R1-4 | focus 分母含所有尝试 | `is_focus_attempt`；flush 与 **overflow** 分支都在清空前判定 | `17f7d35` + `ae150a0` | `test_incomplete_focus_buffer_counts_as_attempt`、`test_buffer_overflow_focus_attempt_is_counted`、`test_overflow_focus_attempt_reaches_state_stats` |
| R1-5 | 采样 vs 已写 KV 时序 | `written_kv_len=P+t`、`attention_kv_len_next=P+t+1`、trace 四列、stop token 不再 forward | `17f7d35` | `test_step_record_follows_the_timing_table`、`test_stop_token_is_sampled_but_not_forwarded` |
| R1-6 | scaffold 覆盖到 prompt 末尾 | local window = question → `prompt_len` | `17f7d35` | `test_local_window_reaches_prompt_end` |
| R1-7 | 分段递归与跨界记账 | 两侧超限仍按最粗边界递归；`straddling` 记账 | `17f7d35`、`54933ff` | segmenter 6 组用例 |
| R1-8 | 撤回"worker CPU parser 无需等待" | 设计文档 §2/§6 重写：D2H ready 依赖 + A/B/C 三路线与代价 | `17f7d35` | `integration-design.md` §2 |
| R1-9 | 保真不可降级；负 ID；伪标签测试 | 有序正文/逐格单元格/词多重集合；拒绝负物理 ID；真实 tokenizer 构造用例 | `17f7d35` | 保真脚本、`test_negative_physical_id_in_mapping_is_rejected`、`test_document_side_tags_stay_data_through_real_construction` |
| R2-a | 撤销"token 边界优先"外层优先级 | 严格按 C1.2 字符层级递归，不插入额外轮次 | `54933ff` | `test_coarse_boundary_through_token_is_not_demoted`（`afd9069`） |
| R2-b | 撤销 `cut_mid_token` 超限豁免 | 上限只用覆盖口径；唯一豁免是 C1.3 无可用切点的单元 | `54933ff` | `test_cap_holds_by_covering_count_on_adversarial_fixture`、`test_cap_verified_by_independent_overlap_count`（`afd9069`） |
| R2-c | 空白两侧都可作合法切点 | 空白层同时评估前/后切点（避免固定归属制造超限） | `54933ff` | 同上 |
| R2-d | overflow focus = 1 | 见 R1-4 | `ae150a0` | 同上 |
| R2-e | 完整原文/segment/IDs/offsets 落盘并纳入 manifest | `demo-fixtures.json` + `prompt-ids-offsets-*.json`，纳入生成索引 `reports/evidence-index.md` | `ae150a0` | 15 件索引 |
| R2-f | 收尾前读 inbox 未回执项 | 已读 `SUP-002-R1.md`、`SUP-002-R2.md`、`SUP-002-review.md`、`SUP-002-closeout.md` | — | 本文件 + outbox 回执 |

## 5. 已知失败 / 限制（照实列）

1. **naive 逐 token 解码不安全**：Qwen3.8-27B 词表 953/248,077 个 token 单独 `decode([id])` 含 U+FFFD → 必须走
   `decode.py` 的字节级增量解码（固定反例 `(64253,121)`：naive `��` / 增量 `딽`）。
2. **sink 的字面偏差**：两臂 16-token sink 与 context/question/local window **无交集**、3 组输入前 32 token 逐 id 相同，
   但**不完全**落在 6-token 的 system 指令正文内（DA 臂落在固定 tool 声明脚手架）。本地判定：不得补字/改模板/改掩码
   （见 SUP-002-review"接受边界"），仅保留事实与合同文字回填。
3. **分段上限的唯一豁免**：只有"该单元在任何层级都没有可用切点"（C1.3 无空白连续串一类）才允许超限，且被独立标记
   （`uncuttable_over_cap`）并计数；`cut_mid_token` 仅元信息、不参与放行。若某输入在既定规则下无法满足上限，
   当前实现会**抛错**（不静默豁免）——最小反例需按 R2 要求单独提交，尚未遇到。
4. **素材仓门禁**：远端旧 `material/attnview` 是纯目录树（无 `.git`），`scripts/verify.sh` 因此
   `FAIL 文件清单为空`；我只跑了它的分项检查（JSON 语法、>5 MiB、密钥模式、链接、必需文件），并未声称原入口通过。
   旧树缺少本地已有的 `docs/stage-03-protocol-read-view.md`、`results/p0-model/preflight-model-facts.md`（我未补写）。
5. **本阶段全部为 CPU 证据**：114 次 forward 与独立参考一致是**模拟**，不是 GPU 执行/计时证据；无任何性能、质量、收益结论。

## 6. 待决项（留给本地/用户）

sink 字面解释的合同回填；多/嵌套答案语义与截断答案的 HTTP 语义；`finish_reason` 与错误映射；
上下文/问题的请求封装字段（不得硬编码文档、不得退化为只读最后一句）；CJK 句末标点是否进 C1.2 层级；
附录 F 表格是否用 PDF 列提取重做基线；解析路线 A/B/C 与同步代价（R10）；`-1` 的最终填充约定；
设备侧 parser、图模式、prefix caching/抢占（均未放行）。

## 7. 下一步与当前授权

- **下一步工作单**：`/path/to/supervision/inbox/` 中**尚未发布的 SUP-003**（小张量 GPU 独立验证阶段）；
  阶段 04 仍待本地放行。未收到 SUP-003 前不自行开 GPU、不自行扩范围。
- **当前授权（用户）**：07:30 CST 前由本地逐阶段验收后可自主推进开发与 GPU 实验；**07:15 起优先收尾**；
  阶段 03 本身保持 CPU 边界。不改关机/计费设置，不操作用户另一个配置终端。
- **compact 后首次继续**：先重读本 closeout + 最新 `inbox/`（review/工作单）+ `protocol.md`，回一条恢复 ACK
  （当前 commit、范围、第一步动作），然后才按 SUP-003 启动（届时才涉及 GPU）。
- **恢复入口命令**：`source /root/attnview/env.sh` → `git -C /root/attnview log -1 --oneline` →
  `bash /root/attnview/tools/p1cpu-run-tests.sh`（仅需确认环境未漂移时再跑）。
- **安全边界**：控制链接与密钥不进交接；协调目录 `/path/to/supervision` 与实现仓分开。

## 8. 关键指纹（本文件写就时）

- 逐字保真：`demo-fixtures.json` = `97dd1a4834dac8c1b58b…`、`prompt-ids-offsets-da.json` = `3dd4feecc9c9738c046b…`
  （与 SUP-002-review 表一致），`prompt-ids-offsets-da_no_mask.json` = `fe235d66882168e62820…`、
  `prompt-ids-offsets-vanilla.json` = `5292ab4329448f4d2698…`
- 报告：`stage-03-report.md` = `e9fac04de22be3fa13ed…`、`integration-design.md` = `3bcac161fcdc8676d379…`
- 证据索引：生成索引 `reports/evidence-index.md`（`evidence/p1-cpu/` 15 件，逐个 sha256；原机械表已移出仓内发行，status=`superseded`）
