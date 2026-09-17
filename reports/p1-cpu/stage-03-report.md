# 阶段 03 报告：M1 接入设计与 CPU 协议准备

- 日期：2026-09-18（远端 CPU 会话；GPU 未参与，服务未启动）
- 执行者：远端执行 Agent；素材 commit `0a4c392a2bc989f80f98057853abed31e7044f1c`（38 文件逐条哈希校验通过）
- 执行 commit：见 `outbox/SUP-002-result.md`（本节写就后提交）
- 状态：**完成**（工作单 E0–E4 全部执行；无阻塞项；若干语义待决项见 §5）

## 1. 交付物与测试入口

| 产物 | 路径 |
| --- | --- |
| 实现（CPU 协议层） | `src/attnview/{segmenter,prompts,prompt,parser,state,readview,reference,extract,trace}.py` |
| 单测（61 项） | `tests/test_{segmenter,parser,readview,extract,state,prompt_template}.py` |
| 测试入口 | `bash tools/p1cpu-run-tests.sh`（内部 `CUDA_VISIBLE_DEVICES=''`，不联网、不加载权重） |
| 演示与轨迹生成 | `python3 tools/p1cpu-demo.py`（真实 tokenizer，本地快照 `local_files_only=True`） |
| prompt 逐字保真核对 | `python3 tools/p1cpu-check-prompt-fidelity.py` |
| 证据 | `evidence/p1-cpu/`（索引：`evidence-index.md`，含逐个文件 sha256） |
| 接入设计 | `reports/p1-cpu/integration-design.md`（含 M1 API 子集建议表） |

命令与退出码：`bash tools/p1cpu-run-tests.sh` → `Ran 61 tests ... OK`（exit 0）；
`python3 tools/p1cpu-demo.py` → exit 0；`python3 tools/p1cpu-check-prompt-fidelity.py` → exit 0。原始输出见 `evidence/p1-cpu/run.log`。

## 2. 逐项结论（对应工作单 §4 验收清单）

### 2.1 分段与模板

- **无损划分**：`join_segments(segments) == document`（**实测** `demo-input.json.lossless_join_ok = true`；单测含中文/emoji/重复片段/无空白原子串）。
- **边界**：2000/2560 → 单 segment；2561 → 2 段（单测，每词 1 token 的确定性夹具）；整篇 3×~1900 token → 3 段。
- **无空白长串原子**：实现细节与论文一致——首版实现只接受"左段 ≤ 上限"的切点，会让原子串与后文合并；
  已改为**允许"右段 ≤ 上限"的切点**，使原子超限串成为**它自己**的 segment（单测 `test_whitespace_free_run_is_atomic`）。
- **层级**：空行段落优先于单换行（单测 `test_coarsest_boundary_preferred_over_finer`）。
- **真实 tokenizer + 真实 spans**：3 段（1955/1891/1899 token）经 chat template 后映射为半开 token 区间；
  拼接复原；区间顺序单调不重叠（`prompt-facts.json`）。
- **三臂 prompt 来源一致**：`da` 与 `da_no_mask` 渲染结果**逐字相同**（`da_no_mask_same_prompt_as_da = true`）；
  `vanilla` 用 Vanilla Instruction Prompt + 内联 context，无 magic chunk（单测断言）。
- **prompt 逐字保真**：与论文摘录件（附录 F）做**非空白字符多重集**核对**完全一致**，7 段散文**严格子串**通过
  （`prompt-fidelity.txt`）。差异仅限表格版式重排（摘录件是 `pdftotext -layout` 的双栏展开），已记录为保真项。
- **sink 检查（合同 C4.2）**：**不满足字面要求，且已定位原因**——
  - 两臂前 16 token 均**不含**任何文档 token（`sink_overlaps_context = false`）；
  - 但两臂的 16 token 都**不完全**落在"fixed system instruction"正文上：该模板的 system 正文只有 **6 个 token**
    （`You are a helpful assistant.`），sink 必然延伸到角色包装 / 后续脚手架；
  - **DA 臂更明显**：模型的**原生 tool 声明格式**把 `# Tools … <tools>…</tools>` 排在 system 正文**之前**，
    因此 sink 落在**固定 tool 声明脚手架**上（`<|im_start|> system \n # Tools …`），vanilla 臂落在 system 正文+角色包装。
  - 处理：**未改 prompt、未补字**（工作单禁止）；实测事实写进单测 `test_sink_deviation_from_literal_contract`，
    列为**待决项**（§5）。
- **thinking 关闭**：`enable_thinking=False` 必须**直接**作为 kwargs 传入；传 `chat_template_kwargs={...}` 会被
  transformers 5.17 **静默忽略**（实测三变体对照：`template-kwargs-check.txt`）。修正后渲染中不再出现
  "Reasoning effort is set to xhigh"，生成提示为 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。
  这是 C2.6 的上机回填项之一（本阶段用 CPU + 真实 template 定下传参方式）。
- **长度预算**：demo 包装后 prompt 7280 token + 生成 115 token = 7395 ≤ 8192（余量 797），**在已测配置内**；
  `vanilla` 臂同文档 prompt 为 5960 token → DA 侧协议包装（tool 脚手架 + DA 指令）额外约 **1320 token**，
  是本阶段可量化的一条协议开销（不代表端到端成本结论）。

### 2.2 增量解析

- 标签在**每个字符边界**拆开喂入都只产生一次转移，且 `effect_step` 恰为包含 `>` 的那个片段索引 +1（单测遍历所有切点）。
- `>` 到达前不转移（单测 `test_no_premature_transition_before_gt`）。
- 同一 token 内多事件（`<global><local>` → noop + transition，effect 相同）✓。
- 重复/乱序引用 → 集合去重 + 稳定排序；越界引用 → **整条声明作废、保持当前模式** + `invalid_reference`（C3.6/C6.2）✓。
- 缺少属性 / 属性非法 / 未知标签 / 未闭合 / 缓冲超限 → 均有独立原因码，且**保持当前模式**、不偷偷回 global ✓。
- `<global>` / `</global>` / `<answer>` 不产生模式转移；`</focus>` / `</local>` 回 global ✓。
- 真实 tokenizer 分片下，脚本的 6 次转移顺序与 `effect_step` 全部符合预期（单测 `test_real_tokenization_splits_tags…`）。
- **输入伪标签不触发**：状态对象**没有**接收输入侧文本的接口；正文中出现同名标签字面串时模式不变、无异常事件（单测断言，并配对照证明正文里确有该串）。

### 2.3 视图（读/写分离与独立参考）

- **独立参考逐位置对照**：115 个生成步 + prefill 全部一致（`reference_mismatch_steps = []`）。
  参考实现不 import `readview`、不复用其合并/对齐工具，直接从三区域语义逐位置构造布尔 mask 再取块外扩。
- **读清单变化、写位置不变**：读清单在一条轨迹里变化 **6 次**；写位置 7280 → 7394 **严格 +1**；
  同一 `kv_len` 下读清单可不同（`fixed-trace.md` 的 33/43/68/77/79/94 行）。
- **t 解析、t+1 生效**：轨迹表把"本步输入 token 内闭合的事件 → effect_step"与"下步模式"并列。
  例：step 33 的输入 token（idx 32）闭合 `<focus magic_chunks="1">` → `effect_step = 33`，该行 visible blocks
  即刻变为 `[0,1,2,7,8,9]`；step 79 输入 token 闭合 `<local>` → 同一行 visible 变为 `[0,7,8,9]`。
- **784 下真实存在被排除的已写块**：34 个 step 有已写入块被排除，例：step 33（focus 1）已写 10 块、排除 `[3,4,5,6]`；
  step 68（focus 2,3）排除 `[1,2]`。**演示未改分段阈值或块大小来凑示例**。
- I1（声明位置一个不丢，`declared_positions_all_visible = true`）、I2（顺序无关）、I4（有效前缀无 `-1`）、
  I5（`max_width` 恒定）、I6（尾块有效长度）、I8（越界抛错）均有单测。

### 2.4 状态隔离与输出隔离

- 两个 request_id 交错喂入互不影响；释放后复用同一 id **无残留**（模式/引用/步数/异常全为空）；
  这是 **CPU 对象级**结论，**不代表** vLLM 混批/取消已支持（R07 仍开）。
- 公开提取只返回最终答案区：分析文本不外露；patch/代码缩进与空行逐字保留；普通词 `focus`/`answer` 与
  非协议标签（`<div>`/`<p>`）不被删除；缺失/截断答案返回内部失败码且**不回传原始流**、错误消息不含内部文本。
- **同名保留标签碰撞**按已接受限制记录（单测 `test_same_name_literal_collision_is_a_documented_limitation`），
  不冒充正文保真通过。

## 3. 接入设计要点（详见 `integration-design.md`）

- 运行期是 V2 runner（`v1/worker/gpu/model_runner.py`）+ FA2；安装包与源码 checkout 同源（`g98dff2a81`）。
- **推荐连接点**：worker 内 `postprocess_sampled`（`model_runner.py:1921`）之后更新本请求视图，
  下一次 `build_attn_metadata`（`attn_utils.py:247`）/`FlashAttentionMetadataBuilder.build`（`flash_attn.py:545`）
  只改全注意力组的对应行；**不必关闭异步调度**（默认开启，`config/vllm.py:1231-1311`）。
- 若改在 host 侧解析，则声明会晚一拍，必须二选一：对 DA 请求关闭异步调度，或每步强制 D2H 同步并计量损失——
  **不允许静默延迟生效冒充满足合同**。
- `-1` 是内部约定：FA2 路径必须翻译成后端接受的填充（建议 `NULL_BLOCK_ID=0`），并只靠 `seqused_k` 界定有效前缀。
- GDN（3×MambaSpec）与全注意力分页 KV 分离，DA 只作用于全注意力组；不释放/跳过块 → 首版不声称显存下降。

## 4. 与阶段 02 基线的边界

- 未改 vLLM/kernel、未接 HTTP、未加载权重、未启动 GPU 计算、未下载第二候选、未跑 no-mask 模型筛查或容量/性能矩阵。
- 未运行 `verify-runtime.sh`（会启动 CUDA）；所有 CPU 命令显式 `CUDA_VISIBLE_DEVICES=''`。
- 未升级任何依赖（只用标准库 + 既有 transformers/tokenizers）；tokenizer 只从已落盘快照加载。

## 5. 待决项（需要用户/本地判断，我不自行决定）

1. **sink 与合同 C4.2 的字面偏差**（§2.1）：是接受"固定脚手架、无文档内容"的实质满足，还是通过加长 system 指令 /
   自定义模板来让 16 token 完全落在 system 正文？后者会改 prompt 结构，属语义变更。
2. **表格版式重建**：附录 F 的表格在摘录件里是双栏展开，我按单元格逐字重排。是否需要以 PDF 列提取重新生成基线？
3. **CJK 句末标点**是否纳入分段层级：合同 §C1.2 只列 `. ! ?` 与 `; : ,`；当前实现**不扩展**，中文长文只能靠段落/空白层级切分。
4. **协议额外 token 口径**：本报告给出"包装后 prompt 增量"(≈1320) 与"标签字符占比"两种可算口径；正式指标用哪个待定。
5. **多个答案区的选择**：当前取**最后一个**完整 `<answer>`（论文要求以 `</answer>` 结尾）；是否改为报错待定。
6. **`</global>` 与 `<focus>` 属性空白的容错**：合同未逐字列举闭标签全表与空白容忍度；两项均为已记录的项目决定。
7. **`-1` 填充约定**、**`finish_reason`/HTTP 映射**、**截断答案的 HTTP 语义**（设计文档 §9 表中"待决"行）。

## 6. 未验证（不得据此宣称）

真实引擎接入、数值等价（R03）、CUDA Graph 兼容（R08）、混批/取消/抢占（R07）、prefix caching 交互、
热路径开销与是否必须关闭异步调度（R10）、任何质量或收益结论（R05/R09）、`kernel_block_size` 的普遍性
（784 仍只是该配置的观测）。本阶段产物均为 **CPU 数学/索引/协议**证据，**不是**实机支持声明。
