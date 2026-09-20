# masked 数值合同草案（SUP-004 masked-prep，仅设计/CPU）

状态：**草案，供本地审定**。不构成 masked GPU 授权，不含冻结阈值。基线 HEAD `dada813`；pin `98dff2a8`；模型 revision `1d4bf0f2…c60c0`。

## 1. 三类比较必须分清（不能互相替代）

| # | 比较对象 | 判据 | 说明 |
| --- | --- | --- | --- |
| A | **global 退化路径** vs 原版 | 同 prompt、同消费 prefix、同配置下逐 token 与逐张量比较 | 已由四臂完成：原版两轮 136 对、候选 vs 原版 272 对张量**逐元素相同**（max_abs=0）|
| B | **masked attention**（候选本步真实 Q + canonical K/V + **独立可见集合**）vs **dense masked 参考** | 逐元素绝对误差 / RMS / 相对 L2 | 这是 masked 的**主判据**：参考必须自己按原始 span 与位置算出可见集合，**不得**调用候选的筛选 helper 或压缩块表 |
| C | masked **端到端 logits** vs global 原版 logits | **不可**判为实现错 | 可见集合不同 ⇒ 数值本就不同；只有当需要验证"全模型等价"时才另立独立参考执行 |

**若**要校验 C（masked 端到端 logits），最小可行方案必须说明**独立参考执行**如何在同一固定 token 轨迹上传播：
逐层隐藏状态由 mask 后的 attention 输出驱动、**GDN/Mamba 状态**按 canonical 写路径推进（不因 mask 改变写槽与原位置）。
- 最小方案：在现有 harness 内、**同一进程**用同一权重实现"参考前向"（mask 由独立代码按 span 计算、通过一个显式注入点替换 FA 组的可见集合），
  逐层比对 hidden/logits；两臂共享 token 轨迹与 GDN 更新顺序。
- 补丁面：仅在现有 `da_fa_override` 通道旁加一条**显式参考注入**（不改已验收 runtime 语义、默认关闭）。
- 成本：与一次前向同量级 ×（层数×步数）的 CPU/GPU 记录量；**局部单层检查不能冒充全模型等价**。
- **C 属于既有计划的一部分**（contract §5.2 的 P2 已包含必要的 logits 验证），**不是**"用户另立项才考虑"的事项；本草案只给**最小独立参考执行**方案，不实现框架。
- **最小独立参考执行（建议）**：在测试路径上，用**独立 span 显式构造 mask** 的 dense attention **替换 FA 输出**（该替换代码自带可见集合计算，
  不复用候选的 `da_fa_override` 块表转换/任一候选筛选 helper —— **仅复用其块表转换不构成独立 reference**）；
  驱动一次**独立请求**、在**重置 GDN 状态**后按**同一权重与同一更新顺序**重放相同 token 轨迹，逐层比对 hidden 与最终 logits。
- **口径区分**：**层局部**（选定 q/位置上的 QKV→attention 输出）与**端到端**（各层传播 + GDN 状态 → logits）是两类不同结论，
  层局部的符合**不能**推出端到端上界；在**尚无 masked 数据**时，**不得**以 global 输出相等去推 masked 容差。

## 2. 容差依据（先于 masked 结果；只补少量代表点）

- 复用现有全局捕获（`run-original-3a/capture/layers.npz`，SHA256 `e60ec7d1…437008`）与首份 oracle 报告（`oracle-original-3a.json`，SHA256 `3615eaa6…110225`）；
  **不重跑** 24192 完整比较。补点方式：**每层至少一个 decode 代表点** + 已知最坏 prefill 点（层 14/13/10 的 max 点）。
- 明确公式（草案沿用并补全）：
  - `max_abs_err = max_i |out_i − ref_i|`
  - `rms_err = sqrt(mean_i (out_i − ref_i)²)`
  - `rel_l2_out = ‖out − ref‖₂ / ‖out‖₂`，`rel_l2_ref = ‖out − ref‖₂ / ‖ref‖₂`
  - 参考幅度：`|ref|` 的 p50/p99、**接近零占比**（如 `|ref| ≤ 1e-3` 的比例）——近零处绝对误差天然小、相对误差会放大，必须并列报告。
  - **既有报告的 `rel_err = max_abs_err / L2(out)` 不是 allclose 的逐元素 rtol**，草案不再以它当逐元素判据。
- **上游可用性核对（已完成，只读侦察）**：结论 **不能直接用作**本项目"BF16 模型层 vs 独立 FP32 dense"的容差依据，**只能作量级参考**。证据：
  - 唯一明确的"BF16 FA 输出 vs **独立 FP32 dense 参考**"是 `vllm/tests/v1/attention/test_mm_prefix.py`（`DTYPE=bf16` L67，`_dense_reference` 全 FP32 L204-255），
    断言 **atol=rtol=2e-2**（L372/L398/L417/L633），并含 mask 有效性反证（L393-395/L637-640）。
  - kernel 级唯一系统测试 `tests/kernels/attention/test_flash_attn.py`：paged 参考 `ref_paged_attn`（L41-92，仅 scores/softmax 用 FP32），
    **bf16 用 atol=1.5e-2 / rtol=1e-2**（fp8 放宽到 1.5e-1/1.5e-1）。
  - backend-impl 级 `tests/v1/attention/test_attention_backends.py`：默认 **atol=rtol=1e-2**（L386-387），fp8 KV 6e-2/1e-1；mask 参考是同 dtype flex_attention，**不做 FP32 upcast**。
  - 默认表 `tests/kernels/allclose_default.py`：bf16 **atol=1e-3 / rtol=1.6e-2**；生产实现 `vllm/v1/attention/backends/flash_attn.py` **全文件无数值断言**、无 FP32 upcast、`supported_dtypes=[fp16,bf16]`（L82）。
  - **模型层（logits）没有逐元素容差**：`tests/models/utils.py` 的 `check_logprobs_close` 只做 **top-N 成员性**（L241-244），PPL 只有单向 `PPL_TOL=0.01`。
  - 与 masked 直接相关的上游事实：`mask_mod` **仅 FA4 支持**（`vllm_flash_attn/flash_attn_interface.py:313-314`，FA2 报 `NotImplementedError`）⇒ 本项目的 masked 只能经**块表/`seqused_k`** 表达，不能依赖 FA2 的 `mask_mod`（与我们的落位方式一致）。
  - 缺口：上游没有"BF16 模型层 logits vs 独立 FP32 dense"的逐元素先例；因此**我们的 logits 容差必须由自带有界校准形成**，不得直接搬用 2e-2/1e-2。
- **代表点实测（已完成：**80 次选点 / 74 个唯一 (scope,layer,step,position)**；6 个 decode 点重复：L3s2、L5s1、L7s1、L8s2、L12s1、L13s3
  —— 因"每层最坏步"与"每层 step1–3"选择集重叠；原始 80 条保留，去重派生统计见 `evidence/p3-calib/masked-prep/scale-metrics-dedup.json`）**：
  | scope | n | max_abs_err (max / median) | rms_err (max) | **rel_l2_out (max)** | rel_l2_ref (max) | 近零占比 (max) |
  | --- | --- | --- | --- | --- | --- | --- |
  | prefill | 16 | 0.260048 / 0.072150 | 0.010253 | **0.00213574** | 0.00213571 | 0.0055 |
  | decode | 64 | 0.117996 / 0.014357 | 0.008497 | **0.00174387** | 0.00174393 | 0.0062 |
  全部 finite（`non_finite = 0`）；报告含 `definitions` 公式与 `rel_err_is_allclose_rtol=false`；**无任何阈值判定**。
  **去重后（74 唯一）**：prefill `rel_l2_out` max **2.13574e-3**；decode `rel_l2_out` max **1.74387e-3**（中位见 dedup 文件）。
  本表数值是 **global 量级诊断**，**不是** masked 阈值依据。
  **可提的候选（仅建议、未冻结、供用户决定）**：以实测最大值为参照，`rel_l2_out` 的候选量级约 **2e-3**（来自 **global** 对照）；
  上游同类先例（`test_mm_prefix.py`，bf16 vs 独立 FP32 dense）用 **atol=rtol=2e-2**，属**逐元素 allclose**口径、与本表的 `rel_l2` **不同量纲**，不可直接换算。
  **但**该 2e-3 是"global vs 原版/FP32 参考"的观测，**不能**直接充当 masked 的容差；masked 需自己的有界校准（见下）。
- **本轮不给数值候选区间**（更正前一版）：此前的 `out_l2 ≈ 5` 无来源，且 `1e-3~3e-3` 无依据 —— 撤回。
  已有事实（来自 `oracle-original-3a.json` 的 decode 逐比较 `out_norm`）：**中位 46.6、最大 401.7**（对应上表 `ref_l2` 量级），说明输出幅度分位跨度很大，
  不能用一个量级去反推容差。**候选区间必须等"代表点结果 + 上游断言方式核对"完成后再提**，且届时仍只作**建议**交用户决定；
  **不得**把 0.118/0.260 乘安全系数当数学保证，**不得**在看到 masked 结果后回调，**也不得**用 global 输出相等去推 masked 容差。
- 需要的**有界校准**（另行批准，本草案不自行执行）：在固定的、**真正排除已写块**的短轨迹上，对 B 类比较补
  16 层 ×（1 prefill 末 token + 3 decode）个代表点，报逐元素绝对误差、RMS、相对 L2、参考幅度分位与近零占比。

## 3. 正确性轨迹与独立性

- 条件沿用：≤8192、BF16、TP1、FA2（显式）、eager、同步、单活跃请求、关 prefix/投机/图。
- **轨迹与位置的具体方案（已撤回前一版"排除已写块"结论，更正如下）**：
  - **撤回**：前一版称 `focus=[362,431)` 可确证排除块 1、以及"local 延长 ≥64 token 可排除块 2" —— **不成立**。
    依据实现合同：`src/attnview/gpuoracle.py:76-78` 与 `src/attnview/readview.py:147-150` —— **focus 与 local 的可见集合都包含
    sink ∪ local_window ∪ 整段 response**，focus 只在此外**追加 refs**；因此
    ① 现 `local_window=[440,1505)` 已覆盖块 0、1，focus 加 refs 也覆盖同一区域 ⇒ **不排除块 1**；
    ② 生成到 1568 之后，块 2 属 **response** 段、**必须可见** ⇒ 延长 64 步也**制造不出**排除；
    ③ 位置 1505 时"当前尾块"本就是块 1 ⇒ `{块0, 当前尾块}` 亦不可能排除块 1。
  - 另更正：`[362,431)` 在块内结束 **≠** 跨越 KV 尾块边界（两者是不同事实，前一版混用）。
- **更正后的设计（已完成，CPU 实测；脚本与配置先提交后运行）**：
  - 复用阶段 03 **现成 da fixture**（`evidence/p1-cpu/demo-fixtures.json`）与**真实 renderer** 重渲染（不新造框架、不改分段）：
    实测 `prompt_len = 7280`、`segment_spans = [[362,2316),[2365,4255),[4304,6202)]`、`local_window_span = [6211,7280)`、`sink_span = [0,16)`，
    与工作单给出的事实**逐项一致**（`facts_match_order` 全 True）；token 哈希记于 `evidence/p3-calib/masked-prep/sets.json`。
  - 按 **b = 784** 独立展开（脚本自行按协议 span 展开，不调用候选筛选实现）：
    | 步 | 已写块 | global 可见 | local 可见 | focus1 可见 | local 排除已写块 | focus1 排除已写块 |
    | --- | --- | --- | --- | --- | --- | --- |
    | decode_step0 (KV 7280) | 0..9 | 0..9 | **{0,7,8,9}** | **{0,1,2,7,8,9}** | {1..6} | **{3,4,5,6}** |
    | decode_step1 (KV 7281) | 0..9 | 0..9 | {0,7,8,9} | {0,1,2,7,8,9} | {1..6} | {3,4,5,6} |
  - **结论**：`local` 与 `focus1` 的可见集合**确实整体排除了未选上下文的完整块 3..6**（且这些块都是**已写**块）——
    这就是"排除已写块"的可核验实例；`global` 步无排除（对照）。
  - **边界分列**（不混用）：段边界是 **token 粒度**（如段 1 结束于 2316）；canonical 有效尾块边界是**块粒度**（`(b+1)*784`，见 `tail_block_boundaries_token`）；
    "段在块内结束"与"跨越 KV 尾块边界"是两件事，脚本分别输出。
  - **下一块起点 7840**：需写满 561 个 decode 步才自然跨入块 10；若要让**少量生成步**即跨界，可**调整公开合成 context 长度**使 prompt 末尾接近块边界
    （总仍 ≤8192）——普通夹具选择由执行者与本地 review 处理，**不向用户抛不成立的选项**。
- **跨 token 声明**：把 `<focus>`/`</focus>` 放在会被 tokenizer 切分/跨块边界的文本处（用阶段 03 的 `_tokenize_with_offsets` 定位），并在轨迹中记录 parse_step/effect_step/applied_step。
- **独立性**：预期可见集合由**原始 span 与位置**独立算出（脚本自带，不 import 候选 helper、不读压缩块表）；
  保留 canonical 写 slot、原位置与 GDN 路径证据（写侧不得因 mask 改变）。
- **限制**：强制轨迹只用于数值对照，**不代表**模型自主声明遵从率；后者留后续工作单。

## 4. 失败检出能力（负对照）

- 数值负对照（安全、不越界）：
  1. **读错一个已分配块**：在可见集合中把某个已分配逻辑块替换为另一个已分配块（不引入越界块号），比对 B 类参考；
  2. **错有效尾长**：把 `seqused_k` 改为「正确值 ± 一个块内偏移」，仍在已分配范围内。
- 结构性判据（独立于数值）：可见块集合、`seqused_k`、`max_seq_len`、写槽与原位置、GDN 路径逐项与独立计算值比对。
- **结构门禁是无条件的、精确的**（不依赖任何数值容差）：可见块集合、`seqused_k`、`max_seq_len`、写槽/原位置、GDN 路径逐项与独立计算值
  **必须完全一致**，不一致即拒绝 —— 这是错块/错尾长的**第一道也是决定性**判据。
- **不得假设数值影响天然大于 BF16 误差**（更正前一版）：低 attention 权重、重复 KV 等情形下，读错块/错尾长可能只造成很小的数值差，
  因而**数值负对照不能当作检出手段的唯一依据**；若要使用数值负对照，须**从敏感样本中单独挑选**（例如 attention 权重集中处、跨块边界的尾块）
  并**实测**其误差量级后才可主张可检出性。若某负对照与正常误差不可区分 ⇒ **不得**以"结果差不多"接受，需改判据或补有界校准。
- 本草案只用现有 CPU 原始张量做量级分析；**任何新 GPU 或真正 masked 运行须另发工作单**。

## 实施边界与下一步

- 新增分析只落 `evidence/p3-calib/masked-prep/`（目录名不代表并发 P3）；记录 HEAD、输入指纹、命令与派生指标；**先提交脚本/配置再运行**。
- 不改已验收 runtime、不部署、不加载模型、不跑 GPU、不追改旧失败件；不为报告复跑四臂或 308 项。
- 待批准项：①是否实施 §1 的 B 类比较与 §4 负对照；②§1 的**最小独立参考执行**（C 属既有 P2 计划内，需批准的是"是否按该最小方案实施"及其成本）；③§2 的候选区间是否作为 masked 判定基准（**正式阈值仍由用户决定**）。
