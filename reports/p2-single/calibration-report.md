# 单请求模型校准报告（SUP-004 calibration / R2 后）

- **执行 HEAD**：`757eddd`（含 R2/NATIVE-005/006/011 的驱动修正提交链：`07de025` → `c65d8e5` → `757eddd`）；pin vLLM `98dff2a8`；
  模型 `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（固定 revision）；GPU：RTX PRO 6000 Blackwell SM120 `(12,0)`。
- **真实生效配置**：BF16、TP1、**显式 `--attention-config.flash_attn_version=2`**、eager（图关）、同步调度、关 prefix 缓存、关投机/MTP、
  `max_model_len=8192`、`max_num_seqs=1`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、固定 seed、temperature 0。
- **输入**：阶段 03 `render_arm` 渲染的最终 prompt（协议 prompt），token 长度 1505；**直接提交该 token_ids**（不重新 tokenize）；
  每 run 的 prompt/token 哈希与 arm.json 记在该 run 的 `manifest.json`（含 `script_sha256` 与 `source/` 源码快照）。

## 1. 四臂主对照（同 prompt、同强制轨迹）

| 臂 | 目录 | exit | startup / 请求 | cleanup | token 轨迹 |
| --- | --- | --- | --- | --- | --- |
| `original` #1 | `evidence/p3-calib/run-original-3a` | 0 | 48.2 s / 2.83 s | ok | `[27,9461,29,7081,799,10642,11540,557]` |
| `original` #2 | `evidence/p3-calib/run-original-3b` | 0 | 48.4 s / 2.76 s | ok | **逐 token 相同**（`identical:true`） |
| `patched-disabled` | `evidence/p3-calib/run-disabled-5` | 0 | 49.2 s / 2.90 s | ok | **逐 token 相同** |
| `patched-global` | `evidence/p3-calib/run-global-5` | 0 | 49.5 s / 3.76 s | ok | **逐 token 相同**；**override 记录 = None（零受限覆写）** |

- 上限满足：启动 ≤15 min（实测 ≈49 s）、单请求 ≤180 s（实测 ≤3.76 s）。
- `patched` 两臂的 cleanup 验收按可观察效果判定：带载荷请求 B 的登记/解析/无受限计划/生命周期释放（registry/config/detok/pending）
  与真实取消均通过；`enforce_global` 的 mark 检查按**真实协议模式条件**判定（全程 global ⇒ `marks=[]` 正确；非 global ⇒ 必须有 mark）。
- 四臂均以 `--compare-to` 机械断言逐 token 相同；候选强制 token 与原始采样分列记录（`force.jsonl`）。

## 2. 重复性与等价性（数值，CPU 读取已保存张量）

| 比较 | 结论 | 证据 |
| --- | --- | --- |
| `original` #1 vs #2 | 8 logits + 16 FA 层输出 × 8 步 = **136 对张量**：全部有限、shape 一致、**逐元素相同（max_abs=0）** | 主代理独立复核 `original-repeat-tensors.json` |
| `patched-disabled`/`patched-global` vs `original` #1 | 共 **272 对张量**：全部有限、**逐元素相同（max_abs=0）** | 主代理独立复核 `patched-v4-main-tensors.json` |
| 四臂 capture 字节一致性 | 四臂 `capture/layers.npz` **字节完全相同**（含 Q/K/V/位置/输出/scale 元数据） | 主代理独立复核 `capture-byte-identity.json`（SHA256 `e60ec7d1…437008`） |

⇒ **主请求范围内，DA/global 路径引入的附加数值偏差为 0**（与关闭臂、原版逐元素相同）。

## 3. dense FP32 全局参考（独立 oracle）

- 输入：`run-original-3a/capture/layers.npz`（7.9 MB 报告 / 1323 MiB 捕获）；**同输入 hash 复用**：另两臂捕获字节相同，
  故**未**重复运行 oracle（不声称三次独立运行）。
- 覆盖：**24,192 comparisons = 24,080 prefill + 112 decode**；16 个全注意力层；**prefill 步 1 的全部 1505 个位置**
  （源码逐位置遍历，非仅末 token）+ **decode 步 1–7 全部**；`non_finite = 0`。
- 误差（FP32 dense 参考 vs 真实 FA2 输出）：**prefill 最大 `max_abs_err` 0.2600479126**（层 14；层 13 0.2295、层 10 0.1456）；
  **decode 最大 0.1179962158**；首个 decode 步 min 0.00414 / median 0.01490 / max 0.09901。
- **跨实现两样本核对**（主代理，NumPy float64，逐 head 重算两个最坏点）：0.2600652519 / 0.1179977043，
  与 FP32 报告差 1.73e-5 / 1.49e-6（证据 `oracle-worstcase-independent.json`）。**仅为两样本核对，不是全量独立 oracle。**
- **口径**（不越界）：报告字段 `rel_err` = `max_abs_err / L2(out)`，**不是**逐元素 `allclose` 的 rtol；
  `dtype_name=None` 不得读作"模型 dtype 未知"——模型 dtype 由 manifest 的 BF16 与逐层 `capture_dtype_L{n}` 给出来源。
  误差含 BF16 舍入与 FA2 与 dense FP32 的实现差异；**本轮不设容差、不套用阶段 04 阈值**。

## 4. 未验证 / 待后续

- 质量、净收益、性能均未测；masked 轨迹、自由生成、公共 HTTP、长上下文（>8192）、混批**未放行**。
- FA2 **生效值**以显式配置为门禁；运行期 `Using FlashAttention version 2` 的日志核对仍待补。
- canonical 块表与 FA metadata 的块号单位未独立核验（只作记录）；硬判据是 FA `seq_lens[0] == 本步最后位置 + 1`。
- 正式 masked 容差须后续**预冻结**（依据本报告与后续取样），不得直接沿用旧阈值。
