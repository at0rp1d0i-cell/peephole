# 单请求模型校准报告（SUP-004 calibration / R2 后）

- **执行 HEAD**：`757eddd`（含 R2/NATIVE-005/006/011 的驱动修正提交链：`07de025` → `c65d8e5` → `757eddd`）；pin vLLM `98dff2a8`；
  模型 `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（固定 revision）；GPU：RTX PRO 6000 Blackwell SM120 `(12,0)`。
- **真实生效配置**：BF16、TP1、**显式 `--attention-config.flash_attn_version=2`**、eager（图关）、同步调度、关 prefix 缓存、关投机/MTP、
  `max_model_len=8192`、`max_num_seqs=1`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、固定 seed、temperature 0。
- **输入**：阶段 03 `render_arm` 渲染的最终 prompt（协议 prompt），token 长度 1505；**直接提交该 token_ids**（不重新 tokenize）；
  每 run 的 prompt/token 哈希与 arm.json 记在该 run 的 `manifest.json`（含 `script_sha256` 与 `source/` 源码快照）。

## 1. 四臂主对照（同一最终 prompt、同一 token 轨迹；原版自然 greedy，候选 teacher forcing 回放）

| 臂 | 目录 | exit | startup / 请求 | cleanup | token 轨迹 |
| --- | --- | --- | --- | --- | --- |
| `original` #1 | `evidence/p3-calib/run-original-3a` | 0 | 48.2 s / 2.83 s | ok | `[27,9461,29,7081,799,10642,11540,557]` |
| `original` #2 | `evidence/p3-calib/run-original-3b` | 0 | 48.4 s / 2.76 s | ok | **逐 token 相同**（`identical:true`） |
| `patched-disabled` | `evidence/p3-calib/run-disabled-5` | 0 | 49.2 s / 2.90 s | ok | **逐 token 相同** |
| `patched-global` | `evidence/p3-calib/run-global-5` | 0 | 49.5 s / 3.76 s | ok | **逐 token 相同**；**override 记录 = None（零受限覆写）** |

- 上限满足：启动 ≤15 min（实测 ≈49 s）、单请求 ≤180 s（实测 ≤3.76 s）。
- `patched` 两臂的 cleanup 验收按可观察效果判定：带载荷请求 B 的登记/解析/无受限计划/生命周期释放（registry/config/detok/pending）
  与真实取消均通过；`enforce_global` 的 mark 检查按**真实协议模式条件**判定（全程 global ⇒ `marks=[]` 正确；非 global ⇒ 必须有 mark）。
- **轨迹来源与比对口径**：`original` 两轮走**自然 greedy**，其中**首轮**（`run-original-3a`）生成轨迹基线、**不带** `--compare-to`；
  第二轮（`run-original-3b`）才用 `--compare-to` 与首轮逐 token 比对。两候选臂走**测试专用 teacher forcing**
  （`--force-trajectory` 回放同一轨迹）并各自 `--compare-to`；候选的强制 token 与原始采样分列记录于 `force.jsonl`。
- **各 run 实际 HEAD**：`run-original-3a`/`3b` = `07de025`；`run-disabled-4`/`run-global-4` = `c65d8e5`；
  `run-disabled-5`/`run-global-5` = `757eddd`。运行期间各 run 的脚本哈希不变（起时快照见 `<run>/source/`，`manifest.script_sha256`）。

## 2. 重复性与等价性（数值，CPU 读取已保存张量）

| 比较 | 结论 | 证据 |
| --- | --- | --- |
| `original` #1 vs #2 | 8 logits + 16 FA 层输出 × 8 步 = **136 对张量**：全部有限、shape 一致、**逐元素相同（max_abs=0）** | 主代理独立复核 `original-repeat-tensors.json` |
| **v4** 候选（`run-disabled-4`/`run-global-4`）vs `original` #1 | 共 **272 对张量**：全部有限、**逐元素相同（max_abs=0）** | 主代理独立复核 `patched-v4-main-tensors.json`（SHA256 `887bae9b…a1b29756`）|
| **v5 最终** 候选（`run-disabled-5`/`run-global-5`）| 两 run **exit 0 / failures [] / cleanup true**；全层捕获与 8 logits 亦逐元素相同 | `final-v5-artifacts.json`（SHA256 `79e1a40f…84c4e9`）|
| 四臂 capture 字节一致性 | 四臂 `capture/layers.npz` **字节完全相同**（含 Q/K/V/位置/输出/scale 元数据） | 主代理独立复核 `capture-byte-identity.json`（SHA256 `e60ec7d1…437008`） |

⇒ **主请求范围内，DA/global 路径引入的附加数值偏差为 0**（与关闭臂、原版逐元素相同）。
- 本节引用的独立复核记录（`original-repeat-tensors.json`、`patched-v4-main-tensors.json`、`final-v5-artifacts.json`、`capture-byte-identity.json`）均已移出仓内发行（independent-review）；
  原始字节归档在数据盘，路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表。

## 3. dense FP32 全局参考（独立 oracle）

- 输入：`run-original-3a/capture/layers.npz`（7.9 MB 报告 / 1323 MiB 捕获 = 1,387,436,946 B）；
  报告全量件 `oracle-original-3a.json` 已移出仓内发行（detail-only），入库版为其聚合段 `oracle-original-3a.aggregates.json`（见 §5）。
- 覆盖：**24,192 comparisons = 24,080 prefill + 112 decode**；16 个全注意力层；**prefill 步 1 的全部 1505 个位置**
  （源码逐位置遍历，非仅末 token）+ **decode 步 1–7 全部**；`non_finite = 0`。
- 误差（FP32 dense 参考 vs 真实 FA2 输出）：**prefill 最大 `max_abs_err` 0.2600479126**（层 14；层 13 0.2295、层 10 0.1456）；
  **decode 最大 0.1179962158**；首个 decode 步 min 0.00414 / median 0.01490 / max 0.09901。
- **跨实现两样本核对**（主代理，NumPy float64，逐 head 重算两个最坏点）：0.2600652519 / 0.1179977043，
  与 FP32 报告差 1.73e-5 / 1.49e-6（证据 `oracle-worstcase-independent.json`，已移出仓内发行（independent-review）；
  路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表）。**仅为两样本核对，不是全量独立 oracle。**
- **口径**（不越界）：报告字段 `rel_err` = `max_abs_err / L2(out)`，**不是**逐元素 `allclose` 的 rtol；
  `dtype_name=None` 不得读作"模型 dtype 未知"——模型 dtype 由 manifest 的 BF16 与逐层 `capture_dtype_L{n}` 给出来源。
  误差含 BF16 舍入与 FA2 与 dense FP32 的实现差异；**本轮不设容差、不套用阶段 04 阈值**。

## 4. 未验证 / 待后续

- 质量、净收益、性能均未测；masked 轨迹、自由生成、公共 HTTP、长上下文（>8192）、混批**未放行**。
- FA2 **生效值**以显式配置为门禁；运行期 `Using FlashAttention version 2` 的日志核对仍待补。
- canonical 块表与 FA metadata 的块号单位未独立核验（只作记录）；硬判据是 FA `seq_lens[0] == 本步最后位置 + 1`。
- 正式 masked 容差须后续**预冻结**（依据本报告与后续取样），不得直接沿用旧阈值。

## 5. 复跑入口与当前环境

- **CPU 测试**（复用已通过记录，不为报告重跑）：
  ```bash
  cd /root/attnview && source ./env.sh
  CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" -m unittest tests.test_p2_calib_hooks tests.test_p2_calib_oracle
  bash tools/p1cpu-run-tests.sh        # 未部署源码树 → 308 项 OK（29.494 s / 复审后 29.903 s 两次记录）
  ```
  现存工具输出仅**末尾摘要**，未保存完整日志 —— 不伪造完整日志。
- **四臂复跑命令（按**已执行入口**整理；目录已换新，`run-original-4a`/`run-disabled-6`/`run-global-6` 尚未执行）**：
  ```bash
  cd /root/attnview && source env.sh && export CUDA_VISIBLE_DEVICES=0
  # 1) 未部署原版：第一轮自然 greedy 产生轨迹基线（不带 compare-to），第二轮比对
  "$ATTNVIEW_PYTHON" tools/p2-calib-run.py --arm original --out evidence/p3-calib/run-original-4a \
    --max-tokens 8 --emit-trajectory evidence/p3-calib/traj-original-2.json --capture-host-logits --record-layers --cleanup-check
  "$ATTNVIEW_PYTHON" tools/p2-calib-run.py --arm original --out evidence/p3-calib/run-original-4b \
    --max-tokens 8 --compare-to evidence/p3-calib/traj-original-2.json --cleanup-check
  # 2) 部署补丁（两副本一并改写）并核验
  "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py apply
  "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py verify
  # 3) 候选两臂：teacher forcing 回放同一轨迹
  "$ATTNVIEW_PYTHON" tools/p2-calib-run.py --arm patched-disabled --out evidence/p3-calib/run-disabled-6 \
    --max-tokens 8 --force-trajectory evidence/p3-calib/traj-original-2.json --compare-to evidence/p3-calib/traj-original-2.json --cleanup-check
  "$ATTNVIEW_PYTHON" tools/p2-calib-run.py --arm patched-global --out evidence/p3-calib/run-global-6 \
    --max-tokens 8 --force-trajectory evidence/p3-calib/traj-original-2.json --compare-to evidence/p3-calib/traj-original-2.json --cleanup-check
  # 4) 撤销回未部署态
  "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py revert
  ```
  上述命令的目标路径（`run-original-4a`、`run-original-4b`、`run-disabled-6`、`run-global-6`、`traj-original-2.json`）尚未产出；
  登记见 `reports/evidence-index.md` 的"已声明缺失/未执行"表。
- **oracle 参考（写新文件，不复写既有报告）**：
  ```bash
  CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-calib-oracle.py \
    --capture evidence/p3-calib/run-original-3a/capture/layers.npz \
    --out evidence/p3-calib/oracle-original-3a-rerun.json
  ```
  首份实测报告的聚合段为 `evidence/p3-calib/oracle-original-3a.aggregates.json`（保留结论所需全部数值段与 decode 逐比较 112 行，
  并自带全量件的 size + sha256；全量件 `oracle-original-3a.json` 已移出仓内发行（detail-only），原始字节归档在数据盘，
  路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表）。
  `oracle-original-3a-rerun.json` 目标路径尚未产出；登记见 `reports/evidence-index.md` 的"已声明缺失/未执行"表。
- **当前环境**（**原交付 HEAD `85cfacf`**；当前报告版本见 `outbox/SUP-004-calibration-result.md`）：源码树**未部署**（checkout 与安装副本逐字节一致、无已部署 `attnview_engine.py`、
  无 `orig/`、无事务记录）；工作区干净；GPU 0 MiB。
- **诊断时间口径**：四臂的 startup/请求耗时**包含**捕获与校准专用同步（强制钩子含 D2H/H2D、logits 捕获含显式 D2H），
  **不得**作为性能结论。
