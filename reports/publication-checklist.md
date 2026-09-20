# 首推入库清单（草案）

日期：2026-09-20。状态：**草案，待用户确认**。范围：第一次 push 到 `github.com/at0rp1d0i-cell/peephole`
时「什么进 git」。关联：[内核/适配分界计划](open-source-split-plan.md)、台账 2026-09-20 命名决策记录。

## 0. 已定前提

| 项 | 决定 | 来源 |
| --- | --- | --- |
| 仓库 | 已建 `at0rp1d0i-cell/peephole`，**public 空仓**（size 0、无分支/commit/README） | 2026-09-20 实查 GitHub API |
| 推送节奏 | **补完再一次首推**（历史与全部证据同时入库） | 用户 2026-09-20 |
| 脱敏 | **本轮只登记不改**；56 个跟踪文件带容器内部标识，规则待定 | 用户 2026-09-20 |
| 许可 | Apache-2.0（2026-09-17 已定），本轮落到仓根并补 `NOTICE` | 素材仓 README / `LICENSE` |

## 1. 入库规则（机械可检）

| 入库 | 排除 |
| --- | --- |
| 文本与结构化产物：`*.md` `*.py` `*.sh` `*.json` `*.jsonl` `*.txt` `*.log` 及脚本快照 | 二进制张量与序列化：`*.npz` `*.npy` `*.pt` `*.bin` `*.safetensors` `*.gguf` |
| 单文件 ≤ 10 MB | 单文件 > 10 MB（**含文本**）：登记 sha256 + 摘要，原始件留远端 |

依据：现有 269 个跟踪文件合计 6.5 MB、最大 236 KB，全部满足该规则（历史一致，无需重写）。
唯一会咬人的缺口已修：见 §4。

## 2. 已跟踪部分（269 文件）

| 区域 | 文件数 | 处置 |
| --- | --- | --- |
| `evidence/` | 156 | 入库（脱敏登记项）。含 `p0-model` 46、`p3-calib` 18、`p1-cpu` 16、`after` 11、`p1-gpu-v{2..6}` 等 |
| `tools/` | 25 | 入库（含 gen/apply patch、calib 驱动、探针） |
| `reports/` | 21 | 入库（含失败归因与复核记录） |
| `tests/` | 16 | 入库 |
| `src/attnview/` | 15 | 入库（内核） |
| `logs/` | 13 | 入库：`install-runtime-*` 与 `serve-e4*`/`serve-e6*`（含三次失败尝试的原日志） |
| `vllm-patch/` | 11 | 入库：`manifest.json` + 2 个新增文件 + 8 个上游改写副本 |
| `configs/` | 4 | 入库 |
| 根文件 | 8 | `env.sh`、`install-runtime.sh`、`verify-runtime.sh`、`setup-local-cuda.sh`、`requirements.freeze*.txt`、`AGENTS.md`、`.gitignore` |
| 本轮新增 | 3 | `LICENSE`、`NOTICE`、`README.md`（未 add） |

## 3. 未跟踪项：逐项处置

本表是 2026-09-20 的**快照**；工作区在活动（stage-05 校准仍在跑，写本表期间就新出现了
`run-global-4`），因此真正生效的是 §1 的规则加 §4 的 `.gitignore`，不是这张枚举表。每个新 run
目录按同一方式处理：**文本层入库、`capture/*.npz` 与 `*.pt` 排除**。

| 路径 | 体量 | 内容 | 处置 |
| --- | --- | --- | --- |
| `evidence/p1-gpu-local-review-20260918-1011/` | 444 K / 5 | 本地复核产物 | 入库 |
| `evidence/p2-single-local-review-params-20260918/` | 4 K / 1 | 复核记录 | 入库 |
| `evidence/p2-single-local-review-r1-20260918/` | 4 K / 1 | 复核记录 | 入库 |
| `evidence/p2-single-local-review-r2-20260918/` | 4 K / 1 | 复核记录 | 入库 |
| `evidence/p3-calib-local-review-20260918/` | 168 K / 11 | 复核记录 | 入库 |
| `evidence/p3-calib/oracle-original-3a.json` | 7.9 M / 1 | oracle 输出（文本） | 入库（≤10 MB；后续同类件按 §1 阈值处理） |
| `evidence/p3-calib/run-disabled-3/` | 124 K / 2 | `error.txt` + 运行脚本快照 | 入库 |
| `evidence/p3-calib/run-disabled-4/` | 2.6 G / 16 | 见下拆分 | **文本层入库，dump 排除** |
| `evidence/p3-calib/run-disabled-5/` | 2.6 G / 16 | 与 4 同构 | **文本层入库，dump 排除** |
| `evidence/p3-calib/run-global-4/` | 2.6 G / 16 | 与 4 同构（写本表期间新出现） | **文本层入库，dump 排除** |

`run-disabled-4`（`-5` 同构）的拆分依据（实测文件清单）：

| 文件 | 大小 | 处置 |
| --- | --- | --- |
| `manifest.json` | 120 K | 入库 |
| `source/p2-calib-run.py` | 124 K | 入库（运行源冻结的脚本快照） |
| `engine-traces.json` | 7.6 K | 入库 |
| `cleanup.json` | 6.7 K | 入库 |
| `force.jsonl` | 2.6 K | 入库 |
| `arm.json` | 813 B | 入库 |
| `capture/layers.npz` | 1.39 G | **排除**（激活 dump） |
| `capture/forward1.npz` | 1.38 G | **排除** |
| `capture/forward{2..8}.npz` | 各 0.95 M | **排除** |
| `logits.pt` | 7.9 M | **排除**（序列化张量；原 `*.pt` 规则已覆盖） |

被排除的大件在证据索引里登记：文件名、大小、sha256、生成命令；本仓保留同一 run 的
`manifest.json`（含逐文件 sha256）与之对应，所以「跑过什么、产出什么、结论如何」可核对，
字节本身不随仓发行。

## 4. `.gitignore` 变更（已落盘）

新增 `*.npz`、`*.npy`——**本轮唯一会咬人的缺口**：`git add -A` 原本会试图提交 5.2 G。
`*.pt` 原规则已覆盖 `logits.pt`。新增处带注释说明「文本 ≤10 MB 才入库，大件登记哈希」。

已核验（2026-09-20）：三个 run 目录各 16 个文件 → **10 个 dump 被忽略、6 个文本层仍会入库**
（`manifest.json`、`source/p2-calib-run.py`、`engine-traces.json`、`cleanup.json`、`force.jsonl`、
`arm.json`）；`git ls-files | git check-ignore` = **0**，即没有任何已跟踪文件被新规则误伤。

## 5. 许可证与署名（「证书」）

| 项 | 状态 |
| --- | --- |
| `LICENSE`（Apache-2.0 全文） | 已落仓根；与素材仓那份仅差附录版权行（其余逐字节一致） |
| `NOTICE`（派生自 pin `98dff2a8`、8 个改写文件、2 个新增文件、引用边界） | 已落；顶部含我方版权行 |
| `LICENSE` 附录版权行 | **已填**：`Copyright 2026 at0rp1d0i-cell`（用户 2026-09-20 决定） |
| `NOTICE` 我方版权行 | **已填**：同上 |
| 自有源文件 SPDX 头（`src/attnview/*.py`、`tools/`、`tests/`、2 个新增补丁文件） | **待改名批次**：加头会改文件哈希 → 打穿 `manifest.json` 的 `package_files` 与部署指纹；必须与改名/清单重生成同批做，不能单独插队 |
| 贡献者授权 | 走 Apache-2.0 第 5 条（inbound=outbound），**不需要 CLA**；若要 DCO 再单独定 |
| 商标边界 | `NOTICE` 已声明派生关系、不暗示 vLLM 官方背书；项目名未使用上游商标 |
| 上游形式义务 | §4(b) 改动声明（文件内注释）✓、§4(c) 保留归属（SPDX 头）✓、§4(d) 上游无 `NOTICE`（已核实）→ 本仓 `NOTICE` 为自述派生 |

## 6. 悬空引用与仓库合并（公开前必须处理）

- **11 个跟踪文件引用 `material/attnview/...`**（`env.sh`、`evidence/p0-model/e0-baseline.txt`、
  `evidence/p1-cpu/prompt-fidelity.txt`、`reports/environment-report.md`、
  `reports/environment-setup.md` 等），而 `material/` 被 `.gitignore` 排除 → 公开仓里这些路径没有对应物。
  处置：把素材仓 `docs/` 并入本仓 `docs/` 并统一引用，或先在 README 说明来源。
- **`AGENTS.md`（已跟踪）含内部流程**：第 3 行引用 `/path/to/supervision/protocol.md`
  与 inbox/outbox 回执约定。处置：公开版删该段，保留硬边界条款（不得自动开通付费资源、不得把推导
  写成实测、不上传权重与私密 trace）与验证入口。

## 7. 首推序列（待 §5 版权行与 §6 处置确认后执行）

```bash
git remote add origin git@github.com:at0rp1d0i-cell/peephole.git   # 协议按用户偏好（SSH/HTTPS）
git add -A && git status --short                                    # 确认待入库集合符合 §1–§3
git ls-files -z | xargs -0 grep -lIE 'autodl-container|GPU-[0-9a-f]{8}-|/root/autodl-tmp/attnview' | sort -u   # 期望：已知的 56 个（本轮只登记）
git ls-files -z | xargs -0 grep -lIE '(ghp_|github_pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)'  # 期望：空
git ls-files -z | xargs -0 du -b | sort -rn | head -3               # 期望：最大 ≤10 MB
git commit -m "..." && git push -u origin main
```

commit message 已扫过：73 条历史信息中 0 命中容器标识，历史本身干净。

## 8. 未决项（用户）

1. **版权署名**：`LICENSE` 附录与 `NOTICE` 的版权行填什么（真名 / 账号 ID / 项目名 / 不填）。
2. 脱敏规则（本轮只登记；56 文件、主机名 28 / GPU UUID 5 / 绝对路径 39）。
3. `docs/` 合并方式与 11 处 `material/attnview` 引用如何改。
4. `AGENTS.md` 公开版是改写还是排除。
