# 首推入库清单（草案）

日期：2026-09-20；更新：2026-09-21。状态：**草案，待用户确认**。范围：第一次 push 到 `github.com/at0rp1d0i-cell/peephole`
时「什么进 git」。关联：[内核/适配分界计划](open-source-split-plan.md)、台账 2026-09-20 命名决策记录。

2026-09-21 已执行：历史两次重写——① `AGENTS.md` 全历史移除；② 树内容脱敏（容器主机名 / GPU UUID /
内部协调目录三类替换，含 tag）。处置记录与 old→new SHA 映射见 [`git-history-map-20260921.txt`](git-history-map-20260921.txt)，
脱敏规则与逐文件改前/改后 sha256 见 [`desensitization.md`](desensitization.md)。本清单按 2026-09-21 实测刷新；
未注明日期的计数均为当日实测。

## 0. 已定前提

| 项 | 决定 | 来源 |
| --- | --- | --- |
| 仓库 | 已建 `at0rp1d0i-cell/peephole`，**public 空仓**（size 0、无分支/commit/README） | 2026-09-20 实查 GitHub API |
| 推送节奏 | **补完再一次首推**（历史与全部证据同时入库） | 用户 2026-09-20 |
| 脱敏 | **已执行（2026-09-21）**：三类替换规则已定并应用于全历史（含 4 个 tag）；规则=`tools/pub-desensitize.py`／`/root/autodl-tmp/sanitize-tree.sh`，逐文件哈希见 `reports/desensitization.md`。当前树与全历史复扫命中 0（§7）。功能性的 `/root/autodl-tmp/attnview` 绝对路径按规则有意保留 | 用户 2026-09-20 登记；2026-09-21 执行并验证 |
| 许可 | Apache-2.0（2026-09-17 已定），**已落仓根并提交**：`LICENSE` + `NOTICE` | 素材仓 README / `LICENSE` |

## 1. 入库规则（机械可检）

| 入库 | 排除 |
| --- | --- |
| 文本与结构化产物：`*.md` `*.py` `*.sh` `*.json` `*.jsonl` `*.txt` `*.log` 及脚本快照 | 二进制张量与序列化：`*.npz` `*.npy` `*.pt` `*.bin` `*.safetensors` `*.gguf` |
| 单文件 ≤ 10 MB | 单文件 > 10 MB（**含文本**）：登记 sha256 + 摘要，原始件留远端 |

`.gitignore` **已提交**（仓根），两段发布相关规则：① 证据大件兜底 `*.npz` / `*.npy`（见 §4）；
② `AGENTS.md`（内部协作入口，排除出公开仓，见 §6）。

依据：2026-09-21 实测 360 个跟踪文件合计 16.9 MiB（17,701,865 B）、单文件最大 8.27 MB
（`evidence/p3-calib/oracle-original-3a.json`），全部满足该规则；当时唯一会咬人的缺口已修：见 §4。

## 2. 已跟踪部分（360 文件，2026-09-21 实测 `git ls-files | wc -l`）

| 区域 | 文件数 | 处置 |
| --- | --- | --- |
| `evidence/` | 228 | 入库（含 §3 快照的 40 个文件与 p3-calib 六个 run 的文本层；大件 dump 按 §4 排除） |
| `tools/` | 28 | 入库（含 gen/apply patch、calib 驱动、探针） |
| `reports/` | 24 | 入库（含失败归因、复核记录、`git-history-map-20260921.txt`） |
| `tests/` | 21 | 入库 |
| `src/attnview/` | 20 | 入库（内核） |
| `logs/` | 13 | 入库：`install-runtime-*` 与 `serve-e4*`/`serve-e6*`（含三次失败尝试的原日志） |
| `vllm-patch/` | 11 | 入库：`manifest.json` + 2 个新增文件 + 8 个上游改写副本 |
| `configs/` | 6 | 入库 |
| 根文件 | 9 | `.gitignore`、`LICENSE`、`NOTICE`、`env.sh`、`install-runtime.sh`、`verify-runtime.sh`、`setup-local-cuda.sh`、`requirements.freeze*.txt` |
| 2026-09-21 工作区新增（待 Main 提交） | 3 | `README.md`、`docs/references/da-paper-extract.md`、`tools/pub-desensitize.py`；另有本清单 §0 引用的 `reports/desensitization.md`（脱敏逐文件哈希）同批落盘 |

## 3. 未跟踪项快照（2026-09-20）：已全部按 §1 规则入库

本表是 2026-09-20 的**快照**；2026-09-21 复核：下列 10 条路径共 40 个文件全部已在 `git ls-files`
中，工作区不再有未跟踪的 `evidence/` 文件。真正生效的是 §1 的规则加 §4 的 `.gitignore`，不是这张
枚举表；每个新 run 目录按同一方式处理：**文本层入库、`capture/*.npz` 与 `*.pt` 排除**。

| 路径 | 体量 | 内容 | 处置 |
| --- | --- | --- | --- |
| `evidence/p1-gpu-local-review-20260918-1011/` | 444 K / 5 | 本地复核产物 | 已入库（5） |
| `evidence/p2-single-local-review-params-20260918/` | 4 K / 1 | 复核记录 | 已入库（1） |
| `evidence/p2-single-local-review-r1-20260918/` | 4 K / 1 | 复核记录 | 已入库（1） |
| `evidence/p2-single-local-review-r2-20260918/` | 4 K / 1 | 复核记录 | 已入库（1） |
| `evidence/p3-calib-local-review-20260918/` | 168 K / 11 | 复核记录 | 已入库（11） |
| `evidence/p3-calib/oracle-original-3a.json` | 7.9 M / 1 | oracle 输出（文本） | 已入库（8.27 MB，≤10 MB） |
| `evidence/p3-calib/run-disabled-3/` | 124 K / 2 | `error.txt` + 运行脚本快照 | 已入库（2） |
| `evidence/p3-calib/run-disabled-4/` | 2.6 G / 16 | 见下拆分 | 已入库：文本层 6、dump 10 排除 |
| `evidence/p3-calib/run-disabled-5/` | 2.6 G / 16 | 与 4 同构 | 已入库：文本层 6、dump 10 排除 |
| `evidence/p3-calib/run-global-4/` | 2.6 G / 16 | 与 4 同构（写本表期间新出现） | 已入库：文本层 6、dump 10 排除 |

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

## 4. `.gitignore` 变更（已提交）

新增 `*.npz`、`*.npy`——**当时唯一会咬人的缺口**：`git add -A` 原本会试图提交 5.2 G。
`*.pt` 原规则已覆盖 `logits.pt`。新增处带注释说明「文本 ≤10 MB 才入库，大件登记哈希」。

已核验（2026-09-21 复核，口径变多）：`p3-calib` 下现有 **6 个 run 目录带 capture dump**
（`run-disabled-4`、`run-disabled-5`、`run-global-4`、`run-global-5`、`run-original-3a`、`run-original-3b`）：
每个 **10 个 dump 被忽略**；四个 16 文件目录各 **6 个文本层已入库**（`manifest.json`、
`source/p2-calib-run.py`、`engine-traces.json`、`cleanup.json`、`force.jsonl`、`arm.json`），
两个 14 文件目录（`run-original-3a`/`-3b`）各 **4 个文本层已入库**（无 `engine-traces.json`、`force.jsonl`）。
`git ls-files | git check-ignore --stdin` = **0**，即没有任何已跟踪文件被新规则误伤。

（2026-09-20 首次核验时只有三个 run 目录；2026-09-21 已增至六个，排除文件数相应增加。）

## 5. 许可证与署名（「证书」）

| 项 | 状态 |
| --- | --- |
| `LICENSE`（Apache-2.0 全文） | 已落仓根并提交；与素材仓那份仅差附录版权行（其余逐字节一致） |
| `NOTICE`（派生自 pin `98dff2a8`、8 个改写文件、2 个新增文件、引用边界） | 已落并提交；顶部含我方版权行 |
| `LICENSE` 附录版权行 | **已填**：`Copyright 2026 at0rp1d0i-cell`（用户 2026-09-20 决定） |
| `NOTICE` 我方版权行 | **已填**：同上 |
| 自有源文件 SPDX 头（`src/attnview/*.py`、`tools/`、`tests/`、2 个新增补丁文件） | **待办（2026-09-21 复核仍未动）**：加头会改文件哈希 → 打穿 `vllm-patch/manifest.json` 的 `package_files` 与部署指纹；必须与改名/清单重生成**同批**做（`vllm-patch/manifest.json` 指纹同批重生成），不能单独插队 |
| 贡献者授权 | 走 Apache-2.0 第 5 条（inbound=outbound），**不需要 CLA**；若要 DCO 再单独定 |
| 商标边界 | `NOTICE` 已声明派生关系、不暗示 vLLM 官方背书；项目名未使用上游商标 |
| 上游形式义务 | §4(b) 改动声明（文件内注释）✓、§4(c) 保留归属（SPDX 头）✓、§4(d) 上游无 `NOTICE`（已核实）→ 本仓 `NOTICE` 为自述派生 |

## 6. 悬空引用与仓库合并（2026-09-21 已定并执行）

- **`material/attnview` 引用**：2026-09-21 实测 **12 个跟踪文件、19 处**（含本清单自身 2 处；原报
  「11 个」未计入本清单），而 `material/` 被 `.gitignore` 排除 → 公开仓里这些路径没有对应物。
  处置（已定）：**不合并素材仓 `docs/`**，改由 README「Design documents」段的 provenance 声明说明
  这些设计文档位于素材仓、随发布并入 `docs/`；跟踪文件的实质读取需求是论文摘录件，已导入仓内
  `docs/references/da-paper-extract.md`（附录 B 第 1514–1555 行、附录 F 第 2257–2534 行逐字摘录；
  2026-09-17 经 `curl -sSL https://arxiv.org/pdf/2609.02737v1` + `pdftotext -layout` 提取；CC BY 4.0，
  已按署名要求标注来源与作者）。其余 `material/attnview` 引用为来源性说明或指向私有素材仓快照的
  可选回退（公开 checkout 中不存在）。
- **`AGENTS.md`（原跟踪）**：决定（2026-09-21）：**排除出公开仓**——不再是「删一段后公开」。已从
  **全部历史**移除（pass 1 `filter-branch`，`git log --all -- AGENTS.md` 为空），并写入 `.gitignore`
  （工作区本地副本留存、不入库）。已处置：README 不再引用 `AGENTS.md`（验证入口改为
  `bash verify-runtime.sh` + `python -m pytest tests/ -q`，边界材料指向 `LICENSE`/`NOTICE`/`reports/`）。

## 7. 首推序列（2026-09-21 更新，待用户确认后执行）

脱敏已在树与全历史完成，推送前只剩复扫、提交与推送：

```bash
git remote add origin git@github.com:at0rp1d0i-cell/peephole.git   # 协议按用户偏好（SSH/HTTPS）
git add -A && git status --short                                    # 确认待入库集合符合 §1–§3
# 1) 历史残留模式（脱敏目标）在全部 commit 上期望 0：
git grep -lIE 'autodl-container-[a-z0-9]+-[a-z0-9]+|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|/root/autodl-tmp/attnview[-]supervision' $(git rev-list --all)   # 实测：空（0）
# 2) 密钥模式期望空：
git ls-files -z | xargs -0 grep -lIE '(ghp_|github_pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)'   # 期望：空
# 3) 二进制/大件兜底：单文件 ≤10 MB（二进制类型由 §1/§4 的 .gitignore 规则排除）：
git ls-files -z | xargs -0 du -b | sort -rn | head -3               # 实测：最大 8.27 MB（evidence/p3-calib/oracle-original-3a.json）
git commit -m "..." && git push -u origin main
```

- 容器标识复扫（2026-09-21 实测）：上述模式在**跟踪树命中 0 个文件**（旧清单的 56 为脱敏前计数），
  在**全历史 148 个 commit** 的树上同样 0。功能性的 `/root/autodl-tmp/attnview` 绝对路径按 §0 的规则
  有意保留（52 个跟踪文件、354 处），不属该模式。
- commit 信息：148 条历史信息实测 0 命中同一标识模式，历史本身干净。

## 8. 未决项（用户）

原四项（2026-09-20）已全部关闭：

| 原未决项 | 结论（2026-09-21） |
| --- | --- |
| 版权署名填什么 | `Copyright 2026 at0rp1d0i-cell`，已填 `LICENSE` 附录与 `NOTICE`（§5） |
| 脱敏规则 | 三类替换规则已定并应用于全历史（含 tag）；复扫命中 0，逐文件哈希见 `reports/desensitization.md`（§0/§7） |
| `docs/` 合并与 `material/attnview` 引用 | 不合并素材仓；README provenance 声明 + 导入 `docs/references/da-paper-extract.md`（§6） |
| `AGENTS.md` 公开版改写还是排除 | 排除出公开仓，已从全部历史移除（§6） |

仍未决（用户）：

1. **首推执行**：`git add -A` 后确认提交信息与推送协议（SSH/HTTPS），再执行 push（§7）。
2. **DCO**：是否要求提交署名（§5「若要 DCO 再单独定」）。
3. **SPDX 头批次**：自有源文件加 SPDX 头须与 `vllm-patch/manifest.json` 指纹重生成同批，批次时间未定（不阻塞首推，§5）。
