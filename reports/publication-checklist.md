# 公开发布清单（2026-09-21：已首推 + 证据范围收缩）

日期：2026-09-20；更新：2026-09-21。状态：**生效中**。范围：公开仓 `github.com/at0rp1d0i-cell/peephole`「什么进 git、什么只登记」。
关联：[内核/适配分界计划](open-source-split-plan.md)、[证据索引（生成物）](evidence-index.md)、
[证据范围规则与登记表](evidence-registry.json)。

**首推已执行**：`origin/main` tip `57f1871`，推送时间 2026-09-21 01:03:17 +0800，含 4 个 tag。
此后**不再改写已推送历史**（用户 2026-09-21 决定）：发布范围的每次收缩都以**新提交**落地，
已推送的字节留在历史里，由登记表声明其当前状态。

2026-09-21 已执行：历史两次重写——① `AGENTS.md` 全历史移除；② 树内容脱敏（容器主机名 / GPU UUID /
内部协调目录三类替换，含 tag）。处置记录与 old→new SHA 映射见 [`git-history-map-20260921.txt`](git-history-map-20260921.txt)，
脱敏规则与逐文件改前/改后 sha256 见 [`desensitization.md`](desensitization.md)。本清单按 2026-09-21 实测刷新；
未注明日期的计数均为当日实测。

## 0. 已定前提

| 项 | 决定 | 来源 |
| --- | --- | --- |
| 仓库 | 已建 `at0rp1d0i-cell/peephole`，**public 空仓**（size 0、无分支/commit/README） | 2026-09-20 实查 GitHub API |
| 推送节奏 | 首推已执行（2026-09-21 01:03，tip `57f1871`）；此后**增量推送新提交、不改写已推送历史** | 用户 2026-09-20 定首推；2026-09-21 定增量 |
| 脱敏 | **已执行（2026-09-21）**：三类替换规则已定并应用于全历史（含 4 个 tag）；规则=`tools/pub-desensitize.py`／`/root/autodl-tmp/sanitize-tree.sh`，逐文件哈希见 `reports/desensitization.md`。当前树与全历史复扫命中 0（§7）。功能性的 `/root/autodl-tmp/attnview` 绝对路径按规则有意保留 | 用户 2026-09-20 登记；2026-09-21 执行并验证 |
| 许可 | Apache-2.0（2026-09-17 已定），**已落仓根并提交**：`LICENSE` + `NOTICE` | 素材仓 README / `LICENSE` |

## 1. 入库规则（机械可检）

| 入库 | 排除 |
| --- | --- |
| 文本与结构化产物：`*.md` `*.py` `*.sh` `*.json` `*.jsonl` `*.txt` `*.log` 及脚本快照 | 二进制张量与序列化：`*.npz` `*.npy` `*.pt` `*.bin` `*.safetensors` `*.gguf` |
| 单文件 ≤ 10 MB | 单文件 > 10 MB（**含文本**）：登记 sha256 + 摘要，原始件留远端 |

**语义规则**（什么算证据、什么只登记）以 `reports/evidence-registry.json` 的 **R1–R9** 为准，本节只管类型与体积；
机械事实（路径 / 字节 / sha256 / 状态）的唯一出处是生成物 [`reports/evidence-index.md`](evidence-index.md)，
用 `python3 tools/pub-evidence-registry.py --check` 复核。

`.gitignore` **已提交**（仓根），两段发布相关规则：① 证据大件兜底 `*.npz` / `*.npy`（见 §4）；
② `AGENTS.md`（内部协作入口，排除出公开仓，见 §6）。

依据（**唯一时点**，见 §2 的表）：当时入库文件单文件最大 3.74 MiB（远低于阈值），全部满足该规则；
当时唯一会咬人的缺口已修：见 §4。本节不再抄第二份计数——同一时点抄两遍正是过去出现「360 对 393」的原因；
复算方式见 §2 的命令。

## 2. 当前树规模（机械复算）

复算方式（`git ls-files` + 逐文件 `os.path.getsize`，与 `tools/gates.sh` 的「入库体积」项同源）：

```bash
git ls-files | awk -F/ '{print ($1=="src" ? "src/attnview/" : (NF==1 ? "根文件" : $1"/"))}' | sort | uniq -c
python3 - <<'EOF'
import os, subprocess
files = subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split()
print(len(files), sum(os.path.getsize(f) for f in files))
EOF
```

| 区域 | 文件数 | 体量 | 说明 |
| --- | ---: | ---: | --- |
| `evidence/` | 256 | 13.59 MiB | 历史证据层 + stage-05 冒烟批（§3 末尾待裁项） |
| `reports/` | 31 | 0.51 MiB | 含生成物 `evidence-index.md`、规则与登记表 `evidence-registry.json` |
| `tools/` | 36 | 0.51 MiB | 含 `_lib.py`（工具层共享实现）、`gates.sh`（门禁入口）、`git-hooks/pre-commit` |
| `vllm-patch/` | 11 | 0.39 MiB | `manifest.json` + 2 个新增文件 + 8 个上游改写副本 |
| `tests/` | 24 | 0.38 MiB | 含 `conftest.py`（统一 `sys.path`）与 `_support.py`（共享装载器） |
| `logs/` | 12 | 0.30 MiB | 安装与 serve 日志（含三次失败尝试原日志） |
| `src/attnview/` | 20 | 0.17 MiB | 内核 |
| 根文件 | 14 | 0.07 MiB | `.gitignore`、`LICENSE`、`NOTICE`、`env.sh`、`install-runtime.sh`、`verify-runtime.sh`、`setup-local-cuda.sh`、`requirements*.txt`、`pyproject.toml`、`README.md`、`CONTRIBUTING.md`、`.gitmessage` |
| `configs/` | 7 | 0.04 MiB | — |
| `docs/` | 1 | 0.02 MiB | `references/da-paper-extract.md` |
| **合计** | **412** | **15.97 MiB** | 单文件最大 3.74 MiB；可再收缩项见 §3 末尾与 `reports/evidence-index.md` 自检段 |

计数时点：本条提交的**索引状态**（`git ls-files` 的集合 + 逐文件当前字节）。此后任何提交都会
改变这些数字——改完请用上面的命令复算本节，不要凭记忆改。

## 3. 证据范围收缩（2026-09-21 执行，以新提交落地）

用户 2026-09-21 决定：公开仓证据按「交付依据 + 论证性失败件 + 报告」发行，其余**只登记不发行**；
不重写已推送历史，收缩以新提交落地。本轮移出 **75 个文件 / 10.64 MiB**，原件逐字节归档在数据盘
`/root/autodl-tmp/attnview-evidence-archive`（与仓内相对路径同构，哈希以登记表为准）：

| 处置 | 件数 | 体量 | 代表项 |
| --- | ---: | ---: | --- |
| `detail-only` 明细只登记 | 1 | 7.89 MiB | `oracle-original-3a.json`（24,192 行逐比较）；入库版是 69 KB 聚合段 `oracle-original-3a.aggregates.json`，后者自带全量件 size + sha256，`tools/p2-oracle-aggregate.py --check` 可用全量件复现 |
| `superseded` 被取代代次 | 42 | 1.76 MiB | `evidence/p1-gpu/`、`p1-gpu-v2/`…`-v5/`（v6 索引已声明「历史，不作为交付依据」）、`p1-cpu/evidence-index.md`（机械表由生成索引取代） |
| `independent-review` 复核过程记录 | 23 | 0.71 MiB | 7 个 `*local-review*` 目录；结论要点留在 `reports/p3-calib/evidence-index.md` §2/§5 与 `reports/p1-gpu/read-view-report.md` |
| `snapshot` 中间快照 | 3 | 0.25 MiB | 环境探针 `1637`/`1655`/`1910`（首末两份仍在库内：`evidence/before/env-report-20260917-1559.md`、`evidence/after/env-report-20260917-1716.md`） |
| `duplicate` 重复副本 | 6 | 0.03 MiB | `logs/console-install.log`、`evidence/verify-after-{expand,move,hf-endpoint}.log`、`cuda-upgrade-freeze-{before,final}.txt`（均与仍在库内的文件逐字节相同） |

收缩后历史证据层为 **154 个 / 4.28 MiB**（原 228 个 / 14.86 MiB）。逐件哈希与状态见生成物
[`reports/evidence-index.md`](evidence-index.md) 的「树外登记」表；规则见 `reports/evidence-registry.json`。

原「未跟踪项快照（2026-09-20）」枚举表已删除：未跟踪项的判定改为机械门禁——`--check` 会列出工作区
未分类产物，**入库前必须先在规则表里给出规则**，不再靠人工维护清单。


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

2026-09-21 起大件登记改为机械：`reports/evidence-registry.json` 的 `registered_only`（在盘未入库，
必须被 `.gitignore` 覆盖，由 A2 校验）与 `gone`（已声明缺失 / 未执行）；`*.npz` / `*.pt` 规则不变。

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
- **树外登记件引用（2026-09-21 新增）**：本轮移出仓内的 75 个件，文档里不得再断言「字节在仓内」；
  引用处改为指向 [`reports/evidence-index.md`](evidence-index.md) 的树外登记表（状态 + 哈希 + 归档根目录）。
  门禁：`--check` 的 **A4**（悬空引用，硬失败）与 **A7**（引用了树外件的文件清单，提示人工核对表述）。

## 7. 推送前门禁（2026-09-21 起）

```bash
python3 tools/pub-evidence-registry.py --check     # A1–A6 硬门禁 + A7 表述清单；期望 OK（pending 只警告）
python3 tools/pub-desensitize.py --check           # 三类内部标识复扫；期望 0 命中（exit 0）
git ls-files -z | xargs -0 grep -lIE '(gh[p]_|github[_]pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)'   # 期望：空
git ls-files -z | xargs -0 du -b | sort -rn | head -3    # 单文件 ≤ 10 MB
git status --short                                 # 待入库集合必须已被 registry 规则覆盖
```

- **不改写已推送历史**：范围的增减一律以新提交表达；`git push` 前不需要、也不允许 `filter-branch`/force-push。
- **不再使用 `git add -A`**：它会把工作区未分类的产物一起带走（历史上正是它差点提交 5.2 G）。新增产物
  先在 `reports/evidence-registry.json` 里给出规则或登记，再 `git add <明确的路径>`。
- 仍有效的复扫事实（2026-09-21）：三类内部标识在跟踪树与全部历史 commit 上命中 0；功能性的
  `/root/autodl-tmp/attnview` 绝对路径按 §0 规则有意保留。


## 8. 未决项（用户）

已关闭：

| 原未决项 | 结论 |
| --- | --- |
| 首推执行 | **已执行**：2026-09-21 01:03:17 +0800 推送 `57f1871`（含 4 个 tag） |
| 证据范围 | **已执行（2026-09-21）**：按「交付依据 + 论证性失败件 + 报告」发行，75 个件移出为树外登记（§3）；oracle 全量件改聚合段 |
| 版权署名 / 脱敏规则 / `docs/` 合并 / `AGENTS.md` 公开与否 | 见 §5、§6（均已于 2026-09-21 关闭） |

仍未决（用户）：

1. **stage-05 冒烟批的明细裁定**：该批 94 个 / 9.30 MiB 已随 `40f404f` 入库，其中
   `run/capture/structure.json` 两份（3.9 MB / 3.6 MB）、`run/manifest.json`（1.3 MB）、`run/steps.jsonl`、
   以及 `-first` / `-retry1` 两轮各自 23 个逐字节相同的源码快照，按 R6/R7/R8 是否降为
   `detail-only` / `registered_only` / `pending`，待该阶段验收后裁定（结论候选与字节数见
   `reports/evidence-index.md` 自检段与阶段 05 报告）。
2. **DCO**：是否要求提交署名（§5「若要 DCO 再单独定」）。
3. **SPDX 头批次**：自有源文件加 SPDX 头须与 `vllm-patch/manifest.json` 指纹重生成同批，批次时间未定（§5）。
