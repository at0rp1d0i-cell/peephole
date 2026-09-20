# 贡献与提交规范

适用：`peephole` 仓库（原 `attnview` 实验仓）。自 2026-09-21 起适用，此前历史保持原样。
工作语言：提交信息与报告用中文，**标识符、命令、路径保留原文**。
> Commit messages and reports are written in Chinese; identifiers, commands and paths are kept verbatim.

## 1 提交信息

### 1.1 骨架

```
<type>(<scope>): <≤50 字中文摘要>     ← subject 整行 ≤72 字符；不含阶段、不含工作单编号

做了什么：<一两句；微小改动可省略>
证据：<可复跑的命令，或产物路径>
未验证：<明确写出，不留空>

Work-order: SUP-004, NATIVE-053
Evidence: evidence/p3-calib/run-disabled-5, reports/p1-gpu/session-closeout.md
Verified: python -m pytest tests/ -q (exit 0)
Unverified: 实机 GPU 路径未运行；数值口径未复核
```

非平凡改动（改了行为、改了门禁、入了证据）**必带正文**。正文写不出来，通常说明这个提交该拆。

### 1.2 type 与 scope

| type | 用于 | 例 |
| --- | --- | --- |
| `feat` | 新增行为/接口 | `feat(protocol): 增量声明解析器` |
| `fix` | 修复缺陷 | `fix(smoke): 逐层快照改在本步 forward 内采集` |
| `docs` | 报告、设计、清单 | `docs(publication): 首推入库清单` |
| `evidence` | 产物入库、索引登记 | `evidence(calib): 校准 run 文本层入库` |
| `test` | 测试与门禁 | `test(smoke): 补 fail-closed 反例` |
| `refactor` | 不改行为的重构 | `refactor(step-plan): 抽出纯函数` |
| `chore` | 脚手架、忽略规则、配置 | `chore(gitignore): 排除证据张量大件` |

`scope` = 子系统/区域，取短名词：`protocol` / `readview` / `smoke` / `calib` / `adapter` / `patch` / `deploy` / `publication` / `readme`。
**阶段与工作单编号都不进 subject**：阶段由 tag 标识（§1.4），编号进 `Work-order:` trailer——subject 留给"读得懂的一句话"。

### 1.3 trailer 词表

固定 4 个键，放消息末尾，供机器过滤（`git log --grep='^Work-order: SUP-004'`、`git log --format='%(trailers:key=Verified,valueonly)'`）：

| 键 | 内容 | 必填 |
| --- | --- | --- |
| `Work-order` | 对应工作单编号，逗号分隔 | 有工作单时必填 |
| `Evidence` | 产物路径，逗号分隔 | 产生/引用了证据时必填 |
| `Verified` | 实际跑过的命令与退出码 | 有可跑验证时必填 |
| `Unverified` | 明确没做的部分 | **必填**；确实全验时写 `无` |

### 1.4 阶段由 tag 标识

`stage-NN` tag 打在阶段边界上，是阶段的权威标识；最近祖先 tag 就是"当前阶段"，因此提交信息里不重复阶段：

```bash
git describe --tags --abbrev=0            # 当前阶段，例：stage-05
git log --oneline stage-04..stage-05      # 某阶段的提交范围
git log --oneline --decorate              # 边界处直接看到 tag
```

### 1.5 反例（都来自本仓历史，供对照）

- **单行超长**：`stage-05 NATIVE-052:结构检查——q_len 取自搬运前标量…` 320 字符挤在 subject 里 → `git log --oneline` 不可读，证据与未验证项没有位置。2026-09-21 实测：156 条提交里 62 条 subject >100 字符，最长 320。
- **第二套风格**：`fix(scope): describe the change in english`——英文 Conventional 风格、无正文、无证据、无未验证项。同一仓库里并存两套提交风格本身就是"乱"的来源；要换风格就整仓换，别一条一条换。
- **无可核对内容**：`docs: 修一些文档`——没说改了什么、凭什么算完成。

### 1.6 证据发布范围

证据分**随仓发行**与**树外登记**两类，判定不看"文件在不在 `evidence/` 下"，只看机械来源：索引 `reports/evidence-index.md`"树内：随仓发行"表列出的字节才随仓发行，未列出的不随仓发行。

引用树外登记件统一写"已移出仓内发行（<status>）；原始字节归档在数据盘，路径/字节/sha256 见索引的树外登记表"——只指向索引，不逐处抄哈希。树外登记件不写进 `Evidence:` trailer（那里只写随仓发行的产物路径）。

规则全文（R1–R9）与状态词只保留两处：索引的"发布规则""状态表"两节与状态源 `reports/evidence-registry.json`，此处不复制；索引的生成与复核入口见 §6。

## 2 一条提交一件事

格式改动（改名、格式化、重排）与行为改动分开提交。一次提交能独立回滚、独立复核。

**共享工作区（多 agent 并行）额外约束**：`git commit --amend`、`rebase`、`reset --hard` 只允许作用在**自己的、未推送的**提交上，且执行前先确认 `HEAD` 就是你那条提交（`git log --oneline -1`）——HEAD 可能已被别人推进。要改别人的提交一律先问；需要撤销自己的改动时优先用 `reset --soft`（只动 HEAD，不动工作区与暂存区），避免覆盖他人未提交的工作。

## 3 机械门禁（提交前自跑）

测试仍使用现有 unittest 用例，由 pytest 统一收集。首次运行时在项目虚拟环境补齐固定的开发依赖：

```bash
source env.sh
uv pip install --python "$ATTNVIEW_PYTHON" -c requirements.freeze.txt -r requirements.dev.txt
```

```bash
python -m pytest tests/ -q                                     # CPU 套件
python3 tools/pub-desensitize.py --check                       # 必须 exit 0
git ls-files -z | xargs -0 du -b | sort -rn | head -3          # 最大文件 ≤10 MB
git ls-files -z | xargs -0 grep -lIE '(gh[p]_|github[_]pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)'   # 期望空
```

`pub-desensitize.py --check` 覆盖三类内部标识：容器主机名、GPU UUID、内部协调目录。绝对路径（`/root/autodl-tmp/attnview`）是有意保留的功能性活值，不在该工具的默认规则内。

## 4 硬边界

1. **不自动开通/购买付费资源**（GPU 实例、存储扩容、外部服务）；改动要用户明确授权。
2. **不得把推导数字写成实测数字**；失败与负结果保留原始日志（`evidence/*/error.txt`、失败尝试目录都不删）。
3. **不上传权重、数据集与私密 trace**；`*.npz` / `*.npy` / `*.pt` / `*.safetensors` / `*.gguf` 一律不入库，只在证据索引登记 size + sha256。
4. **不宣称未经质量约束测量支持的加速或收益**；未验证项必须写进提交与报告。

## 5 推送与发布节奏

- 开发提交留在本地；**阶段边界**（审计通过 / 本地 READY）才 push 一次。
- 阶段边界打 annotated tag `stage-NN`。
- **不改写已推送历史**。唯一例外：2026-09-21 首推前的两次重写（AGENTS.md 全历史移除 + 树内容脱敏），映射见 `reports/git-history-map-20260921.txt`。
- **未推送的提交可以自由整理**（改写、合并、补正文）——这正是"阶段边界才推"换来的空间。
- 推送前跑 §3 四道扫描 + 全历史残留扫描；推送后从远端重新 clone 复核（证明公开的字节就是你以为的字节）：

```bash
git grep -lIE 'autodl-container-[a-z0-9]+-[a-z0-9]+|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|/root/autodl-tmp/attnview[-]supervision' $(git rev-list --all) | wc -l   # 期望 0
```

### 5.1 历史消息规范的边界（2026-09-21 决定）

本规范生效点之前的 **156 条提交**（含 2 条英文 Conventional 风格、无正文的提交）**保留原始消息，不回溯改写**。理由：其中 141 条是"阶段 + 工作单"打头的长单行，规范化等于逐条重写文案——机械拆分只能产出 `收尾`、`R1 返工` 这类空壳 subject；而代价是二次 `force-push`、二次 SHA 映射（证据 manifest 的 `head` 字段再次失效）与一次完整重核。收益只是观感统一，风险落在已经公开的历史上。风格分界点即 §1 的生效日期。

### 5.2 推送前的尾部整理

未推送的提交在推送前整理到新模板——这正是"阶段边界才推"换来的空间：

- 各作者整理**自己的**未推送提交；共享工作区里不改写他人的提交（见 §2）。
- 整理清单：subject 同构且 ≤72 字符；正文与 `Work-order` / `Evidence` / `Verified` / `Unverified` 四个 trailer 齐备。
- 推送前跑 §3 门禁；推送后按本节开头做 clone 复核。

## 6 验证入口

| 层 | 命令 |
| --- | --- |
| CPU 协议层与检查 | `python -m pytest tests/ -q` |
| 运行环境 | `bash verify-runtime.sh` |
| 补丁部署（事务化） | `python tools/p2-apply-patch.py apply \| verify \| revert` |
| 证据索引（生成物，勿手改） | 复核 `python3 tools/pub-evidence-registry.py --check`；重生成 `python3 tools/pub-evidence-registry.py --write` |

证据索引的机械事实来自工作区与 `git ls-files`，**规则与登记**写在状态源 `reports/evidence-registry.json`，`reports/evidence-index.md` 是它的生成物（改状态源后必须重生成，勿手改）。复核项：索引与磁盘是否一致（过期即失败）、文档/配置里的 `evidence/...` 悬空引用、未经 `allow_duplicate` 声明的逐字节重复、登记项是否真的被 git 忽略。索引"树外登记"表里的件（§1.6 的 R3–R7）原件归档在数据盘 `/root/autodl-tmp/attnview-evidence-archive`，与仓内相对路径同构、逐字节一致；R8 的未验收件与 R9 的大件不入库、留在盘上并被 `.gitignore` 排除——两者都不随仓发行。

## 7 启用提交模板

```bash
git config commit.template .gitmessage    # 本仓已设置；新 clone 需各自执行一次
```

模板只提供骨架与提示，不替代 §1 的规则；`#` 开头的行在提交时会被 git 去掉。
