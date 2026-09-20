# 脱敏报告：规则、受影响文件与哈希钉住审计

日期：2026-09-21。执行者：agent（历史重写作业 pass 2 + 仓内工具复核）。
范围：首次公开发布前的**树内容脱敏**。规则源：`/root/autodl-tmp/sanitize-tree.sh`（历史重写时在 `git filter-branch --tree-filter` 里执行的那套规则）；仓内可复跑版本：`tools/pub-desensitize.py`。
受影响文件数据来源：`/root/autodl-tmp/pass2-hash-table.tsv`（逐文件改前/改后字节数与 sha256 的作业记录；本报告直接引用，未重算）。

**结论**

1. 三类内部标识在**工作树与全部 148 个 commit** 的文本内容里都已不存在：工具 `--check` 0 命中（exit `0`），全历史 `git grep` 0（§6）。
2. **31 个入库文件**被改写（§3），逐文件改前/改后字节数与 sha256 可对照；改写只做三条替换，没有改动任何数值与结论。
3. 脱敏同时打破了两类**已入库的哈希登记**：`reports/p0-model/evidence-index.md` 的 15 行 sha256 登记（§4.3）与 28 个文件的 29 处 commit 登记（§4.4）。这不是笔误，是重写的必然结果，处置见 §4.5。
4. `vllm-patch/manifest.json` 与 `evidence/p3-calib/run-*/manifest.json` 的钉**没有被打破**（§4.1、§4.2）；`/root/...` 工作路径按功能性默认值保留（§2）；§7 列出一项需要人工决策的范围外敏感项。

## 1. 为什么历史必须一起脱敏

只把 HEAD 上的标识清掉等于没清：`git log -p`、`git show <旧 commit>`、`git checkout <旧 tag>` 都能把旧字节原样取回，克隆又把全部对象一起带走。因此脱敏不是"在 HEAD 上跑一遍 sed"，而是用 `git filter-branch --tree-filter` 重写**所有 refs**（含 tag）。

| pass | 动作 | 影响 |
| --- | --- | --- |
| 1 | `--index-filter 'git rm --cached --ignore-unmatch AGENTS.md' -- --all` | `AGENTS.md` 从全历史消失 |
| 2 | `--tree-filter '/root/autodl-tmp/sanitize-tree.sh' --tag-name-filter cat -- --all` | 31 个文件的字节被改写（§3） |

两次重写后**所有 SHA 都变了**：`reports/git-history-map-20260921.txt` 的 142 行 old→new 无一相同。任何钉住旧 commit 的登记都因此失效（§4.4）。

`refs/original` 已删除（`git for-each-ref refs/original` 为空）——保留它，旧对象仍然可达，脱敏等于没做。

## 2. 规则表

工具 `RULES` 与 `sanitize-tree.sh` 的 3 条 sed 表达式逐条一致：

| # | 模式 | 占位符 | 说明 |
| --- | --- | --- | --- |
| 1 | `autodl-container-[a-z0-9]+-[a-z0-9]+` | `container-host` | 只吃两段 id：`autodl-container-<id>-<id>-storage` 这类带后缀的写法保留后缀（→ `container-host-storage`），后缀是卷命名而不是标识的一部分 |
| 2 | `GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}` | `GPU-<redacted>` | 固定 token：无编号、无映射表，因而不存在需要保管的"真实值 → 占位符"键；大小写敏感（与 sed 一致，大写 UUID 不匹配） |
| 3 | `/root/autodl-tmp/attnview[-]supervision` | `/path/to/supervision` | 内部协调目录；源码与文档统一用 `[-]` 书写，见下 |

**`[-]` 不是笔误。** 第 3 条的字面量一旦以裸形式出现在**被扫描的文件**里（工具自己的规则表、本文、检查命令），规则就会命中自己：`--check` 永远非 0，`--apply` 还会把规则表/文档改写成占位符（自毁）。`[-]` 是单字符类，语义与 `-` 完全等价，所以本文与工具统一用 `attnview[-]supervision` 指代该路径；同理，本文引用的全历史检查命令也把该分支写成 `attnview[-]supervision`（§6）。

**为什么 `/root/...` 路径默认保留。** 这些绝对路径是 `env.sh` / `install-runtime.sh` / `tools` 里的**活值**：部署指纹与本地可复现链条依赖准确的路径，改写会让脚本在本机失效。因此路径归一化放在显式开关之后，默认关闭：

| 规则 | 命中对象 | 占位符 |
| --- | --- | --- |
| `abs-attnview-root` | `/root/attnview`、`/root/autodl-tmp/attnview`（仓库自身绝对根） | `/path/to/attnview` |
| `abs-data-disk` | `/root/autodl-tmp`（数据盘） | `/path/to/data` |
| `abs-root-home` | 其余 `/root` 路径与裸 `/root` | `/path/to/home` |

三条按"先长后短"排列（前一条吃掉的区间后一条看不到）。`--include-paths` 下工具**跳过自身**：规则表与注释是策略文本、不是数据，若被自己的路径规则改写，第 3 条规则会失去真实字面量。默认三条规则仍然扫描自身，真实标识不会被漏掉。

### 工具行为

| 项 | 行为 |
| --- | --- |
| 默认动作 | `--check`：只读扫描，打印每文件命中计数与汇总；有命中 exit `1`，干净 exit `0` |
| 改写 | `--apply`：就地字节级替换（等价 `sed -i`），打印每文件改动数与改前/改后 sha256，并在写后自检（重扫必须 0 命中）；若被改写的文件曾被本仓其他文件登记过 sha256，会一并列出需要同步的登记处（§4.3 就是这类登记） |
| 路径规则 | `--include-paths`（默认关闭，见上） |
| 自定义集合 | `--paths <glob…>`；默认集合 = `git ls-files -z --cached --others --exclude-standard`（已跟踪 ∪ 未跟踪非忽略） |
| 跳过 | 二进制（含 NUL）与 > 10 MiB 的文件，计入跳过清单 |
| 退出码 | `0` 无命中 / `1` `--check` 有命中 / `2` 用法或路径错误 / `3` `--apply` 写盘或写后自检失败 |
| 幂等 | `--apply` 之后再 `--check` 必为 0 命中（占位符不含任何规则模式） |

```bash
python3 tools/pub-desensitize.py --check                             # 默认动作；0 命中 exit 0，有命中 exit 1
python3 tools/pub-desensitize.py --apply                             # 就地改写 + 前后 sha256
python3 tools/pub-desensitize.py --check --include-paths             # 连路径规则一起查（默认关闭）
python3 tools/pub-desensitize.py --check --paths 'evidence/**/*.md'  # 自定义文件集合
```

## 3. 受影响文件（31 个）

数据取自 `/root/autodl-tmp/pass2-hash-table.tsv`。列含义：`bytes_before` / `sha256_before` 是**脱敏前**的字节数与哈希（在 HEAD 上不可复现——它们只存在于重写前的字节里，这正是本表存在的意义），`bytes_after` / `sha256_after` 是 pass 2 产出的字节数与哈希。

| # | path | bytes_before | sha256_before | bytes_after | sha256_after |
| ---: | --- | ---: | --- | ---: | --- |
| 1 | `evidence/after-model/env-report-20260917-1910.md` | 86,971 | `739abdb065112bddd8e3b19ba47eb912f6ccade9daadcfebb282db19803a709c` | 86,901 | `83fff8f023aeccd5d7cd071cdb29f0bc71c0d8e985ababa00c2703ff96bfb06a` |
| 2 | `evidence/after/env-report-20260917-1637.md` | 86,901 | `4cf8e066ca9116040069312bba55568139ee39bda9ef490084b620470c6cffe7` | 86,831 | `af384bff870225c4a3e50d11a57e791492923589f05a71cd5b89797a1631980a` |
| 3 | `evidence/after/env-report-20260917-1655.md` | 86,925 | `5d3be6f45f830028a4751c0f3a2eeced8b84db6a13f11c9db65349e17b180edf` | 86,855 | `3605428293f62a78c4b2a0153a2f121f48e6d0758b8c343212902ffe06a73da4` |
| 4 | `evidence/after/env-report-20260917-1716.md` | 86,925 | `395353c9ccf94e39aa60e23fd638b8df357a4631e2a63ee8254e4f63106b43d7` | 86,855 | `8925202467a84dd8dc9ca583d63732402be8d793db3e1187256d913e191d283b` |
| 5 | `evidence/before/env-report-20260917-1559.md` | 86,675 | `adc4a09fde231161d508e7cf346099f4d3b9607e3223de294719d78fba4c73d7` | 86,605 | `5eda90017629ba69c0c9e8dd23a2dc258c5627a871f64b8e16d5980b42e8b4ae` |
| 6 | `evidence/p0-model/e0-baseline.txt` | 1,153 | `8bd30dba9e4ec8a6bf0b4e4c63589e953aca17cb3104f1c88a7c8e3896e79f77` | 1,131 | `1abd77b6f767afc9ce525e743291bfc074f561c6697d3e06a3e8ba62c5a357c5` |
| 7 | `evidence/p3-calib-local-review-20260918/metadata-retry.chrome.json` | 34,112 | `452f9e1b524f58c5b5c6aed833b79917ab9ee93429872556d4f9ff2a4c5df2f9` | 34,090 | `4e610ad31539b932068feb9efd2c0fa220fad98adfdc5a8dd0d444c638bfe0af` |
| 8 | `evidence/p3-calib/metadata-probe-gate2.chrome.json` | 32,952 | `939f2e91f09992ba3c62aed99d18435c3688b0807587c466d1c64e9fcbbb53e6` | 32,930 | `332116a2e839abe85509bd554474984e0ed310aa97917fe945a1d31f8204ebe0` |
| 9 | `evidence/p3-calib/metadata-probe-gate3.chrome.json` | 34,096 | `f6dfe2375f8bb8ff446dc8fde1c2c989d351760eca522c03a078863742a536d7` | 34,074 | `426c8d26c4590f057e489de3bcf44319b3bba375cc3c7e3a897de017e878f5fa` |
| 10 | `evidence/p3-calib/metadata-probe-gate4.chrome.json` | 34,082 | `79f434b1366593ac11613a04df0b6562c7a95fa5cf50bf8e504229b09dc42713` | 34,060 | `d2be21748e325117a33c6bc9bcbd26736652a0ce322e5c4c5dbb0ff99be61065` |
| 11 | `evidence/p3-calib/metadata-probe-gate5.chrome.json` | 34,092 | `4ba7c0570146112c9197f91e27763ff13ffc35793d35874e3d1e8af1a4550372` | 34,070 | `524545a739f399d3d1c100398bc5fe6f503b5bd8372d977f2b37d0300cac1b78` |
| 12 | `evidence/post-expansion.txt` | 2,238 | `ee8dd8d51895f4480d89f2bf93832068d962be59598abf2a338b87dd921c70ce` | 2,216 | `1743027caa0130df2a9f55108ef708c6c38cdb2d471003121f984ab8ff4ae449` |
| 13 | `logs/console-install.log` | 2,443 | `dfb21eec4d637bdfbade4bdf39592a20c6c27b7c3b8859ccdcb50bfdf2dacd75` | 2,421 | `189d9b27c5e5ba7178ba12bc7e093e3ff8addde3c794dc969a4a8ec6122f6d57` |
| 14 | `logs/install-runtime-20260917-155930.log` | 568 | `eab4ee2f92cd15ea566082d6da870a13722fa033ed5f1b7aec8bef3191c5d015` | 546 | `05fa3fbdf86fecce46c6d56e1795f4c63241c461dd0b54c8a0bfe48f8b238c6c` |
| 15 | `logs/install-runtime-20260917-160026.log` | 631 | `eb3d69f62c610482080e2774abe135b62f6923badb6f1988ba3df85b41972f7d` | 609 | `8205e1959176798d6e262dd9d2d47ecaf40bf68cdcb4aad1cf0fbf4e56a430f4` |
| 16 | `logs/install-runtime-20260917-160325.log` | 10,932 | `5f117102a8a757acabfea7c23867bf58e1b0737929fe277ebd920084cae819ca` | 10,910 | `c4d38fc3afa671b1065fca8e60fc7537350d5785792697ac3b8910a97df5dc24` |
| 17 | `logs/install-runtime-20260917-163051.log` | 2,443 | `dfb21eec4d637bdfbade4bdf39592a20c6c27b7c3b8859ccdcb50bfdf2dacd75` | 2,421 | `189d9b27c5e5ba7178ba12bc7e093e3ff8addde3c794dc969a4a8ec6122f6d57` |
| 18 | `logs/serve-e4-attempt1-offline-resolve-fail.log` | 6,678 | `273f307a19630bdd8d6cdcc6bcd2a3b0cc4d6a9564fafd8fd211cd587666ade1` | 6,656 | `21b1868f07fc01ce842d3525a6a5d6f126b2823422055060d7a487d96d36f502` |
| 19 | `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | 36,272 | `47991d5814d648555e0aa5d5e0f02aa17207eefd78975a84c4d17a4c9c8e610d` | 36,250 | `6c0717b64d88fe212cd5879437af4cf9deb5cd76d33254d1cef0f3b913448b0a` |
| 20 | `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | 88,720 | `5511c07f40217a0473662df4ab52b1e01c5e2d1f8c459c4931741e51d57a2877` | 88,698 | `3296f64bfb58f44f58a32930ec1a8ec175db0a34ca1d2046009f3f5e5d805736` |
| 21 | `logs/serve-e4.log` | 34,463 | `7d12122512d8d970b7418ef7665639b44e6c1f8d3f41a3fbe5555cc03bd39898` | 34,441 | `571a01c89c29e0045f80d6a360076da27a41e15f07e054a1b66c548275b9b3c9` |
| 22 | `logs/serve-e4b.log` | 33,937 | `f363f5ebc105e32d11134b48385a9f835d6c3682dc105cc94a777836469eb856` | 33,915 | `3d906c46a2432b142a659aaa93ee0dc1ad96fe15a7d55e12ee69effcd22d014e` |
| 23 | `logs/serve-e4c.log` | 28,667 | `ffe9b7d8eebc27fee80f01fc497c3bdc62c4492c9316a71aa163ecae87d7e688` | 28,645 | `f5b852f90ad14786d6ab4a03d2e34eb8dfd6a6bffdcf88e70e5f3855a74ed182` |
| 24 | `logs/serve-e6.log` | 33,642 | `9d18826928de284bcff9a00e9fc4d0ba826d86b660ef9705b69c2ff4c99413f3` | 33,620 | `3021d17d8b9b98e2a8f6e2503c136e04f7e2dec6c578be0c71efdef233eef7e7` |
| 25 | `logs/serve-e6b.log` | 33,974 | `0d018b91febfcc6512efc9effe79a1db49161ed6bde526ecc5132d368d7840fe` | 33,952 | `2cf48986ff32803bf97c4b033ecd741147793a9ec861fec126e18f2d075bf211` |
| 26 | `reports/environment-report.md` | 14,210 | `cb72f41443a1e4c40656a7380659000526dcc5436bbdd2095f2edf6bb74a9f73` | 14,188 | `4c0c33200e244a9f15bfe8b1d67b00324936768ef5b060edd06256041afecec8` |
| 27 | `reports/evidence-index.md` | 6,660 | `157cddeeca29ba6c14319d814c7c776b51b4321ea70443139733576e6ea94b0b` | 6,638 | `3f559beb759dd235eda0cab8906050539d921715265e863fa55218d3f67292ef` |
| 28 | `reports/p0-model/model-report.md` | 30,353 | `2b949b3f7d40644900b38d99c14f903a66308d0404ca00cd8a277977b176b377` | 30,331 | `6a044950a315d5283315a74abe6d622a8b6f069cd308aeb901659783945cd3b8` |
| 29 | `reports/p0-model/serve-command.sh` | 2,836 | `adb3d2415a676aec132a8fd23ff9496c1e2144841b0e5e0ea4f91336f662d979` | 2,814 | `b6aeec54ad346f88801c6e47c6adf9e56437f2aec1867659ec30ccc111c6ad52` |
| 30 | `reports/p1-cpu/session-closeout.md` | 10,722 | `2672d08c9ca4d6d7af3b81ddf7de0314dcf6dbe9bdee4577441dbc9669d1ded4` | 10,688 | `5d4c21d96aa57dc5bdf08b2a23b0f537c89a063ae64fb692fc5902629124ef49` |
| 31 | `reports/publication-checklist.md` | 8,471 | `8945a251d53ff8da75df69512affb737cf169c33fbbdc7312864202acdc18701` | 8,454 | `764e6c22964d1f054206c9260666ef055f7fbe67ecb3c4ff3814736a9727b5a3` † |

校验（实测）：30/31 行的 `sha256_after` 与**当前 HEAD 字节**一致。

† 第 31 行 `reports/publication-checklist.md` 在脱敏后又被其属主继续编辑（mtime 2026-09-21 00:56 之后），所以它的当前工作区值与表中的 `sha256_after` 不同：**表中值对应重写后即刻状态，当前值以工作区为准**。这是后续编辑，不是脱敏口径差异；其余 30 行不受影响。

## 4. 哈希钉住审计

判定标准只有一条：**被钉的文件是否在 §3 的 31 个文件里**。审计覆盖全部已跟踪文本文件（360 个）与未跟踪非忽略文件，做法是把每个 64 位十六进制 token 与 §3 的改前/改后哈希集合比对。

### 4.1 `vllm-patch/manifest.json` —— 未被打破（17/17）

| 字段 | 钉住的对象 | 数量 | 校验 |
| --- | --- | ---: | --- |
| `package_files[].sha256` | `src/attnview/{__init__,decode,parser,readview,gpukv,state,step_plan}.py` | 7 | 全部 MATCH |
| `edits[].post_sha256` | `vllm-patch/patched/**`（8 个文件） | 8 | 全部 MATCH |
| `new_files[].sha256` | `vllm-patch/files/vllm/**` | 2 | 全部 MATCH |
| `edits[].pre_sha256` | vLLM checkout（仓外，`pin_commit 98dff2a8…`） | 8 | 不适用：不在本仓，重写未触碰 |

这 17 个入库对象都不含三类标识（命中数 0），因此**逐一字节未变**。旁证：`tools/p2-apply-patch.py` 的预检本来就要求 `sha256(src) == post`，这些文件按设计不可改。

### 4.2 `evidence/p3-calib/run-*/manifest.json` —— 未被打破，另有一处既存漂移

| 字段 | 钉住的对象 | 结论 |
| --- | --- | --- |
| `source.source_snapshot_sha256` | 同 run 的 `source/p2-calib-run.py` 快照 | **6/6 MATCH**（快照在仓内，钉仍有效） |
| `source.script_sha256` | `tools/p2-calib-run.py` | **MISMATCH，但与脱敏无关**：钉的是各 run 当时的脚本版本（`8758db24…`、`0f2040a0…`、`e2d5d044…`），该文件当前为 `8f00cde8…`——脚本在 run 之后被继续开发（这些版本作为历史 blob 仍在）；该文件命中数 0，重写前后字节相同 |

这条漂移符合 manifest 自己声明的口径（"运行期不得覆写本脚本；结束后如需修改，请用新输出目录起新 run"）：权威钉是随 run 冻结的 `source/` 快照，它仍然成立。

### 4.3 真正被打破的钉：`reports/p0-model/evidence-index.md` 的 15 行

该文件是**机器生成的证据清单**（`| 文件 | 字节 | sha256 |`，90 行数据行）。其中 15 行登记的是含标识的文件——全部落在 §3 的 31 个文件里——所以那 15 行的字节数与 sha256 都指脱敏**前**的字节，现已失效。它自身命中数 0，因此字节未变、行号稳定。

| 行 | 被登记的文件 | 记录值（脱敏前，已失效） | 应改为（脱敏后） |
| ---: | --- | --- | --- |
| L110 | `evidence/p0-model/e0-baseline.txt` | `8bd30dba9e4ec8a6…`／1,153 B | `1abd77b6f767afc9…`／1,131 B |
| L153 | `logs/serve-e4.log` | `7d12122512d8d970…`／34,463 B | `571a01c89c29e004…`／34,441 B |
| L154 | `logs/serve-e4b.log` | `f363f5ebc105e32d…`／33,937 B | `3d906c46a2432b14…`／33,915 B |
| L155 | `logs/serve-e4c.log` | `ffe9b7d8eebc27fe…`／28,667 B | `f5b852f90ad14786…`／28,645 B |
| L156 | `logs/serve-e6.log` | `9d18826928de284b…`／33,642 B | `3021d17d8b9b98e2…`／33,620 B |
| L157 | `logs/serve-e6b.log` | `0d018b91febfcc65…`／33,974 B | `2cf48986ff32803b…`／33,952 B |
| L158 | `logs/serve-e4-attempt1-offline-resolve-fail.log` | `273f307a19630bdd…`／6,678 B | `21b1868f07fc01ce…`／6,656 B |
| L159 | `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | `47991d5814d64855…`／36,272 B | `6c0717b64d88fe21…`／36,250 B |
| L160 | `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | `5511c07f40217a04…`／88,720 B | `3296f64bfb58f44f…`／88,698 B |
| L161 | `evidence/after-model/env-report-20260917-1910.md` | `739abdb065112bdd…`／86,971 B | `83fff8f023aeccd5…`／86,901 B |
| L163 | `evidence/after/env-report-20260917-1637.md` | `4cf8e066ca911604…`／86,901 B | `af384bff870225c4…`／86,831 B |
| L164 | `evidence/after/env-report-20260917-1655.md` | `5d3be6f45f830028…`／86,925 B | `3605428293f62a78…`／86,855 B |
| L165 | `evidence/after/env-report-20260917-1716.md` | `395353c9ccf94e39…`／86,925 B | `8925202467a84dd8…`／86,855 B |
| L187 | `reports/p0-model/model-report.md` | `2b949b3f7d406449…`／30,353 B | `6a044950a315d528…`／30,331 B |
| L189 | `reports/p0-model/serve-command.sh` | `adb3d2415a676aec…`／2,836 B | `b6aeec54ad346f88…`／2,814 B |

可直接替换的行（新值取自同一份作业记录，非人工誊写）：

```text
| `evidence/p0-model/e0-baseline.txt` | 1,131 | `1abd77b6f767afc9ce525e743291bfc074f561c6697d3e06a3e8ba62c5a357c5` |
| `logs/serve-e4.log` | 34,441 | `571a01c89c29e0045f80d6a360076da27a41e15f07e054a1b66c548275b9b3c9` |
| `logs/serve-e4b.log` | 33,915 | `3d906c46a2432b142a659aaa93ee0dc1ad96fe15a7d55e12ee69effcd22d014e` |
| `logs/serve-e4c.log` | 28,645 | `f5b852f90ad14786d6ab4a03d2e34eb8dfd6a6bffdcf88e70e5f3855a74ed182` |
| `logs/serve-e6.log` | 33,620 | `3021d17d8b9b98e2a8f6e2503c136e04f7e2dec6c578be0c71efdef233eef7e7` |
| `logs/serve-e6b.log` | 33,952 | `2cf48986ff32803bf97c4b033ecd741147793a9ec861fec126e18f2d075bf211` |
| `logs/serve-e4-attempt1-offline-resolve-fail.log` | 6,656 | `21b1868f07fc01ce842d3525a6a5d6f126b2823422055060d7a487d96d36f502` |
| `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | 36,250 | `6c0717b64d88fe212cd5879437af4cf9deb5cd76d33254d1cef0f3b913448b0a` |
| `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | 88,698 | `3296f64bfb58f44f58a32930ec1a8ec175db0a34ca1d2046009f3f5e5d805736` |
| `evidence/after-model/env-report-20260917-1910.md` | 86,901 | `83fff8f023aeccd5d7cd071cdb29f0bc71c0d8e985ababa00c2703ff96bfb06a` |
| `evidence/after/env-report-20260917-1637.md` | 86,831 | `af384bff870225c4a3e50d11a57e791492923589f05a71cd5b89797a1631980a` |
| `evidence/after/env-report-20260917-1655.md` | 86,855 | `3605428293f62a78c4b2a0153a2f121f48e6d0758b8c343212902ffe06a73da4` |
| `evidence/after/env-report-20260917-1716.md` | 86,855 | `8925202467a84dd8dc9ca583d63732402be8d793db3e1187256d913e191d283b` |
| `reports/p0-model/model-report.md` | 30,331 | `6a044950a315d5283315a74abe6d622a8b6f069cd308aeb901659783945cd3b8` |
| `reports/p0-model/serve-command.sh` | 2,814 | `b6aeec54ad346f88801c6e47c6adf9e56437f2aec1867659ec30ccc111c6ad52` |
```

方法论提醒：这一处是**全文本反查**才发现的——只扫 `*.json` / `*.jsonl` / `*.txt` 会漏掉它，因为登记写在 `.md` 里。

### 4.4 commit 登记：28 个文件 29 处

下列登记钉的是**重写前的 commit**，重写后已不可解析（`git cat-file -t <旧 sha>` 报 not a valid object）。完整 40 位对照见 `reports/git-history-map-20260921.txt`（142 行 old→new）。

| 文件 | 字段 | 旧 commit（已失效） | 新 commit（映射后） |
| --- | --- | --- | --- |
| `configs/p1-gpu/read-view-check-v2.json` | `cpu_stage_accepted_commit` | `54933ff81517969ffe7674d5842f742c78cdf23b` | `ca85482534e0c809754c83719d8d340cc9a8834e` |
| `configs/p1-gpu/read-view-check-v3.json` | `cpu_stage_accepted_commit` | `54933ff81517969ffe7674d5842f742c78cdf23b` | `ca85482534e0c809754c83719d8d340cc9a8834e` |
| `configs/p1-gpu/read-view-check.json` | `cpu_stage_accepted_commit` | `54933ff81517969ffe7674d5842f742c78cdf23b` | `ca85482534e0c809754c83719d8d340cc9a8834e` |
| `evidence/p1-gpu-local-review-20260918-1011/run-manifest.json` | `head_commit` | `d0a970656ae76754fd5ef6d6b4da7e0e802d44d5` | `9e20013d041501310e7cd1ae878654dfe752159b` |
| `evidence/p1-gpu-v4/run-manifest.json` | `head_commit` | `e9134526d2d07a5c5a2ffec68be8e040258b6e4b` | `10df1c2a74f6a9bb6bdc09ef9eb250ee3a3a9478` |
| `evidence/p1-gpu-v5/run-manifest.json` | `head_commit` | `afaf24fca1358970f9960e590c5947ed71599ff4` | `29b817ce5bd47e97d3f196af78c6847aa04c78a4` |
| `evidence/p1-gpu-v6/control-manifest.json` | `head_commit` | `38982cf773790ab936db120e0dd0a0c52990b870` | `c3cc6650d53881fe2e672bfd143ea828a5f12a43` |
| `evidence/p1-gpu-v6/evidence-index.md` | `HEAD` | `38982cf773790ab936db120e0dd0a0c52990b870` | `c3cc6650d53881fe2e672bfd143ea828a5f12a43` |
| `evidence/p1-gpu-v6/run-manifest.json` | `head_commit` | `38982cf773790ab936db120e0dd0a0c52990b870` | `c3cc6650d53881fe2e672bfd143ea828a5f12a43` |
| `evidence/p2-single-local-review-r1-20260918/findings.json` | `reviewed_head` | `fc593203c9f03fad435cc28aa8cae7a02ee68599` | `a6bcce6a614fcc45d072b6f7adee7e0283ad2020` |
| `evidence/p2-single-local-review-r2-20260918/deployment-unknown-state.json` | `head` | `a208ed0b84f8d41d23716d703fea75b0d91c7f2c` | `ff9529504a9d09ceea27befb0e4bdeb718793fff` |
| `evidence/p3-calib-local-review-20260918/final-v5-artifacts.json` | `head`（L7、L21） | `757edddb3d26ab130634dc487b3ac494e8fa9d68` | `a9fff6ab6cb126716500604df7633f2cbd1508f5` |
| `evidence/p3-calib-local-review-20260918/output-mode-findings.json` | `head` | `cb57a5d710636e38189f66bb1f5cff0491eadbf5` | `429f4a2ef0dafdd2c9ba3c8d82f016b0c32f2468` |
| `evidence/p3-calib-local-review-20260918/r2-note-recheck.json` | `head` | `cb57a5d710636e38189f66bb1f5cff0491eadbf5` | `429f4a2ef0dafdd2c9ba3c8d82f016b0c32f2468` |
| `evidence/p3-calib/gate-record.json` | `head` | `eceb695dedb4cb665d4603340736f8970d20f748` | `50fac5602f65274e509f865deff53ddb91847d9d` |
| `evidence/p3-calib/run-disabled-4/manifest.json` | `head` | `c65d8e52907f62596d43ba9b4fceb16196650a76` | `c76ef5cfe62bded7aac8f27a1502140727c761d3` |
| `evidence/p3-calib/run-disabled-5/manifest.json` | `head` | `757edddb3d26ab130634dc487b3ac494e8fa9d68` | `a9fff6ab6cb126716500604df7633f2cbd1508f5` |
| `evidence/p3-calib/run-global-4/manifest.json` | `head` | `c65d8e52907f62596d43ba9b4fceb16196650a76` | `c76ef5cfe62bded7aac8f27a1502140727c761d3` |
| `evidence/p3-calib/run-global-5/manifest.json` | `head` | `757edddb3d26ab130634dc487b3ac494e8fa9d68` | `a9fff6ab6cb126716500604df7633f2cbd1508f5` |
| `evidence/p3-calib/run-original-2a/manifest.json` | `head` | `cb57a5d710636e38189f66bb1f5cff0491eadbf5` | `429f4a2ef0dafdd2c9ba3c8d82f016b0c32f2468` |
| `evidence/p3-calib/run-original-3a/manifest.json` | `head` | `07de025b662b5ee3af4cd2d504562c6fab84ea76` | `0668e288046ea2d2bc22287657b4e309643b9d2b` |
| `evidence/p3-calib/run-original-3b/manifest.json` | `head` | `07de025b662b5ee3af4cd2d504562c6fab84ea76` | `0668e288046ea2d2bc22287657b4e309643b9d2b` |
| `evidence/p3-calib/run-original-r1-1/manifest.json` | `head` | `3d3f994c1bce228e6885e04fc8d001db2b46ffb9` | `56cdd90124390907f409816f8a34a160154b2fcd` |
| `evidence/p3-calib/run-original-r1-2/manifest.json` | `head` | `291dcbbd1b419588064904120863095a0e003601` | `b6d8756ca2fa11a9b27a6f1ea676c3985c066bdf` |
| `evidence/p3-calib/run-vanilla-1/manifest.json` | `head` | `eceb695dedb4cb665d4603340736f8970d20f748` | `50fac5602f65274e509f865deff53ddb91847d9d` |
| `evidence/p3-masked-prep-local-review-20260920/manifest.json` | `head` | `e5ca9aeba07038bc572f589f2effc4988487fd92` | `205fa013dbc4c5ef57ed7bdd09b5fbb9a6d525f6` |
| `evidence/p3-masked-ref-local-review-20260920/preintegration-findings.json` | `head` | `9118dafc6832199e281b092b1a3af15f600c6010` | `61a905cb689e8580e6becc5c6291ec0c9fbf884d` |
| `reports/p1-cpu/stage-03-report.md` | `初交 commit` | `cb7bce19de6f2b71bf0fcf03b9b4a432a88d035a` | `73555a3b8cd6b045da08c8e5e9f82d6ad5bfd5ff` |

这两类 commit 引用**不受影响**（不属于本仓历史）：`vllm-patch/manifest.json` 的 `pin_commit`（vLLM `98dff2a8…`）与 `vllm-patch/patched/**` 注释里的上游 diff 哈希；`reports/p1-cpu/stage-03-report.md` 里的素材仓快照 commit 属于另一个仓。

### 4.5 处置建议

| 项 | 建议 | 归属 |
| --- | --- | --- |
| §4.3 的 15 行 sha256 登记 | 逐行替换为脱敏后值，或在清单头部声明"本清单 sha256 为脱敏前值，见本报告" | `reports/p0-model/evidence-index.md` 的所有者 |
| §4.4 的 29 处 commit 登记 | 保持原样（它们是 run 当时的 HEAD 事实），读者用 `reports/git-history-map-20260921.txt` 对照即可；若要更显眼，可在对应清单加一行指向映射表 | 各清单所有者 |
| §4.2 的 `source.script_sha256` 漂移 | 登记为既存漂移；不改历史 run 的 manifest（权威钉是 `source/` 快照） | 无需动作 |

## 5. 跳过与边界

| 项 | 工具行为 | 历史重写侧 | 本仓实测 |
| --- | --- | --- | --- |
| 二进制（含 NUL） | 跳过并计数 | `grep -I` 同样不选二进制 | 363 个文件中 0 个 |
| > 10 MiB | 跳过并计数 | sed 不设大小限制（会改写） | 0 个文件超过 10 MiB |
| 非 UTF-8 文本 | 跳过（`not-utf8`） | sed 会改写 | 0 个文件 |
| `--include-paths` 下的工具自身 | 跳过（`self(policy)`） | 不适用（路径规则不在重写规则内） | 1 个 |

**等价性**：把 3 条 sed 表达式与工具规则分别作用在合成的模式样例（两段 id 的主机名及其 `-storage` 变体、8-4-4-4-12 的 UUID 与小写/大写变体、协调目录路径、一行多处命中、UTF-8 文本、CRLF）上，逐字节比较——与 tree-filter 的 `grep -lIZ | sed -i` 管线结果一致；唯一差异是 > 10 MiB 的文件：工具按策略跳过，管线会改写（本仓无此类文件）。

**原始记录原则**：脱敏只做上面三条替换，不改数值与结论；`evidence/**` 里的路径字符串除命中三条规则的部分外保持字节不变。

## 6. 可重跑证据

工作树（含未跟踪文件；实测时点为本报告落地后，此后新落地的文件会让计数增长）：

```console
$ python3 tools/pub-desensitize.py --check
扫描 365 个文本文件；规则集 3/6 条（默认三条：主机名 / GPU UUID / 协调目录）；模式 check
合计 0 处 / 0 个文件
[check] 0 命中
$ echo $?
0
```

含路径规则（预期非 0：路径是刻意保留的功能性默认值）：

```console
$ python3 tools/pub-desensitize.py --check --include-paths
扫描 364 个文本文件，跳过 1 个；规则集 6/6 条（含绝对路径）；模式 check
  abs-attnview-root    命中   940  文件 106  去重值 191
  abs-data-disk        命中    27  文件  14  去重值   5
  abs-root-home        命中    87  文件  24  去重值  27
合计 1054 处 / 109 个文件
[check] 存在命中；用 --apply 就地改写
$ echo $?
1
```

全历史（`refs/original` 已删除，故旧对象不可达；`git rev-list --all` = 148 个 commit）。注意命令的第三个分支写成 `attnview[-]supervision`：字符类在这里等价于 `-`，这样命令自身不会命中自己（§2）：

```console
$ git grep -lIE 'autodl-container-[a-z0-9]+-[a-z0-9]+|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|/root/autodl-tmp/attnview[-]supervision' $(git rev-list --all) | wc -l
0
```

正控（证明 harness 真的在工作，不是空跑）：同一条命令下 `attnview` → 18718、`container-host` → 3933、`GPU-<redacted>` → 739。

`git grep ... HEAD` 会命中 `reports/publication-checklist.md:115`，但那一行是**检查命令自身的字面文本**（写的是 `autodl-container` 而非 `autodl-container-<id>-<id>`），不满足三条模式中的任何一条（实测 0），所以上面的 0 成立。

**提交前必须重跑 `--check`**：未跟踪文件也会被扫描，任何新落地的文件都可能带回命中。

## 7. 范围外清点（未列入规则表）

在入库文本文件上清点（正则形态匹配，非人工阅读；实测时点同 §6）。结论：**本轮全部不处理**；唯一曾待决策的项（平台内部地址）已按下表决策为已知保留项。

| 类别 | 实测 | 说明 |
| --- | ---: | --- |
| 私人邮箱 | 0 | 唯一邮箱样命中是 `git@github.com`（SSH 地址，非个人邮箱） |
| `gh[p]_` / `github[_]pat_` / `hf_` / `sk-` / `AKIA` / 私钥块 / Slack token | 0 | 全部 0 |
| MAC 地址 | 0 | — |
| IPv4 形态命中 | 216 | 74 处是 `127.0.0.1`（回环）；124 处是**版本号**（如 `nvidia-*==<x.y.z.w>`、`cuda-toolkit==…`）；18 处落在私网段，其中 13 处仍是版本号（`nvidia-curand==10.4.0.35`），5 处是平台内部地址（下方单列；把本报告自身的 2 处提及算上共 7 处） |
| IPv4 中的真实公开地址 | 3 | `evidence/after/network-tools.txt` 里 DNS 解析结果：`github.com`、`hf-mirror.com`、`objects.githubusercontent.com`（公开服务地址，不构成泄露） |

### 平台内部地址 `172.30.54.6`：已知保留项（本轮不脱敏）

| 项 | 内容 |
| --- | --- |
| 实测 | **7 处**：5 个 `evidence/{before,after}/env-report-*.md` 各 1 处（`df`/mount 输出里的 `172.30.54.6:/data … /autodl-pub`），加本报告 2 处（记录该决策本身） |
| 为什么保留 | RFC1918 私网地址、不可路由；不是身份标识（不是主机名/UUID/账号），与已保留的功能性 `/root/...` 工作路径同类。另外：本轮重跑 `git filter-branch` 会打断正在飞的实现工作（重写要求干净工作区，暂存/还原有覆盖风险），收益与风险不成比例 |
| 不处理的方式 | 现有三条规则不覆盖它，工具与报告默认都不改；本节即登记 |
| 将来启用时的确切规则 | 在 `sanitize-tree.sh` 的 sed 里**按字面地址**加一条：`s\|172\.30\.54\.6\|<internal-host>\|g`（转义点号）。**不要**用通用私网段正则：`pip_freeze` 等文件里的版本号（如 `nvidia-curand==10.4.0.35`）会被误伤 |
| 启用后怎么验 | 重跑 tree-filter 后，用 §6 的全历史命令加一个 `172\.30\.54\.6` 分支复查为 0；随后重跑 `tools/pub-desensitize.py --check` 确认工作树仍为 0 命中 |

## 8. 已知保留项（不再处理）

| 项 | 为什么不处理 |
| --- | --- |
| `SUP-nnn` 工作单编号 | 编号只有在协调目录（已改写为 `/path/to/supervision`）的上下文里才构成可达引用；无协调目录时它不指向任何可访问资源 |
| `/root/autodl-tmp/attnview`、`/root/autodl-tmp`、`/root` 等工作路径 | 功能性活值（`env.sh` / `install-runtime.sh` / `tools`）：改写会打断本地可复现链条与部署指纹。需要面向公开产物归一化时用 `--include-paths`（§2） |
| `material/attnview` 素材仓、vLLM checkout 的仓外引用 | 不在本仓历史内；两次重写只作用于本仓 |
| 平台内部地址 `172.30.54.6`（7 处，见 §7） | RFC1918、不可路由、非身份标识；与功能性 `/root/...` 路径同类。将来要处理时的确切规则与验证方式见 §7（本轮不重跑历史：需干净工作区，会打断在飞的实现工作） |
| 本文与检查命令里的 `attnview[-]supervision` 写法 | 是刻意的自指规避（§2），语义与裸字面量等价 |

## 9. 附：本次审计用到的命令

```bash
python3 tools/pub-desensitize.py --check                              # 工作树，默认三条规则
python3 tools/pub-desensitize.py --check --include-paths              # 含路径规则
python3 tools/pub-desensitize.py --check --paths 'evidence/**/*.md'   # 自定义集合（本例 12 个文件）
git grep -lIE 'autodl-container-[a-z0-9]+-[a-z0-9]+|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|/root/autodl-tmp/attnview[-]supervision' $(git rev-list --all) | wc -l
git for-each-ref refs/original | wc -l                                 # 必须为 0
```
