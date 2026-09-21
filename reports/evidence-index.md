# 证据索引（生成物，勿手改）

生成器：`python3 tools/pub-evidence-registry.py --write`；规则与登记：`reports/evidence-registry.json`。
本表覆盖 `evidence/` 与 `logs/`。机械事实（路径/字节/sha256/状态）一律现算，复核：`python3 tools/pub-evidence-registry.py --check`。

## 发布规则

- **R1** 交付依据入库：被结论引用的 run 文本层（manifest / summary / arm / cleanup / force / engine-traces / error.txt / run 日志 / 源快照）入库。
- **R2** 论证性失败件入库：失败尝试的目录与日志必须保留，不删、不改写。
- **R3** 独立校验分两类：被结论引用的独立复算件入库；由监督/复核流程产出的过程记录（`*local-review*`）只登记 path + size + sha256，原件留归档目录（`registered_removed` 的 `independent-review`）。
- **R4** 被取代代次只登记：同一实验的早期代次（结论已撤回或口径未定稿）字节不入库，只登记 path + size + sha256，原始字节归档在数据盘；撤回理由记在对应阶段的报告里，状态与哈希见 reports/evidence-index.md 的树外登记表。
- **R5** 中间快照只登记：同一状态的多份刷新（环境探针等）只留首末两份，其余登记。
- **R6** 明细只登记：逐位置/逐比较的原始明细以聚合段入库、明细只登记（聚合段自带明细的 size + sha256）；失败 run 的部分结构 dump 同样只登记，失败记录本身按 R2 入库。
- **R7** 逐字节重复只保留一份（≥16 KiB 才值得摘出，小文件保留并在索引里标注同哈希）。例外：run 自带的源码快照是按协议冻结的逐 run provenance，允许重复，由 allow_duplicate 声明。
- **R8** 未验收件不入库：尚未通过阶段验收的运行产物只登记为 pending，验收后按 R1 的 run 文本层入库。
- **R9** 大件不入库：*.npz / *.npy / *.pt / *.bin / *.safetensors / *.gguf 与 >10 MB 的文件只登记 size + sha256；与任何文档逐条绑定的成员逐个登记，其余按组登记（glob + 组内文件数 + 合计字节）。

## 状态表

| status | 含义 |
| --- | --- |
| `delivered` | 随仓发行 |
| `independent-review` | 复核过程记录（树外登记） |
| `superseded` | 被取代代次（树外登记） |
| `snapshot` | 中间快照（树外登记） |
| `detail-only` | 明细只登记（树外登记） |
| `duplicate` | 重复副本（树外登记） |
| `unclassified` | 未分类——门禁失败 |
| `pending` | 工作区未跟踪——入库前先分类 |

树内产物 **356 个 / 34,209,639 B**；树外登记 **78 个**；在盘未跟踪 **0 个**。
树外登记件的原件归档在 `/root/autodl-tmp/attnview-evidence-archive`（远端数据盘），字节与 sha256 以本表为准；仓内不发行其字节。

## 树内：随仓发行（356 个 / 34,209,639 B）

| 文件 | 字节 | sha256 | 说明 |
| --- | ---: | --- | --- |
| `evidence/after/cuda_tensor.txt` | 252 | `e89655c964f0a4a24b379b58b5af458ad6a8cc4ea3fd854a0acec9547de6bb3c` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/env-report-20260917-1716.md` | 86,855 | `8925202467a84dd8dc9ca583d63732402be8d793db3e1187256d913e191d283b` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/import_versions.txt` | 587 | `61b623a6b1dba2e0c08d4afd8389fc8158d5266fe2e1017879f219028631e589` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/network-tools.txt` | 1,150 | `e3f026504fa94ef9009a21078e89d65e0b1df3c432fde69ff5f78cd05d8bc207` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/nvcc.txt` | 208 | `d5e3a195eebabdcebf990838dda4ca273b10cf09c7e01080333f03d60cc1db56` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/pip_check.txt` | 30 | `9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/pip_freeze.txt` | 4,118 | `33585d37b124ca2dc86518d8c6606a03795ce3f21bd8e7e18139871a5d34a07f` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/python_version.txt` | 15 | `55ae85cf4bdb38743edbcd53ea68ff36511997ec6c21b1e83d8bebc939bf056b` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/after/vllm_cli.txt` | 1,162 | `40960bd7c950d219c9837b46a703b4dacbf625530008c6aff6d9ea9e874307e0` | 阶段 01 环境探针最终态与 verify-runtime.sh 输出 |
| `evidence/before/env-report-20260917-1559.md` | 86,605 | `5eda90017629ba69c0c9e8dd23a2dc258c5627a871f64b8e16d5980b42e8b4ae` | 阶段 01 环境探针基线段（未经 env.sh） |
| `evidence/cuda-setup.log` | 1,382 | `885a838e7724f9d31ac90b9e0801f35b247fba1a2c2f3e7258c34034479812ad` | 环境与迁移记录 |
| `evidence/git-message-reword-20260921/log-after.txt` | 671 | `fe02dd27b5d6f4be2c4c0c124e3ac967a52061426884469b71e4f9d79583b9b0` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/log-before.txt` | 682 | `9e27fbd0c7436b230140d6d73f9edca7e94dcb466a780c99a5176446fc852dd7` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/mapping.json` | 2,758 | `f33390860217288555a092d80d5157c14a2ecb6f781a253c22c2d11439a39cae` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/result.json` | 199 | `669ec59f17e3832c54a10c396ef1e1026075c330c4f018ddb2fda46eccaf13f3` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/status-after.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/status-before.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/trees-after.txt` | 369 | `7ae324509c41667b8611fd56a54fcc8241304a54eae0ba305a749d7e32a8a1fa` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/git-message-reword-20260921/trees-before.txt` | 369 | `7ae324509c41667b8611fd56a54fcc8241304a54eae0ba305a749d7e32a8a1fa` | 提交信息重写的过程记录（before/after 对照、mapping、result、trees/status 快照）；随仓发行以便核对重写前后一致 |
| `evidence/move-to-data-disk.txt` | 969 | `2503b5bee95f8d7f6855ad6905f129d7b0314a977e6d8239fe265057dc044fe4` | 环境与迁移记录 |
| `evidence/p0-model/cuda-upgrade-freeze-after.txt` | 4,118 | `33585d37b124ca2dc86518d8c6606a03795ce3f21bd8e7e18139871a5d34a07f` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e0-baseline.txt` | 1,131 | `1abd77b6f767afc9ce525e743291bfc074f561c6697d3e06a3e8ba62c5a357c5` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e1-console.txt` | 447 | `8c84bc1341eaac32349c9a212e7003d210003ebb3a7de564031dd364adbdce0c` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e1-model-identity.json` | 15,698 | `9efa1bfda57e5e8a3df1afddaaed1df136cd6765c814aaf0b6ce82fdee362575` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e2-download.log` | 2,946 | `015da4f15589f5248e6215ccdc55da5c3a4813f63cbe65c4a9f9d8b8b66824be` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e2-small-files.txt` | 265 | `2d37ae9a3bc647254e5d7c500791348e0f4011346fa4d7926f5201c3b9e77255` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e2-snapshot-files.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e2-verify-console.txt` | 513 | `35753e435aa92f0134bf32dd77f1231fdb964f8774eeb99ddf4671413d13f25e` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e2-verify.json` | 11,451 | `27a4e8357e0dcef143faf0eebc287d793bc7f11127b2a1f2d0eef01199a8ba40` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-config-audit.json` | 12,824 | `9fc950575fb05b045b0a1f441a59adf2914e2993ccdaedb6d67b6116f10007ab` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-console.txt` | 6,796 | `69e4ebcac10dbfe7380df3b63a091a46d3048b1c8b9b52872bb6a2f72164dbf7` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-prompt-protocol-mixed.txt` | 258 | `08449a7b4a981da02f00cbd540f6f603027469111a3a398ec87e3c339439bdf2` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-prompt-thinking_default.txt` | 389 | `1cb7f1ed8bb2efc808ca0426512e0642a672f9b877faeb6e57b6ec417682655c` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-prompt-thinking_disabled.txt` | 191 | `7733165a3e19f7b199fce722c4e5f822dbb798057212d17e2e2a2b2fe78c8519` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-protocol-console.txt` | 4,625 | `4c3e43fa61a7d5af7285a14eff00e97729852f95b4be21e2804517e56c903ff1` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e3-protocol-tokens.json` | 4,625 | `4c3e43fa61a7d5af7285a14eff00e97729852f95b4be21e2804517e56c903ff1` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-expected-baseline.md` | 8,860 | `114015df7698daa3f2ee61f78639f279a1244bacd2311c0925df05f3dd221d9b` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-extract-console.txt` | 20,504 | `47be0b6e42a74bafe0a6e454c64f704918622d1deb710b9c71820f0a2877add2` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-extract.json` | 43,509 | `f61ca30baa1a2316a1084f91801c848f65355c803e4ae0e250cedf6920197b88` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-kernel-block-probe-attempt1-nofinding.json` | 238 | `61e159108cb612e5a82f5b1123cb13695b8659635ee444e8417331e9968591d5` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-kernel-block-probe.json` | 1,355 | `0d4e2b0cc5ebfa2d5ddcb4978783114114af702f17937d0dc50944884f3ed9e9` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4-metrics.txt` | 53,262 | `3a9ea6568ecb8bc217c4c911b5186c535f65cfdf5e29332b075388afcca25a31` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e4b-metrics.txt` | 49,786 | `20cf6708e2598cbca1a35cb9ab841dffb516d772d9e8827058d8fe44ba0cf05c` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-prompt-check.json` | 5,262 | `5dc760e66390b4024be79cbf4d140eda6b2cde87feae9ece7e78c9ba846e672e` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-prompt-server-thinking_disabled.txt` | 191 | `7733165a3e19f7b199fce722c4e5f822dbb798057212d17e2e2a2b2fe78c8519` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-prompt-server-thinking_enabled.txt` | 389 | `1cb7f1ed8bb2efc808ca0426512e0642a672f9b877faeb6e57b6ec417682655c` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-record.json` | 1,968 | `0188d79f400243f0659614ac1410e847c5805f7ec6c78d27724908e1b2eaba33` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5-response-raw.json` | 713 | `c4707698bbbfe4dad0507ae9c042731e675ec8fc6ce7ab55b2800152d39401a1` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-record.json` | 1,968 | `6227e28d42d3c9c08da60a051956dddd061ecfc01473c03f3537cb09873edfb0` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-response-raw.json` | 713 | `e254f5089661b7d0e9df15afca75ded3ad8a7e3ee4ebeead3c0a1b17432326e3` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-sampling-record.json` | 1,986 | `4c759bbe57eea760ccbf2cd80d6a2cf610358bf0364c76d8002555dbfbf5a640` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-sampling-request.json` | 316 | `210d408a12cee00f40e7c33fd32829c4f82ff33cfc001b619aa167d8b595518a` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e5b-sampling-response-raw.json` | 713 | `9ec5f3c08ca2672d8363eb46de8ca4daf995be786588cada1ade03817a86b554` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6-extract-console.txt` | 20,495 | `e79a5775a8ef9eece120455012ad23ed7518e123d06c3f9be61d43b3a57a737b` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6-metrics.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6-record.json` | 1,968 | `0139768c741f6fd89aaef926415b5a42adbd54788092f18a69dc6656c05eea0a` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6-response-raw.json` | 713 | `a883331644e0ab68583d52acb11f42559f1035846073aa8454452ea168d03553` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6b-metrics.txt` | 49,749 | `1ea51d99fe3980194441bad15c24b67e0eb1387d1579b9e056bdb7527972b736` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6b-record.json` | 1,968 | `50c1fc5d4400db68241e44174f827df1200c259ff9f2b1355a11ea2e543a32ea` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6b-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p0-model/e6b-response-raw.json` | 713 | `3eb98c5d21478eb4bbafd0118e82b71aa0b10b4dc666dfa6486840c2e8657b37` | 阶段 02 模型身份/配置/协议事实与冻结点 |
| `evidence/p1-cpu/demo-fixtures.json` | 40,435 | `97dd1a4834dac8c1b58be8e8f6746723b71d64179e595807657758b9132182a1` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/demo-input.json` | 8,340 | `8f476f905efb7034d7ead1417f8be86fac655fed52891680b37543c6263c5e2f` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/extraction.json` | 933 | `47d85d6f1d4a09e7183629fff477a9a4c0264b06dfa90660850c20e765e401f8` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/fixed-trace.json` | 225,153 | `2aab9723de3dc0301a484e3893dc9968154ea033841baf7023f0b16f3ccec590` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/fixed-trace.md` | 18,529 | `aa097476ff12b3639014c20beb79cdeb612348a6d67868acac232272fd206257` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-da.txt` | 25,943 | `1f5e2361f32622b4bb513847a7a7ca54789beb8e7daf71b8b2d68a93749ae60e` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-da_no_mask.txt` | 25,943 | `1f5e2361f32622b4bb513847a7a7ca54789beb8e7daf71b8b2d68a93749ae60e` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-facts.json` | 6,556 | `44af7fa453e5173263d6f75f7d75e77dff15196d3e3bd3fac3e2ec2e4dfd7ad7` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-fidelity.txt` | 771 | `1654068d1f3c83dcee3f165e3cc97a0918227fe3da00b0c2288db146d9f2d074` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-ids-offsets-da.json` | 148,550 | `3dd4feecc9c9738c046b100fd85d108f118f131960af686a47ffe9020ee04e9a` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-ids-offsets-da_no_mask.json` | 148,558 | `fe235d66882168e628209c7e22f924c5ab06c735eededb856ef01476d4202f47` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-ids-offsets-vanilla.json` | 119,347 | `5292ab4329448f4d269859c4db8d0f9f79e588da1fa110571b97d9d37b810488` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/prompt-vanilla.txt` | 20,157 | `34c33bf6c38ce8d0e7b482e2195afe4b52990a7de01512acd337218031fec5a8` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/run.log` | 354 | `2a1ba05497c2fa6516c1b387388268429da376540380ae752a255a45c0f4a6ad` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-cpu/template-kwargs-check.txt` | 883 | `b610e633bec6564e19621a7a18c1b7120eb64c5ebd25bd2a11aaeecdac953129` | 阶段 03 CPU 侧夹具、prompt 保真与偏移记录 |
| `evidence/p1-gpu-v6/cases-seed0.json` | 69,001 | `aa7520c26456e741f867191dc57b128956f3f2c7b201045e9f98b7af2211962a` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/cases-seed1.json` | 69,007 | `9209cb789526e1f424874d8b2f4d2776cff2034c6b3059f8753f21567c731bbc` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/cases-seed2.json` | 68,991 | `4d94754caea9226f10cd06abb2691b2f89ae75349fd4ffa14c317aaaa76648e8` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/control-manifest.json` | 1,028 | `86a4369820b4e2fbfb9911fbf34477097d90826d3901b47d41b3ae0978db2bbc` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/evidence-index.md` | 1,902 | `bd12b069fe2a66c9e351f5800ba9626fa3960512bfe3f248dd9a287d347ab8ec` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/negative-control.json` | 3,438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/negative-control.log` | 551 | `147d7d9dcd0dba62ca6e7988f80176d1470b06cff3efc2d397c459640bdb2d10` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/run-manifest.json` | 1,030 | `422d5e673d737bacc9604bc5b8389647eeadb20e47a36d94350375e2d8acdc83` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/run.log` | 8,300 | `5a1146ddebca69ca99da0759722e26b7de4b17d9ff128b82467ccda491344bfb` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p1-gpu-v6/summary.json` | 238,574 | `7953a1eb80474fd0ebcdca3fbc7619f834824c5b8a7486fd9d2cdbc7df614bbb` | 阶段 04 交付依据轮（R2 提交时序合规）；v1–v5 见树外登记 |
| `evidence/p2-single/deploy-drill-final.log` | 3,582 | `0d321e3a6804d0afb32868fa9a6e602efdd7be10610fe1be2a6b5e32b3d7d3c0` | 阶段 05 部署事务与单请求校准 |
| `evidence/p2-single/deploy-drill-postcommit.log` | 1,287 | `22ff182c8747a2d096b8df7e010e0238b7ae92ad97bb207d3fb3d418de4acb3f` | 阶段 05 部署事务与单请求校准 |
| `evidence/p2-single/deploy-drill.log` | 2,931 | `3054ee9f9de4fc27839bc02a185f8f69c2165a9f0a9949dcb5c85a342bd8001c` | 阶段 05 部署事务与单请求校准 |
| `evidence/p2-single/params-channel-probe.json` | 2,576 | `0fd53360e3781fe0cecc936976b89a0341e7e207882f233e13a785b9e632790f` | 阶段 05 部署事务与单请求校准 |
| `evidence/p3-calib/gate-record.json` | 1,078 | `efd0e337034a55a35cee90af1af2c99be355720055919d2621ed521beb614c00` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/crossblock-events.json` | 156,414 | `b8f544fa3b04e2edb3d973f11495c04cd350e7817f9b207d318bc7f610df0b5a` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/crossblock-note.json` | 153,051 | `ca842c58579d63cb40e623e7cf0628b986b7c21f669afb17653c30d3c1c6e921` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/crossblock-r2.json` | 147,148 | `30a41c2889846175babb08976ed1e0f311cbd6c80bd3b6085e11b41087f4bc3a` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/crossblock.json` | 102,935 | `b8a109e75c903cb9a0fa463e0e15925725613a6045f304322cd0442d6a048423` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/points.json` | 3,915 | `d4790b03c504334aa6eab1a6902e28b24a9bcd3dfd32f62369a7f758817ce234` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/ref-cpu-checks.json` | 9,718 | `eda2d12a9a4f128e77970256b0e6982afbce96a4efc1ef3f56bb6db451593b8b` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/scale-metrics-dedup.json` | 1,371 | `04530cd02694bd50cec961e8b291338cfa63965fbe91cad474db51c2de0ad375` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/scale-metrics.json` | 98,318 | `850bda144bb5ba90ba24d5eea9b2b3fa69e443db8f862d9714e607e22f855f37` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/masked-prep/sets.json` | 14,721 | `2df7bb74a2e09f20180fbbe5e6811b9f11840a2b91f5cdb6405e870af69e8cfb` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate.json` | 3,478 | `93d902d3bd7ac1a2bae1fd507b8556cf227f3ff102926ee391bff94df56b1237` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate2.chrome.json` | 32,930 | `332116a2e839abe85509bd554474984e0ed310aa97917fe945a1d31f8204ebe0` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate2.json` | 5,349 | `25dd36324c4563b714448d45763713092e172dfb49d605f72d08e497ffda796e` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate3.chrome.json` | 34,074 | `426c8d26c4590f057e489de3bcf44319b3bba375cc3c7e3a897de017e878f5fa` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate3.json` | 5,145 | `97b294c42a4c1b6cb0a6068a09a90c670aebcea494e88e21f2bdf90cd76bbec3` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate4.chrome.json` | 34,060 | `d2be21748e325117a33c6bc9bcbd26736652a0ce322e5c4c5dbb0ff99be61065` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate4.events.json` | 5,435 | `7199a91e847bf56264aa37f1005d2a8199c9009d26e4c8e1e08ed608c5806dc6` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate4.json` | 5,194 | `357b22ee1c703e21c976e3b3d1782be2f3f6ac7ab11c167eb6de943851a8978b` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate5.chrome.json` | 34,070 | `524545a739f399d3d1c100398bc5fe6f503b5bd8372d977f2b37d0300cac1b78` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe-gate5.json` | 5,354 | `076fb27be3d6ce90f4fff36872e0208b1e004880d2475dc3d8666fce86ef702b` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/metadata-probe.json` | 3,478 | `b248242410e5b8b1e15166d76086d4aae4c7b97b9d0f0121825e64711a460611` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/oracle-original-3a.aggregates.json` | 69,215 | `c1ce771697c9d4e1a571198f92e5d1acf3631547795ec780f352cf294097830b` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/affected-unittest.log` | 19,821 | `560b0374843b30e5ba421dc7010762afc755e2f4733ab0c8f52b7c79f30659d8` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/bridge-checks-postcommit.json` | 9,667 | `6759e259e244b99d56dec1a342116e46010f03a37f553821ab18220ac483df28` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/bridge-checks-postcommit.log` | 41 | `c603c816b9a8661124ddea52a9ea30c49c109b8edd8178d98722fcf8bef133c2` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/bridge-checks.json` | 9,665 | `d0de8e62ece98bcb8a2662490cd96f1b84a40dcd94108e1c114638009b1af826` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/bridge-checks.log` | 169 | `75ddb84b790930d52b474a9af270020526365b075080dcd24a2fa485948098b4` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/deployment-verify.log` | 31 | `91bef6e2d6db9bdce8093a4e2d90a158c82a07490d42ad52114336ac76d424ef` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/precommit-gates.log` | 1,731 | `3f1cad87264cd76d40b0c049e3351cb2cdefba292cfffa2f6751fd0fb115ff3e` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/pytest-collection.log` | 33,579 | `52d7836820e2f09d10d9d531047f332f972d0f71f52829f258451910cebdb8f1` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/reference-cpu-pre-gpu.json` | 9,667 | `4f86ed8665de0a1833051bb7c84f8e93cb3458262479643675ed18a4ddc470d3` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/summary.json` | 2,030 | `95261d989567989cbcbd21ee1ac1eb0c96e0e3062f3f5846678b7532e8b78591` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/reference-integration-20260921/validation.json` | 3,977 | `0015fd11792e0942b8089e949de9bc626c50ff693a07b66cd767be7632657c56` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-3/error.txt` | 1,294 | `479194f61d0f8f0fc109d14b63715691d3eff74085a8d2190007fd8c5e48332d` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-3/source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6ff89af372ae9b6fbb5fa391bab86633b0bc3e9eb763cf2e3` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/arm.json` | 813 | `771161e7a4eba15ecb0c5d0566e3a4281a4fd4ae732f1aa8df55b55d23a46f59` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/cleanup.json` | 6,693 | `56c698f762b41300eef425223dc8b0dfb94da453597c516863ef1607a6d9064e` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/engine-traces.json` | 7,600 | `3b7f1492be9e22ac929c40e10df91d03f888be37415158fca9bfb232fceb2933` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/force.jsonl` | 2,608 | `0ea885d19aac2d1f7b15f10d743e25ea0d5fc33d33e6a4aa79d630a533be30f9` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/manifest.json` | 120,313 | `d16403764805cd316009a2c8092bff101e0c53ebcb3da31c86ce31366527b234` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-4/source/p2-calib-run.py` | 124,517 | `0f2040a04770d841e053f9528b581b112e01004de3ad74a456abe1a729686754` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/arm.json` | 813 | `08fcbc6d9e890d81a50191d3d6f563448362a3e4b3dc30ac75eb5e327cf7b8e9` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/cleanup.json` | 7,240 | `aa9ae9e37b627a331e7589448bae0685bd8302017ea21be65d6a42ee1a36ad53` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/engine-traces.json` | 7,600 | `7eb4aafc5346a3acc2402a07aab3d1225ae628a6184bdb108c2696068bc8c9fe` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/force.jsonl` | 2,608 | `aba861b766577c0ecbcc36a5ade89aef0959d1212d7ac9a03bcdd88bbe31072e` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/manifest.json` | 120,965 | `5cf9576262d835221b9fae726b0d6921811af0910c71980f999bce3e0ff5b400` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-disabled-5/source/p2-calib-run.py` | 127,908 | `e2d5d0440bb8aabad971ce6842270a890f3c616c499138566cba2c6272632a70` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/arm.json` | 803 | `274d44795116f421e93fbbf6f7bddd59884283d01dbda5e721bb37781a50ddd8` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/cleanup.json` | 6,697 | `9f8888854e01710ac3391c00420e6b1eb0e21179864b7cf51d07097bd6a7763a` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/engine-traces.json` | 12,764 | `bada76543c35c0f40d248c3acdf3cde418b85d74e28503f7134ab2b840f8369f` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/force.jsonl` | 2,576 | `3bcabe65d6114523882b74d5bac2d073c1aafe5afa319b2f696b3d09617a49bf` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/manifest.json` | 120,667 | `8ce6b52dbfb5ca842815bc28d7ed4c204957494850a76c8dae18c07b94ac1f42` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-4/source/p2-calib-run.py` | 124,517 | `0f2040a04770d841e053f9528b581b112e01004de3ad74a456abe1a729686754` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/arm.json` | 803 | `0f8e9aed468a3081708efa67ef622844be266ecadb41c9904c275f2f0fe32bad` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/cleanup.json` | 7,244 | `c5b00f7fb720a2dc075c180e334acbe12df5e61e36bc2d74fa1e080b027a009c` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/engine-traces.json` | 12,764 | `b8caf2627143970db482d6f73f8830f4efd4b5e40f7077f173ce08db4f0980f0` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/force.jsonl` | 2,576 | `6864b9fcff18878b49face3b4133f6dd3b41521a3491f5e24833a8f487abde56` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/manifest.json` | 121,318 | `ba12082e1051376fd076fa0fc7bfafa361f3e87a8d1cb9be35ca01b900cb1a06` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-global-5/source/p2-calib-run.py` | 127,908 | `e2d5d0440bb8aabad971ce6842270a890f3c616c499138566cba2c6272632a70` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-2a/manifest.json` | 24,815 | `2e648615e8dae167756a2286d1e2132495d709d5e3bb1a3ee37e3ec7cb7caac6` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3a/arm.json` | 323 | `7e001ca614aa318250a8e328d664b7d8a28e1431d248234d051e784c6135953f` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3a/cleanup.json` | 5,457 | `df5126624862fd529fd81362f5cf99bc3b47c39f94555fe773e2a56ef2acd553` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3a/manifest.json` | 115,377 | `6362678768d876a3df321cde05971257192266c3bc911074431289c829c9abfc` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3a/source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6ff89af372ae9b6fbb5fa391bab86633b0bc3e9eb763cf2e3` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3b/arm.json` | 323 | `6c699c676abd4f3eb37ef20e2a4d053fe7e30d12fdb1cbbdc8bf4f8b4677d89d` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3b/cleanup.json` | 5,457 | `46c64f4a2d0cae77cb8a14bd29814a3d80f56c3e9f8f2018d6aeaffa20896ecf` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3b/manifest.json` | 115,106 | `30724ec1df368bcdaa485f04b89297867987f500f7e81a1cec7ecf591d8703e7` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-3b/source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6ff89af372ae9b6fbb5fa391bab86633b0bc3e9eb763cf2e3` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-1/arm.json` | 254 | `224bae8178aa3bf0c77420aa5cb337c30b4216f3139c2072a31dd6e03b4c05f7` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-1/error.txt` | 4,013 | `f614804991a7fbbbabd8798c05765748d6db9eb0fddb047868e3a19f2f9677e8` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-1/manifest.json` | 28,247 | `d277243669328083a3863a8120825a5372d8897fbb49b5d335af55e81739209a` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-2/arm.json` | 323 | `0e18b782ba461a506d27fdd3c4cb0c765bbff86fe3767ad3600ab02ad35b561f` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-2/error.txt` | 4,345 | `9e9c07d85e64493734c3d27127e0a8db276a5125726a891e7f52452d1e3c5196` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-original-r1-2/manifest.json` | 28,673 | `89001215d1876664639dfdd64ff98d042e75c43ba974935711cf2d8c0eed2478` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/run-vanilla-1/manifest.json` | 2,465 | `a07654f4636846631be7567e3ac75a48aadc79457f5794f2681259954312c2d0` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/traj-original-1.json` | 924 | `da01f10eeae306ddd5f599180a4b56192fe399dc0ab3cd86460098eee5e923af` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-calib/traj.json` | 88 | `019e8493a3681e6a76d0106b117e922f2933d6d5188d39e99aea76d2f8c7f8c4` | 阶段 05 校准 run 文本层与独立复算（含 oracle 聚合段） |
| `evidence/p3-masked-reference/reference-20260921-run2/run/source/p2-calib-run.py` | 154,482 | `03069799700c31788a78d3adf301a815284f0b65a8135d2c80d8ae4fe8c840b7` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run3/run/source/p2-calib-run.py` | 155,241 | `d1051431ec1bca8d8135041d466f51ee3413e1de24b25eea42579ef4aa008f27` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run4/run/source/p2-calib-run.py` | 155,394 | `65d231037d0c1b3f1a550f95bfe8ddba799e5531d0fdd767927c414ab3ada55a` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run4/summary.json` | 7,642 | `c61fe1041b8ac95ac5b07c262c6c49b61e3f26b997eb8097a3e5264ad4561cb7` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/apply.log` | 112 | `885073a35b563e6480d1fd7c792863b67fe7605ae684146b7a9ada57f460b7b9` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/gpu.log` | 13,923 | `d731f6105d7a55eed73511da411587d0a428971998f78320f450bab69975742f` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/revert.log` | 1,200 | `49d83ba335c96218a346a1855b50495f1e8e2229ecf6c4e0b9eca6b83894fd14` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/arm.json` | 1,861 | `807174a607c58edcd1d483ffb3381ac62b2003d84efa2a0205810afb48625224` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/capture/structure.json` | 3,930,507 | `b0e7874fefb5a78fa2111405605e0ce2572155f76fe54869c656dadc2980ad96` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/cleanup.json` | 7,293 | `14ad973d269fe86377b399c5aa9dbdbbe20c25f60cfabfeadcf86dd66eb26036` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/engine-traces-main.json` | 2 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/engine-traces.json` | 7,601 | `f2fd15ef61d72f0a166cc4b8e024f40f01090fb57e3ee5c4ab6d8ae680b4f1cc` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/force.jsonl` | 9,552 | `186d35919b4b7743e1433148c9d2ec81590347e5d3be5ccdbde11d04ae6412b1` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/manifest.json` | 7,600,220 | `214a07252c7ac82cfd69b002708e53ad57dde03769c26ce45fa216c8355a41ae` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/_lib.py` | 4,103 | `90108628112acac9bbb15a0490c8feb5f45ac61913923c54b0b3a94b2d8da1cb` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/__init__.py` | 788 | `0e3bb337c8b3339945cb2486b010a955887421b1331d5781d4022f365b3675ce` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/decode.py` | 2,903 | `61a903674fdc24466cb5b8155a0536f9a9e8c0af9e0cb7c0ddce5bab94fe4f4f` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/extract.py` | 5,811 | `68eef6a60b1a8666155d15210704804038ccf01d18adbcd887b72552c610b308` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/fixture_rebuild.py` | 3,247 | `0693be69ed45346c611d7f04c4d63d621b8cad3a19bf918e4e67d5b8ff53ebf9` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpucheck.py` | 8,504 | `4c9a2f0da31b9fa87dd5c6eca2bedb1e6ea73e5680e3d57814c3b7b924229f5e` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpukv.py` | 7,462 | `285fd0dee588a1febd476d51ad61c022bdbf9ece136e02e311d1ad26b498964c` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpuoracle.py` | 6,264 | `2c75f780fa8bf9aee09d2658dd78c3cda59d3e0a6e11dddd9eca446053656052` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/parser.py` | 15,069 | `2cac16f70a78094eccd45962406c3e95643a1d0cf467f7f06f219b0bfeb1e60d` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/prompt.py` | 13,715 | `438591ad038bbdad653039afc65d8e4949d630cc43d63edcf6b4ebcc76d7023b` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/prompts.py` | 8,038 | `943d22a4148cfa87c2c9d8797dab0ebb98a68fa66f2605d2d0900f09e913dd65` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/readview.py` | 12,167 | `b11a971e20e2568d427675a4ef71b73012dacf9604c67d5646c90e13a5db2f5c` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference.py` | 2,956 | `434d924d290ad97031effcc8b868c56047ef144ce3393be516283f48f844cafd` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_bridge.py` | 13,929 | `32280a1e95c01e368259ec6edd1523a2903ebdf392ccdbc5eb07317ad6149e1b` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_dense.py` | 12,528 | `5d52877a25bfb65da3da228e181aea640455df3208b604c668aa0d5bfa1b758a` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_hook.py` | 7,759 | `855e4ad531d1d023ed1ffd3b43198d341084943476476049a32fb8e6daf9b6dd` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_run.py` | 9,316 | `c592877429a4636bf314cc322459c2fb1262957a3db19a6be54f47732787d9d6` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/segmenter.py` | 12,245 | `c3bcc218c78141a60ee750a64a0508a62057555a430e5508184b24945e19129b` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/smoke_structure.py` | 10,757 | `f63eb7969b841f56b7f5aae4f78be2e0d0e8ea58a7289627956da6a68903513c` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/state.py` | 10,157 | `e1dc6dab84d9108a7062b3527fb2f658720c8e8c401d31f0d7131f6cb559946a` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/step_plan.py` | 18,702 | `f86f7ee703feaa78c57c48edfb86671020d974b0b8b1290b010bc709cc3b8714` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/trace.py` | 6,764 | `bdc3f309bfcf3303c5c98e5bfdfd4b718059dce97bb2c8d194591ea0e1692b54` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/source/p2-calib-run.py` | 155,574 | `a9da137f6350b79977e225c6592f69e7d140ea9353539571dad24fb961eb6635` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/summary.json` | 35,819 | `da370745f77a380e92bb8411c1a238b4e15db3d7b608131c27b59438b8772182` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/transaction.json` | 3,196 | `68ca628c8c97fb726b415c48f91b073e370d5203b767387544ac2c96df069458` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-reference/reference-20260921-run5/verify.log` | 57 | `bed9e93562842c263733f182cc81679a20d17bed745e068362b244fac9c2b6ad` | 真实独立 masked reference run（run5）的文本层；被取代代次的字节留数据盘，见在盘未入库集合 |
| `evidence/p3-masked-smoke/closeout-gates-20260921/dependencies-after.txt` | 4,163 | `613edbca75157296fb0f1720a254c671d380fec44f7f1aaab319cb02100003a3` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/dependencies-before.txt` | 4,118 | `350d32f4804ada87ad8617cabe64c07743fc43a1685f57d1b7a34ab374c89d01` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/desensitize.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/desensitize.log` | 413 | `1adcb13513a0f1d5b3f6c96b271906e3e21df3cfd2f94ac03449499c20ca2eed` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/head.txt` | 41 | `7e1bc4d56a9e0d601328feda23ea2cd2f9f5a672850bf41c2873eeec257533d9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/install.log` | 206 | `0c8ced7c224b3814be8ee17e2fb6fb6e6871098417605ecee836665409c74b6b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/largest.txt` | 128 | `84f353dd191558d916ad312ba00e02eb7a9823838cc90ee2e6945ee28ab05dd0` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/pytest.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/pytest.log` | 867 | `b55d9448da5183e98caafa959a0b06e8bb5213562cf17b6bf1e7b8408a207c09` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/secrets-command.exit` | 4 | `181210f8f9c779c26da1d9b2075bde0127302ee0e3fca38c9a83f5b1dd8e5d3b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/closeout-gates-20260921/secrets.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/command.txt` | 227 | `ffd74d41216c37cba75194262582daca9d3b7491688f274e29cb97e213f9b155` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/deployment-before.log` | 31 | `91bef6e2d6db9bdce8093a4e2d90a158c82a07490d42ad52114336ac76d424ef` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/exit_code.txt` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/head.txt` | 41 | `639968db3c69694b7870e5c8575426226ff29ab158b47fea5021acc2ac639970` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/launch-inputs.json` | 1,320 | `d2f78a507e97327685cbfe3fd1a4ad6ae682f5dcef9fe5e501336ea6688ee315` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/patch-check.log` | 31 | `4c47d62cff001df1424605477a46d53e13285ea8ff470729cffb446984fa9d91` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-20260920T170715Z/unittest.log` | 754 | `98b476695e69265af7a1c260240e50762d67a5370b1e305f46457384cdb9aee6` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/command.txt` | 185 | `7fa9cd1d1d06be671249042cd433eea5ecaec09b3cd6fc67be5145cf78c3af7f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/exit_code.txt` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/head.txt` | 41 | `c4e9f85f9787d5b6ce82655853401e3434f3d5132721616d71729a62a7fd4532` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/unittest.log` | 986 | `682c9e194d6e2769133fe94d9a6cae3fad20319bc967ad64e7b0aaf4b06cfa90` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/apply.log` | 112 | `885073a35b563e6480d1fd7c792863b67fe7605ae684146b7a9ada57f460b7b9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/gpu.log` | 13,921 | `e5a52e0b75a9e8d620ff75d49f8c07d7d0ed77ff9d290fa7d6c4264b34772e24` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/numeric-audit.json` | 31,142 | `51baf96f59bd6c048f19b99737164c058f46b92e67e2d8dcf8972f8f0328ac91` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/revert.log` | 1,200 | `49d83ba335c96218a346a1855b50495f1e8e2229ecf6c4e0b9eca6b83894fd14` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/arm.json` | 1,855 | `baf1ec2f03dd394c55170f06515922f3c5b679a6bd79df1dc2fbeee23182bb69` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/capture/structure.json` | 3,916,459 | `68ac7e6b9ef60425860e046d05a9e6ffd5460111c53354cdb5ef8e6dfda212ac` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/cleanup.json` | 7,306 | `a294226b08e0804699a91b28c21928ac201eef9e076be55952edf8f01b5f64d7` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/engine-traces-main.json` | 17,961 | `02d9739914eb3f0a5837300b2c417fbc3a7921e84dcf1868400914ae0c690a9f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/engine-traces.json` | 25,560 | `c1c99bd60af5c1d0a56ec04b8e8db1b5b7f967dfce115b005cbd6a8666250931` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/force.jsonl` | 9,378 | `a91f816f7d1e5e9081e5e7a9aa6ee96220b2ba886f4f048615258237e29e6087` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/manifest.json` | 2,737,856 | `b7f6d5f88b826f854b0e5df1966c9140fe945896a82d4d0a696fdab4d05d2dce` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/_lib.py` | 4,103 | `90108628112acac9bbb15a0490c8feb5f45ac61913923c54b0b3a94b2d8da1cb` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/__init__.py` | 788 | `0e3bb337c8b3339945cb2486b010a955887421b1331d5781d4022f365b3675ce` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/decode.py` | 2,903 | `61a903674fdc24466cb5b8155a0536f9a9e8c0af9e0cb7c0ddce5bab94fe4f4f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/extract.py` | 5,811 | `68eef6a60b1a8666155d15210704804038ccf01d18adbcd887b72552c610b308` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/fixture_rebuild.py` | 3,247 | `0693be69ed45346c611d7f04c4d63d621b8cad3a19bf918e4e67d5b8ff53ebf9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpucheck.py` | 8,504 | `4c9a2f0da31b9fa87dd5c6eca2bedb1e6ea73e5680e3d57814c3b7b924229f5e` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpukv.py` | 7,462 | `285fd0dee588a1febd476d51ad61c022bdbf9ece136e02e311d1ad26b498964c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpuoracle.py` | 6,264 | `2c75f780fa8bf9aee09d2658dd78c3cda59d3e0a6e11dddd9eca446053656052` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/parser.py` | 15,069 | `2cac16f70a78094eccd45962406c3e95643a1d0cf467f7f06f219b0bfeb1e60d` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/prompt.py` | 13,715 | `438591ad038bbdad653039afc65d8e4949d630cc43d63edcf6b4ebcc76d7023b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/prompts.py` | 8,038 | `943d22a4148cfa87c2c9d8797dab0ebb98a68fa66f2605d2d0900f09e913dd65` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/readview.py` | 12,167 | `b11a971e20e2568d427675a4ef71b73012dacf9604c67d5646c90e13a5db2f5c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference.py` | 2,956 | `434d924d290ad97031effcc8b868c56047ef144ce3393be516283f48f844cafd` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_bridge.py` | 13,929 | `32280a1e95c01e368259ec6edd1523a2903ebdf392ccdbc5eb07317ad6149e1b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_dense.py` | 12,528 | `5d52877a25bfb65da3da228e181aea640455df3208b604c668aa0d5bfa1b758a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_hook.py` | 7,759 | `855e4ad531d1d023ed1ffd3b43198d341084943476476049a32fb8e6daf9b6dd` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_run.py` | 9,316 | `c592877429a4636bf314cc322459c2fb1262957a3db19a6be54f47732787d9d6` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/segmenter.py` | 12,245 | `c3bcc218c78141a60ee750a64a0508a62057555a430e5508184b24945e19129b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/smoke_structure.py` | 10,757 | `f63eb7969b841f56b7f5aae4f78be2e0d0e8ea58a7289627956da6a68903513c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/state.py` | 10,157 | `e1dc6dab84d9108a7062b3527fb2f658720c8e8c401d31f0d7131f6cb559946a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/step_plan.py` | 18,702 | `f86f7ee703feaa78c57c48edfb86671020d974b0b8b1290b010bc709cc3b8714` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/trace.py` | 6,764 | `bdc3f309bfcf3303c5c98e5bfdfd4b718059dce97bb2c8d194591ea0e1692b54` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/p2-calib-run.py` | 155,574 | `a9da137f6350b79977e225c6592f69e7d140ea9353539571dad24fb961eb6635` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/steps.jsonl` | 13,488 | `24298329ee5c606d5f822f20180caac46abb0739b530d5d3a2dd0b7028bb4141` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/transaction.json` | 3,182 | `06575007a845fedde4c75d7c973948745dedc249ae103e969bd6164133cc75f9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/verify.log` | 57 | `bed9e93562842c263733f182cc81679a20d17bed745e068362b244fac9c2b6ad` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/apply.log` | 112 | `885073a35b563e6480d1fd7c792863b67fe7605ae684146b7a9ada57f460b7b9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/artifact-index.json` | 6,535 | `2b93d661c1a116fd08abd437d4cb3ffd27233fdd9d806a75a6dfce60e5f8812f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/gpu.log` | 13,363 | `4c225d9e966a137b29b687857886d1dc468e29f492dd290968478eb7ec4cbe32` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/revert.log` | 1,200 | `49d83ba335c96218a346a1855b50495f1e8e2229ecf6c4e0b9eca6b83894fd14` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/arm.json` | 1,849 | `3d8d02bb9b75a0acda1e2da71bee262e87e6b62f0bc654978831543afc64cb86` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/capture/structure.json` | 3,592,264 | `ee857b7f0a39fb191edcbed74a0009cf931cec6e87b39d4f44d1223838c03d7a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/error.txt` | 3,896 | `b482ab6f614c30e76caf410d218ac3285d6ca6be0f7cd1943dd2a6efbcc24dfb` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/force.jsonl` | 1,937 | `16635fb9652f76f141d890367bcd818fb5673b0d42760dbc441257115050d9e5` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/manifest.json` | 108,304 | `0ac935585ddb94fb00ee5c12af9875e1586db0bdd96ed05db334ae6a3e4107a1` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/__init__.py` | 788 | `0e3bb337c8b3339945cb2486b010a955887421b1331d5781d4022f365b3675ce` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/decode.py` | 2,881 | `29fb688006bb982a2ee88ff34ab0a657f1f7866cdc20a7a7cab548176feba44f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/extract.py` | 5,802 | `dc7c4e2551fee0af1f00826f023ad8fd5bc5be313290dcf3f122451247193f17` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/fixture_rebuild.py` | 3,247 | `0693be69ed45346c611d7f04c4d63d621b8cad3a19bf918e4e67d5b8ff53ebf9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpucheck.py` | 8,477 | `898857527f3fabb0fbdf4111694a850257fde868194699f50294bb74f17cb42d` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpukv.py` | 7,436 | `e0ce47f9511aefc65bd797e046163686eda6f7b5c4d1b0e425754d300d15f2c6` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpuoracle.py` | 6,264 | `2c75f780fa8bf9aee09d2658dd78c3cda59d3e0a6e11dddd9eca446053656052` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/parser.py` | 15,069 | `2cac16f70a78094eccd45962406c3e95643a1d0cf467f7f06f219b0bfeb1e60d` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/prompt.py` | 13,932 | `192bd07fcd15cbd0741a76e58e107d360e7d831f63602ce87182e94b38b46f0a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/prompts.py` | 8,038 | `943d22a4148cfa87c2c9d8797dab0ebb98a68fa66f2605d2d0900f09e913dd65` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/readview.py` | 12,158 | `ce162dd31d12aeb9475a4239424e2164cf3ed84e30f5168e4a14972e8f4c49a5` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference.py` | 2,947 | `57b8a76b89f414ee968a50ca7965dc3412b95b77cdecc4d5231ac7afeea5d917` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_bridge.py` | 13,033 | `ebbd08f879db6e9269ade91de5f2283c6a1e6a59d2ec47c1b43984f4e40e9995` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_dense.py` | 12,206 | `32845d8f543ea485dcb9ddb0017959c2807e6db797ecf0f3bbd1963dc13b717e` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_hook.py` | 6,979 | `2e05098e9bbe5df1efdb5e11a71cbee3ce94048bb7d65179fab5730b91870bd9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/segmenter.py` | 12,094 | `1dca3d43d7a16b520af7b5799db31bc89f1dc03b0664926048e384d458d78d00` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/smoke_structure.py` | 10,757 | `f63eb7969b841f56b7f5aae4f78be2e0d0e8ea58a7289627956da6a68903513c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/state.py` | 10,155 | `62bfcdaa54c3b9df5d7d4710e6159696e6d6b4d51113f1960b70f8a9370c750c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/step_plan.py` | 18,681 | `d32ac1724d7853945a7139e611079b2612bbadab39921538631c12e1b4d46e23` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/trace.py` | 6,764 | `bdc3f309bfcf3303c5c98e5bfdfd4b718059dce97bb2c8d194591ea0e1692b54` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/p2-calib-run.py` | 141,273 | `8ba523146a36d7d1ea655db5786c424afa73a9af839bf9a7b8844de0fcad4fef` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/transaction.json` | 3,315 | `ec4de7649a6d446ed886dcf081695014439dc3423c7c3a250828fea112ded72a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/verify.log` | 57 | `bed9e93562842c263733f182cc81679a20d17bed745e068362b244fac9c2b6ad` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/apply.log` | 112 | `885073a35b563e6480d1fd7c792863b67fe7605ae684146b7a9ada57f460b7b9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/artifact-index.json` | 11,821 | `534ac923cb1414f43a6407ffbcace6e55732fad159e930d338504c4ad403e79b` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/gpu.log` | 13,906 | `bb1e670460d75ba748da66ee7d32a3c3434bc9c762a17cc0142c0283d2a63b05` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/numeric-audit.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/numeric-audit.json` | 31,142 | `fe36d8807caf082150bc13e3703c855235e547afe5b0ef6847ddf8105bf2b157` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/numeric-audit.log` | 319 | `4c80baa2b669840fa99fdce1078a08b6237331e6dc90f68a0ddca5e4cc848ff3` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/restoration.json` | 6,834 | `6dbd4ed029beddc3c7c9e9130ba55a059a9f23a5bea979166e9d089aee942a4a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/revert.log` | 1,200 | `49d83ba335c96218a346a1855b50495f1e8e2229ecf6c4e0b9eca6b83894fd14` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/arm.json` | 1,852 | `63a0c0be5834d46816fc3462234b17f639dea33ad3b7978404fa56e90c09dfe4` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/capture/structure.json` | 3,916,459 | `811c81956e86842e5f02537b442e7c1a544ab6d990078d6933a57eced161299c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/cleanup.json` | 7,306 | `a28be8736de9952769377f70bd40f81250ca07618179077e30e4f5d265c03475` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/engine-traces-main.json` | 17,961 | `41d2d9cabb0c7694d2d5d72a4a0a04f90ea7ba267bc122ff1d8f7b7193e9baec` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/engine-traces.json` | 25,560 | `f87f8edcf0e1d8dc648244eafd878fae4dd56f11fe610040e501ed6b7bca00e5` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/force.jsonl` | 9,378 | `bc99665a6f2b9b4099e7e7ac768932b9497129c806a4f88f8ae9504dde430cdb` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/manifest.json` | 1,301,387 | `0b9f4c40d1a9c7d9bae98112a771465b06ebe31e558bac36742826363015f29f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/__init__.py` | 788 | `0e3bb337c8b3339945cb2486b010a955887421b1331d5781d4022f365b3675ce` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/decode.py` | 2,881 | `29fb688006bb982a2ee88ff34ab0a657f1f7866cdc20a7a7cab548176feba44f` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/extract.py` | 5,802 | `dc7c4e2551fee0af1f00826f023ad8fd5bc5be313290dcf3f122451247193f17` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/fixture_rebuild.py` | 3,247 | `0693be69ed45346c611d7f04c4d63d621b8cad3a19bf918e4e67d5b8ff53ebf9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpucheck.py` | 8,477 | `898857527f3fabb0fbdf4111694a850257fde868194699f50294bb74f17cb42d` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpukv.py` | 7,436 | `e0ce47f9511aefc65bd797e046163686eda6f7b5c4d1b0e425754d300d15f2c6` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpuoracle.py` | 6,264 | `2c75f780fa8bf9aee09d2658dd78c3cda59d3e0a6e11dddd9eca446053656052` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/parser.py` | 15,069 | `2cac16f70a78094eccd45962406c3e95643a1d0cf467f7f06f219b0bfeb1e60d` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/prompt.py` | 13,932 | `192bd07fcd15cbd0741a76e58e107d360e7d831f63602ce87182e94b38b46f0a` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/prompts.py` | 8,038 | `943d22a4148cfa87c2c9d8797dab0ebb98a68fa66f2605d2d0900f09e913dd65` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/readview.py` | 12,158 | `ce162dd31d12aeb9475a4239424e2164cf3ed84e30f5168e4a14972e8f4c49a5` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference.py` | 2,947 | `57b8a76b89f414ee968a50ca7965dc3412b95b77cdecc4d5231ac7afeea5d917` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_bridge.py` | 13,033 | `ebbd08f879db6e9269ade91de5f2283c6a1e6a59d2ec47c1b43984f4e40e9995` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_dense.py` | 12,206 | `32845d8f543ea485dcb9ddb0017959c2807e6db797ecf0f3bbd1963dc13b717e` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_hook.py` | 6,979 | `2e05098e9bbe5df1efdb5e11a71cbee3ce94048bb7d65179fab5730b91870bd9` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/segmenter.py` | 12,094 | `1dca3d43d7a16b520af7b5799db31bc89f1dc03b0664926048e384d458d78d00` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/smoke_structure.py` | 10,757 | `f63eb7969b841f56b7f5aae4f78be2e0d0e8ea58a7289627956da6a68903513c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/state.py` | 10,155 | `62bfcdaa54c3b9df5d7d4710e6159696e6d6b4d51113f1960b70f8a9370c750c` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/step_plan.py` | 18,681 | `d32ac1724d7853945a7139e611079b2612bbadab39921538631c12e1b4d46e23` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/trace.py` | 6,764 | `bdc3f309bfcf3303c5c98e5bfdfd4b718059dce97bb2c8d194591ea0e1692b54` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/p2-calib-run.py` | 142,432 | `d1f269dbdf5174db905a7a1110c1dec62eb6fd0e30b2b925c4c47aec8fe96305` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/steps.jsonl` | 13,486 | `a89898e1f4d62262e39c64ce4db851f713fa64176c5a1db991e329ae84bca3d1` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/transaction.json` | 3,317 | `be0d583a84aa79250f03aabbbcaaa9c05c23ae69d4f97ea75d66c242c08081e6` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/verify.log` | 57 | `bed9e93562842c263733f182cc81679a20d17bed745e068362b244fac9c2b6ad` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/p3-masked-smoke/traj-7834-generated.json` | 1,812 | `5b15d841fff0436a7714b52dbc36dd140270ded0eb3c9669d72f632cb6f549ea` | stage-05 masked 冒烟；验收状态见 reports/p3-calib/evidence-index.md §0（capture/structure.json 与 steps.jsonl 是否降为 detail-only 待验收裁定） |
| `evidence/paged-kv-before-cleanup.txt` | 547 | `104d9397a23a4bf8a81e1ba16782df14ea4596a380e4b0999d59f40a05d7f1ab` | 环境与迁移记录 |
| `evidence/post-expansion.txt` | 2,216 | `1743027caa0130df2a9f55108ef708c6c38cdb2d471003121f984ab8ff4ae449` | 环境与迁移记录 |
| `evidence/verify-console.log` | 7,355 | `b15107b5b82f2e87aadc4e6dfa950d507eac2f18631cb6e6e12b8e9ca7159528` | 环境与迁移记录 |
| `logs/install-runtime-20260917-155930.log` | 546 | `05fa3fbdf86fecce46c6d56e1795f4c63241c461dd0b54c8a0bfe48f8b238c6c` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/install-runtime-20260917-160026.log` | 609 | `8205e1959176798d6e262dd9d2d47ecaf40bf68cdcb4aad1cf0fbf4e56a430f4` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/install-runtime-20260917-160325.log` | 10,910 | `c4d38fc3afa671b1065fca8e60fc7537350d5785792697ac3b8910a97df5dc24` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/install-runtime-20260917-163051.log` | 2,421 | `189d9b27c5e5ba7178ba12bc7e093e3ff8addde3c794dc969a4a8ec6122f6d57` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4-attempt1-offline-resolve-fail.log` | 6,656 | `21b1868f07fc01ce842d3525a6a5d6f126b2823422055060d7a487d96d36f502` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | 36,250 | `6c0717b64d88fe212cd5879437af4cf9deb5cd76d33254d1cef0f3b913448b0a` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | 88,698 | `3296f64bfb58f44f58a32930ec1a8ec175db0a34ca1d2046009f3f5e5d805736` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4.log` | 34,441 | `571a01c89c29e0045f80d6a360076da27a41e15f07e054a1b66c548275b9b3c9` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4b.log` | 33,915 | `3d906c46a2432b142a659aaa93ee0dc1ad96fe15a7d55e12ee69effcd22d014e` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e4c.log` | 28,645 | `f5b852f90ad14786d6ab4a03d2e34eb8dfd6a6bffdcf88e70e5f3855a74ed182` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e6.log` | 33,620 | `3021d17d8b9b98e2a8f6e2503c136e04f7e2dec6c578be0c71efdef233eef7e7` | 安装与 serve 日志（含三次失败尝试原日志） |
| `logs/serve-e6b.log` | 33,952 | `2cf48986ff32803bf97c4b033ecd741147793a9ec861fec126e18f2d075bf211` | 安装与 serve 日志（含三次失败尝试原日志） |

## 树外登记（78 个 / 2,787,430,460 B）

字节不随仓发行；哈希为剥离当时的实测值。

| 文件 | 字节 | sha256 | 状态 | 原因 |
| --- | ---: | --- | --- | --- |
| `evidence/after-model/env-report-20260917-1910.md` | 86,901 | `83fff8f023aeccd5d7cd071cdb29f0bc71c0d8e985ababa00c2703ff96bfb06a` | snapshot | 模型运行后环境探针（中间快照）；首/末两份保留在 evidence/before 与 evidence/after |
| `evidence/after/env-report-20260917-1637.md` | 86,831 | `af384bff870225c4a3e50d11a57e791492923589f05a71cd5b89797a1631980a` | snapshot | 同一状态（已 source env.sh）的中间刷新；首/末两份为 evidence/before/env-report-20260917-1559.md 与 evidence/after/env-report-20260917-1716.md |
| `evidence/after/env-report-20260917-1655.md` | 86,855 | `3605428293f62a78c4b2a0153a2f121f48e6d0758b8c343212902ffe06a73da4` | snapshot | 同一状态（已 source env.sh）的中间刷新；首/末两份为 evidence/before/env-report-20260917-1559.md 与 evidence/after/env-report-20260917-1716.md |
| `evidence/p0-model/cuda-upgrade-freeze-before.txt` | 4,118 | `a21cb93333fa99960425b2d2ef9a66b4f9677cd59001ea500e3ea6f97eb6e302` | duplicate | 与 requirements.freeze.stage-01.txt 逐字节相同 |
| `evidence/p0-model/cuda-upgrade-freeze-final.txt` | 4,118 | `350d32f4804ada87ad8617cabe64c07743fc43a1685f57d1b7a34ab374c89d01` | duplicate | 与 reports/p0-model/requirements.freeze.txt 逐字节相同 |
| `evidence/p1-cpu/evidence-index.md` | 1,349 | `ca63c3f15553c399bc6104814acf36c07398c9501f3e75f99b9dc13c21c767b4` | superseded | 机械 size/sha256 表，由 reports/evidence-index.md（生成物）取代 |
| `evidence/p1-gpu-local-review-20260918-1011/cases-seed0.json` | 69,001 | `2bb0601b77c8bb0d3103b9594b1f98b2358483bf33814e7253d5441e045c72b8` | independent-review | 本地主代理独立复跑记录（CPU 111 项/4.435 s、GPU 972 判据/0 失败） |
| `evidence/p1-gpu-local-review-20260918-1011/cases-seed1.json` | 69,007 | `4168ff3553b77bd000b44fae182cbada9efb9f109a24f9746e2694c0a81b172f` | independent-review | 本地主代理独立复跑记录（CPU 111 项/4.435 s、GPU 972 判据/0 失败） |
| `evidence/p1-gpu-local-review-20260918-1011/cases-seed2.json` | 68,991 | `b07d7bb9020276495aa8916a6f8cdff27923ea37569c001502a471ba38e51da2` | independent-review | 本地主代理独立复跑记录（CPU 111 项/4.435 s、GPU 972 判据/0 失败） |
| `evidence/p1-gpu-local-review-20260918-1011/run-manifest.json` | 1,102 | `9f0264bdaf7124d74b76209ddfc0c6a9b2a0b80ad26b69db3c9c296712d35a7d` | independent-review | 本地主代理独立复跑记录（CPU 111 项/4.435 s、GPU 972 判据/0 失败） |
| `evidence/p1-gpu-local-review-20260918-1011/summary.json` | 238,574 | `cc30f3de73c698e536f7e9855c30a7af44e8d2058abcffe914c1e9c482ec6301` | independent-review | 本地主代理独立复跑记录（CPU 111 项/4.435 s、GPU 972 判据/0 失败） |
| `evidence/p1-gpu-v2/evidence-index.md` | 1,113 | `b267fd077c9b8c8ed6acea5999c19871dfc9d285d8415ee032d81bb044bf1415` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/negative-control.json` | 3,438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/negative-control.log` | 405 | `4eb57fa301a36dd44ec261d1727ac665898b9fec1db2474ebefb8f4a8edd72a0` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/v2-cases-seed0.json` | 53,906 | `0820dbf7a1340134241eee033f2fe69b94c1382728b9510802c0ca43a20212e3` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/v2-cases-seed1.json` | 53,915 | `36701d2a3b0a7a18f42c7dad1d9f67d575d6604be9ac3c287b9d75acc2c3ea51` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/v2-cases-seed2.json` | 53,898 | `0f789e267dda7e15bf6f9c3987614acf2683471a640cc33decebf2c5100b7a9d` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/v2-run.log` | 7,539 | `71f548748d665ccb0c10ea58ac9060574ba57dcb0d1bf32b0412923300d2e3f6` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v2/v2-summary.json` | 175,989 | `596d3abf98378d8e70125ddcdb3861af3e7ed8a15b786368dc6edd583f269f7f` | superseded | v2：trajectory step2 的 expect_blocks 漏块 5 且未被代码读取，结论已撤回 |
| `evidence/p1-gpu-v3/cases-seed0.json` | 69,001 | `12e20a676c1a067d299c50d9249d4d0ce3791bb05b90a9359918084a5939ebe0` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/cases-seed1.json` | 69,007 | `a585b22b8dd3dec7dd72c07c5d83c15f9d1d3563ed67f48e5d0ef49b14c553ef` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/cases-seed2.json` | 68,991 | `266d3c4f5dc8ec7229a187da9d11f74f4ca73cf23b74afd9fadedc4eb428ce42` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/evidence-index.md` | 1,392 | `dc02e41a9b2fe65afa762605ca8f57b796e0db0e4bfd8898225d7940ad985a57` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/negative-control.json` | 3,438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/negative-control.log` | 405 | `1139e95ee8343872357489b609e83516fef69ce7b3b631b24992c228d73fe247` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/summary.json` | 238,574 | `aa01627e091b42eb7e6eaf95e9e6d918a2e34c2deae4a393131915befaa554fe` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v3/v3-run.log` | 8,085 | `659869990e3dec3ba83d78b7820f1108a335371a8ff8e4eea180949540734a4b` | superseded | v3：配置提交晚于运行（b96d08a 10:07:10 vs 产物 10:06:49），时序不合规 |
| `evidence/p1-gpu-v4/cases-seed0.json` | 69,001 | `c936641179ded70b6371e13c42ec32bf291dfd2a5211e86509fde5695eb0d882` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/cases-seed1.json` | 69,007 | `8e73e7eb4d799d12bf5d0d2aeac8d65cf848a3792f58cf4212627f59adaf1864` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/cases-seed2.json` | 68,991 | `f717a9a025b1e82e45f07eecec31311b497f6cde32eee082ae718d22830294ed` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/negative-control.json` | 3,438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/negative-control.log` | 551 | `ef7160a7adb881563a40e6b6676740356dcb7621dc958cf87f675426e4bc62e7` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/run-manifest.json` | 846 | `57658e0411fd98e5d74d816cf59e9b0e81c5fae2af55cf2e2a3c0404d6724a38` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/run.log` | 8,300 | `92b0676ca92e1e6f121243aa1aeb3242375a24bf971e7cca45de85179dbc675b` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v4/summary.json` | 238,574 | `5e18e9573df1615cfe9e96ee56d8552b09f34737d5b3604b746d34ce42076835` | superseded | v4：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/cases-seed0.json` | 69,001 | `4a8548abd3baea2c5737fe29a96d53820a1017a34e0ee8353985e98c0dfa7015` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/cases-seed1.json` | 69,007 | `2fc7e892e5df96c036c1e77f204d691dc7c8383e74ff49b6ed81ea32eeedd5d4` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/cases-seed2.json` | 68,991 | `1ac90c07c5292d82d6f8151e482ed0520269e7194fd4ff2f4074643469213863` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/negative-control.json` | 3,438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/negative-control.log` | 551 | `a7038bdc01b713dcb9a96d27b4a9b63f76c1fb0bca62de56aa429c066ef305e8` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/run-manifest.json` | 1,028 | `965037217128f1d66507b44b1b35a7a90cd4576c83c7360bbc9e2e861389a5f1` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/run.log` | 8,300 | `937f2bf8dd26b43586f2997ac02d9f3d81d78e1a222a5acd341816a0b830d6e2` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu-v5/summary.json` | 238,574 | `f2711c57e2db08240f9346ab38360c5335c152e2f25e006b0f6fb96a801aa8a5` | superseded | v5：运行清单口径/命名未定稿 |
| `evidence/p1-gpu/cases-seed0.json` | 16,773 | `d825996514160ad9561351ce3df059db44532e1f79f15902592c65a077327a49` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/cases-seed1.json` | 16,781 | `379916273f30b7815e5fc47365aeab814b6385c2347d8c9324cb58aada699860` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/cases-seed2.json` | 16,784 | `d9b721a1a216718530b6eeb68ebadc763fa03fbd5372b507874ce620ab6dad78` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/evidence-index.md` | 1,196 | `dc806f98d7cb91c84ff7d4b394678e494106435a013b7846c9694bc74e790d76` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/negative-control.json` | 2,056 | `d2ff889980024218fc862a8f10309554348ca78d48441347b6fe8834bd94d471` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/negative-control.log` | 402 | `865da10c9a4003a66316a8c444ad6d530c6cff1c0506c6adb5bc61548d2b1566` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/run-partial-v1.log` | 1,781 | `4966fa6f4e5ff8e0a22ffe69ff96323ddacbdeac3567a9119f5d1b6564564b2d` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/run.log` | 3,260 | `9dd7c7482f1b573d024fdb177899c7935f7539478ee9c75b03ed4c01c88d70ba` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p1-gpu/summary.json` | 61,124 | `fad0472659e155fb34e5bf7745299f07aeb0b937f515adc6430a70ed6e513191` | superseded | v1：夹具逻辑/物理错位（按 i 写入、按 l2p[i] 读取），结论已撤回；撤回记录见 evidence/p1-gpu-v6/evidence-index.md |
| `evidence/p2-single-local-review-params-20260918/params-channel-probe.json` | 2,605 | `b0fe8b6fe2404189b8911288a20522f2fee48f213288e4f1762c836574a63dd7` | independent-review | 阶段 02 params 通道探针复核记录 |
| `evidence/p2-single-local-review-r1-20260918/findings.json` | 1,001 | `ef0cc46c3a5cfc2b775ca83ea147d01d60faf9a09361e1ca0eb88d7f11099ea7` | independent-review | 阶段 02 R1 findings 复核记录 |
| `evidence/p2-single-local-review-r2-20260918/deployment-unknown-state.json` | 516 | `60e24446c69e9ad0502604a2f20c4ff7f8106158d463e47df6768aa1ac8b7784` | independent-review | 阶段 02 R2 部署未知态复核记录 |
| `evidence/p3-calib-local-review-20260918/capture-byte-identity.json` | 859 | `655b4a2843cdc5652b0be9fa1cd01f3a632e4bf2ac6a689d40d162c7d8ff93e9` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/driver-findings.json` | 264 | `bcce8eef53b699973dbb424e3db80cdc37781aafafb8279ee30c99a75d2c1a5c` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/engine-contract-findings.json` | 500 | `8c296b1fe6caf78b7a7ff4208c08e8a33f632ecbd9c6da277bedcf3455642a35` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/final-v5-artifacts.json` | 1,089 | `79e1a40f50bb4647fb17254b20c5805e1d8f75aa9d57c5eba5e8a90d6284c4e9` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/metadata-retry.chrome.json` | 34,090 | `4e610ad31539b932068feb9efd2c0fa220fad98adfdc5a8dd0d444c638bfe0af` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/metadata-retry.json` | 5,370 | `b1709a1a134229a286751efca5918b855b632e4e8d9cf46640b7c9826e559a25` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/oracle-worstcase-independent.json` | 1,324 | `b3ea79ddd71cfe90d1efa9ddc4bb0b42062a362b62f5fff262faeca1e582e50e` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/original-repeat-tensors.json` | 28,137 | `e3c7c2f5970f9c8d0041af5941c60f9b66b76673a83bf7e95b169bd5bd30dda9` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/output-mode-findings.json` | 1,532 | `6d5ab34f8053da2f1e4c75925ffeb4d43a4592c4e06562b823a2e07d9f070971` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/patched-v4-main-tensors.json` | 63,815 | `887bae9b0a139e6e8481f72b1f233f72759c0078f36217dc3a90e800a1b29756` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib-local-review-20260918/r2-note-recheck.json` | 1,708 | `8195b79b611fcd2167eaefa1da0048f2b426c307f4b65421f6f61ad3c6f0cbe3` | independent-review | 阶段 03 校准复核记录（含 capture 字节一致性、oracle 最坏 case 独立复算；结论要点留在 reports/p3-calib/evidence-index.md §2/§5） |
| `evidence/p3-calib/oracle-original-3a.json` | 8,268,191 | `3615eaa619134745879b98038a5f8783f925520e2d0ba73e1263cd4ecc110225` | detail-only | 24,192 行逐比较明细；入库版为 evidence/p3-calib/oracle-original-3a.aggregates.json，后者自带本文件 size + sha256，复核对 tools/p2-oracle-aggregate.py --check |
| `evidence/p3-calib/run-disabled-5/capture/forward1.npz` | 1,380,888,464 | `fdb9b06ff8e64b3b4d015ac71a33e68bccf6b572f35aa39861565370e1aff655` | remote-only | 张量 dump：被文档引用（见索引 A7 清单），字节留数据盘、不发行；哈希为本机实测 |
| `evidence/p3-calib/run-original-3a/capture/layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | remote-only | 张量 dump：被文档引用（见索引 A7 清单），字节留数据盘、不发行；哈希为本机实测 |
| `evidence/p3-calib/run-original-3a/logits.pt` | 7,950,079 | `24c6ad2fe504141dc65db3278c0cfdb7874ef9ae1a07064af02b709c2b849974` | remote-only | 张量 dump：被文档引用（见索引 A7 清单），字节留数据盘、不发行；哈希为本机实测 |
| `evidence/p3-masked-prep-local-review-20260920/crossblock.json` | 153,051 | `ca842c58579d63cb40e623e7cf0628b986b7c21f669afb17653c30d3c1c6e921` | independent-review | masked 跨块准备离线复核；crossblock.json 与 evidence/p3-calib/masked-prep/crossblock-note.json 逐字节相同 |
| `evidence/p3-masked-prep-local-review-20260920/manifest.json` | 1,343 | `1fd86238502a3a44f1ce4d3a651080a10b5bc38da80bf3335d4d56331089dceb` | independent-review | masked 跨块准备离线复核；crossblock.json 与 evidence/p3-calib/masked-prep/crossblock-note.json 逐字节相同 |
| `evidence/p3-masked-prep-local-review-20260920/run.log` | 572 | `7f8c64dbf7ea086b2761a87884a150c7bed69404fd8b5c977848b9aaa44b8faa` | independent-review | masked 跨块准备离线复核；crossblock.json 与 evidence/p3-calib/masked-prep/crossblock-note.json 逐字节相同 |
| `evidence/p3-masked-ref-local-review-20260920/preintegration-findings.json` | 820 | `a84017e9ae4f54598457a72c8b276bbb3e98573a0df6b335bee2f19fa810d8ae` | independent-review | masked 参考实现集成前 findings 复核记录 |
| `evidence/verify-after-expand.log` | 7,355 | `b15107b5b82f2e87aadc4e6dfa950d507eac2f18631cb6e6e12b8e9ca7159528` | duplicate | 与 evidence/verify-console.log 逐字节相同（同一校验输出，历史命名不同） |
| `evidence/verify-after-hf-endpoint.log` | 7,355 | `b15107b5b82f2e87aadc4e6dfa950d507eac2f18631cb6e6e12b8e9ca7159528` | duplicate | 与 evidence/verify-console.log 逐字节相同（同一校验输出，历史命名不同） |
| `evidence/verify-after-move.log` | 7,355 | `b15107b5b82f2e87aadc4e6dfa950d507eac2f18631cb6e6e12b8e9ca7159528` | duplicate | 与 evidence/verify-console.log 逐字节相同（同一校验输出，历史命名不同） |
| `logs/console-install.log` | 2,421 | `189d9b27c5e5ba7178ba12bc7e093e3ff8addde3c794dc969a4a8ec6122f6d57` | duplicate | 与 logs/install-runtime-20260917-163051.log 逐字节相同 |

## 在盘未入库集合（13 组 / 27,515,842,531 B）

按 R9 的类型规则排除（张量/序列化大件）；不与任何文档逐条绑定的组只登记组级事实，
被文档引用的成员在上表逐个登记。计数口径：`glob` 展开 **减去**已逐个登记的成员（树内上表 / 树外登记表），由 A9 逐组复算。

| glob | 文件数 | 合计字节 | 状态 | 说明 |
| --- | ---: | ---: | --- | --- |
| `evidence/p3-calib/run-*/capture/**` | 52 | 13,881,579,550 | remote-only | p3-calib 各 run 的 capture dump（layers.npz / forward1.npz / forward2..8.npz）：R9 类型规则排除，字节留数据盘；未被文档逐条引用，只登记组级事实（逐文件哈希可随时复算）；被文档引用的成员已在上表逐个登记 |
| `evidence/p3-calib/run-*/logits.pt` | 6 | 49,696,240 | remote-only | p3-calib 各 run 的 logits 序列化：R9 类型规则排除，字节留数据盘；未被文档逐条引用，只登记组级事实（逐文件哈希可随时复算）；被文档引用的成员已在上表逐个登记 |
| `evidence/p3-masked-smoke/diagnostic-20260921-first/run/capture/**` | 6 | 1,027,686,562 | remote-only | 首次 masked 冒烟 run 的 capture dump：R9 类型规则排除，字节留数据盘；文本层已逐个入库 |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/capture/**` | 30 | 2,068,470,486 | remote-only | 窄修复后 masked 冒烟 run 的 capture dump：R9 类型规则排除，字节留数据盘；文本层已逐个入库 |
| `evidence/p3-masked-smoke/diagnostic-20260921-current/**` | 69 | 2,102,977,859 | remote-only | 被取代的 masked 诊断 run（run4 对照轮）：文本层与 capture 都留数据盘，不随仓发行 |
| `evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/capture/**` | 30 | 2,068,470,486 | remote-only | 终版 masked 诊断 run 的 capture dump：R9 类型规则排除，字节留数据盘；structure.json 与 manifest 入库 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/capture/**` | 31 | 2,073,556,148 | remote-only | 终版独立 reference run 的 capture dump（逐步 KV/Q/out）：R9 类型规则排除，字节留数据盘；structure.json 与 manifest 入库 |
| `evidence/p3-masked-reference/reference-20260921-run5/run/reference.json` | 1 | 5,085,814 | remote-only | reference 报告的重复导出（manifest.reference 的同源副本）：字节留数据盘，避免同一内容入库两份 |
| `evidence/p3-masked-reference/reference-20260921-run1/**` | 76 | 2,117,943,139 | remote-only | 被取代的独立 reference 代次（run1 无 GDN 探针、run2/run3 探针拒绝共享 storage、run4 首版探针）：文本层与 capture 都留数据盘，只登记组级事实 |
| `evidence/p3-masked-reference/reference-20260921-run2/**` | 37 | 313,865 | remote-only | 被取代的独立 reference 代次（run1 无 GDN 探针、run2/run3 探针拒绝共享 storage、run4 首版探针）：文本层与 capture 都留数据盘，只登记组级事实 |
| `evidence/p3-masked-reference/reference-20260921-run3/**` | 37 | 313,866 | remote-only | 被取代的独立 reference 代次（run1 无 GDN 探针、run2/run3 探针拒绝共享 storage、run4 首版探针）：文本层与 capture 都留数据盘，只登记组级事实 |
| `evidence/p3-masked-reference/reference-20260921-run4/**` | 75 | 2,119,717,374 | remote-only | 被取代的独立 reference 代次（run1 无 GDN 探针、run2/run3 探针拒绝共享 storage、run4 首版探针）：文本层与 capture 都留数据盘，只登记组级事实 |
| `evidence/p3-masked-smoke/diagnostic-20260921-retry1/numeric-audit-current.json` | 1 | 31,142 | remote-only | 被取代的 numeric-audit 副本（仅 audit_head/脚本哈希不同）：留数据盘，入库的是 retry1/numeric-audit.json |

## 已声明缺失 / 未执行（14 条）

被文档或 manifest 声明、但任何位置都没有的路径；留档以免每次审计重新判一遍。

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| `evidence/p3-calib/run-disabled-4/steps.jsonl` | missing | arm.json 的 trace_path，该 run 无此文件（reports/p3-calib/evidence-index.md §3.1） |
| `evidence/p3-calib/run-disabled-5/steps.jsonl` | missing | 同构（§3.1） |
| `evidence/p3-calib/run-global-4/steps.jsonl` | missing | 同构（§3.1） |
| `evidence/p3-calib/run-global-5/steps.jsonl` | missing | 同构（§3.1） |
| `evidence/p3-calib/run-original-2a/arm.json` | missing | manifest 自述 arm_file.exists=false（§3.1） |
| `evidence/p3-calib/traj-r1.json` | missing | 失败尝试未写出轨迹（§3.1、§4） |
| `evidence/p3-calib/run-original-1/` | missing | 登记时为空目录并已消失；是否曾产出工件未能判定（§1.12） |
| `evidence/p2-single/patch-manifest.json` | planned | reports/p2-single/integration-design.md 的设计产物名；实际部署记录为 evidence/p2-single/deploy-drill-*.log 与各 run 的 manifest.json |
| `evidence/p3-calib/run-original-4a` | planned | 四臂复跑命令中的目标目录，尚未执行（reports/p2-single/calibration-report.md） |
| `evidence/p3-calib/run-disabled-6` | planned | 同上 |
| `evidence/p3-calib/run-global-6` | planned | 同上 |
| `evidence/p3-calib/traj-original-2.json` | planned | 同上（复跑命令里的轨迹路径） |
| `evidence/p3-calib/oracle-original-3a-rerun.json` | planned | 复跑命令里的 oracle 输出路径，尚未执行；全量件哈希登记见 reports/evidence-index.md 树外登记表（聚合段的 full_file 字段同值） |
| `evidence/p3-calib/run-original-4b` | planned | 四臂复跑命令中的目标目录，尚未执行（reports/p2-single/calibration-report.md） |

## 自检

- A7 reports/evidence-registry.json 引用 78 个树外登记件，表述需与登记状态一致：evidence/after-model/env-report-20260917-1910.md（snapshot）、evidence/after/env-report-20260917-1637.md（snapshot）、evidence/after/env-report-20260917-1655.md（snapshot）、evidence/after（snapshot）、evidence/p0-model/cuda-upgrade-freeze-before.txt（duplicate）、evidence/p0-model/cuda-upgrade-freeze-final.txt（duplicate）、evidence/p1-cpu/evidence-index.md（superseded）、evidence/p1-gpu-local-review-20260918-1011/cases-seed0.json（independent-review）、evidence/p1-gpu-local-review-20260918-1011/cases-seed1.json（independent-review）、evidence/p1-gpu-local-review-20260918-1011/cases-seed2.json（independent-review）、evidence/p1-gpu-local-review-20260918-1011/run-manifest.json（independent-review）、evidence/p1-gpu-local-review-20260918-1011/summary.json（independent-review）、evidence/p1-gpu-v2/evidence-index.md（superseded）、evidence/p1-gpu-v2/negative-control.json（superseded）、evidence/p1-gpu-v2/negative-control.log（superseded）、evidence/p1-gpu-v2/v2-cases-seed0.json（superseded）、evidence/p1-gpu-v2/v2-cases-seed1.json（superseded）、evidence/p1-gpu-v2/v2-cases-seed2.json（superseded）、evidence/p1-gpu-v2/v2-run.log（superseded）、evidence/p1-gpu-v2/v2-summary.json（superseded）、evidence/p1-gpu-v3/cases-seed0.json（superseded）、evidence/p1-gpu-v3/cases-seed1.json（superseded）、evidence/p1-gpu-v3/cases-seed2.json（superseded）、evidence/p1-gpu-v3/evidence-index.md（superseded）、evidence/p1-gpu-v3/negative-control.json（superseded）、evidence/p1-gpu-v3/negative-control.log（superseded）、evidence/p1-gpu-v3/summary.json（superseded）、evidence/p1-gpu-v3/v3-run.log（superseded）、evidence/p1-gpu-v4/cases-seed0.json（superseded）、evidence/p1-gpu-v4/cases-seed1.json（superseded）、evidence/p1-gpu-v4/cases-seed2.json（superseded）、evidence/p1-gpu-v4/negative-control.json（superseded）、evidence/p1-gpu-v4/negative-control.log（superseded）、evidence/p1-gpu-v4/run-manifest.json（superseded）、evidence/p1-gpu-v4/run.log（superseded）、evidence/p1-gpu-v4/summary.json（superseded）、evidence/p1-gpu-v5/cases-seed0.json（superseded）、evidence/p1-gpu-v5/cases-seed1.json（superseded）、evidence/p1-gpu-v5/cases-seed2.json（superseded）、evidence/p1-gpu-v5/negative-control.json（superseded）、evidence/p1-gpu-v5/negative-control.log（superseded）、evidence/p1-gpu-v5/run-manifest.json（superseded）、evidence/p1-gpu-v5/run.log（superseded）、evidence/p1-gpu-v5/summary.json（superseded）、evidence/p1-gpu/cases-seed0.json（superseded）、evidence/p1-gpu/cases-seed1.json（superseded）、evidence/p1-gpu/cases-seed2.json（superseded）、evidence/p1-gpu/evidence-index.md（superseded）、evidence/p1-gpu/negative-control.json（superseded）、evidence/p1-gpu/negative-control.log（superseded）、evidence/p1-gpu/run-partial-v1.log（superseded）、evidence/p1-gpu/run.log（superseded）、evidence/p1-gpu/summary.json（superseded）、evidence/p2-single-local-review-params-20260918/params-channel-probe.json（independent-review）、evidence/p2-single-local-review-r1-20260918/findings.json（independent-review）、evidence/p2-single-local-review-r2-20260918/deployment-unknown-state.json（independent-review）、evidence/p3-calib-local-review-20260918/capture-byte-identity.json（independent-review）、evidence/p3-calib-local-review-20260918/driver-findings.json（independent-review）、evidence/p3-calib-local-review-20260918/engine-contract-findings.json（independent-review）、evidence/p3-calib-local-review-20260918/final-v5-artifacts.json（independent-review）、evidence/p3-calib-local-review-20260918/metadata-retry.chrome.json（independent-review）、evidence/p3-calib-local-review-20260918/metadata-retry.json（independent-review）、evidence/p3-calib-local-review-20260918/oracle-worstcase-independent.json（independent-review）、evidence/p3-calib-local-review-20260918/original-repeat-tensors.json（independent-review）、evidence/p3-calib-local-review-20260918/output-mode-findings.json（independent-review）、evidence/p3-calib-local-review-20260918/patched-v4-main-tensors.json（independent-review）、evidence/p3-calib-local-review-20260918/r2-note-recheck.json（independent-review）、evidence/p3-calib/oracle-original-3a.json（detail-only）、evidence/p3-calib/run-disabled-5/capture/forward1.npz（remote-only）、evidence/p3-calib/run-original-3a/capture/layers.npz（remote-only）、evidence/p3-calib/run-original-3a/logits.pt（remote-only）、evidence/p3-masked-prep-local-review-20260920/crossblock.json（independent-review）、evidence/p3-masked-prep-local-review-20260920/manifest.json（independent-review）、evidence/p3-masked-prep-local-review-20260920/run.log（independent-review）、evidence/p3-masked-ref-local-review-20260920/preintegration-findings.json（independent-review）、evidence/verify-after-expand.log（duplicate）、evidence/verify-after-hf-endpoint.log（duplicate）、evidence/verify-after-move.log（duplicate）
- A7 reports/desensitization.md 引用 14 个树外登记件，表述需与登记状态一致：evidence/after-model/env-report-20260917-1910.md（snapshot）、evidence/after/env-report-20260917-1637.md（snapshot）、evidence/after/env-report-20260917-1655.md（snapshot）、evidence/p1-gpu-local-review-20260918-1011/run-manifest.json（independent-review）、evidence/p1-gpu-v4/run-manifest.json（superseded）、evidence/p1-gpu-v5/run-manifest.json（superseded）、evidence/p2-single-local-review-r1-20260918/findings.json（independent-review）、evidence/p2-single-local-review-r2-20260918/deployment-unknown-state.json（independent-review）、evidence/p3-calib-local-review-20260918/final-v5-artifacts.json（independent-review）、evidence/p3-calib-local-review-20260918/metadata-retry.chrome.json（independent-review）、evidence/p3-calib-local-review-20260918/output-mode-findings.json（independent-review）、evidence/p3-calib-local-review-20260918/r2-note-recheck.json（independent-review）、evidence/p3-masked-prep-local-review-20260920/manifest.json（independent-review）、evidence/p3-masked-ref-local-review-20260920/preintegration-findings.json（independent-review）
- A7 reports/p3-calib/evidence-index.md 引用 14 个树外登记件，表述需与登记状态一致：evidence/p1-gpu-local-review-20260918-1011（independent-review）、evidence/p2-single-local-review-params-20260918（independent-review）、evidence/p2-single-local-review-r1-20260918（independent-review）、evidence/p2-single-local-review-r2-20260918（independent-review）、evidence/p3-calib-local-review-20260918（independent-review）、evidence/p3-calib/oracle-original-3a.json（detail-only）、evidence/p3-calib/run-disabled-5/capture/forward1.npz（remote-only）、evidence/p3-calib/run-disabled-5（remote-only）、evidence/p3-calib/run-original-3a/capture/layers.npz（remote-only）、evidence/p3-calib/run-original-3a/logits.pt（remote-only）、evidence/p3-calib/run-original-3a（remote-only）、evidence/p3-calib（remote-only）、evidence/p3-masked-prep-local-review-20260920（independent-review）、evidence/p3-masked-ref-local-review-20260920（independent-review）
- A7 reports/p1-gpu/read-view-report.md 引用 6 个树外登记件，表述需与登记状态一致：evidence/p1-gpu-local-review-20260918-1011（independent-review）、evidence/p1-gpu-v2（superseded）、evidence/p1-gpu-v3（superseded）、evidence/p1-gpu-v4（superseded）、evidence/p1-gpu-v5（superseded）、evidence/p1-gpu（superseded）
- A7 reports/p0-model/dependency-delta.md 引用 3 个树外登记件，表述需与登记状态一致：evidence/p0-model/cuda-upgrade-freeze-before.txt（duplicate）、evidence/p0-model/cuda-upgrade-freeze-final.txt（duplicate）、evidence/p0-model（duplicate）
- A7 reports/p0-model/evidence-index.md 引用 3 个树外登记件，表述需与登记状态一致：evidence/after-model/env-report-20260917-1910.md（snapshot）、evidence/after-model（snapshot）、evidence/p0-model（duplicate）
- A7 reports/p2-single/calibration-report.md 引用 3 个树外登记件，表述需与登记状态一致：evidence/p3-calib/run-disabled-5（remote-only）、evidence/p3-calib/run-original-3a/capture/layers.npz（remote-only）、evidence/p3-calib/run-original-3a（remote-only）
- A7 CONTRIBUTING.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/p3-calib/run-disabled-5（remote-only）、evidence（duplicate）
- A7 configs/p1-gpu/read-view-check-v2.json 引用 2 个树外登记件，表述需与登记状态一致：evidence/p1-gpu-v2（superseded）、evidence/p1-gpu（superseded）
- A7 configs/p1-gpu/read-view-check-v3.json 引用 2 个树外登记件，表述需与登记状态一致：evidence/p1-gpu-v2（superseded）、evidence/p1-gpu-v3（superseded）
- A7 reports/environment-setup.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/after（snapshot）、evidence/verify-after-move.log（duplicate）
- A7 reports/p0-model/model-report.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/after-model/env-report-20260917-1910.md（snapshot）、evidence/p0-model（duplicate）
- A7 reports/p1-cpu/session-closeout.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/p1-cpu/evidence-index.md（superseded）、evidence/p1-cpu（superseded）
- A7 reports/p1-cpu/stage-03-report.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/p1-cpu/evidence-index.md（superseded）、evidence/p1-cpu（superseded）
- A7 reports/p1-gpu/session-closeout.md 引用 2 个树外登记件，表述需与登记状态一致：evidence/p1-gpu-local-review-20260918-1011（independent-review）、evidence/p1-gpu（superseded）
- A7 .gitmessage 引用 1 个树外登记件，表述需与登记状态一致：evidence/p3-calib/run-disabled-5（remote-only）
- A7 README.md 引用 1 个树外登记件，表述需与登记状态一致：evidence（duplicate）
- A7 configs/p1-gpu/read-view-check.json 引用 1 个树外登记件，表述需与登记状态一致：evidence/p1-gpu（superseded）
- A7 reports/publication-checklist.md 引用 1 个树外登记件，表述需与登记状态一致：evidence/p1-gpu（superseded）
- 同哈希（4,118 B，小于阈值、保留）：evidence/after/pip_freeze.txt, evidence/p0-model/cuda-upgrade-freeze-after.txt
- 同哈希（4,625 B，小于阈值、保留）：evidence/p0-model/e3-protocol-console.txt, evidence/p0-model/e3-protocol-tokens.json
- 同哈希（allow_duplicate 声明）：evidence/p1-cpu/prompt-da.txt, evidence/p1-cpu/prompt-da_no_mask.txt
- 同哈希（allow_duplicate 声明）：evidence/p3-calib/run-disabled-3/source/p2-calib-run.py, evidence/p3-calib/run-original-3a/source/p2-calib-run.py, evidence/p3-calib/run-original-3b/source/p2-calib-run.py
- 同哈希（allow_duplicate 声明）：evidence/p3-calib/run-disabled-4/source/p2-calib-run.py, evidence/p3-calib/run-global-4/source/p2-calib-run.py
- 同哈希（allow_duplicate 声明）：evidence/p3-calib/run-disabled-5/source/p2-calib-run.py, evidence/p3-calib/run-global-5/source/p2-calib-run.py
- 同哈希（1,200 B，小于阈值、保留）：evidence/p3-masked-reference/reference-20260921-run5/revert.log, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/revert.log, evidence/p3-masked-smoke/diagnostic-20260921-first/revert.log, evidence/p3-masked-smoke/diagnostic-20260921-retry1/revert.log
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/_lib.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/_lib.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/decode.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/decode.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/extract.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/extract.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/fixture_rebuild.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/fixture_rebuild.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/fixture_rebuild.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/fixture_rebuild.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpucheck.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpucheck.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpukv.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpukv.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/gpuoracle.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/gpuoracle.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpuoracle.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpuoracle.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/parser.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/parser.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/parser.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/parser.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/prompt.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/prompt.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/prompts.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/prompts.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/prompts.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/prompts.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/readview.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/readview.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_bridge.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_bridge.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_dense.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_dense.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_hook.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_hook.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/reference_run.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/reference_run.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/segmenter.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/segmenter.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/smoke_structure.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/smoke_structure.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/smoke_structure.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/smoke_structure.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/state.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/state.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/step_plan.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/step_plan.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/attnview/trace.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/attnview/trace.py, evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/trace.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/trace.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-reference/reference-20260921-run5/run/source/p2-calib-run.py, evidence/p3-masked-smoke/diagnostic-20260921-4d32a16/run/source/p2-calib-run.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/decode.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/decode.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/extract.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/extract.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpucheck.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpucheck.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/gpukv.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/gpukv.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/prompt.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/prompt.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/readview.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/readview.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_bridge.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_bridge.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_dense.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_dense.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/reference_hook.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/reference_hook.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/segmenter.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/segmenter.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/state.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/state.py
- 同哈希（allow_duplicate 声明）：evidence/p3-masked-smoke/diagnostic-20260921-first/run/source/attnview/step_plan.py, evidence/p3-masked-smoke/diagnostic-20260921-retry1/run/source/attnview/step_plan.py
- 入库 356 个 / 树外登记 78 个 / 在盘未入库 0 个
- 引用扫描面：reports, configs + README.md, CONTRIBUTING.md, .gitmessage
