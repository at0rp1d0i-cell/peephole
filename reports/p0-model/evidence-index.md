# 阶段 02 证据索引（远端 `lab`，2026-09-17 UTC）

规则：每条给出**实际命令**、**退出码**、**远端路径**与**内容摘要**。原始日志只留远端；
本目录（素材仓 `results/p0-model/`）只放脱敏报告、可复跑命令与索引。
所有产物落**数据盘**（`/root/attnview` → `/root/autodl-tmp/attnview`），系统盘只被 `torch.compile` 缓存写入（见 model-report §6 阻塞项 2）。

## E0 现状快照与环境复核

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `source /root/attnview/env.sh; bash verify-runtime.sh` | 0 | `evidence/after/*.txt`、`evidence/p0-model/e0-baseline.txt` | 全部检查通过（解释器 / `pip check` / import+设备 / 小张量 CUDA / vLLM CLI / 项目自带 nvcc） |
| `nvidia-smi --query-gpu=...`、`df -h`、`cat material/attnview/SNAPSHOT-COMMIT.txt` | 0 | 同上 | GPU 0 MiB/0%、cc 12.0；数据盘 192 GiB 可用；快照 commit 记录 |
| `git -C /root/attnview/vllm rev-parse HEAD; git status --porcelain` | 0 | 同上 | `98dff2a8…`（tag v0.29.0），工作区干净 |

## E1 模型 identity 复核

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `"$ATTNVIEW_PYTHON" tools/e1-model-identity.py --out evidence/p0-model` | 0 | `e1-model-identity.json`、`e1-console.txt` | 两候选 revision 与文件数/总字节**零差异**；3.8-27B 32 文件 / 55,586,114,863 B；均 public/非 gated/apache-2.0 |

## E2 下载与校验

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2… --exclude '*.safetensors'` | 0 | `e2-small-files.txt` | 9.4 s 取回 14 个非权重文件，供 E3 先行核对 |
| `bash tools/e2-download.sh`（内部执行上条命令的完整版，hub 托管） | 0 | `e2-download.log`、`e2-snapshot-files.txt` | **6135 s**（17:10:35Z→18:52:50Z），均速 ≈9.06 MB/s；`df` 前后 +55.57 GB |
| `hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2…`（重跑） | 0 | `e2-small-files.txt` 之后追加 | 0.32 s，命中缓存（"已校验并跳过"） |
| `"$ATTNVIEW_PYTHON" tools/e2-verify.py --snapshot <snapshot> --identity …` | 0 | `e2-verify.json`、`e2-verify-console.txt` | 32/32 大小一致、19/19 LFS sha256 一致、总字节精确相等、无缺失/多余文件 |

## E3 配置、tokenizer 与模板

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `"$ATTNVIEW_PYTHON" tools/e3-config-audit.py --snapshot <snapshot> --out evidence/p0-model` | 0 | `e3-config-audit.json`、`e3-console.txt`、`e3-prompt-thinking_{disabled,default}.txt` | 层结构/KV 算术/上下文/dtype/模板开关/特殊 token/文件 sha256；两种模板渲染样例与 token 数（36 / 72） |
| `"$ATTNVIEW_PYTHON" tools/e3-probe-protocol-tokens.py --snapshot <snapshot> --out evidence/p0-model` | 0 | `e3-protocol-tokens.json`、`e3-prompt-protocol-mixed.txt` | 8 个协议标签均不在词表，切分长度 3–8 token |
| （本地增量核对）13 个非 LFS 文件与 Hub `blob_id`（git blob sha1）比对 | 0 | 见 `e2-verify-console.txt` 同批次输出 | 13/13 一致；由此判定上游 `crc32.txt` 陈旧 |

## E4 原版服务启动（含三次失败）

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| 预置基线（启动前） | — | `e4-expected-baseline.md` | 默认值出处、KV/GDN 显存算术、失败分类特征、FA4-hd256 判定链 |
| `bash tools/serve-vanilla.sh --tag e4-first`（首次，未加 `--revision`） | **1** | `logs/serve-e4-attempt1-offline-resolve-fail.log` | `LocalEntryNotFoundError`：离线 + `revision=None` |
| 同上（加 `--revision`，默认 `max_num_seqs=1024`） | **1** | `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (635)` |
| 同上（加 `--max-num-seqs 256`，仍启用 FlashInfer 采样） | **1** | `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | FlashInfer sampling JIT：`ninja` 失败，根因 CCCL `CUDA compiler and CUDA toolkit headers are incompatible`（nvcc 13.4 vs cuda.h 13.0） |
| `bash tools/serve-vanilla.sh --tag e4`（+`VLLM_USE_FLASHINFER_SAMPLER=0`） | 0（服务就绪；停止时 SIGTERM → 143） | `logs/serve-e4.log`、`e4-extract.json`、`e4-extract-console.txt`、`e4-metrics.txt` | backend/块大小/图模式/dtype/KV 预算逐项取证；`/health` 200；`/metrics` 200 |
| `"$ATTNVIEW_PYTHON" tools/e4-extract.py logs/serve-e4.log --out evidence/p0-model` | 0 | `e4-extract.json` | 8 组关键词命中行（含行号），供逐项引用 |

## E5 一次最小生成

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `"$ATTNVIEW_PYTHON" tools/e5-request.py --out evidence/p0-model --tag e5` | 0 | `e5-request.json`、`e5-response-raw.json`、`e5-record.json` | HTTP 200、0.397 s、`finish_reason=stop`、usage 25/8/33、输出 "The capital of France is Paris." |
| `"$ATTNVIEW_PYTHON" tools/e5-prompt-check.py --out evidence/p0-model --snapshot <snapshot>` | 0 | `e5-prompt-check.json`、`e5-prompt-server-thinking_{disabled,enabled}.txt` | 服务端 `enable_thinking=false`→36 token、`true`→72 token；服务端渲染与本地渲染 token id **完全一致** |

## E6 干净会话复跑与探针

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `env -i PATH=/usr/local/sbin:…:/bin HOME=/root bash --noprofile --norc tools/serve-vanilla.sh --tag e6`（hub 托管） | 0（停止时 143） | `logs/serve-e6.log`、`e6-extract.json`、`e6-metrics.txt` | 空环境 + 最小 PATH 下自举成功；KV/块大小/backend/图模式与 E4 **逐项一致**；`init engine` 44.06 s（compilation 1.11 s，缓存热） |
| `"$ATTNVIEW_PYTHON" tools/e5-request.py --out evidence/p0-model --tag e6` | 0 | `e6-request.json`、`e6-response-raw.json`、`e6-record.json` | 与 E5 同输入同输出（usage 25/8/33、`stop`、0.404 s） |
| `bash material/attnview/scripts/env-probe.sh --out /root/attnview/evidence/after-model` | 0 | `evidence/after-model/env-report-20260917-1910.md` | 模型运行后的机器/GPU/磁盘快照；结束后 GPU 回到 0 MiB |

## kernel 块大小运行时取证（2026-09-18，推翻早前推断）

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `PYTHONPATH=tools/kbs-probe KBS_PROBE_OUT=… bash tools/e4-kernel-block-probe.sh`（hub 托管，日志 `serve-e4c.log`） | 0（停止 143） | `evidence/p0-model/e4-kernel-block-probe.json`、`logs/serve-e4c.log` | **实测** `kernel_block_sizes=[784,784,784,784]`、manager 块同为 784、`hybrid_splitting_used=false`、`num_blocks=652`；KV 组 = 3×MambaSpec(`GDNAttentionBackend`, `[MultipleOf(1)]`) + 1×FullAttentionSpec(`FlashAttentionBackend`, `[MultipleOf(16)]`)，四组 `supports_manager_block_size=true` |
| 早期两次失败尝试（保留第二次的失败前产物） | 1 / 1 | `e4-kernel-block-probe-attempt1-nofinding.json` | 第一次（对象图遍历）没找到该值；第二次暴露真因：**EngineCore 在子进程**，父进程 monkeypatch 不生效 → 最终改用 sitecustomize 注入（脚本已弃用删除，结论保留） |

## 最终配置的干净会话复跑（E6b，取代修复前的 E6）

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `env -i PATH=<最小> HOME=/root bash --noprofile --norc tools/serve-vanilla.sh --tag e6b` | 0（停止 143） | `logs/serve-e6b.log`、`e6b-metrics.txt` | `Application startup complete.`；`init engine` 44.40 s；FlashInfer 采样启用；KV 31.24 GiB / 652 / 314,187；缓存落 `/root/attnview/caches/vllm/…`（新 `VLLM_CACHE_ROOT` 生效） |
| `tools/e5-request.py --tag e6b` | 0 | `e6b-{request,response-raw,record}.json` | 200 / `stop` / usage 25-8-33 / 0.371 s，输出与 e5/e5b/e6 一致 |

## 环境修复与修复后复验（2026-09-18，经用户授权）

| 命令 | 退出码 | 产物 | 摘要 |
| --- | --- | --- | --- |
| `pip freeze`（修复前基线） | 0 | `cuda-upgrade-freeze-before.txt` | `nvidia-cuda-runtime==13.0.96` |
| `"$ATTNVIEW_PYTHON" -m pip install --upgrade "nvidia-cuda-runtime==13.4.92"` | 0 | `cuda-upgrade-freeze-after.txt` | 与 `nvidia-cuda-nvcc/crt 13.4.92` 对齐；`pip check` 干净；freeze 仅此一行变化 |
| `ninja -C /root/.cache/flashinfer/0.6.18/120f/cached_ops/sampling -f build.ninja` | 0 | 会话输出（`.o` 三个 + `sampling.so` 2.3 MB） | 修复前同一命令两次失败：先 CCCL 编译期 `#error`，后 `ld: cannot find -lcudart` |
| 下载官方 stub：`curl -o cudart.tar.xz https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.4.92-archive.tar.xz` | 0 | sha256 `0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5` | 提取 `lib/stubs/libcuda.so` 装入项目前缀 |
| `bash /root/attnview/setup-local-cuda.sh`（两次） | 0 / 0 | 会话输出 | 第 4 步：版本一致性校验通过（13.4 = 13.4）、补 dev 链接、stub 就位、链接自检 ok（26672 B）；幂等 |
| `bash /root/attnview/verify-runtime.sh` | 0 | `evidence/after/*.txt` | 修复后全项通过 |
| `bash tools/serve-vanilla.sh --tag e4b`（无任何规避 flag） | 0（停止 143） | `logs/serve-e4b.log`、`e4b-metrics.txt` | 日志出现 `Using FlashInfer for top-p & top-k sampling.`；KV 31.24 GiB / 652 blocks / 314,187 tokens 与修复前逐项相同；`init engine` 44.87 s |
| `tools/e5-request.py --tag e5b`（贪心，与 E5 同参） | 0 | `e5b-{request,response-raw,record}.json` | 200 / `stop` / usage 25-8-33 / 0.394 s；输出与 E5 相同 |
| `tools/e5-request.py --tag e5b-sampling --temperature 0.7 --top-p 0.95 --top-k 20` | 0 | `e5b-sampling-{request,response-raw,record}.json` | 200 / `stop` / 0.422 s；证明 FlashInfer 采样路径运行时确实被走到（仅冒烟，不作质量结论） |

## 本阶段交付文档

| 文件 | 内容 |
| --- | --- |
| `model-report.md` | 主报告：机器、identity、配置核对、启动逐项取证、失败链、显存账、短生成、复跑、差异表、阻塞项、结论标签 |
| `serve-command.sh` | 实测可复跑的启动命令（含环境变量、端口、全部 flag 与两处必需说明） |
| `model-identity.md` | revision、32 个文件的大小与 sha256、tokenizer/模板哈希、特殊 token 表、渲染后 prompt 样例、协议标签切分 |
| `evidence-index.md` | 本文件 |
| `requirements.freeze.txt` | **最终权威依赖清单**（197 行；与阶段 01 版本仅 3 行 CUDA 组件差异） |
| `dependency-delta.md` | 依赖差异记录（修复前/中间态/最终态三段快照 + 复现命令 + 为何必须成组对齐） |
| `environment-rebuild.md` | 从空容器重建本环境的命令顺序与核对点 |

## 文件清单（路径 / 字节 / sha256，机器生成）
| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `evidence/p0-model/cuda-upgrade-freeze-after.txt` | 4,118 | `33585d37b124ca2dc86518d8c6606a03795ce3f21bd8e7e18139871a5d34a07f` |
| `evidence/p0-model/cuda-upgrade-freeze-before.txt` | 4,118 | `a21cb93333fa99960425b2d2ef9a66b4f9677cd59001ea500e3ea6f97eb6e302` |
| `evidence/p0-model/cuda-upgrade-freeze-final.txt` | 4,118 | `350d32f4804ada87ad8617cabe64c07743fc43a1685f57d1b7a34ab374c89d01` |
| `evidence/p0-model/e0-baseline.txt` | 1,153 | `8bd30dba9e4ec8a6bf0b4e4c63589e953aca17cb3104f1c88a7c8e3896e79f77` |
| `evidence/p0-model/e1-console.txt` | 447 | `8c84bc1341eaac32349c9a212e7003d210003ebb3a7de564031dd364adbdce0c` |
| `evidence/p0-model/e1-model-identity.json` | 15,698 | `9efa1bfda57e5e8a3df1afddaaed1df136cd6765c814aaf0b6ce82fdee362575` |
| `evidence/p0-model/e2-download.log` | 2,946 | `015da4f15589f5248e6215ccdc55da5c3a4813f63cbe65c4a9f9d8b8b66824be` |
| `evidence/p0-model/e2-small-files.txt` | 265 | `2d37ae9a3bc647254e5d7c500791348e0f4011346fa4d7926f5201c3b9e77255` |
| `evidence/p0-model/e2-snapshot-files.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `evidence/p0-model/e2-verify-console.txt` | 513 | `35753e435aa92f0134bf32dd77f1231fdb964f8774eeb99ddf4671413d13f25e` |
| `evidence/p0-model/e2-verify.json` | 11,451 | `27a4e8357e0dcef143faf0eebc287d793bc7f11127b2a1f2d0eef01199a8ba40` |
| `evidence/p0-model/e3-config-audit.json` | 12,824 | `9fc950575fb05b045b0a1f441a59adf2914e2993ccdaedb6d67b6116f10007ab` |
| `evidence/p0-model/e3-console.txt` | 6,796 | `69e4ebcac10dbfe7380df3b63a091a46d3048b1c8b9b52872bb6a2f72164dbf7` |
| `evidence/p0-model/e3-prompt-protocol-mixed.txt` | 258 | `08449a7b4a981da02f00cbd540f6f603027469111a3a398ec87e3c339439bdf2` |
| `evidence/p0-model/e3-prompt-thinking_default.txt` | 389 | `1cb7f1ed8bb2efc808ca0426512e0642a672f9b877faeb6e57b6ec417682655c` |
| `evidence/p0-model/e3-prompt-thinking_disabled.txt` | 191 | `7733165a3e19f7b199fce722c4e5f822dbb798057212d17e2e2a2b2fe78c8519` |
| `evidence/p0-model/e3-protocol-console.txt` | 4,625 | `4c3e43fa61a7d5af7285a14eff00e97729852f95b4be21e2804517e56c903ff1` |
| `evidence/p0-model/e3-protocol-tokens.json` | 4,625 | `4c3e43fa61a7d5af7285a14eff00e97729852f95b4be21e2804517e56c903ff1` |
| `evidence/p0-model/e4-expected-baseline.md` | 8,860 | `114015df7698daa3f2ee61f78639f279a1244bacd2311c0925df05f3dd221d9b` |
| `evidence/p0-model/e4-extract-console.txt` | 20,504 | `47be0b6e42a74bafe0a6e454c64f704918622d1deb710b9c71820f0a2877add2` |
| `evidence/p0-model/e4-extract.json` | 43,509 | `f61ca30baa1a2316a1084f91801c848f65355c803e4ae0e250cedf6920197b88` |
| `evidence/p0-model/e4-kernel-block-probe-attempt1-nofinding.json` | 238 | `61e159108cb612e5a82f5b1123cb13695b8659635ee444e8417331e9968591d5` |
| `evidence/p0-model/e4-kernel-block-probe.json` | 1,355 | `0d4e2b0cc5ebfa2d5ddcb4978783114114af702f17937d0dc50944884f3ed9e9` |
| `evidence/p0-model/e4-metrics.txt` | 53,262 | `3a9ea6568ecb8bc217c4c911b5186c535f65cfdf5e29332b075388afcca25a31` |
| `evidence/p0-model/e4b-metrics.txt` | 49,786 | `20cf6708e2598cbca1a35cb9ab841dffb516d772d9e8827058d8fe44ba0cf05c` |
| `evidence/p0-model/e5-prompt-check.json` | 5,262 | `5dc760e66390b4024be79cbf4d140eda6b2cde87feae9ece7e78c9ba846e672e` |
| `evidence/p0-model/e5-prompt-server-thinking_disabled.txt` | 191 | `7733165a3e19f7b199fce722c4e5f822dbb798057212d17e2e2a2b2fe78c8519` |
| `evidence/p0-model/e5-prompt-server-thinking_enabled.txt` | 389 | `1cb7f1ed8bb2efc808ca0426512e0642a672f9b877faeb6e57b6ec417682655c` |
| `evidence/p0-model/e5-record.json` | 1,968 | `0188d79f400243f0659614ac1410e847c5805f7ec6c78d27724908e1b2eaba33` |
| `evidence/p0-model/e5-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` |
| `evidence/p0-model/e5-response-raw.json` | 713 | `c4707698bbbfe4dad0507ae9c042731e675ec8fc6ce7ab55b2800152d39401a1` |
| `evidence/p0-model/e5b-record.json` | 1,968 | `6227e28d42d3c9c08da60a051956dddd061ecfc01473c03f3537cb09873edfb0` |
| `evidence/p0-model/e5b-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` |
| `evidence/p0-model/e5b-response-raw.json` | 713 | `e254f5089661b7d0e9df15afca75ded3ad8a7e3ee4ebeead3c0a1b17432326e3` |
| `evidence/p0-model/e5b-sampling-record.json` | 1,986 | `4c759bbe57eea760ccbf2cd80d6a2cf610358bf0364c76d8002555dbfbf5a640` |
| `evidence/p0-model/e5b-sampling-request.json` | 316 | `210d408a12cee00f40e7c33fd32829c4f82ff33cfc001b619aa167d8b595518a` |
| `evidence/p0-model/e5b-sampling-response-raw.json` | 713 | `9ec5f3c08ca2672d8363eb46de8ca4daf995be786588cada1ade03817a86b554` |
| `evidence/p0-model/e6-extract-console.txt` | 20,495 | `e79a5775a8ef9eece120455012ad23ed7518e123d06c3f9be61d43b3a57a737b` |
| `evidence/p0-model/e6-metrics.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `evidence/p0-model/e6-record.json` | 1,968 | `0139768c741f6fd89aaef926415b5a42adbd54788092f18a69dc6656c05eea0a` |
| `evidence/p0-model/e6-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` |
| `evidence/p0-model/e6-response-raw.json` | 713 | `a883331644e0ab68583d52acb11f42559f1035846073aa8454452ea168d03553` |
| `evidence/p0-model/e6b-metrics.txt` | 49,749 | `1ea51d99fe3980194441bad15c24b67e0eb1387d1579b9e056bdb7527972b736` |
| `evidence/p0-model/e6b-record.json` | 1,968 | `50c1fc5d4400db68241e44174f827df1200c259ff9f2b1355a11ea2e543a32ea` |
| `evidence/p0-model/e6b-request.json` | 300 | `c9c61047a823e819136049591ab086109fe46667edef3931dd05bf752b8daadf` |
| `evidence/p0-model/e6b-response-raw.json` | 713 | `3eb98c5d21478eb4bbafd0118e82b71aa0b10b4dc666dfa6486840c2e8657b37` |
| `logs/serve-e4.log` | 34,463 | `7d12122512d8d970b7418ef7665639b44e6c1f8d3f41a3fbe5555cc03bd39898` |
| `logs/serve-e4b.log` | 33,937 | `f363f5ebc105e32d11134b48385a9f835d6c3682dc105cc94a777836469eb856` |
| `logs/serve-e4c.log` | 28,667 | `ffe9b7d8eebc27fee80f01fc497c3bdc62c4492c9316a71aa163ecae87d7e688` |
| `logs/serve-e6.log` | 33,642 | `9d18826928de284bcff9a00e9fc4d0ba826d86b660ef9705b69c2ff4c99413f3` |
| `logs/serve-e6b.log` | 33,974 | `0d018b91febfcc6512efc9effe79a1db49161ed6bde526ecc5132d368d7840fe` |
| `logs/serve-e4-attempt1-offline-resolve-fail.log` | 6,678 | `273f307a19630bdd8d6cdcc6bcd2a3b0cc4d6a9564fafd8fd211cd587666ade1` |
| `logs/serve-e4-attempt2-maxnumseqs-default-fail.log` | 36,272 | `47991d5814d648555e0aa5d5e0f02aa17207eefd78975a84c4d17a4c9c8e610d` |
| `logs/serve-e4-attempt3-flashinfer-jit-fail.log` | 88,720 | `5511c07f40217a0473662df4ab52b1e01c5e2d1f8c459c4931741e51d57a2877` |
| `evidence/after-model/env-report-20260917-1910.md` | 86,971 | `739abdb065112bddd8e3b19ba47eb912f6ccade9daadcfebb282db19803a709c` |
| `evidence/after/cuda_tensor.txt` | 252 | `e89655c964f0a4a24b379b58b5af458ad6a8cc4ea3fd854a0acec9547de6bb3c` |
| `evidence/after/env-report-20260917-1637.md` | 86,901 | `4cf8e066ca9116040069312bba55568139ee39bda9ef490084b620470c6cffe7` |
| `evidence/after/env-report-20260917-1655.md` | 86,925 | `5d3be6f45f830028a4751c0f3a2eeced8b84db6a13f11c9db65349e17b180edf` |
| `evidence/after/env-report-20260917-1716.md` | 86,925 | `395353c9ccf94e39aa60e23fd638b8df357a4631e2a63ee8254e4f63106b43d7` |
| `evidence/after/import_versions.txt` | 587 | `61b623a6b1dba2e0c08d4afd8389fc8158d5266fe2e1017879f219028631e589` |
| `evidence/after/network-tools.txt` | 1,150 | `e3f026504fa94ef9009a21078e89d65e0b1df3c432fde69ff5f78cd05d8bc207` |
| `evidence/after/nvcc.txt` | 208 | `d5e3a195eebabdcebf990838dda4ca273b10cf09c7e01080333f03d60cc1db56` |
| `evidence/after/pip_check.txt` | 30 | `9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09` |
| `evidence/after/pip_freeze.txt` | 4,118 | `33585d37b124ca2dc86518d8c6606a03795ce3f21bd8e7e18139871a5d34a07f` |
| `evidence/after/python_version.txt` | 15 | `55ae85cf4bdb38743edbcd53ea68ff36511997ec6c21b1e83d8bebc939bf056b` |
| `evidence/after/vllm_cli.txt` | 1,162 | `40960bd7c950d219c9837b46a703b4dacbf625530008c6aff6d9ea9e874307e0` |
| `tools/e1-model-identity.py` | 3,866 | `8a70f734f83cda4204dbc6c48d121c3eeaa41cf36dadba8f24ecc6da21e7ca5d` |
| `tools/e2-download.sh` | 1,674 | `3b23233375513abe34a89d0066140b634b44207ed110d90e972781a9ca7a411d` |
| `tools/e2-verify.py` | 3,595 | `949872c37f5b3f96eda67b3816aea98368ad48ddf19c4b1c39dc6b44f72a791c` |
| `tools/e3-config-audit.py` | 9,167 | `2721bb737ca17a868710c8bd6dfc96f273b3e44770a6271f7ab51c6f4e80b525` |
| `tools/e3-probe-protocol-tokens.py` | 2,741 | `495a84ea4ef185bb6c106f72f44e19671e2b8b121b392ee72e5fa4e28afce469` |
| `tools/e4-extract.py` | 3,594 | `404061e357ccf5202c0ab7258231ae56543d8888b57be2206f482d892d51b648` |
| `tools/e4-kernel-block-probe.sh` | 1,066 | `155b1c2b804c9555c0193c1351b63155065fb58415182ecfae56ca9d124a3106` |
| `tools/e5-prompt-check.py` | 4,154 | `09b9160ae766c3d4ea3a360f810fbe224807473bd3aafa8e39970ccb79e46f67` |
| `tools/e5-request.py` | 3,188 | `ee36679942fc3a1dc323fc145bb6c7d31d84466de8227411bfa301f33d5a44ff` |
| `tools/gen-file-inventory.py` | 1,656 | `71763efe3bfc1167c44aba8f1beed9ef7bbac5e8745ddc0a191f191826d3ab12` |
| `tools/gen-model-identity.py` | 5,510 | `ac77274054dc26ae2cf0a02f49508c7080593ad77060f5a3ea1014ffdfca5d6f` |
| `tools/kbs-probe/__pycache__/sitecustomize.cpython-312.pyc` | 5,139 | `5988a562c1ddef6393969a9cdc4c78fec7c76444aed4b2bedd55be353aa334e3` |
| `tools/kbs-probe/sitecustomize.py` | 4,384 | `2e2307faccc833047cd60e19073569bff8c5c7ade63fea6dddb60ab7245b3c5d` |
| `tools/serve-vanilla.sh` | 3,291 | `ff8c798d438144fc14becacf8a78d0de8abddd59f0cdbc18a06fda57e2a9d1b1` |
| `reports/p0-model/model-report.md` | 28,634 | `275cfc8951fa61ebd7b3ff97bea7044198466158fd9ed52242c8667204310dd6` |
| `reports/p0-model/model-identity.md` | 8,530 | `2ebe5f7df1a5faa575408ff4a89f4aaed31f34ded072345119914f2c63e29a94` |
| `reports/p0-model/serve-command.sh` | 2,836 | `adb3d2415a676aec132a8fd23ff9496c1e2144841b0e5e0ea4f91336f662d979` |
| `reports/p0-model/requirements.freeze.txt` | 4,118 | `350d32f4804ada87ad8617cabe64c07743fc43a1685f57d1b7a34ab374c89d01` |
| `reports/p0-model/dependency-delta.md` | 2,513 | `d60b7c2c6df0607991b499e76d46fc727bf952139cc19ceab3ad1318472abb16` |
| `reports/p0-model/environment-rebuild.md` | 3,877 | `11dc2774cf52c4d9bcd4c6ba66e643dbbfc4ae70051edea837071ea5c9f6d43a` |
| `setup-local-cuda.sh` | 7,497 | `0ca255e719387d9c37edf7f62f9866ef02a309ba9f783e8ac8f8b17a46da5b94` |
| `env.sh` | 4,808 | `796f638fb247f194a5cd931473d79b7d27e476c6e483182d9ebfc8b8863572e0` |
