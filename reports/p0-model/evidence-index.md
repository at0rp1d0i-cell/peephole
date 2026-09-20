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
| `bash material/attnview/scripts/env-probe.sh --out /root/attnview/evidence/after-model` | 0 | `evidence/after-model/env-report-20260917-1910.md`（已移出仓内发行，见 `reports/evidence-index.md` 树外登记表） | 模型运行后的机器/GPU/磁盘快照；结束后 GPU 回到 0 MiB |

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
| `pip freeze`（修复前基线） | 0 | `cuda-upgrade-freeze-before.txt`（已移出仓内发行，见 `reports/evidence-index.md` 树外登记表） | `nvidia-cuda-runtime==13.0.96` |
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

## 文件清单

机械 size/sha256 表见生成物 `reports/evidence-index.md`；本文件保留命令/退出码/摘要级 provenance。（该生成索引覆盖 `evidence/` 与 `logs/`——本阶段 `evidence/p0-model/*`、`logs/*`、`evidence/after/*` 的字节与 sha256 登记在其树内/树外两张表中；`tools/` 与 `reports/` 下的文件不在该索引收录范围。）

> **哈希登记说明（2026-09-21）**
> - 原机械表 **15 行**登记的文件在 2026-09-21 公开发布前的脱敏中被改写（三类替换：容器主机名 / GPU UUID / 协调目录路径），其上 sha256 与字节数已按改写后的**当前字节**更新；脱敏前哈希见 `reports/desensitization.md` §3 受影响文件表（逐行对照见该报告 §4.3）。
> - `env.sh` 行**不是**脱敏所致：该文件在索引生成后被 commit `f9d834f`（2026-09-21）继续编辑（4,808 → 4,836 B），同样已更新为当前字节。
> - 本索引记录的部分证据在 2026-09-21 公开发布前经过脱敏（三类替换），哈希以当前字节为准。
