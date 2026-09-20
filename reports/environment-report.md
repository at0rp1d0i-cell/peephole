# 阶段 01 环境报告（远端实机）

日期：2026-09-17（UTC）。执行者：agent（远端 `lab` 容器内命令执行 + 本机素材仓复核）。
工作单：[`docs/stage-01-environment.md`](../../docs/stage-01-environment.md)。素材基线：`fb293e8`。

**结论：支持（环境层就绪）** —— 隔离解释器与依赖可用、版本与来源可追溯、`pip check` / import / CLI / 小张量 CUDA / 项目自带 nvcc 全部通过、干净非交互会话可重复执行。
**不属于本阶段验收的部分一律未验证**：模型下载与加载、实际 attention backend、`kernel_block_size`、图模式、生成、性能、质量、DA 支持。

## 1. 目标机器与资源

| 项 | 实测值 |
| --- | --- |
| 主机 | `container-host`（AutoDL 容器，SSH 别名 `lab`） |
| OS / 内核 | Ubuntu 22.04.5 LTS / `5.15.0-78-generic` / x86_64 |
| CPU | Intel Xeon Platinum 8470Q，208 逻辑 CPU（2 socket × 52 core × 2 thread） |
| 内存 | MemTotal 1,056,444,572 kB（≈1007 GiB）；MemAvailable 855,391,512 kB |
| `/dev/shm` | 60 GiB |
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition，compute capability **12.0**，97887 MiB |
| GPU 状态 | 0 MiB 已用、0% 利用率、33 °C、功耗上限 600 W（空闲 35 W）、ECC 关闭、MIG 关闭、无其他进程 |
| 驱动 | 580.142（驱动侧 CUDA 13.0） |
| 磁盘 | `/`（容器 overlay，非持久）30 GiB，环境迁出后剩 18 GiB；`/root/autodl-tmp`（`/dev/md0`，本机数据盘）**已扩容 50 → 200 GiB**，已用 8.4 GiB、剩 **192 GiB**；`/autodl-fs/data`（网络文件存储）200 GiB，已用 4.6 GiB、剩 195 GiB；`/autodl-pub` 7.3 TiB 只读 |

宿主驱动、系统 CUDA、系统 Python、镜像自带软件栈均未修改。镜像自带的 `/usr/local/cuda`（ldconfig 中为 CUDA 12.8 库）保持原样且**本环境不使用**。

## 2. 目录与隔离方式

持久根目录为用户在目标机器上指定的 **`/root/attnview`**（工作单 §2 的"持久目录"以用户指定为准）。
**物理位置在本机数据盘**：`/root/attnview` 是指向 `/root/autodl-tmp/attnview` 的符号链接——用户确认的原则是"与性能有关的资产优先放数据盘"（`/dev/md0` 读写更快，且可跨实例复制），同时保留 `/root/attnview` 这个路径，使 venv 的 `pyvenv.cfg`、脚本 shebang 与脚本内绝对路径无需重写。实例重建后重建链接即可：`ln -s /root/autodl-tmp/attnview /root/attnview`。

```text
/root/attnview -> /root/autodl-tmp/attnview/     （数据盘，50 GiB 配额）
├── env.sh                 环境定义（唯一来源，交互与非交互会话都 source 它）
├── install-runtime.sh     装 venv + vLLM + 源码 checkout
├── setup-local-cuda.sh    建立项目自带 CUDA 前缀
├── verify-runtime.sh      环境层最小验证（E5）
├── material/attnview/     素材仓快照（commit fb293e8，仅 tracked 文件，744 KiB）
├── venvs/attnview/        uv venv（7.6 GiB）
├── pythons/               uv 托管 CPython 3.12.13（106 MiB）
├── vllm/                  vLLM 源码 checkout @ pin（411 MiB）
├── cuda/                  项目自带 CUDA 前缀（符号链接）
├── caches/                uv / pip / triton / torch / cuda / huggingface 缓存
├── logs/                  安装与运行日志
├── evidence/{before,after}/ 环境探针与逐项检查原始输出
└── reports/               本阶段交付报告
```

- 解释器与依赖**全部由 uv 管理**：venv 与 `uv` 自身都在项目目录内，不依赖镜像自带的 conda 环境。
- CUDA 工具链**由项目自带**：`CUDA_HOME=/root/attnview/cuda`，内容来自 venv 内的 CUDA 13 轮子，不经系统 `/usr/local/cuda`。
- 未改 `/etc/environment`、未改系统 PATH、未装 apt 包、未动驱动。使用方式为显式 `source /root/attnview/env.sh` 或直接使用绝对解释器路径。
- 缓存与 venv 同一文件系统 + `UV_LINK_MODE=hardlink`，因此缓存不额外占盘。

## 3. 版本、来源与对应关系

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| Python | 3.12.13 | uv 托管 CPython（`UV_PYTHON_INSTALL_DIR=/root/attnview/pythons`） |
| uv | 引导 0.11.26（镜像 `/root/miniconda3/bin/uv`）→ 项目内 0.12.15（`venvs/attnview/bin/uv`） | 引导一次后自托管，随 freeze 锁定 |
| torch | 2.13.0+cu130（`torch.version.cuda=13.0`，cuDNN 92000） | PyPI 镜像 `torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl`，sha256 `796633c4cdf0fe2cdced72d8f88f22e73dbcfce83132763162f6d4bff13b820b` |
| vLLM（运行） | 0.29.0 | PyPI 镜像 `vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl`，sha256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635` |
| vLLM（源码） | tag `v0.29.0` = `98dff2a81d747d1dba01a47f939f48c3526d4206` | `https://github.com/vllm-project/vllm.git`，工作区干净（`## HEAD (no branch)`） |
| triton | 3.7.1 | sha256 `7e40869937a68206ec70d7f25bb7ec6433cb083f9135e1f36dbd318dc449a728` |
| transformers / tokenizers | 5.17.0 / 0.23.2 | sha256 `78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801`（transformers） |
| flashinfer-python | 0.6.18 | sha256 `d5d26edb48f8def0bf28b2c94415aef7f0754e2cd9e3cb3d6e7081262d068aad` |
| numpy / xgrammar | 2.3.5 / 0.2.7 | PyPI 镜像 |
| nvcc | 13.4（V13.4.92） | venv 内 CUDA 13 轮子的合并布局 `site-packages/nvidia/cu13` |
| 包索引 | `https://pypi.tuna.tsinghua.edu.cn/simple` | 环境变量级覆盖，未改 `/etc/pip.conf` |

**`clone` 不等于 Python 正在跑的代码**（工作单 E3.5 要求）：

```text
sys.executable   = /root/attnview/venvs/attnview/bin/python
vllm.__version__ = 0.29.0
vllm.__file__    = /root/attnview/venvs/attnview/lib/python3.12/site-packages/vllm/__init__.py
```

即：运行期 vLLM 来自安装轮子；`/root/attnview/vllm` 是供 P1/P2 阅读与打补丁的**独立源码 checkout**，本阶段不做 editable 构建。

依赖清单：`requirements.freeze.txt`（197 行，`python -m pip freeze`；锁文件只是解析结果清单，不等于可重建证明，重建步骤见 `environment-setup.md`）。

## 4. 逐项检查结果

| 检查（工作单条目） | 结果 | 结论标签 | 原始产物 |
| --- | --- | --- | --- |
| E1 现状快照 | 见 §1，GPU/驱动/磁盘/`/dev/shm` 全部取到；安装前 `/` 剩 19 GiB | 支持 | `evidence/before/env-report-20260917-1559.md` |
| E2 目录与隔离 | `/root/attnview` 下自洽；无系统级改动 | 支持 | 本报告 §2 |
| E3 pin 核对与安装 | `vllm==0.29.0` 在索引中存在且可解析；源码 tag/commit 与工作单候选一致 | 支持 | `logs/install-runtime-20260917-163051.log` |
| E4 网络与工具 | 见 §5；镜像与 hf-mirror 直连可用，github 直连不通、经平台加速可用 | 支持 | `evidence/after/network-tools.txt` |
| E5 最小验证 | 见下表 | 支持 | `evidence/after/*.txt`、`evidence/verify-console.log` |

E5 明细（干净非交互会话，`source /root/attnview/env.sh` 后执行）：

| 检查 | 命令要点 | 结果 |
| --- | --- | --- |
| 解释器 | `"$ATTNVIEW_PYTHON" -V` | `Python 3.12.13`，exit 0 |
| 依赖一致性 | `python -m pip check` | `No broken requirements found.`，exit 0 |
| 依赖冻结 | `python -m pip freeze` | 197 行，exit 0 |
| 模块 import 与设备 | `import torch, vllm` | `torch.cuda.is_available()=True`；1 设备，cc 12.0，94.97 GiB，188 SM |
| 小张量 CUDA | `arange(1024).sum()` 与 512×512 fp32 matmul 对 CPU 参考 | 已知结果 523776 ✓；最大绝对误差 `5.341e-05`；`torch.cuda.synchronize()` 正常 |
| vLLM CLI | `venvs/attnview/bin/vllm --help` | exit 0，列出 `chat/complete/serve/launch/bench/collect-env/run-batch` |
| 项目自带 nvcc | `cuda/bin/nvcc --version`、`nvcc -cubin -arch=sm_120` | 13.4；sm_120 cubin 生成成功（5488 字节） |

其余取证：安装成功后再次执行 `scripts/env-probe.sh` 刷新环境快照（安装后那次属中间刷新，已移出仓内发行，status=`snapshot`）；仓内保留首/末两份 `evidence/before/env-report-20260917-1559.md`、`evidence/after/env-report-20260917-1716.md`，其余刷新见 `reports/evidence-index.md` 的树外登记表。项目自带 CUDA 前缀建立日志 → `evidence/cuda-setup.log`。

## 5. 网络事实（原始观测）

| 目标 | 直连 | 经 `/etc/network_turbo` |
| --- | --- | --- |
| `pypi.tuna.tsinghua.edu.cn/simple/vllm/` | HTTP 200，0.51 s | 未用（代理反而更慢） |
| `pypi.org/simple/vllm/` | HTTP 200，2.24 s | — |
| `hf-mirror.com` | HTTP 200，1.56 s | — |
| `github.com/vllm-project/vllm` | **超时（25 s，无数据）** | HTTP 200，4.45 s |
| `objects.githubusercontent.com`（release 资产） | **超时（25 s，0 字节）** | 用于 CPython 下载，成功 |

因此：**PyPI 走镜像不经代理；github 与 github release 资产必须经平台学术加速**。uv 托管 CPython 与 vLLM 源码检出两处都据此处理（见 `environment-setup.md`）。

DNS 解析均正常。工具可用性：`git` / `curl` / `jq` / `tar` 有；`zstd`、`docker`、`nsys`、`ncu`、`py-spy`、`perf` **缺失**。按工作单 §3 E4，Docker/NCU 缺失不作为本阶段失败，也不与模型运行失败混为一谈。

## 6. 本次修改范围

| 项 | 说明 |
| --- | --- |
| 新增 | `/root/attnview` 下的全部内容（约 8.3 GiB）：venv、托管解释器、vLLM 源码、项目 CUDA 前缀、缓存、日志、证据、报告 |
| 未改 | 宿主驱动、系统 CUDA（`/usr/local/cuda`）、系统 Python、`/etc/environment`、系统 PATH、`/etc/pip.conf`；未装 apt 包；未改他人项目 |
| 空间回收 | 删除 `/root/paged-kv-attention-kernel-lab/.venv`（5.5 GiB，**经用户授权**）；该项目源码与 `.git` 保留，HEAD `8e6b0dc` 且 `main...origin/main` 无未推送提交，可用 `uv sync --locked` 重建 venv |
| 环境迁移 | 按用户"性能相关资产优先放数据盘"的口径，把整个 `/root/attnview` 复制到 `/root/autodl-tmp/attnview`（8.3 GiB / 121244 文件 / 7.7 s，约 1.1 GB/s），`/root/attnview` 改为符号链接；迁移后重跑 `verify-runtime.sh` 全项 exit 0，再删除系统盘冗余副本。`/` 由剩 8.1 GiB 变为剩 17 GiB |
| 观测到的他人改动 | `/usr/local/bin/{uv,uvx,gh,rtk}` 在 00:36（+0800）出现，与本阶段脚本无关（本阶段只写 `/root/attnview`）；其 `uv` 0.12.15 与项目内 `uv` 是**不同 inode**，裸 shell 下 `command -v uv` 可能命中它 |
| 数据盘扩容 | 用户在平台控制台把 `/root/autodl-tmp` 从 50 GiB 扩到 200 GiB（付费动作，agent 未代做）。扩容后同一实例、链接与文件系统均未变，无需重装 |
| 权重与缓存落位 | 新建 `models/{hf-home,weights}`；`env.sh` 的 `HF_HOME` 由 `caches/huggingface` 改为 `models/hf-home`（原目录不存在），并显式设 `HF_HUB_CACHE=$HF_HOME/hub`；新增 `ATTNVIEW_MODELS`。收尾后重跑 `verify-runtime.sh` 全项 exit 0 |
| 容量回收授权 | 用户授权"容量不足时把 KV 项目暂时删掉（应已推送）"。实测**条件不成立、未删除**：唯一 KV 项目 `/root/paged-kv-attention-kernel-lab` 现仅 **15 MB**（其 5.5 GiB 的 `.venv` 已在上一轮按授权删除），`## main...origin/main` 未推送 0 个。数据盘剩 192 GiB、系统盘剩 18 GiB，均无需回收 |

远端脚本 `env.sh` / `install-runtime.sh` / `setup-local-cuda.sh` / `verify-runtime.sh` 已在素材仓 `remote/` 下同源更新（工作单 §4 要求：按 UV 管理与项目自带 CUDA 的口径逐项复核后修正）。

## 7. 已知限制与未验证项

- **本阶段的"支持"只覆盖环境层**。以下全部未验证，且不得由本报告推断：实际 attention backend、`kernel_block_size`、CUDA Graph 模式、`qwen3_5` 在 SM120 的可运行性、模型能否下载/加载/生成、性能与质量、DA 机制的任何行为。
- 无 NCU / 无 `nsys`：后续性能归因按 plan §6.3，只报告有证据的执行时间与消融归因。
- 无 Docker：SWE-bench 类容器化验收需要另行准备（工作单 §3 E4 单列待办）。
- 无 `zstd`：与素材仓打包/解包流程相关，使用时再补。
- **权重落盘（2026-09-17，用户决定 + 已落地）**：Qwen3.8-27B checkpoint 为 `55,562,855,904 B ≈ 51.75 GiB`，P0 两个 27B 候选并存需约 103.5 GiB，而数据盘原配额只有 50.0 GiB（**清空也放不下单个 checkpoint**）。容量算术：两候选 ≈114 GiB（含 10% 余量）+ 环境 8.3 GiB + 编译缓存约 5 GiB + 数据集与评测产物 15–30 GiB ⇒ 142–157 GiB。用户据此**把数据盘扩容到 200 GiB**（付费动作，用户在控制台执行），扩容后剩 192 GiB。收尾：新建 `models/{hf-home,weights}`，`HF_HOME` 指向 `models/hf-home`、`HF_HUB_CACHE=$HF_HOME/hub`，收尾后 `verify-runtime.sh` 全项 exit 0。
- 容器与持久性：环境已放数据盘（AutoDL 释放实例时数据盘保留），因此环境本身可跨实例带走；**但 `/root/attnview` 这个符号链接在系统盘上，实例重建后需重建**：`ln -s /root/autodl-tmp/attnview /root/attnview`（重建后无需重装，venv 内绝对路径在链接恢复后即生效）。
- 索引与 `freeze` 的可复现边界：`pip freeze` 不含索引来源与哈希；重建以 `environment-setup.md` 的命令 + 本报告 §3 的 wheel sha256 为准。

### 磁盘 I/O 实测（2026-09-17，各 3 次、2 GiB/次）

"数据盘更快"这一前提在本实例**未被测量支持**：

| 目标 | 顺序写（`dd conv=fdatasync`，中位） | 顺序读（`dd iflag=direct`，中位） |
| --- | ---: | ---: |
| 数据盘 `/root/autodl-tmp` | 699 MB/s | 2.0 GB/s |
| 系统盘 `/` | 729 MB/s | 2.0 GB/s |

两者差异在噪声内，原因是**同一物理设备池**：数据盘是 `/dev/md0[/docker/volumes/<container>-storage/_data]`，系统盘 overlay 的上层也在同一块 `/data` NVMe 上。因此把环境放在数据盘的**实际收益是持久性（跨实例保留）、可复制性与容量（192 GiB vs 18 GiB）**，而不是 I/O 速度；后续据此做取舍（例如数据集放哪）时不要假定数据盘更快。限制：单流 `dd`、本机无 `fio`、未测随机/小文件/多队列深度、宿主 load average 17–24（208 逻辑 CPU，可能有邻租户干扰）、样本 3 次。
