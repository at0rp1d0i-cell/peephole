# 阶段 01 环境建立与复跑说明

配套：[`environment-report.md`](environment-report.md)（事实与结果）、[`evidence-index.md`](evidence-index.md)（生成索引：证据发布范围与机械事实）。
目标机 SSH 别名：`lab`（`~/.ssh/config`，AutoDL 容器）。全部命令在远端执行。

## 0. 设计要点

- **uv 管理环境**：uv 建 venv、装依赖、并自托管在 venv 内；Python 用 uv 托管 CPython，不依赖镜像自带的 conda 解释器。
- **物理位置在本机数据盘**：`/root/attnview` 是指向 `/root/autodl-tmp/attnview` 的符号链接。用户口径是"与性能有关的资产优先放数据盘"——该盘（`/dev/md0`）读写更快且可跨实例复制，实测复制 8.3 GiB 用 7.7 s（约 1.1 GB/s）。保留原路径使 venv 的 `pyvenv.cfg`、脚本 shebang 与脚本内绝对路径无需重写。
- **项目自带 CUDA**：`CUDA_HOME=/root/attnview/cuda`，来源是 venv 内 CUDA 13 轮子的合并布局，不使用系统 `/usr/local/cuda`。
- **不改系统**：不写 `/etc/environment`、不改系统 PATH、不装 apt 包、不动驱动。交互与非交互会话都用显式 `source /root/attnview/env.sh`。
- **缓存不额外占盘**：缓存与 venv 同一文件系统，`UV_LINK_MODE=hardlink` 生效。

## 1. 一次性建立（已执行，可幂等重跑）

```bash
# 0) 上传脚本与素材快照（从本机素材仓）
scp -q remote/{env.sh,install-runtime.sh,setup-local-cuda.sh,verify-runtime.sh} lab:/root/attnview/
git archive --format=tar HEAD | ssh lab 'mkdir -p /root/attnview/material/attnview && tar -xf - -C /root/attnview/material/attnview'

# 1) 环境定义 → venv → vLLM → 源码 checkout（约 25 分钟，绝大部分是下载）
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/install-runtime.sh'

# 2) 项目自带 CUDA 前缀
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/setup-local-cuda.sh'

# 3) 环境层最小验证
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/verify-runtime.sh'
```

`install-runtime.sh` 的实际动作：`uv venv --python 3.12 --python-preference only-managed --seed` → `uv pip install --python "$ATTNVIEW_PYTHON" uv`（自托管）→ `uv pip install --python "$ATTNVIEW_PYTHON" "vllm==0.29.0"` → `python -m pip freeze` → `git clone https://github.com/vllm-project/vllm.git` + `git checkout --detach 98dff2a8…`。可用 `--deps` 只做依赖、跳过源码检出；`VLLM_PIN=` / `VLLM_COMMIT=` 可覆盖版本。

**这条链上有两个必须绕开的直连阻塞**（原始证据见 `evidence/after/network-tools.txt`）：

1. uv 托管 CPython 要下载 `github.com/astral-sh/python-build-standalone` 的 release 资产，直连 25 s 无任何字节；
2. `git clone github.com` 直连同样超时。

脚本的处理是**只在这两步** `source /etc/network_turbo`（平台学术加速），PyPI 仍走清华镜像、不经代理。若加速不可用，`install-runtime.sh` 会回退镜像自带 `/root/miniconda3/bin/python` 建 venv（可用 `ATTNVIEW_BASE_PYTHON=` 显式指定）。

## 2. 日常使用（重开会话后）

```bash
ssh lab
source /root/attnview/env.sh          # 唯一入口：PATH/缓存/索引/CUDA_HOME 全部就位
python -V                             # → Python 3.12.13（venv 内）
vllm --help                           # → vLLM 0.29.0 CLI
nvcc --version                        # → 13.4（项目自带，非 /usr/local/cuda）
```

非交互命令用同一条路径，或直接用绝对解释器（不依赖任何环境变量）：

```bash
ssh lab 'source /root/attnview/env.sh && /root/attnview/venvs/attnview/bin/python -c "import vllm; print(vllm.__file__)"'
ssh lab '/root/attnview/venvs/attnview/bin/python -m pip check'
```

关键变量（`env.sh` 内，均有注释）：`ATTNVIEW_HOME`、`ATTNVIEW_PYTHON`、`ATTNVIEW_VLLM_SRC`、`ATTNVIEW_CUDA`、`ATTNVIEW_MODELS`、`CUDA_HOME`、`HF_HOME`（`models/hf-home`）、`HF_HUB_CACHE`、`UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR`、`UV_LINK_MODE=hardlink`、`UV_DEFAULT_INDEX`、`PIP_INDEX_URL`、`TMPDIR`。下载权重前用 `echo $HF_HOME` 确认落在数据盘上；权重也可显式落到 `$ATTNVIEW_MODELS/weights/<模型>/`。

## 3. 重建与回退

**路径失效时**（实例重建后系统盘复原，数据盘内容仍在）：只需重建符号链接，无需重装。

```bash
ssh lab 'ln -s /root/autodl-tmp/attnview /root/attnview && source /root/attnview/env.sh && bash /root/attnview/verify-runtime.sh'
```

**整体重建**（数据盘也丢了、或 venv 损坏时；约 25 分钟）：

```bash
ssh lab 'rm -rf /root/attnview/{venvs,pythons,cuda} && source /root/attnview/env.sh \
  && bash /root/attnview/install-runtime.sh && bash /root/attnview/setup-local-cuda.sh \
  && bash /root/attnview/verify-runtime.sh'
```

**只回退 CUDA 前缀**：`rm -rf /root/attnview/cuda` 后重跑 `setup-local-cuda.sh`（纯符号链接，不含独有数据）。
**只回退 venv**：`rm -rf /root/attnview/venvs /root/attnview/pythons` 后重跑 `install-runtime.sh`（`vllm/` 源码 checkout 与素材快照不受影响）。
**不要**手工删 `caches/uv`：venv 内的文件与缓存是硬链接关系，删除缓存目录不会损坏 venv，但会让下次安装重新下载约 8 GiB。

依赖精确版本见 [`requirements.freeze.txt`](requirements.freeze.txt)（197 行，解释器 `/root/attnview/venvs/attnview/bin/python`，来源为清华 PyPI 镜像）。`pip freeze` 不是可重建证明：重建以本文件 §1 的命令 + `environment-report.md` §3 的 wheel sha256 为准。

### 3.1 数据盘扩容与收尾（**已完成 2026-09-17**）

用户按 `environment-report.md` §7 的容量算术（142–157 GiB）在平台控制台把 `/root/autodl-tmp` 扩到 **200 GiB**（付费动作，agent 未代做）。扩容后同一实例、同一挂载点、符号链接与环境全部保留，**无需重装**。收尾四步及结果：

```bash
# 1) 确认配额与挂载点 → 200G，/dev/md0 xfs，/root/attnview 链接仍在
ssh lab 'df -h /root/autodl-tmp; findmnt -no SOURCE,FSTYPE,TARGET /root/autodl-tmp; ls -ld /root/attnview'
# 2) 符号链接无需重建（容器未重启）
# 3) 建权重与缓存目录
ssh lab 'mkdir -p /root/attnview/models/{hf-home,weights}'
# 4) 把 HF 缓存指向数据盘固定位置（见 remote/env.sh）
#    HF_HOME=$ATTNVIEW_HOME/models/hf-home、HF_HUB_CACHE=$HF_HOME/hub
# 5) 自检 → exit 0，7 项检查全通过（输出见 evidence/verify-console.log）
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/verify-runtime.sh'
```

第 5 步的校验输出见 `evidence/verify-console.log`；迁移/扩容/HF 端点三次的副本（`evidence/verify-after-move.log`、`verify-after-expand.log`、`verify-after-hf-endpoint.log`）已移出仓内发行（同内容，status=`duplicate`），原始字节归档在数据盘，路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表。该校验结论逐字为：7 项检查全部 `[exit 0]`，末尾"全部检查通过"。

顺序 I/O 顺带实测（各 3 次、2 GiB/次）：数据盘写中位 699 MB/s、读 2.0 GB/s；系统盘写 729 MB/s、读 2.0 GB/s——**两盘同速，因为同属一块 `/data` NVMe**（数据盘是 `/dev/md0` 上的 docker volume 子目录）。所以"放数据盘"的价值是持久性、可复制性与容量，不是速度；完整数据见 `environment-report.md` §7 与本机 `evidence/post-expansion.txt`。

## 4. 空间核算（迁移 + 扩容之后）

| 路径 | 体积 | 说明 |
| --- | --- | --- |
| `/root/autodl-tmp/attnview/venvs/attnview` | 7.6 GiB | 含 torch(cu130)、vLLM、flashinfer、CUDA 13 运行时与 nvcc |
| `…/pythons/` | 106 MiB | uv 托管 CPython 3.12.13 |
| `…/vllm/` | 411 MiB | 源码 checkout（含 `.git`） |
| `…/models/` | 空（已建 `hf-home`/`weights`） | 权重与 HF 缓存的目标位置 |
| 其余（缓存/素材/日志/证据） | 约 0.2 GiB | 缓存因硬链接不重复计 |
| **环境合计（数据盘）** | **8.3 GiB** | 数据盘 200 GiB 配额，已用 8.4 GiB、**剩 192 GiB** |
| 系统盘 `/` | — | 迁移后剩 18 GiB |

容量与用途的对应（2026-09-17 实测）：

| 位置 | 容量 | 本项目的用法 |
| --- | --- | --- |
| `/root/autodl-tmp`（本机数据盘，`/dev/md0`） | **200 GiB**（剩 192 GiB） | 环境、缓存、编译产物、**权重与 HF 缓存**；跨实例保留、可复制 |
| `/autodl-fs/data`（网络文件存储） | 200 GiB（剩 195 GiB） | 备选（本项目暂未使用） |
| `/`（容器 overlay，非持久） | 30 GiB（剩 18 GiB） | 只放镜像与临时物；实例重建即失效 |

权重空间核对：Qwen3.8-27B 与 Qwen3.6-27B 各 `55,562,855,904 B ≈ 51.75 GiB`，两候选约 103.5 GiB，加上环境与数据集后有充足余量。

## 5. 脚本清单

| 文件 | 作用 | 幂等性 |
| --- | --- | --- |
| `env.sh` | 环境定义唯一来源；只做变量赋值与幂等 PATH 追加，不输出、不做命令替换、不依赖 cwd | 可反复 source |
| `install-runtime.sh` | venv + vLLM 依赖 + 源码 checkout + freeze | venv 存在则复用；依赖已满足则跳过；源码目录存在则只 fetch/checkout |
| `setup-local-cuda.sh` | 把 venv 内 CUDA 13 工具链暴露为 `$ATTNVIEW_CUDA` 前缀并编译自检 | 前缀每次重建，结果一致 |
| `verify-runtime.sh` | E5 逐项验证，原始输出落 `evidence/after/`（逐项 `.txt` 与 `evidence/verify-console.log`；其中环境探针中间刷新已移出仓内发行，见 §6） | 只读检查，可重复 |

素材仓内同一份脚本位于 `remote/`（相对 `docs/stage-01-environment.md` §4 的旧版：旧版基于 miniconda + `venv`、写 `/etc/environment`、`repurpose` 变量 `home`，已按 uv + 项目自带 CUDA + `/root/attnview` 重写）。

## 6. 原始产物、退出码与复现入口（阶段 01 原始记录）

安装与 CUDA 前缀建立（命令、退出码、内容）：

| 产物 | 命令 | 退出码 | 内容 |
| --- | --- | --- | --- |
| `logs/install-runtime-20260917-163051.log` | `source env.sh && bash install-runtime.sh` | 0 | 本次有效安装全程：venv 建立、依赖解析与安装、freeze、源码 checkout、import 来源核对、磁盘 |
| `logs/install-runtime-20260917-155930.log` | 同命令（失败一次） | 1 | `uv: command not found`——非交互 PATH 里没有镜像 uv，据此把 uv 解析与自托管写进脚本 |
| `logs/install-runtime-20260917-160026.log` | 同命令（失败一次） | — | uv 托管 CPython 直连下载停滞（`.temp` 无字节落盘），据此改为仅该步走平台加速 |
| `evidence/cuda-setup.log` | `bash setup-local-cuda.sh` | 0 | 工具链目录、`nvcc --version`、前缀符号链接、sm_120 cubin 编译结果 |

同一次有效安装另有 `nohup` 控制台副本 `logs/console-install.log`（exit 0，与上表第一条逐字节相同，末尾为"安装完成"），已移出仓内发行（status=`duplicate`）；原始字节归档在数据盘，路径/字节/sha256 见 `reports/evidence-index.md` 的树外登记表。

E5 自检（`verify-runtime.sh` 产生；这里只登记产物、命令与退出码，逐项读数与结论见 [`environment-report.md`](environment-report.md) §4）：

| 产物 | 命令 | 退出码 |
| --- | --- | --- |
| `evidence/after/python_version.txt` | `"$ATTNVIEW_PYTHON" -V` | 0 |
| `evidence/after/pip_check.txt` | `python -m pip check` | 0 |
| `evidence/after/pip_freeze.txt` | `python -m pip freeze` | 0 |
| `evidence/after/import_versions.txt` | `import torch, vllm` + 设备属性 | 0 |
| `evidence/after/cuda_tensor.txt` | 小张量 CUDA 运算 + 同步 | 0 |
| `evidence/after/vllm_cli.txt` | `venvs/attnview/bin/vllm --help` | 0 |
| `evidence/after/nvcc.txt` | `cuda/bin/nvcc --version` | 0 |
| `evidence/after/network-tools.txt` | 见文件内命令记录 | 0 |

`evidence/verify-console.log`（`bash verify-runtime.sh`，exit 0）是上述各项的命令、输出与 `[exit n]` 汇总，末尾"全部检查通过"。

环境探针（`evidence/before|after/env-report-*.md`）由 `bash material/attnview/scripts/env-probe.sh --out evidence/before|after` 采集：uname/用户、`nvidia-smi` 全量、ECC/计算模式、GPU 拓扑、CUDA 工具链、Python/框架版本、vLLM 侧环境变量、主机资源、磁盘、计时工具可用性；探针为只读采集，不安装、不下载、不改系统配置。`evidence/before/` 的一份**未** source `env.sh`（纯基线），`evidence/after/` 的刷新在 source 之后采集。脚本成功退出**不代表**每项检查成功，逐项结论见 `environment-report.md` §1、§4。`evidence/after/` 内的环境探针中间刷新已移出仓内发行（status=`snapshot`）：仓内保留首份 `evidence/before/env-report-20260917-1559.md` 与末份 `evidence/after/env-report-20260917-1716.md`，其余刷新见 `reports/evidence-index.md` 的树外登记表。

### 6.1 迁移到数据盘（`evidence/move-to-data-disk.txt`）

该文件的命令记录：迁移动机（用户口径：性能相关资产优先数据盘）、`cp -a` 统计（8.3 GiB / 121244 文件 / 7.7 s）、符号链接切换、迁移前后 `df`。动作序列：`cp -a /root/attnview/. /root/autodl-tmp/attnview/` → `mv /root/attnview /root/attnview.on-system-disk` → `ln -s /root/autodl-tmp/attnview /root/attnview` →（走 `/root/attnview` 路径）`source env.sh && bash verify-runtime.sh` → `rm -rf /root/attnview.on-system-disk`（删除系统盘冗余副本）。迁移后 7 项检查全部 `[exit 0]`，末尾"全部检查通过"——证明符号链接路径下 venv、CLI、nvcc 全部可用；该校验输出见 `evidence/verify-console.log`（迁移副本已移出仓内发行，同内容，status=`duplicate`，见 §3.1）。系统盘可用空间由 8.1 GiB 变为 17 GiB。

### 6.2 空间回收（`evidence/paged-kv-before-cleanup.txt`）

删除前的快照：大小 5.5 GiB（全部是 `.venv`）、git HEAD `8e6b0dc`、`## main...origin/main`、未推送提交 0、stash 0、origin URL；处置说明（只删 `.venv`，源码与 `.git` 保留，可用 `uv sync --locked` 重建）。

### 6.3 复现入口

```bash
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/verify-runtime.sh'   # 环境层全部检查，原始输出见本节 E5 表
ssh lab 'git -C /root/attnview/vllm rev-parse HEAD'                              # → 98dff2a81d747d1dba01a47f939f48c3526d4206
ssh lab '/root/attnview/venvs/attnview/bin/python -c "import vllm; print(vllm.__file__)"'
```

证据发布范围与机械事实以生成索引 `reports/evidence-index.md` 为准。
