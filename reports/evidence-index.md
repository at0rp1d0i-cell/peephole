# 阶段 01 证据索引（远端原始产物）

远端根：`/root/attnview`。主机 `container-host`，日期 2026-09-17（UTC）。
原始产物一律留在远端；本素材仓只保留脱敏后的报告与清单。主机名、GPU UUID、绝对路径属容器内部标识，未做进一步脱敏处理，入库前如需对外发布请再确认。

## 安装与配置

| 产物 | 命令 | 退出码 | 内容 |
| --- | --- | --- | --- |
| `logs/install-runtime-20260917-163051.log` | `source env.sh && bash install-runtime.sh` | 0 | 本次有效安装全程：venv 建立、依赖解析与安装、freeze、源码 checkout、import 来源核对、磁盘 |
| `logs/console-install.log` | 同上（`nohup` 控制台副本） | 0 | 与上同源，末尾为"安装完成" |
| `logs/install-runtime-20260917-155930.log` | 同命令（失败一次） | 1 | `uv: command not found`——非交互 PATH 里没有镜像 uv，据此把 uv 解析与自托管写进脚本 |
| `logs/install-runtime-20260917-160026.log` | 同命令（失败一次） | — | uv 托管 CPython 直连下载停滞（`.temp` 无字节落盘），据此改为仅该步走平台加速 |
| `evidence/cuda-setup.log` | `bash setup-local-cuda.sh` | 0 | 工具链目录、nvcc --version、前缀符号链接、sm_120 cubin 编译结果 |

## 环境探针（E1 / E5）

| 产物 | 命令 | 内容 |
| --- | --- | --- |
| `evidence/before/env-report-20260917-1559.md` | `bash material/attnview/scripts/env-probe.sh --out evidence/before`（**未** source env.sh，纯基线） | uname/用户、nvidia-smi 全量、ECC/计算模式、GPU 拓扑、CUDA 工具链、Python/框架版本、vLLM 侧环境变量、主机资源、磁盘、计时工具可用性 |
| `evidence/after/env-report-20260917-1637.md` | 同上，`--out evidence/after`（已 source env.sh） | 同上，且 `CUDA_HOME=/root/attnview/cuda`、nvcc 13.4、`HF_HOME` 等指向项目内 |

探针为只读采集，不安装、不下载、不改系统配置；脚本成功退出不代表每项检查成功，逐项结论见 `environment-report.md` §1、§4。

## E5 最小验证（`evidence/after/`，由 `verify-runtime.sh` 产生）

| 产物 | 命令 | 退出码 | 关键内容 |
| --- | --- | --- | --- |
| `python_version.txt` | `"$ATTNVIEW_PYTHON" -V` | 0 | `Python 3.12.13` |
| `pip_check.txt` | `python -m pip check` | 0 | `No broken requirements found.` |
| `pip_freeze.txt` | `python -m pip freeze` | 0 | 197 行精确依赖版本 |
| `import_versions.txt` | `import torch, vllm` + 设备属性 | 0 | `torch 2.13.0+cu130`、`vllm 0.29.0`、`vllm.__file__` 指向 venv 站点包、`torch.cuda.is_available()=True`、`cc=12.0`、`94.97 GiB`、188 SM |
| `cuda_tensor.txt` | 小张量 CUDA 运算 + 同步 | 0 | `arange(1024).sum()=523776`（等于已知期望值）；512×512 fp32 matmul 对 CPU 参考最大绝对误差 `5.341e-05` |
| `vllm_cli.txt` | `venvs/attnview/bin/vllm --help` | 0 | 子命令 `chat/complete/serve/launch/bench/collect-env/run-batch` |
| `nvcc.txt` | `cuda/bin/nvcc --version` | 0 | release 13.4, V13.4.92 |
| `network-tools.txt` | 见下 | 0 | E4 网络与工具事实 |
| `verify-console.log` | `bash verify-runtime.sh` | 0 | 上述各项的命令、输出与 `[exit n]` 汇总，末尾"全部检查通过" |

## E4 网络与工具（`evidence/after/network-tools.txt`）

含 DNS 解析结果、四个目标的直连 HTTP 码与耗时、`/etc/network_turbo` 下 github 的对照、直连 github release 资产的结果、以及 `git/curl/jq/tar/zstd/uv/nvcc/docker/nsys/ncu/py-spy/perf` 的存在性。要点：`pypi.tuna` 200/0.51 s、`pypi.org` 200/2.24 s、`hf-mirror.com` 200/1.56 s；`github.com` 直连超时（000 @25 s）、经加速 200/4.45 s；release 资产直连 0 字节 @25 s。

## 数据盘扩容与收尾（2026-09-17T16:54Z 起）

用户把 `/root/autodl-tmp` 从 50 GiB 扩到 **200 GiB**（付费动作，用户执行）。

| 产物 | 命令 | 退出码 | 内容 |
| --- | --- | --- | --- |
| `evidence/post-expansion.txt` | 见文件内命令记录 | 0 | 扩容后的配额与挂载、收尾五步、各 3 次顺序 I/O 实测（数据盘写中位 699 MB/s / 读 2.0 GB/s；系统盘 729 MB/s / 2.0 GB/s）、宿主 load、`fio` 缺失、结论（两盘同速，同一 `/data` NVMe） |
| `evidence/verify-after-expand.log` | `source env.sh && bash verify-runtime.sh` | 0 | `HF_HOME` 改为 `models/hf-home` 之后 7 项检查全部 `[exit 0]` |
| `evidence/after/env-report-20260917-16*.md` | `bash material/attnview/scripts/env-probe.sh --out evidence/after` | 0 | 刷新后的环境快照，含 `HF_HOME=/root/attnview/models/hf-home` |

## 环境迁移到数据盘（2026-09-17T16:45Z 前后）

| 产物 | 命令 | 退出码 | 内容 |
| --- | --- | --- | --- |
| `evidence/move-to-data-disk.txt` | 见文件内命令记录 | 0 | 迁移动机（用户口径：性能相关资产优先数据盘）、`cp -a` 统计（8.3 GiB / 121244 文件 / 7.7 s）、符号链接切换、迁移前后 `df` |
| `evidence/verify-after-move.log` | `source env.sh && bash verify-runtime.sh`（走 `/root/attnview` 路径） | 0 | 迁移后 7 项检查全部 `[exit 0]`，末尾"全部检查通过"——证明符号链接路径下 venv、CLI、nvcc 全部可用 |

迁移动作本身：`cp -a /root/attnview/. /root/autodl-tmp/attnview/` → `mv /root/attnview /root/attnview.on-system-disk` → `ln -s /root/autodl-tmp/attnview /root/attnview` → 重跑验证 → `rm -rf /root/attnview.on-system-disk`（删除系统盘冗余副本）。系统盘可用空间由 8.1 GiB 变为 17 GiB。

## 空间回收

| 产物 | 内容 |
| --- | --- |
| `evidence/paged-kv-before-cleanup.txt` | 删除前的快照：大小 5.5 GiB（全部是 `.venv`）、git HEAD `8e6b0dc`、`## main...origin/main`、未推送提交 0、stash 0、origin URL；处置说明（只删 `.venv`，源码与 `.git` 保留，可用 `uv sync --locked` 重建） |

## 复现入口

```bash
ssh lab 'source /root/attnview/env.sh && bash /root/attnview/verify-runtime.sh'   # 环境层全部检查，原始输出重写 evidence/after/
ssh lab 'git -C /root/attnview/vllm rev-parse HEAD'                              # → 98dff2a81d747d1dba01a47f939f48c3526d4206
ssh lab '/root/attnview/venvs/attnview/bin/python -c "import vllm; print(vllm.__file__)"'
```

## 未留证据 / 未验证

- 模型权重、数据集、生成输出、性能计时：本阶段边界之外，**没有**任何产物。
- 实际 attention backend、`kernel_block_size`、CUDA Graph 模式：未采集（需启动服务，属下一阶段）。
- NCU / Nsight Systems / Docker：工具不存在，因此无相关证据。
