# attnview 远端环境定义（AutoDL 容器 / RTX PRO 6000 Blackwell 96GB）。
#
# 单一来源：交互 shell 与非交互 ssh 命令都显式 source 本文件，避免"手动登录能用、
# ssh 一句命令不生效"两套环境。用法：
#
#   source /root/attnview/env.sh
#   "$ATTNVIEW_PYTHON" -c 'import vllm; print(vllm.__file__)'
#
# 约束：只做变量赋值与幂等的 PATH 追加；不输出、不做命令替换、不依赖 cwd，
# 因为非交互命令可能反复 source 本文件。不改 /etc/environment，也不改系统 PATH。
#
# 解释器与依赖全部由 uv 管理；CUDA 工具链由本项目自带（venv 内的 nvidia-*-cu12
# 轮子 + $ATTNVIEW_CUDA 前缀，见 setup-local-cuda.sh），不使用系统 /usr/local/cuda。
#
# 物理位置：/root/attnview 是指向数据盘 /root/autodl-tmp/attnview 的符号链接
# （性能相关资产优先放本机数据盘，且该盘可跨实例复制）。变量一律用 /root/attnview，
# 使 venv 内绝对路径与脚本 shebang 不因物理位置而失效。实例重建后若链接丢失：
#   ln -s /root/autodl-tmp/attnview /root/attnview

export ATTNVIEW_HOME="/root/attnview"
export ATTNVIEW_MATERIAL="$ATTNVIEW_HOME/material/attnview"   # 素材仓快照（只读参考）
export ATTNVIEW_VENV="$ATTNVIEW_HOME/venvs/attnview"
export ATTNVIEW_PYTHON="$ATTNVIEW_VENV/bin/python"
export ATTNVIEW_VLLM_SRC="$ATTNVIEW_HOME/vllm"               # vLLM 源码 checkout（pin）
export ATTNVIEW_CUDA="$ATTNVIEW_HOME/cuda"                   # 项目自带 CUDA 前缀
export ATTNVIEW_MODELS="$ATTNVIEW_HOME/models"               # 权重与 HF 缓存（数据盘，200 GiB）
export ATTNVIEW_EVIDENCE_DIR="$ATTNVIEW_HOME/evidence"
export ATTNVIEW_REPORTS_DIR="$ATTNVIEW_HOME/reports"
export ATTNVIEW_LOGS_DIR="$ATTNVIEW_HOME/logs"

# 项目自带 CUDA：不使用 /usr/local/cuda 系统工具链。
export CUDA_HOME="$ATTNVIEW_CUDA"
export CUDA_PATH="$ATTNVIEW_CUDA"

# uv：解释器安装位置、缓存、索引与链接模式。
# UV_LINK_MODE=hardlink 只在缓存与 venv 同一文件系统时生效——两者都在本项目内。
#
# 引导副本：建 venv 之前镜像里只有这一份 uv（非交互 shell 不在 PATH 上）。
# 环境建好后 uv 自托管在 venv 内（$ATTNVIEW_VENV/bin/uv，随 freeze 一起锁版本），
# PATH 里 venv/bin 在前，后续步骤自动用项目自带的那份。
export ATTNVIEW_UV_BOOTSTRAP="/root/miniconda3/bin/uv"

# 可选：显式指定基座解释器（跳过 uv 托管 CPython 的下载）。留空时由 uv 托管。
# export ATTNVIEW_BASE_PYTHON="/root/miniconda3/bin/python"
export UV_PYTHON_INSTALL_DIR="$ATTNVIEW_HOME/pythons"
export UV_CACHE_DIR="$ATTNVIEW_HOME/caches/uv"
export UV_LINK_MODE="hardlink"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_INDEX_URL="$UV_DEFAULT_INDEX"

# 可重建的缓存与临时目录统一放项目内（默认 /root/.cache 与 /root/autodl-tmp 混放）。
export PIP_CACHE_DIR="$ATTNVIEW_HOME/caches/pip"
export TRITON_CACHE_DIR="$ATTNVIEW_HOME/caches/triton"
# vLLM 的 torch.compile / AOT / FlashInfer autotune 缓存默认落 ~/.cache/vllm（系统盘）；
# 2026-09-18 实测一次运行即写 243 MB，而系统盘只剩 18 GiB，故显式改到数据盘。
export VLLM_CACHE_ROOT="$ATTNVIEW_HOME/caches/vllm"
# FlashInfer 的 JIT 缓存 = $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer（沿用上游布局）。
export FLASHINFER_WORKSPACE_BASE="$ATTNVIEW_HOME/caches"
export TORCH_EXTENSIONS_DIR="$ATTNVIEW_HOME/caches/torch-extensions"
export TORCH_HOME="$ATTNVIEW_HOME/caches/torch"
export CUDA_CACHE_PATH="$ATTNVIEW_HOME/caches/cuda"
# 权重与 HF 缓存放数据盘上的固定位置（2026-09-17 扩容到 200 GiB 后启用；
# 之前 50 GiB 配额放不下 51.75 GiB 的单个 27B checkpoint）。
export HF_HOME="$ATTNVIEW_MODELS/hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"

# HF 下载源必须固定：2026-09-17 实测 huggingface.co 直连不通（curl 000/10 s、0 字节），
# 经平台学术加速也只有 0.32 MB/s；hf-mirror.com 可达，实测吞吐 3.5–4.2 MB/s
# （100 MiB 单流 3.49 MB/s；200 MiB 四路聚合 4.17 MB/s）——即 51.77 GiB 的 27B 权重
# 约需 3.7–4.4 小时。不设 HF_ENDPOINT 时 `hf download` 会走直连并失败。
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$ATTNVIEW_HOME/tmp"

# 与镜像 ~/.bashrc 一致，避免交互/非交互两条路径线程数不同。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 幂等前置，防止反复 source 撑大 PATH。
case ":$PATH:" in
  *":$ATTNVIEW_VENV/bin:"*) ;;
  *) PATH="$ATTNVIEW_VENV/bin:$PATH" ;;
esac
case ":$PATH:" in
  *":$ATTNVIEW_CUDA/bin:"*) ;;
  *) PATH="$ATTNVIEW_CUDA/bin:$PATH" ;;
esac
export PATH
