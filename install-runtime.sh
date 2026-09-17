#!/usr/bin/env bash
# attnview 远端运行时安装：uv 建 venv + 安装锁定版 vLLM + 检出 pin 的源码。
#
# 不下载模型权重/数据集；不改系统 CUDA、不装 apt 包、不写 /etc/environment。
# 项目自带的 CUDA 工具链由 setup-local-cuda.sh 单独建立。
#
# 幂等：venv 存在则复用；依赖已满足则 uv 直接跳过；源码目录存在则只 fetch/checkout。
#
# 用法（远端机上）：
#   bash /root/attnview/install-runtime.sh            # 全流程
#   bash /root/attnview/install-runtime.sh --deps     # 只做 venv + 依赖，跳过源码检出
#   VLLM_PIN=0.29.0 bash /root/attnview/install-runtime.sh
set -euo pipefail

env_file=${ATTNVIEW_ENV_FILE:-/root/attnview/env.sh}
[ -r "$env_file" ] || { echo "缺少环境文件：$env_file" >&2; exit 2; }
# shellcheck disable=SC1090
. "$env_file"

VLLM_PIN=${VLLM_PIN:-0.29.0}
VLLM_COMMIT=${VLLM_COMMIT:-98dff2a81d747d1dba01a47f939f48c3526d4206}
PY_MINOR=${PY_MINOR:-3.12}
VLLM_REPO=${VLLM_REPO:-https://github.com/vllm-project/vllm.git}
deps_only=0
[ "${1:-}" = "--deps" ] && deps_only=1

# 解释器来源：默认 uv 托管 CPython（环境完全由 uv 管理，不依赖镜像自带软件栈）。
# 托管解释器需从 github release 下载，直连实测长时间无进展，仅这一步借平台学术
# 加速；仍失败则回退镜像自带的 /root/miniconda3/bin/python。用 ATTNVIEW_BASE_PYTHON
# 可显式指定基座解释器，跳过下载。
PY_BASE=${ATTNVIEW_BASE_PYTHON:-}
if [ -n "$PY_BASE" ] && ! "$PY_BASE" -c "import sys; sys.exit(0 if sys.version_info[:2] >= tuple(int(p) for p in '$PY_MINOR'.split('.')) else 1)" 2>/dev/null; then
  printf '指定的基座解释器 %s 版本低于 %s，忽略\n' "$PY_BASE" "$PY_MINOR" >&2
  PY_BASE=""
fi

mkdir -p "$ATTNVIEW_LOGS_DIR" "$ATTNVIEW_REPORTS_DIR" "$ATTNVIEW_HOME/tmp"
stamp=$(date -u +%Y%m%d-%H%M%S)
log="$ATTNVIEW_LOGS_DIR/install-runtime-$stamp.log"
exec > >(tee -a "$log") 2>&1

say() { printf '\n== %s ==\n' "$*"; }

# uv 解析：优先 PATH（环境建好后是 venv 内自托管的那份），否则用镜像里的引导副本。
uvbin=$(command -v uv || true)
[ -n "$uvbin" ] || uvbin=${ATTNVIEW_UV_BOOTSTRAP:-}
[ -x "$uvbin" ] || { echo "找不到 uv（PATH 与 ${ATTNVIEW_UV_BOOTSTRAP:-未设置} 均无）" >&2; exit 2; }

# uv 托管 CPython：下载走 github release，直连停滞时借平台学术加速（仅覆盖这一步，
# 不要让代理影响后面的 PyPI 镜像下载）。
mkvenv_managed() {
  if [ -r /etc/network_turbo ]; then
    ( set -a; . /etc/network_turbo >/dev/null 2>&1; set +a
      "$uvbin" venv --python "$PY_MINOR" --python-preference only-managed --seed "$ATTNVIEW_VENV" )
  else
    "$uvbin" venv --python "$PY_MINOR" --python-preference only-managed --seed "$ATTNVIEW_VENV"
  fi
}

# git 访问 github：实测直连 clone 停滞，走平台学术加速；PyPI 走镜像，不经代理。
git_remote() {
  if [ -r /etc/network_turbo ]; then
    ( set -a; . /etc/network_turbo >/dev/null 2>&1; set +a; git "$@" )
  else
    git "$@"
  fi
}

say "环境"
printf 'host=%s\n' "$(hostname)"
printf 'venv=%s\npython=%s\nsrc=%s\nvllm=%s @ %s\nlog=%s\n' \
  "$ATTNVIEW_VENV" "$ATTNVIEW_PYTHON" "$ATTNVIEW_VLLM_SRC" "$VLLM_PIN" "$VLLM_COMMIT" "$log"
printf 'uv=%s（%s）\n' "$uvbin" "$("$uvbin" --version 2>/dev/null || echo -)"
printf '安装前磁盘：\n'; df -h / "$ATTNVIEW_HOME" | sed 's/^/  /'

say "1/4 uv venv（Python $PY_MINOR）"
if [ -x "$ATTNVIEW_PYTHON" ]; then
  printf '复用已有 venv：%s\n' "$ATTNVIEW_PYTHON"
elif [ -n "$PY_BASE" ]; then
  printf '基座解释器：显式指定 %s（%s）\n' "$PY_BASE" "$("$PY_BASE" -V 2>&1)"
  "$uvbin" venv --python "$PY_BASE" --seed "$ATTNVIEW_VENV"
else
  printf '基座解释器：uv 托管 CPython %s\n' "$PY_MINOR"
  if ! mkvenv_managed; then
    PY_BASE=/root/miniconda3/bin/python
    printf 'uv 托管解释器不可用，回退镜像自带：%s\n' "$PY_BASE" >&2
    "$uvbin" venv --python "$PY_BASE" --seed "$ATTNVIEW_VENV"
  fi
fi
printf 'python: %s\n' "$("$ATTNVIEW_PYTHON" -V 2>&1)"
printf 'pip   : %s\n' "$("$ATTNVIEW_PYTHON" -m pip -V 2>&1)"
if [ -x "$ATTNVIEW_VENV/bin/uv" ]; then
  printf 'uv（项目自带）: %s\n' "$("$ATTNVIEW_VENV/bin/uv" --version)"
else
  "$uvbin" pip install --python "$ATTNVIEW_PYTHON" uv
  printf 'uv（项目自带）: %s → %s\n' "$("$ATTNVIEW_VENV/bin/uv" --version)" "$ATTNVIEW_VENV/bin/uv"
fi

say "2/4 安装 vLLM $VLLM_PIN（含 torch 与 nvidia-*-cu12 运行时）"
printf '索引：%s\n' "$UV_DEFAULT_INDEX"
"$uvbin" pip install --python "$ATTNVIEW_PYTHON" "vllm==$VLLM_PIN"
"$ATTNVIEW_PYTHON" - <<'PY'
import importlib.metadata as md
for name in ("vllm", "torch", "torchvision", "torchaudio", "triton", "transformers",
             "tokenizers", "numpy", "xformers", "flashinfer-python", "ray", "xgrammar"):
    try:
        print(f"  {name:20s} {md.version(name)}")
    except Exception:
        print(f"  {name:20s} 未安装")
PY

say "3/4 冻结实际解析到的版本"
freeze="$ATTNVIEW_REPORTS_DIR/requirements.freeze.txt"
"$ATTNVIEW_PYTHON" -m pip freeze > "$freeze"
printf 'freeze=%s（%s 行）\n' "$freeze" "$(wc -l < "$freeze" | tr -d ' ')"
cp -f "$freeze" "$ATTNVIEW_HOME/requirements.freeze.txt"

if [ "$deps_only" -eq 1 ]; then
  say "跳过源码检出（--deps）"
else
  say "4/4 检出 vLLM 源码 @ $VLLM_COMMIT"
  src=$ATTNVIEW_VLLM_SRC
  if [ -d "$src/.git" ]; then
    printf '复用已有 checkout：%s\n' "$src"
    git -C "$src" remote set-url origin "$VLLM_REPO"
    git_remote -C "$src" fetch --tags --force origin
  else
    git_remote clone "$VLLM_REPO" "$src"
  fi
  git -C "$src" checkout --detach "$VLLM_COMMIT"
  printf 'HEAD          = %s\n' "$(git -C "$src" rev-parse HEAD)"
  printf 'HEAD 描述     = %s\n' "$(git -C "$src" describe --tags --always 2>&1)"
  printf '工作区状态    :\n'; git -C "$src" status --porcelain=v1 -b | sed 's/^/  /'
  printf '源码体积      : %s\n' "$(du -sh "$src" | cut -f1)"
fi

say "import 来源核对（clone 不等于 Python 正在跑的代码）"
"$ATTNVIEW_PYTHON" -c 'import vllm, sys; print("sys.executable =", sys.executable); print("vllm.__file__  =", vllm.__file__); print("vllm.__version__ =", vllm.__version__)'

say "安装后磁盘"
df -h / "$ATTNVIEW_HOME" | sed 's/^/  /'
printf 'venv 体积：%s\n' "$(du -sh "$ATTNVIEW_VENV" | cut -f1)"

printf '\n安装完成。日志：%s\n' "$log"
printf '下一步：bash %s/setup-local-cuda.sh，然后 bash %s/verify-runtime.sh\n' "$ATTNVIEW_HOME" "$ATTNVIEW_HOME"
