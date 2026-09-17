#!/usr/bin/env bash
# 为 attnview 建立"项目自带"的 CUDA 工具链前缀，不触碰系统 /usr/local/cuda。
#
# 现状（2026-09-17 实机核对）：本栈的 torch 是 cu130 构建，CUDA 13 的轮子把整套
# 工具链以合并布局装在 site-packages/nvidia/cu13/（nvcc 13.4、ptxas、nvlink、
# cudart、cublas…）。因此不需要额外装编译器，也不需要拼装散落目录，本脚本只把
# 它按 CUDA_HOME 的常规形状（bin/ include/ lib64/ nvvm/）暴露到 $ATTNVIEW_CUDA。
#
# 幂等：前缀用符号链接重建，重复执行结果一致；缺工具链时才补装 nvidia-cuda-nvcc。
#
# 用法（远端机上）：
#   bash /root/attnview/setup-local-cuda.sh
set -euo pipefail

env_file=${ATTNVIEW_ENV_FILE:-/root/attnview/env.sh}
[ -r "$env_file" ] || { echo "缺少环境文件：$env_file" >&2; exit 2; }
# shellcheck disable=SC1090
. "$env_file"

py=${ATTNVIEW_PYTHON:?}
cuda=${ATTNVIEW_CUDA:?}
site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

uvbin=$(command -v uv || true)
[ -n "$uvbin" ] || uvbin=${ATTNVIEW_UV_BOOTSTRAP:-}
[ -x "$uvbin" ] || { echo "找不到 uv（PATH 与 ${ATTNVIEW_UV_BOOTSTRAP:-未设置} 均无）" >&2; exit 2; }

say() { printf '\n== %s ==\n' "$*"; }
find_toolkit() { # 合并布局的 CUDA 工具链目录，如 <site>/nvidia/cu13
  local d
  for d in "$site"/nvidia/cu[0-9]*; do
    [ -d "$d/bin" ] && { printf '%s' "$d"; return 0; }
  done
  return 1
}

say "1/3 定位 venv 内的 CUDA 工具链"
if ! tk=$(find_toolkit); then
  printf '未随依赖带入，补装 nvidia-cuda-nvcc\n'
  "$uvbin" pip install --python "$py" nvidia-cuda-nvcc
  tk=$(find_toolkit) || { echo "补装后仍未找到 CUDA 工具链目录" >&2; exit 2; }
fi
printf '工具链目录：%s\n' "$tk"
printf 'nvcc      ：%s\n' "$("$tk/bin/nvcc" --version 2>&1 | tail -1)"
printf '运行时    ：%s\n' "$("$py" -c 'import torch; print("torch", torch.__version__, "| CUDA", torch.version.cuda)')"

say "2/3 暴露为 $cuda"
rm -rf "$cuda"
mkdir -p "$cuda"
for sub in bin include lib nvvm; do
  [ -e "$tk/$sub" ] || continue
  ln -sfn "$tk/$sub" "$cuda/$sub"
done
# CUDA_HOME 的常规布局用 lib64；指向同一份 lib，避免复制。
ln -sfn "$tk/lib" "$cuda/lib64"
printf '符号链接：\n'; ls -l "$cuda" | sed -n '2,$p' | sed 's/^/  /'

say "3/3 自检：nvcc 编一个 sm_120 cubin（只编不链，不需要系统 libcuda）"
printf 'CUDA_HOME   : %s\n' "$cuda"
printf 'nvcc --version:\n'; "$cuda/bin/nvcc" --version 2>&1 | sed 's/^/  /'
tmp=$(mktemp -d "$ATTNVIEW_HOME/tmp/nvcc-XXXXXX")
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/probe.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void add_one(float *x) { x[threadIdx.x] += 1.0f; }
int main() { return 0; }
CU
if "$cuda/bin/nvcc" -cubin -arch=sm_120 "$tmp/probe.cu" -o "$tmp/probe.cubin"; then
  printf '编译结果    : ok，sm_120 cubin %s 字节\n' "$(stat -c %s "$tmp/probe.cubin")"
else
  printf '编译结果    : 失败（见上方输出）\n' >&2
  exit 1
fi
printf '\n项目自带 CUDA 前缀就绪：%s\n' "$cuda"
