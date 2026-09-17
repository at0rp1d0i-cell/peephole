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
# 第 4 步（2026-09-18 追加，经用户授权）：补齐 JIT 编译/链接所需的东西，并做版本一致性检查。
# 背景：pip 轮子的 CUDA 套件允许 nvcc 与 cudart 处于不同 minor（实测曾出现 nvcc 13.4 + cuda.h 13.0），
# flashinfer 内置 CCCL 的编译期检查（cuda/std/__cccl/cuda_toolkit.h:41）会直接 #error；
# 且 nvidia-cuda-runtime 轮子只提供带 SONAME 的 libcudart.so.13、不含开发用 libcudart.so 与驱动 stub，
# 于是 "-lcudart -lcuda" 这类 JIT 链接必然失败。第 4 步把这两类问题一次性堵住。
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

say "1/4 定位 venv 内的 CUDA 工具链"
if ! tk=$(find_toolkit); then
  printf '未随依赖带入，补装 nvidia-cuda-nvcc\n'
  "$uvbin" pip install --python "$py" nvidia-cuda-nvcc
  tk=$(find_toolkit) || { echo "补装后仍未找到 CUDA 工具链目录" >&2; exit 2; }
fi
printf '工具链目录：%s\n' "$tk"
printf 'nvcc      ：%s\n' "$("$tk/bin/nvcc" --version 2>&1 | tail -1)"
printf '运行时    ：%s\n' "$("$py" -c 'import torch; print("torch", torch.__version__, "| CUDA", torch.version.cuda)')"

say "2/4 暴露为 $cuda"
rm -rf "$cuda"
mkdir -p "$cuda"
for sub in bin include lib nvvm; do
  [ -e "$tk/$sub" ] || continue
  ln -sfn "$tk/$sub" "$cuda/$sub"
done
# CUDA_HOME 的常规布局用 lib64；指向同一份 lib，避免复制。
ln -sfn "$tk/lib" "$cuda/lib64"
printf '符号链接：\n'; ls -l "$cuda" | sed -n '2,$p' | sed 's/^/  /'

say "3/4 自检：nvcc 编一个 sm_120 cubin（只编不链，不需要系统 libcuda）"
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

say "4/4 补齐 JIT 的 dev 链接与驱动 stub，并做版本一致性检查"

# --- 4a 版本一致性：nvcc 的 major.minor 必须与 cuda.h 的 CUDA_VERSION major.minor 相同 ---
nvcc_pkg_ver=$("$py" -c 'import importlib.metadata as m;print(m.version("nvidia-cuda-nvcc"))')
check_cuda_versions() {
  nvcc_rel=$("$cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)
  hdr_ver=$(sed -n 's/^#define CUDA_VERSION \([0-9]\+\).*/\1/p' "$cuda/include/cuda.h" | head -1)
  hdr_rel="$((hdr_ver / 1000)).$(( (hdr_ver % 1000) / 10 ))"
}

check_cuda_versions
printf 'nvcc 版本   : %s (pip nvidia-cuda-nvcc %s)\ncuda.h 版本 : %s (CUDA_VERSION=%s)\n' \
  "$nvcc_rel" "$nvcc_pkg_ver" "$hdr_rel" "$hdr_ver"

if [ "$nvcc_rel" != "$hdr_rel" ]; then
  cat <<EOF

检测到 CUDA 编译器与头文件 minor 版本不一致（nvcc $nvcc_rel vs cuda.h $hdr_rel）。
pip 允许这种组合（nvidia-cuda-nvcc 对 nvidia-cuda-runtime 未锁版本），但 flashinfer 内置 CCCL 的
编译期兼容性检查（cuda/std/__cccl/cuda_toolkit.h:41）会直接 #error，导致 JIT 路径不可用。
按用户 2026-09-18 的授权，自动把项目 venv 内的 CUDA 组件对齐到 nvcc 的版本：
  nvidia-cuda-runtime / nvidia-cuda-nvrtc / nvidia-cuda-cupti == $nvcc_pkg_ver
（只动项目 venv，不碰系统 CUDA/驱动；不升级任何无关包。）
EOF
  "$py" -m pip install --upgrade \
    "nvidia-cuda-runtime==$nvcc_pkg_ver" \
    "nvidia-cuda-nvrtc==$nvcc_pkg_ver" \
    "nvidia-cuda-cupti==$nvcc_pkg_ver" || {
    echo "错误：对齐安装失败，请检查索引与网络后重跑本脚本。" >&2
    exit 1
  }
  "$py" -m pip check || { echo "错误：对齐后 pip check 失败。" >&2; exit 1; }
  check_cuda_versions
  printf '对齐后：nvcc %s / cuda.h %s\n' "$nvcc_rel" "$hdr_rel"
  if [ "$nvcc_rel" != "$hdr_rel" ]; then
    echo "错误：对齐后版本仍不一致（nvcc $nvcc_rel vs cuda.h $hdr_rel），停止。" >&2
    exit 1
  fi
fi

# --- 4b 开发用链接：轮子只给 libcudart.so.<major>，JIT 用 -lcudart ---
libdir=$tk/lib
if [ -e "$libdir/libcudart.so.13" ] && [ ! -e "$libdir/libcudart.so" ]; then
  ln -sfn libcudart.so.13 "$libdir/libcudart.so"
  printf '已补 libcudart.so -> libcudart.so.13\n'
fi

# --- 4c 驱动 stub：lib64/stubs/libcuda.so 供 -lcuda 解析 ---
# 优先用官方 CUDA redist 的同版本 stub（纯符号桩、不含实现，运行时由宿主驱动提供 libcuda.so.1）；
# 下载不可用时退回宿主驱动库并告警。
stub_dir=$libdir/stubs
mkdir -p "$stub_dir"
stub=$stub_dir/libcuda.so
build_ver=$("$py" -c 'import importlib.metadata as m;print(m.version("nvidia-cuda-nvcc"))')
if [ ! -e "$stub" ]; then
  url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-${build_ver}-archive.tar.xz"
  # 记录过的已知版本校验值（13.4.92）；其它版本只打印不校验。
  known_ver=13.4.92
  known_sha=0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5
  printf '驱动 stub 缺失，尝试从官方 redist 下载：%s\n' "$url"
  if timeout 120 curl -sS -o "$tmp/cudart.tar.xz" "$url"; then
    got_sha=$(sha256sum "$tmp/cudart.tar.xz" | awk '{print $1}')
    if [ "$build_ver" = "$known_ver" ] && [ "$got_sha" != "$known_sha" ]; then
      echo "错误：cudart 归档 sha256 与记录不符（期望 $known_sha，实得 $got_sha）" >&2
      exit 1
    fi
    tar -xJf "$tmp/cudart.tar.xz" -C "$tmp" --strip-components=1 \
      "cuda_cudart-linux-x86_64-${build_ver}-archive/lib/stubs/libcuda.so"
    cp -a "$tmp/lib/stubs/libcuda.so" "$stub"
    printf '已安装官方 stub（%s，sha256=%s）\n' "$build_ver" "$got_sha"
  elif [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$stub"
    printf '警告：官方下载不可用，暂用宿主驱动库 %s 作为链接目标\n' "$(readlink -f "$stub")" >&2
  else
    echo "错误：既无法下载官方 stub，也未找到宿主 libcuda.so.1" >&2
    exit 1
  fi
fi

# --- 4d 链接自检：复刻 JIT 的链接形态（-L lib64 -L lib64/stubs -lcudart -lcuda）---
cat > "$tmp/link_probe.cu" <<'CU'
#include <cuda_runtime.h>
#include <cuda.h>
__global__ void k(float *x) { x[threadIdx.x] += 1.0f; }
extern "C" int probe() { void *p = nullptr; cudaMalloc(&p, 8); cuInit(0); return (int)(p != nullptr); }
CU
if "$cuda/bin/nvcc" -shared -arch=sm_120 -std=c++17 "$tmp/link_probe.cu" \
     -L"$cuda/lib64" -L"$cuda/lib64/stubs" -lcudart -lcuda -o "$tmp/link_probe.so"; then
  printf '链接结果    : ok，%s 字节（JIT 链接能力可用）\n' "$(stat -c %s "$tmp/link_probe.so")"
else
  printf '链接结果    : 失败（见上方输出）\n' >&2
  exit 1
fi

printf '\n项目自带 CUDA 前缀就绪：%s\n' "$cuda"
