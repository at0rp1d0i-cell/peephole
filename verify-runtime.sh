#!/usr/bin/env bash
# attnview 环境层最小验证（对应工作单 E5）：干净会话下逐项检查并保留原始输出。
#
# 只做环境验证：小张量运算、模块 import、CLI 加载。不加载模型、不计时、不下载权重。
# 用法（远端机上）：
#   bash /root/attnview/verify-runtime.sh
set -uo pipefail

env_file=${ATTNVIEW_ENV_FILE:-/root/attnview/env.sh}
[ -r "$env_file" ] || { echo "缺少环境文件：$env_file" >&2; exit 2; }
# shellcheck disable=SC1090
. "$env_file"

out=${ATTNVIEW_EVIDENCE_DIR:-$ATTNVIEW_HOME/evidence}/after
mkdir -p "$out"

fail=0
run() { # run <检查名> <命令...>
  local name=$1; shift
  printf '\n=== %s ===\n$ %s\n' "$name" "$*"
  "$@" > "$out/$name.txt" 2>&1
  local rc=$?
  cat "$out/$name.txt"
  printf '[exit %s] → %s\n' "$rc" "$out/$name.txt"
  [ "$rc" -eq 0 ] || fail=1
  return 0
}

py=$ATTNVIEW_PYTHON
[ -x "$py" ] || { echo "解释器不存在：$py（先跑 install-runtime.sh）" >&2; exit 2; }

run python_version "$py" -V
run pip_check "$py" -m pip check
run pip_freeze "$py" -m pip freeze
run import_versions "$py" - <<'PY'
import sys, importlib.metadata as md
print("sys.executable =", sys.executable)
import torch
print("torch          =", torch.__version__, "| torch.version.cuda =", torch.version.cuda,
      "| cudnn =", torch.backends.cudnn.version())
import vllm
print("vllm.__version__ =", vllm.__version__)
print("vllm.__file__    =", vllm.__file__)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} cc={p.major}.{p.minor} "
          f"mem={p.total_memory/2**30:.2f} GiB sm_count={p.multi_processor_count}")
for name in ("transformers", "tokenizers", "triton", "numpy", "ray"):
    try:
        print(f"{name:14s} = {md.version(name)}")
    except Exception:
        print(f"{name:14s} = 未安装")
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset: 全部可见>"))
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
PY
run cuda_tensor "$py" - <<'PY'
import torch
print("设备：", torch.cuda.get_device_name(0))
# 1) 预先知道结果的整数运算
n = torch.arange(1024, device="cuda", dtype=torch.int64).sum().item()
expected = 1024 * 1023 // 2
print(f"arange(1024).sum() = {n}（期望 {expected}）→ {'ok' if n == expected else '不符'}")
assert n == expected
# 2) 小张量矩阵乘，与 CPU 参考对照
torch.manual_seed(0)
a = torch.randn(512, 512, dtype=torch.float32)
b = torch.randn(512, 512, dtype=torch.float32)
ref = a @ b
got = (a.cuda() @ b.cuda()).cpu()
torch.cuda.synchronize()
diff = (got - ref).abs().max().item()
print(f"512x512 fp32 matmul 与 CPU 参考的最大绝对误差 = {diff:.3e}")
assert diff < 1e-3, diff
print("显存：已分配 %.1f MiB / 保留 %.1f MiB" % (
    torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20))
print("小张量 CUDA 与同步：ok")
PY
run vllm_cli timeout 300 "$ATTNVIEW_VENV/bin/vllm" --help
run nvcc "$ATTNVIEW_CUDA/bin/nvcc" --version

printf '\n=== 汇总 ===\n'
if [ "$fail" -eq 0 ]; then
  printf '全部检查通过（原始输出：%s）\n' "$out"
else
  printf '存在失败项：见上方 [exit != 0] 标注与 %s 中对应文件\n' "$out"
fi
exit "$fail"
