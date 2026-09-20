#!/usr/bin/env bash
# CPU 测试入口（不联网、不需要 GPU）。
#
# 只是 `python -m pytest tests/` 的薄包装：仓库里跑测试只有这一个入口，
# 提交前的完整门禁见 tools/gates.sh（本脚本是它的子集）。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=''
source ./env.sh
: "${ATTNVIEW_PYTHON:?env.sh 未设置 ATTNVIEW_PYTHON}"
exec "$ATTNVIEW_PYTHON" -m pytest tests/ "$@"
