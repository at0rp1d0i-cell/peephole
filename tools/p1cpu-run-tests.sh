#!/usr/bin/env bash
# 阶段 03 CPU 测试入口（不联网、不需要 GPU）。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=''
source ./env.sh
: "${ATTNVIEW_PYTHON:?env.sh 未设置 ATTNVIEW_PYTHON}"
exec "$ATTNVIEW_PYTHON" -m unittest discover -s tests -t tests "$@"
