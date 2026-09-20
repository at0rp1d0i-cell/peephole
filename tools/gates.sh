#!/usr/bin/env bash
# 提交前的单一门禁入口（CONTRIBUTING §3）。用法：`bash tools/gates.sh`
#
# 每一项都会执行（不因前一项失败而中断），最后按累计结果给退出码：
#   0 = 全通过；非 0 = 至少一项失败。版本化钩子 `tools/git-hooks/pre-commit` 直接调用本脚本。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck source=/dev/null
source ./env.sh
: "${ATTNVIEW_PYTHON:?env.sh 未设置 ATTNVIEW_PYTHON}"

fail=0
run() {
  local name="$1"
  shift
  printf '== %s\n' "$name"
  if "$@"; then
    printf '   ok: %s\n' "$name"
  else
    printf '   FAIL: %s（见上）\n' "$name" >&2
    fail=1
  fi
}

# R9：大件只登记、不发行（阈值 10 MB 与 CONTRIBUTING §3 一致）。
check_size() {
  local big
  big="$(git ls-files -z | xargs -0 du -b 2>/dev/null | awk '$1 > 10485760 {print $2" ("$1" B)"}')"
  if [ -n "$big" ]; then
    printf '   以下入库文件超过 10 MB：\n%s\n' "$big" >&2
    return 1
  fi
}

# 密钥/token 模式扫描：与 CONTRIBUTING §3 的表达式一致，期望空。
check_secrets() {
  local hits
  hits="$(git ls-files -z | xargs -0 grep -lIE \
    '(gh[p]_|github[_]pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)' \
    2>/dev/null || true)"
  if [ -n "$hits" ]; then
    printf '   命中密钥/token 模式：\n%s\n' "$hits" >&2
    return 1
  fi
}

run "CPU 测试"       "$ATTNVIEW_PYTHON" -m pytest tests/ -q
run "静态检查"       "$ATTNVIEW_PYTHON" -m ruff check .
run "补丁树一致"     "$ATTNVIEW_PYTHON" "$REPO/tools/p2-gen-patch.py" --check
run "证据索引"       "$ATTNVIEW_PYTHON" "$REPO/tools/pub-evidence-registry.py" --check
run "脱敏"           "$ATTNVIEW_PYTHON" "$REPO/tools/pub-desensitize.py" --check
run "入库体积 ≤10MB" check_size
run "密钥模式扫描"   check_secrets

if [ "$fail" -ne 0 ]; then
  printf '\n门禁未通过：按上面 FAIL 项修正后重跑（临时跳过用 git commit --no-verify）\n' >&2
fi
exit "$fail"
