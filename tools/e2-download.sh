#!/usr/bin/env bash
# E2: 下载主候选权重（可断点续传）。证据落 /root/attnview/evidence/p0-model/。
# 用法：bash /root/attnview/tools/e2-download.sh [追加的 hf download 参数]
set -uo pipefail

source /root/attnview/env.sh

REPO="Qwen/Qwen3.8-27B"
REV="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
OUT="$ATTNVIEW_EVIDENCE_DIR/p0-model"
LOG="$OUT/e2-download.log"
mkdir -p "$OUT"

snapshot="$HF_HUB_CACHE/models--Qwen--Qwen3.8-27B/snapshots/$REV"

{
  echo "=== E2 download run ==="
  echo "CMD: hf download $REPO --revision $REV $*"
  echo "HF_ENDPOINT=$HF_ENDPOINT"
  echo "HF_HOME=$HF_HOME"
  echo "HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "START_EPOCH=$(date +%s)"
  echo "--- df before (bytes) ---"
  df -B1 /root/attnview/models | cat
  echo "--- du before ---"
  du -sb "$HF_HUB_CACHE" 2>/dev/null || echo "0	(none)"
  echo "--- hf download output ---"
} >>"$LOG"

hf download "$REPO" --revision "$REV" "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

end_epoch=$(date +%s)
start_epoch=$(grep -m1 '^START_EPOCH=' "$LOG" | cut -d= -f2)
elapsed=$((end_epoch - start_epoch))
{
  echo "--- end ---"
  echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "END_EPOCH=$end_epoch"
  echo "ELAPSED_SECONDS=$elapsed"
  echo "EXIT=$rc"
  echo "--- df after (bytes) ---"
  df -B1 /root/attnview/models | cat
  echo "--- du after ---"
  du -sb "$HF_HUB_CACHE" 2>/dev/null || echo "n/a"
  du -sb "$snapshot" 2>/dev/null || echo "n/a"
  echo "--- snapshot file count ---"
  find "$snapshot" -maxdepth 1 -type f -printf '%s\t%f\n' 2>/dev/null | sort -k2 | tee "$OUT/e2-snapshot-files.txt" | wc -l
} >>"$LOG"

exit "$rc"
