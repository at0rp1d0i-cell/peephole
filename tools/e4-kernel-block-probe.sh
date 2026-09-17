#!/usr/bin/env bash
# 观测 kernel_block_size（真实值）：注入只读 sitecustomize 钩子后跑一次原版 serve。
#
# 探针配置（报告需注明）：
#   PYTHONPATH=tools/kbs-probe（含 sitecustomize.py） + KBS_PROBE_OUT=<输出 JSON>
#   服务参数与正式基线**完全一致**（serve-vanilla.sh 的全部 flag，含 --revision / --max-num-seqs 256）
#   EngineCore 子进程同样会加载该 sitecustomize，因此钩子对真实服务路径有效。
#   钩子只读：调用原 initialize_kv_cache 后把 self._kernel_block_sizes 抄到文件，不改行为。
#
# 用法：
#   bash tools/e4-kernel-block-probe.sh            # 前台；就绪后由调用方停服
set -uo pipefail

source /root/attnview/env.sh

OUT="$ATTNVIEW_EVIDENCE_DIR/p0-model/e4-kernel-block-probe.json"
rm -f "$OUT" "$OUT.error"
export PYTHONPATH="$ATTNVIEW_HOME/tools/kbs-probe${PYTHONPATH:+:$PYTHONPATH}"
export KBS_PROBE_OUT="$OUT"

echo "probe: PYTHONPATH=$PYTHONPATH"
echo "probe: KBS_PROBE_OUT=$OUT"

bash "$ATTNVIEW_HOME/tools/serve-vanilla.sh" --tag e4c
