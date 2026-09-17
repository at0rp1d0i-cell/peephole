#!/usr/bin/env python
"""E4 取证：从 vLLM serve 日志里抽取本阶段要求的逐项证据行。

用法：
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e4-extract.py LOG [LOG2 ...] --out DIR

只做抽取（行号 + 原文），不做结论；无法命中的组会明确列出为空，便于人工补查。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GROUPS: dict[str, list[str]] = {
    "dtype_and_loading": [
        r"dtype",
        r"[Ll]oading (model )?weights",
        r"[Mm]odel loading took",
        r"[Ll]oading took",
        r"[Mm]emory profiling",
        r"torch\.dtype",
    ],
    "attention_backend": [
        r"[Aa]ttention backend",
        r"Using .*[Bb]ackend",
        r"flash_attn",
        r"[Ff]lash[Aa]ttention",
        r"[Ff]lash[Ii]nfer",
        r"TRITON_ATTN",
        r"[Bb]ackend.*select",
        r"invalid_reasons",
        r"GDN|GatedDeltaNet|gated_delta",
        r"mamba",
    ],
    "block_size": [
        r"block[\s_\-]?size",
        r"kernel_block",
        r"mamba[_-]block[_-]size",
        r"page size",
    ],
    "cudagraph": [
        r"cudagraph",
        r"CUDA graph",
        r"[Gg]raph capture",
        r"Capturing",
        r"compil",
        r"UNIFORM_BATCH",
        r"PIECEWISE",
    ],
    "kv_cache": [
        r"KV cache",
        r"kv_cache",
        r"Available KV",
        r"GPU KV cache size",
        r"Maximum concurrency",
        r"num_gpu_blocks",
        r"blocks",
    ],
    "context_and_scheduling": [
        r"max_model_len",
        r"[Cc]hunked prefill",
        r"max_num_(batched_tokens|seqs)",
        r"[Pp]refix caching",
        r"[Ss]cheduler",
    ],
    "startup_timing": [
        r"Starting vLLM",
        r"init engine",
        r"engine.*took",
        r"Started server",
        r"Uvicorn running",
        r"total.*took",
        r"took .*s\b",
    ],
    "warnings_and_degradation": [
        r"WARNING",
        r"[Dd]eprecat",
        r"fallback|fall back|[Dd]isabl|[Dd]egrad|not supported",
        r"[Ee]rror|[Tt]raceback",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-group", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {"logs": {}, "groups": {}}
    compiled = {g: re.compile("|".join(pat), re.IGNORECASE) for g, pat in GROUPS.items()}
    hits: dict[str, list[dict[str, object]]] = {g: [] for g in GROUPS}

    for log in args.logs:
        p = Path(log)
        lines = p.read_text(errors="replace").splitlines()
        result["logs"][str(p)] = {"lines": len(lines)}
        for i, line in enumerate(lines, 1):
            for g, rx in compiled.items():
                if rx.search(line):
                    if len(hits[g]) < args.max_per_group:
                        hits[g].append({"log": str(p), "line": i, "text": line[:400]})

    result["groups"] = hits
    (out / "e4-extract.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for g, rows in hits.items():
        print(f"\n=== {g} ({len(rows)} hits) ===")
        for r in rows[:60]:
            print(f"  {r['log']}:{r['line']}: {r['text']}")
        if len(rows) > 60:
            print(f"  ... {len(rows) - 60} more in {out / 'e4-extract.json'}")
    print("\nwritten:", out / "e4-extract.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
