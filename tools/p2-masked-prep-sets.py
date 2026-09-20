#!/usr/bin/env python3
"""masked-prep：可见集合的**独立** CPU 计算（先提交后运行）。

做法（不新造框架、不改分段规则、不 import 候选筛选实现）：
1) 用阶段 03 真实 tokenizer + `render_arm(arm="da", ...)` 重渲染 da fixture，记录 token 哈希；
2) 与配置里 note 给出的事实逐项核对（不一致即报错，不猜、不近似）；
3) **独立**按协议 span 展开各模式的 KV 位置集合：
   global = [0, kv_len)；local = sink ∪ local_window ∪ response=[prompt_len, kv_len)；
   focus_k = local ∪ 段 k 的 token span（与合同 `readview.semantic_spans` 语义一致，但由本脚本自行展开）；
4) 位置 → canonical 块（向上取整，b 来自配置），给出「已写块 / 可见块 / **被排除的已写块**」；
5) **分别**列出"段边界（token 粒度）"与"canonical 有效尾块边界（块粒度）"，两者不混用。
用法：source env.sh && CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-masked-prep-sets.py --config configs/p2-masked-prep/sets.json --out evidence/p3-calib/masked-prep/sets.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SNAPSHOT = REPO / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def block_of(token_index: int, block_size: int) -> int:
    return token_index // block_size


def blocks_of_span(start: int, end: int, block_size: int) -> list[int]:
    if end <= start:
        return []
    return sorted({block_of(p, block_size) for p in range(start, end)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    fixture = json.loads((REPO / cfg["fixture"]).read_text())
    block_size = int(cfg["block_size"])

    from transformers import AutoTokenizer

    from attnview.prompt import _tokenize_with_offsets, render_arm  # 阶段 03 既有入口
    from attnview.segmenter import build_offsets_index, segment_context

    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    context = fixture["document"]
    question = fixture["question"]
    _ids, offsets = _tokenize_with_offsets(tokenizer, context)
    segments = segment_context(context, build_offsets_index(offsets))
    arm = render_arm("da", segments, question, context, tokenizer, enable_thinking=False)

    prompt_len = len(arm.token_ids)
    seg_spans = [list(s) for s in arm.segment_spans]
    local_span = list(arm.scaffold.local_window_span)
    sink_span = list(arm.scaffold.sink_span)
    prompt_hash = hashlib.sha256(bytes(str(list(arm.token_ids)), "utf-8")).hexdigest()

    exp = cfg["expected_from_order"]
    facts = {
        "prompt_len": prompt_len == exp["prompt_len"],
        "segment_spans": seg_spans == [list(s) for s in exp["segment_spans"]],
        "local_window_span": local_span == exp["local_window_span"],
        "sink_span": sink_span == exp["sink_span"],
    }
    if not all(facts.values()):
        raise RuntimeError(f"重渲染事实与 note 不一致，拒绝继续：{facts}\n实际={dict(prompt_len=prompt_len, segment_spans=seg_spans, local_window_span=local_span, sink_span=sink_span)}")

    report: dict = {
        "config": str(args.config), "fixture": cfg["fixture"], "block_size": block_size,
        "prompt_len": prompt_len, "segment_spans": seg_spans, "local_window_span": local_span,
        "sink_span": sink_span, "token_ids_sha256": prompt_hash, "facts_match_order": facts,
        "steps": {},
    }

    for step in cfg["decode_steps"]:
        kv_len = prompt_len + step
        written = sorted({block_of(p, block_size) for p in range(kv_len)})
        base_spans = [tuple(sink_span), tuple(local_span), (prompt_len, kv_len)]
        per_mode: dict[str, dict] = {}
        for mode in cfg["modes"]:
            spans = list(base_spans)
            if mode == "global":
                spans.append((0, kv_len))
            elif mode.startswith("focus"):
                idx = int(mode.replace("focus", "")) - 1
                spans.append(tuple(seg_spans[idx]))
            positions = sorted({p for s, e in spans for p in range(max(0, s), min(e, kv_len))})
            visible = sorted({block_of(p, block_size) for p in positions})
            excluded = [b for b in written if b not in visible]
            seg_blocks = {
                f"seg{i+1}": blocks_of_span(s, min(e, kv_len), block_size)
                for i, (s, e) in enumerate(seg_spans)
            }
            # effective_read_tokens：**可见块集合**覆盖的 token 数（块粒度、截到 kv_len）；
            # 与 positions_count（语义 span 的 token 数）是两个不同口径，必须并列报告。
            eff = sum(
                max(0, min(kv_len, (b + 1) * block_size) - b * block_size)
                for b in visible
            )
            per_mode[mode] = {
                "positions_count": len(positions),
                "effective_read_tokens": eff,
                "effective_read_tokens_note": "按可见块覆盖的 token 数（块粒度，截到 kv_len）；不同于 positions_count",
                "visible_blocks": visible,
                "excluded_written_blocks": excluded,
                "segment_token_spans": seg_spans,
                "segment_blocks": seg_blocks,
                "tail_block_boundaries_token": [(b + 1) * block_size for b in written],
            }
        per_mode["global"]["excluded_written_blocks"] = []
        report["steps"][f"decode_step{step}_kv_len{kv_len}"] = {
            "written_blocks": written, "modes": per_mode,
            "note": "段边界为 token 粒度；canonical 有效尾块边界为块粒度，两者分列不混用。",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    checks = []
    step0 = report["steps"][f"decode_step0_kv_len{prompt_len}"]
    ln = int(exp["first_decode_kv_len"])
    step1 = report["steps"].get(f"decode_step1_kv_len{ln}")
    if step1:
        exp_eff = (cfg.get("expected_from_order") or {}).get("first_decode_effective_read_tokens") or {}
        for mode, want in exp_eff.items():
            got = step1["modes"][mode]["effective_read_tokens"]
            checks.append({"name": f"首 decode 有效尾长 == note（{mode}）", "ok": got == want,
                           "got": got, "want": want, "kv_len": int(exp["first_decode_kv_len"])})
        checks.append({"name": "local 可见块 == note", "ok": step1["modes"]["local"]["visible_blocks"] == exp["local_blocks"],
                       "got": step1["modes"]["local"]["visible_blocks"]})
        checks.append({"name": "focus1 可见块 == note", "ok": step1["modes"]["focus1"]["visible_blocks"] == exp["focus1_blocks"],
                       "got": step1["modes"]["focus1"]["visible_blocks"]})
        checks.append({"name": "local/focus1 排除的完整上下文块 == note",
                       "ok": sorted(set(step1["modes"]["local"]["excluded_written_blocks"])
                                    & set(step1["modes"]["focus1"]["excluded_written_blocks"]))
                             == exp["excluded_complete_context_blocks_by_local_and_focus1"],
                       "got": sorted(set(step1["modes"]["local"]["excluded_written_blocks"])
                                     & set(step1["modes"]["focus1"]["excluded_written_blocks"]))})
    report["checks"] = checks
    report["failed"] = [c["name"] for c in checks if not c["ok"]]
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"prompt_len": prompt_len, "failed": report["failed"],
                      "checks": [c["name"] for c in checks]}, ensure_ascii=False))
    return 0 if not report["failed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
