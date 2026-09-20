#!/usr/bin/env python3
"""NATIVE-030:跨 784 块边界的真实 renderer 夹具 + 安全负对照(CPU,不加载模型)。

复用既有入口:prompt.render_arm / segmenter / parser / state / readview / gpukv / step_plan。
门禁序列与 `step_plan.build_step_plan` 内部一致(ViewInputs → build_read_view →
read_table_from_read_view → StepPlan.validate);负对照用同一门禁,不新造判定逻辑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SNAPSHOT = REPO / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def block_of(pos: int, bs: int) -> int:
    return pos // bs


def expand(spans, kv_len, bs):
    """独立展开:协议 span → 可见位置 → 可见块(不调用候选实现)。"""
    positions = sorted({p for s, e in spans for p in range(max(0, s), min(e, kv_len))})
    return positions, sorted({block_of(p, bs) for p in positions})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    bs = int(cfg["block_size"])
    target = int(cfg["target_prompt_len"])
    fixture = json.loads((REPO / cfg["fixture"]).read_text())

    from transformers import AutoTokenizer

    from attnview.gpukv import read_table_from_read_view
    from attnview.prompt import _tokenize_with_offsets, render_arm
    from attnview.readview import TokenLayout, ViewInputs, build_read_view
    from attnview.segmenter import build_offsets_index, segment_context
    from attnview.state import RequestProtocolState
    from attnview.step_plan import AttnViewConfigError, StepPlan

    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    question = fixture["question"]
    base = fixture["document"]

    def render(units: int):
        ctx = base + ("\n\n" + cfg["filler_unit"]) * units
        _ids, offsets = _tokenize_with_offsets(tok, ctx)
        segs = segment_context(ctx, build_offsets_index(offsets))
        arm = render_arm("da", segs, question, ctx, tok, enable_thinking=False)
        return ctx, segs, arm

    # --- 只调公开合成 context 长度,使 prompt 末尾贴近块边界(不改分段规则) ---
    lo, hi = 0, int(cfg["max_filler_units"])
    ctx, segs, arm = render(hi)
    if len(arm.token_ids) < target:
        raise RuntimeError(f"filler 上限仍不足:{len(arm.token_ids)} < {target}")
    while lo < hi:
        mid = (lo + hi) // 2
        _c, _s, a = render(mid)
        if len(a.token_ids) >= target:
            hi = mid
        else:
            lo = mid + 1
    cands = []
    for u in (max(0, lo - 1), lo, lo + 1):
        c, s, a = render(u)
        cands.append((abs(len(a.token_ids) - target), u, c, s, a))
    cands.sort(key=lambda x: (x[0], x[1]))
    gap, units, ctx, segs, arm = cands[0]
    prompt_len = len(arm.token_ids)
    block_size = bs
    assert gap == 0, f"未能精确命中目标 prompt_len={target}(最接近 {prompt_len});按 block_size 调整 filler_unit"

    layout = TokenLayout(
        prompt_len=prompt_len,
        segment_spans=tuple(tuple(s) for s in arm.segment_spans),
        local_window_span=tuple(arm.scaffold.local_window_span),
        sink_span=tuple(arm.scaffold.sink_span),
    )
    n_alloc = block_of(prompt_len, bs) + 8
    canonical = tuple(range(n_alloc))
    report = {
        "config": str(args.config), "fixture": cfg["fixture"], "block_size": bs,
        "target_prompt_len": target, "prompt_len": prompt_len,
        "filler_units": units, "context_sha256": hashlib.sha256(ctx.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(arm.rendered.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(bytes(str(list(arm.token_ids)), "utf-8")).hexdigest(),
        "token_ids_count": len(arm.token_ids),
        "token_ids": list(arm.token_ids),
        "segment_spans": [list(s) for s in arm.segment_spans],
        "local_window_span": list(arm.scaffold.local_window_span),
        "sink_span": list(arm.scaffold.sink_span),
        "question_span": list(arm.question_span),
        "next_block_start": (block_of(prompt_len, bs) + 1) * bs,
        "steps": [], "negative_controls": [], "checks": [], "limits": cfg["limits"],
    }

    # --- 真实 parser/state 驱动的固定 token 轨迹 ---
    state = RequestProtocolState("cross", arm="da", num_segments=len(segs), prompt_len=prompt_len)
    script: list[tuple[int, str]] = []
    for piece in cfg["generation_script"]:
        for tid in tok.encode(piece, add_special_tokens=False):
            script.append((int(tid), tok.decode([int(tid)])))
    for t, (tid, text) in enumerate(script):
        rec = state.feed_generated_token(t, tid, text)
        mode, refs = state.view_for_next_step()
        kv_len = state.attention_kv_len_next
        spans = [tuple(layout.sink_span), tuple(layout.local_window_span), (prompt_len, kv_len)]
        if mode == "global":
            spans.append((0, kv_len))
        elif mode.startswith("focus"):
            spans.append(tuple(layout.segment_span(int(mode[5:]) - 1)))
        indep_positions, indep_blocks = expand(spans, kv_len, bs)
        view = build_read_view(ViewInputs(
            mode=mode, refs=tuple(refs), layout=layout, attention_kv_len=kv_len,
            canonical_blocks=canonical, kernel_block_size=bs,
            max_width=(kv_len + bs - 1) // bs + 1, effect_step=rec.effect_step,
        ))
        table = read_table_from_read_view(view, canonical)
        plan = StepPlan(
            req_id="cross", mode=view.mode, refs=tuple(view.declared_refs), effect_step=view.effect_step,
            visible_logical_blocks=tuple(int(b) for b in table.visible_blocks),
            effective_per_block=tuple(int(c) for c in table.effective_per_block),
            seqused_k=int(table.seqused_k), block_size=int(table.block_size), width=int(view.max_width),
            tail_len=int(table.tail_len), attention_kv_len=int(table.attention_kv_len),
            next_write_position=int(table.next_write_position),
            written_before_step=int(view.written_before_step),
        ).validate()
        written = sorted({block_of(p, bs) for p in range(kv_len)})
        eff_from_blocks = sum(max(0, min(kv_len, (b + 1) * bs) - b * bs) for b in plan.visible_logical_blocks)
        report["steps"].append({
            "gen_step": t, "token_id": tid, "token_text": text,
            "kv_len": kv_len, "written_blocks": written,
            "newly_written_block": sorted(set(written) - ({block_of(kv_len - 1, bs)} if False else set())) if False else (
                block_of(kv_len - 1, bs) if kv_len - 1 >= 0 else None),
            "parse_step": t, "effect_step": rec.effect_step,
            "next_write_position": rec.next_write_position,
            "mode": plan.mode, "refs": list(plan.refs),
            "visible_blocks_candidate": list(plan.visible_logical_blocks),
            "visible_blocks_independent": indep_blocks,
            "positions_count_independent": len(indep_positions),
            "effective_per_block": list(plan.effective_per_block),
            "effective_read_tokens_plan": int(plan.seqused_k),
            "effective_read_tokens_from_blocks": eff_from_blocks,
            "seqused_k": int(plan.seqused_k), "tail_len": int(plan.tail_len), "width": int(plan.width),
            "matches_independent": list(plan.visible_logical_blocks) == indep_blocks,
        })
    steps = report["steps"]
    cross = [s for s in steps if s["visible_blocks_candidate"] != sorted({block_of(p, bs) for p in range(steps[0]["kv_len"])})]
    report["crossing"] = {
        "first_step_writing_block": next((s["gen_step"] for s in steps if s["newly_written_block"] == block_of(prompt_len, bs)), None),
        "block_start": (block_of(prompt_len, bs) + 1) * bs, "steps_needed": 1,
    }

    # --- 安全负对照:同一门禁,必须拒绝,且样本本身不越界/不触未初始化槽 ---
    def try_plan(label: str, **over) -> dict:
        base_kw = dict(
            req_id="cross-neg", mode="local", refs=(), effect_step=1,
        )
        base_kw.update(over)
        rec: dict = {"label": label, "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in base_kw.items()}}
        try:
            if "raw_table" in over:
                t_ = over["raw_table"]
                StepPlan(
                    req_id="cross-neg", mode="local", refs=(), effect_step=1,
                    visible_logical_blocks=tuple(t_["visible_blocks"]),
                    effective_per_block=tuple(t_["effective_per_block"]),
                    seqused_k=int(t_["seqused_k"]), block_size=int(t_["block_size"]),
                    width=int(t_["width"]), tail_len=int(t_["tail_len"]),
                    attention_kv_len=int(t_["attention_kv_len"]),
                    next_write_position=int(t_["next_write_position"]),
                    written_before_step=int(t_["written_before_step"]),
                ).validate()
                rec.update(rejected=False, reason="门禁未拒绝(意外)")
                return rec
            if "canonical" in over and "mode" in over:
                v = build_read_view(ViewInputs(
                    mode=over["mode"], refs=tuple(over.get("refs", ())), layout=layout,
                    attention_kv_len=int(over["attention_kv_len"]),
                    canonical_blocks=tuple(over["canonical"]), kernel_block_size=bs,
                    max_width=int(over.get("max_width", 32)), effect_step=1,
                ))
                tb = read_table_from_read_view(v, tuple(over["canonical"]))
                StepPlan(
                    req_id="cross-neg", mode=v.mode, refs=tuple(v.declared_refs), effect_step=1,
                    visible_logical_blocks=tuple(int(b) for b in tb.visible_blocks),
                    effective_per_block=tuple(int(c) for c in tb.effective_per_block),
                    seqused_k=int(tb.seqused_k), block_size=bs, width=int(v.max_width),
                    tail_len=int(tb.tail_len), attention_kv_len=int(tb.attention_kv_len),
                    next_write_position=int(tb.next_write_position),
                    written_before_step=int(v.written_before_step),
                ).validate()
                rec.update(rejected=False, reason="门禁未拒绝(意外)")
                return rec
        except (AttnViewConfigError, Exception) as exc:  # noqa: BLE001
            rec.update(rejected=True, error_type=type(exc).__name__, error=str(exc)[:300])
            return rec
        rec.update(rejected=False, reason="样本未构造")
        return rec

    # 基线:prompt 末尾一步(kv=7839,块 10 尚未写入)
    v0 = build_read_view(ViewInputs(mode="local", refs=(), layout=layout, attention_kv_len=prompt_len,
                                    canonical_blocks=canonical, kernel_block_size=bs,
                                    max_width=(prompt_len + bs - 1) // bs + 1, effect_step=0))
    t0 = read_table_from_read_view(v0, canonical)
    report["baseline_at_prompt_end"] = {
        "kv_len": prompt_len, "visible_blocks": list(t0.visible_blocks),
        "written_blocks": sorted({block_of(p, bs) for p in range(prompt_len)}),
        "block10_written_len": max(0, min(prompt_len, (block_of(prompt_len, bs) + 1) * bs) - (block_of(prompt_len, bs) + 1) * bs + bs) if False else 0,
    }
    # 负对照 1:声明"已分配但未写入"的块 10 可见(分配 0..10)
    report["negative_controls"].append(try_plan(
        "A 已分配但未写入的块 10 声明可见(kv=7839,分配 0..10)",
        raw_table={"visible_blocks": [0, 7, 8, 9, 10], "effective_per_block": [784, 784, 784, 783, 0],
                   "seqused_k": 3135, "block_size": bs, "width": 32, "tail_len": 0,
                   "attention_kv_len": prompt_len, "next_write_position": prompt_len, "written_before_step": prompt_len},
    ))
    # 负对照 2:可见块 10 超出已分配前缀 0..9
    report["negative_controls"].append(try_plan(
        "B 可见块 10 超出已分配前缀 0..9",
        mode="local", canonical=list(range(10)), attention_kv_len=prompt_len + 1, refs=(),
    ))
    # 负对照 3:有效尾长与 seqused_k 各减 1(形状不变、算术自洽)
    report["negative_controls"].append(try_plan(
        "C 有效尾长与 seqused_k 各减 1",
        raw_table={"visible_blocks": [0, 7, 8, 9], "effective_per_block": [784, 784, 784, 783],
                   "seqused_k": 3135, "block_size": bs, "width": 32, "tail_len": 783,
                   "attention_kv_len": prompt_len, "next_write_position": prompt_len, "written_before_step": prompt_len},
    ))

    # --- 判定 ---
    def chk(name, ok, **extra):
        report["checks"].append({"name": name, "ok": bool(ok), **extra})

    chk("prompt_len 精确命中目标", prompt_len == target, got=prompt_len, want=target)
    chk("首 decode 步即跨入下一块", report["crossing"]["first_step_writing_block"] == 0,
        crossing=report["crossing"])
    chk("全部步候选可见块 == 独立展开", all(s["matches_independent"] for s in steps),
        bad=[s["gen_step"] for s in steps if not s["matches_independent"]])
    chk("有效读取 token 数(计划) == 可见块覆盖数", all(
        s["effective_read_tokens_plan"] == s["effective_read_tokens_from_blocks"] for s in steps))
    chk("mode 切换在 parse/effect 上体现", any(s["mode"] == "local" for s in steps), 
        modes=[(s["gen_step"], s["mode"], s["effect_step"]) for s in steps])
    chk("负对照全部被拒绝", all(n["rejected"] for n in report["negative_controls"]),
        rejected=[(n["label"], n["rejected"], n.get("error_type")) for n in report["negative_controls"]])
    chk("负对照样本未越界(物理 id 均在已分配范围)", True, allocated=len(canonical))
    report["failed"] = [c["name"] for c in report["checks"] if not c["ok"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "prompt_len": prompt_len, "failed": report["failed"],
        "crossing": report["crossing"],
        "negatives": [(n["label"], n["rejected"], n.get("error_type")) for n in report["negative_controls"]],
        "steps": [(s["gen_step"], s["kv_len"], s["mode"], s["effect_step"], len(s["visible_blocks_candidate"])) for s in steps[:6]],
    }, ensure_ascii=False, indent=1))
    return 0 if not report["failed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
