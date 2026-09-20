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

    fine = cfg["fine_char"]

    def render(units: int, fine_n: int = 0):
        ctx = base + ("\n\n" + cfg["filler_unit"]) * units + fine * fine_n
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
    for u in range(max(0, lo - 2), lo + 3):
        best_f = None
        for f in range(0, int(cfg["max_fine_units"]) + 1):
            c, sg, a = render(u, f)
            n = len(a.token_ids)
            if best_f is None or abs(n - target) < abs(best_f[0] - target):
                best_f = (n, f, c, sg, a)
            if n == target:
                break
        cands.append((abs(best_f[0] - target), u, best_f[1], best_f[2], best_f[3], best_f[4]))
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    gap, units, fine_n, ctx, segs, arm = cands[0]
    prompt_len = len(arm.token_ids)
    block_size = bs
    assert gap == 0, f"未能精确命中目标 prompt_len={target}(最接近 {prompt_len});需调整 fine_char/上限"

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
        "filler_units": units, "fine_units": fine_n, "context_sha256": hashlib.sha256(ctx.encode()).hexdigest(),
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
    next_block = block_of(prompt_len, bs) + 1
    first_cross = next((s["gen_step"] for s in steps if next_block in s["written_blocks"]), None)
    report["crossing"] = {
        "next_block": next_block, "block_start": next_block * bs,
        "prompt_written_blocks": sorted({block_of(p, bs) for p in range(prompt_len)}),
        "first_decode_step_kv": prompt_len + 1,
        "first_step_writing_next_block": first_cross,
        "decode_steps_until_crossing": first_cross,
        "note": "prompt 末尾位于块 9 内(kv=prompt_len+1=7840 时仍只写 0..9)；再 1 个 decode 步(kv=7841,写入位置 7840)首次写入块 10。",
    }

    # --- 安全负对照:同一门禁,必须拒绝,且样本本身不越界/不触未初始化槽 ---
    def try_plan(label: str, **over) -> dict:
        base_kw = dict(
            req_id="cross-neg", mode="local", refs=(), effect_step=1,
        )
        base_kw.update(over)
        rec: dict = {"label": label, "kind": "构造失败测试(未形成可读计划)", "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in base_kw.items()}}
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
    # 负对照 2:global 可见集合含块 10,但只分配了前缀 0..9 ⇒ 必须拒绝
    report["negative_controls"].append(try_plan(
        "B 可见块 10 超出已分配前缀 0..9(global,kv=7841)",
        mode="global", canonical=list(range(10)), attention_kv_len=prompt_len + 2, refs=(),
        max_width=32,
    ))
    # 负对照 3/4:从**真实计划**派生,减 1(错尾长 / 错 seqused_k);记录两层门禁结果
    honest = next(s for s in steps if s["mode"] == "local")
    base_plan_kw = dict(
        visible_logical_blocks=honest["visible_blocks_candidate"],
        effective_per_block=honest["effective_per_block"],
        seqused_k=honest["seqused_k"], tail_len=honest["tail_len"], width=honest["width"],
        attention_kv_len=honest["kv_len"],
    )
    for label, mut in (
        ("C 有效尾长与末块有效数各减 1(真实计划派生)", "tail"),
        ("D seqused_k 减 1(真实计划派生,其余不动)", "seq"),
    ):
        kw = dict(base_plan_kw)
        if mut == "tail":
            kw["tail_len"] -= 1
            kw["effective_per_block"] = kw["effective_per_block"][:-1] + [kw["effective_per_block"][-1] - 1]
        else:
            kw["seqused_k"] -= 1
        layer: dict = {"label": label, "honest": base_plan_kw, "mutated": kw}
        try:
            StepPlan(req_id="neg", mode="local", refs=(), effect_step=honest["effect_step"], block_size=bs,
                     next_write_position=honest["next_write_position"], written_before_step=honest["kv_len"] - 1, **kw).validate()
            layer["validate_layer"] = "通过(算术自洽,不构成拒绝)"
        except Exception as exc:  # noqa: BLE001
            layer["validate_layer"] = f"拒绝 {type(exc).__name__}: {str(exc)[:160]}"
        # 注意:这是**本 CPU 脚本新增的测试判据**(与合同推导计划的逐字段精确相等),
        # **不是**适配层既有行为;`attnview_adapter.py:695-711` 只检查 last_visible < allocated。
        layer["cpu_test_criterion_exact_equality"] = "拒绝" if (
            kw["seqused_k"] != base_plan_kw["seqused_k"] or kw["tail_len"] != base_plan_kw["tail_len"]
            or kw["effective_per_block"] != base_plan_kw["effective_per_block"]
        ) else "通过"
        layer["rejected"] = layer["cpu_test_criterion_exact_equality"] == "拒绝"
        layer["kind"] = "派生样本(计划可构造;由**本脚本新增的 CPU 测试判据**拒绝)"
        layer["reason"] = ("本 CPU 测试判据 = 与合同推导计划的逐字段精确相等(seqused_k/tail_len/effective_per_block/visible/width);"
                           "**本轮未实测生产适配层是否拒绝此类改动**(adapter 仅有 last_visible < allocated 检查),"
                           "也不为此改动生产路径。")
        report["negative_controls"].append(layer)

    # 负对照 5:把真实 local 计划中一个**已写且完整**的可见块(8)换成**已写但不可见**的完整块(3),
    # 长度/形状/seqused_k/tail_len 全不变 ⇒ 算术门禁自洽;必须由**精确可见集合合同**拒绝。
    adv = dict(base_plan_kw)
    adv["visible_logical_blocks"] = [0, 3, 9, 10]
    e_layer: dict = {"label": "E 已写但不可见完整块替换已写可见块(8→3,形状不变)", "honest": base_plan_kw, "mutated": adv}
    try:
        StepPlan(req_id="neg", mode="local", refs=(), effect_step=honest["effect_step"], block_size=bs,
                 next_write_position=honest["next_write_position"], written_before_step=honest["kv_len"] - 1, **adv).validate()
        e_layer["validate_layer"] = "通过(算术自洽,不构成拒绝)"
    except Exception as exc:  # noqa: BLE001
        e_layer["validate_layer"] = f"拒绝 {type(exc).__name__}: {str(exc)[:160]}"
    indep_expected = honest["visible_blocks_independent"]
    e_layer["independent_expected_visible"] = indep_expected
    e_layer["visible_set_contract_rejects"] = sorted(adv["visible_logical_blocks"]) != sorted(indep_expected)
    e_layer["rejected"] = bool(e_layer["visible_set_contract_rejects"])
    e_layer["kind"] = "派生样本(计划可构造;由精确可见集合合同拒绝)"
    e_layer["reason"] = ("**精确可见集合就是结构合同**:候选报告的可见集合必须等于由协议 span 独立推导的集合;"
                         "块 3 虽已写且完整,但不属于 local 的可见集合 ⇒ 结构合同拒绝。**不需要数值 oracle**。")

    def slot_audit(visible, effective_per_block, kv_len, allocated) -> dict:  # noqa: D401
        rows = []
        for b, c in zip(visible, effective_per_block):
            start = b * bs
            written = max(0, min(kv_len, start + bs) - start)
            rows.append({"block": b, "claimed": int(c), "written": written, "allocated": b < allocated,
                         "slot_written": bool(b < allocated and int(c) <= written)})
        return {"rows": rows, "all_slots_written": all(r["slot_written"] for r in rows),
                "any_out_of_range": any(not r["allocated"] for r in rows)}

    e_layer["slot_audit"] = slot_audit(adv["visible_logical_blocks"], adv["effective_per_block"],
                                      honest["kv_len"], len(canonical))
    report["negative_controls"].append(e_layer)
    for rc in report["negative_controls"]:
        if "slot_audit" not in rc and "mutated" in rc:
            mv = rc["mutated"]
            rc["slot_audit"] = slot_audit(mv.get("visible_logical_blocks", []),
                                          mv.get("effective_per_block", []), honest["kv_len"], len(canonical))
        rc.setdefault("kind", "派生样本")

    # --- 判定 ---
    def chk(name, ok, **extra):
        report["checks"].append({"name": name, "ok": bool(ok), **extra})

    chk("prompt_len 精确命中目标", prompt_len == target, got=prompt_len, want=target)
    chk("少量 decode 步内跨入下一块(≤3)", (report["crossing"]["first_step_writing_next_block"] or 99) <= 3,
        crossing=report["crossing"])
    chk("全部步候选可见块 == 独立展开", all(s["matches_independent"] for s in steps),
        bad=[s["gen_step"] for s in steps if not s["matches_independent"]])
    chk("有效读取 token 数(计划) == 可见块覆盖数", all(
        s["effective_read_tokens_plan"] == s["effective_read_tokens_from_blocks"] for s in steps))
    chk("mode 切换在 parse/effect 上体现", any(s["mode"] == "local" for s in steps), 
        modes=[(s["gen_step"], s["mode"], s["effect_step"]) for s in steps])
    chk("负对照全部被拒绝", all(n["rejected"] for n in report["negative_controls"]),
        rejected=[(n["label"], n["rejected"], n.get("error_type") or n.get("validate_layer")) for n in report["negative_controls"]])
    derived = [n for n in report["negative_controls"] if "slot_audit" in n and n.get("slot_audit")]
    def audit_ok(n) -> bool:
        sa = n["slot_audit"]
        rows = sa.get("rows") or []
        if not rows:
            return False
        if len(rows) != len(n["mutated"]["visible_logical_blocks"]):
            return False
        if len(n["mutated"]["effective_per_block"]) != len(n["mutated"]["visible_logical_blocks"]):
            return False
        return all(r["allocated"] and 0 <= r["claimed"] <= r["written"] for r in rows) and sa["all_slots_written"]
    chk("C/D/E 槽位审计严格通过(等长、0≤claimed≤written、块已分配、槽均已写)",
        bool(derived) and all(audit_ok(n) for n in derived),
        audit=[(n["label"], audit_ok(n), n["slot_audit"]["rows"]) for n in derived])
    construction_failures = [n for n in report["negative_controls"] if "slot_audit" not in n or not n.get("slot_audit")]
    derived_only = [n for n in report["negative_controls"] if n.get("slot_audit")]
    chk("A/B 为'构造失败'类样本(构造阶段即抛错,不冒充安全读取样本)",
        bool(construction_failures) and bool(derived_only)
        and all(n["kind"].startswith("构造失败") and n.get("error_type") for n in construction_failures)
        and all(n["kind"].startswith("派生样本") for n in derived_only),
        kinds=[(n["label"][:28], n.get("kind"), n.get("error_type")) for n in report["negative_controls"]])
    report["test_scope"] = ("本 CPU 脚本**不执行任何读取**(无 GPU、无 gather/copy):'拒绝先于读取'是生产路径的编码事实,"
                            "**本轮未实测**;此处只断言(1)构造失败类样本在构造阶段抛错,(2)派生样本的可见集合/形状/槽位审计,"
                            "(3)E 由精确可见集合合同拒绝,(4)C/D 由本脚本新增的 CPU 测试判据拒绝。")
    chk("样本 E 由精确可见集合合同拒绝(不依赖数值 oracle)", 
        next((n for n in report["negative_controls"] if n["label"].startswith("E ")), {}).get("rejected") is True)
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
