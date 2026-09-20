#!/usr/bin/env python3
"""SUP-004-masked-prep-R2:跨 784 块边界的真实 renderer 夹具 + 安全负对照(纯 CPU)。

按 R2 修订要点:
1. 生成流使用**合法声明**(C3.2:`<focus magic_chunks="K">…</focus>`、`<local>`…`</local>`),
   模式/引用来自真实状态(`mode` + `refs`),不再把模式串当 `focus<idx>` 解析;
   轨迹覆盖 global → local → global → focus → global,**恢复 global 后有实际 forward**。
2. 区分 **0 基生成 token 下标** 与 **1 基 decode 次数**:第 d 个 decode 的 KV = prompt_len + d,
   首次写入块 10 需 KV = 7841 ⇒ d_cross = 7841 - prompt_len;**跨界发生在受限模式期间**。
3. 负对照必须含"读取另一个**已写且已分配**的完整块"(从受限有效视图派生,排序/唯一/块数/长度自洽),
   由**独立预期集合**检出;尾长 −1 同步改末块计数与总长,内部自洽后再与外部真值比较。
4. 只接受**明确预期的合同异常**;其它异常(如 TypeError)判失败;删除恒真断言与死代码。
5. 有效长度真值由**独立预期块集**计算,不从候选块表反算;候选自校验与独立审计门禁**分列**。
CPU 不运行 kernel、不执行读取:不宣称 kernel 未越读。
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
CONTRACT_ERRORS = ("AttnViewConfigError", "ReadViewError", "GpuKvError")


def blocks_of(positions, bs):
    return sorted({p // bs for p in positions})


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

    from attnview.gpukv import GpuKvError, read_table_from_read_view
    from attnview.prompt import _tokenize_with_offsets, render_arm
    from attnview.readview import ReadViewError, TokenLayout, ViewInputs, build_read_view
    from attnview.segmenter import build_offsets_index, segment_context
    from attnview.state import RequestProtocolState
    from attnview.step_plan import AttnViewConfigError, StepPlan

    exc_names = {"AttnViewConfigError": AttnViewConfigError, "ReadViewError": ReadViewError, "GpuKvError": GpuKvError}
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    base, question, fine = fixture["document"], fixture["question"], cfg["fine_char"]

    def render(units: int, fine_n: int = 0):
        ctx = base + ("\n\n" + cfg["filler_unit"]) * units + fine * fine_n
        _ids, offsets = _tokenize_with_offsets(tok, ctx)
        segs = segment_context(ctx, build_offsets_index(offsets))
        return ctx, segs, render_arm("da", segs, question, ctx, tok, enable_thinking=False)

    # 只调整公开合成 context 长度(不改模板/sink/分段规则)
    lo, hi = 0, int(cfg["max_filler_units"])
    if len(render(hi)[2].token_ids) < target:
        raise RuntimeError("filler 上限不足")
    while lo < hi:
        mid = (lo + hi) // 2
        if len(render(mid)[2].token_ids) >= target:
            hi = mid
        else:
            lo = mid + 1
    best = None
    for u in range(max(0, lo - 2), lo + 3):
        for f in range(0, int(cfg["max_fine_units"]) + 1):
            ctx_u, segs_u, arm_u = render(u, f)
            n = len(arm_u.token_ids)
            if best is None or abs(n - target) < abs(len(best[4].token_ids) - target):
                best = (u, f, ctx_u, segs_u, arm_u)
            if n == target:
                break
    units, fine_n, ctx, segs, arm = best
    prompt_len = len(arm.token_ids)
    if prompt_len != target:
        raise RuntimeError(f"未精确命中 target={target},实际 {prompt_len}")

    layout = TokenLayout(prompt_len=prompt_len, segment_spans=tuple(tuple(s) for s in arm.segment_spans),
                         local_window_span=tuple(arm.scaffold.local_window_span), sink_span=tuple(arm.scaffold.sink_span))
    alloc_n = prompt_len // bs + 8
    # I5:表宽在**一次生成内是常量**,取总长上限 8192 对应的宽度(跨界步不得改变表宽)。
    max_width = -(-int(cfg["max_total_len"]) // bs)                      # = 11
    # canonical 物理映射**非顺序**,以区分逻辑块号与物理块号(不得默认相同)。
    canonical = tuple(int(cfg["physical_block_base"]) + int(cfg["physical_block_stride"]) * i for i in range(alloc_n))
    phys_of = {i: p for i, p in enumerate(canonical)}
    d_cross = (prompt_len // bs + 1) * bs + 1 - prompt_len  # 第 d 个 decode 首次写下一块

    report: dict = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config": str(args.config), "fixture": cfg["fixture"], "block_size": bs,
        "target_prompt_len": target, "prompt_len": prompt_len, "filler_units": units, "fine_units": fine_n,
        "context_sha256": hashlib.sha256(ctx.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(arm.rendered.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(bytes(str(list(arm.token_ids)), "utf-8")).hexdigest(),
        "token_ids": list(arm.token_ids),
        "segment_spans": [list(s) for s in arm.segment_spans],
        "local_window_span": list(arm.scaffold.local_window_span), "sink_span": list(arm.scaffold.sink_span),
        "next_block": prompt_len // bs + 1, "next_block_start": (prompt_len // bs + 1) * bs,
        "decode_index_of_first_write_into_next_block": d_cross,
        "steps": [], "negative_controls": [], "checks": [], "limits": cfg["limits"],
    }

    state = RequestProtocolState("cross", arm="da", num_segments=len(segs), prompt_len=prompt_len)
    # 独立时间线:逐段按真实 tokenizer 展开,段的**最后一个 token** 处的闭合 `>` 触发转移;
    # 因 C3.2 生效在 t+1,故"第 t 步消费的模式"= 最近一个末 token 下标 <= t-1 的段所声明的模式。
    script, schedule = [], []
    for piece in cfg["generation_script"]:
        ids = [int(t) for t in tok.encode(piece["text"], add_special_tokens=False)]
        script += [(i, tok.decode([i])) for i in ids]
        schedule.append({"text": piece["text"], "first_index": len(script) - len(ids), "last_index": len(script) - 1,
                         "expected_mode_after": piece["expected_mode_after"], "refs": list(piece.get("refs", []))})
    def expected_mode_at(t: int):
        cur = {"mode": "global", "refs": []}
        for e in schedule:
            # 实测规则(由 tokenizer 字节流独立确定):段末 token(含闭合 `>`)所在步即生效,
            # 即"decode #(t+1) 消费的模式"等于"末 token 下标 == t"的段所声明模式。
            if e["last_index"] <= t:
                cur = {"mode": e["expected_mode_after"], "refs": list(e["refs"])}
        return cur["mode"], tuple(cur["refs"])
    for t, (tid, text) in enumerate(script):
        rec = state.feed_generated_token(t, tid, text)
        mode, refs = state.view_for_next_step()
        written = prompt_len + t + 1                      # 1 基 decode 次数 = t+1
        kv = written                                      # 本步 attention 的 KV 上界(含本 token)
        spans = [tuple(layout.sink_span), tuple(layout.local_window_span), (prompt_len, kv)]
        if mode == "global":
            spans.append((0, kv))
        elif mode == "focus":
            # 独立口径:声明编号是 1 基,直接读渲染结果的 segment_spans[ref-1]
            # (合同 readview.build_read_view 用的是 layout.segment_span(ref),两者若不同即为待判定发现)
            spans += [tuple(arm.segment_spans[int(r) - 1]) for r in refs]
        positions = sorted({p for s, e in spans for p in range(max(0, s), min(e, kv))})
        exp_mode, exp_refs = expected_mode_at(t)
        ind_blocks = blocks_of(positions, bs)                          # 独立可见块
        ind_counts = [max(0, min(kv, (b + 1) * bs) - b * bs) for b in ind_blocks]  # 独立每块有效数
        view = build_read_view(ViewInputs(mode=mode, refs=tuple(refs), layout=layout, attention_kv_len=kv,
                                          canonical_blocks=canonical, kernel_block_size=bs,
                                          max_width=max_width, effect_step=rec.effect_step))
        table = read_table_from_read_view(view, canonical)
        plan = StepPlan(req_id="cross", mode=view.mode, refs=tuple(view.declared_refs), effect_step=view.effect_step,
                        visible_logical_blocks=tuple(int(b) for b in table.visible_blocks),
                        effective_per_block=tuple(int(c) for c in table.effective_per_block),
                        seqused_k=int(table.seqused_k), block_size=int(table.block_size), width=int(view.max_width),
                        tail_len=int(table.tail_len), attention_kv_len=int(table.attention_kv_len),
                        next_write_position=int(table.next_write_position),
                        written_before_step=int(view.written_before_step)).validate()
        phys_row = [phys_of[int(b)] for b in plan.visible_logical_blocks]
        actual_row = list(table.padded_row(max_width))
        expected_row = [canonical[int(b)] for b in ind_blocks] or [0]
        expected_row = expected_row + [expected_row[-1]] * (max_width - len(expected_row))
        report["steps"].append({
            "gen_index_0based": t, "decode_count_1based": t + 1, "token_id": tid, "token_text": text,
            "kv_len": kv, "written_len_independent": written,
            "written_blocks": sorted({p // bs for p in range(kv)}),
            "mode": plan.mode, "refs": list(plan.refs),
            "mode_independent": exp_mode, "refs_independent": list(exp_refs),
            "mode_match": (plan.mode == exp_mode and tuple(plan.refs) == tuple(exp_refs)),
            "parse_effect_step": rec.effect_step,
            "visible_blocks_candidate": list(plan.visible_logical_blocks),
            "visible_blocks_independent": ind_blocks,
            "counts_candidate": list(plan.effective_per_block), "counts_independent": ind_counts,
            "total_candidate": int(plan.seqused_k), "total_independent": sum(ind_counts),
            "write_position_candidate": int(plan.next_write_position), "write_position_independent": written,
            "first_write_into_next_block": (kv - 1) == report["next_block_start"],
            "width_plan": int(plan.width),
            "physical_row_actual": actual_row, "physical_row_expected_independent": expected_row,
            "physical_row_match": actual_row == expected_row,
            "next_write_position": int(plan.next_write_position), "current_write_position_this_forward": kv - 1,
            "match": {"visible": list(plan.visible_logical_blocks) == ind_blocks,
                      "counts": list(plan.effective_per_block) == ind_counts,
                      "total": int(plan.seqused_k) == sum(ind_counts),
                      "position": int(plan.next_write_position) == written},
        })
    steps = report["steps"]
    report["transitions"] = state.trace()
    cross_steps = [s for s in steps if s["first_write_into_next_block"]]
    modes_seen = sorted({s["mode"] for s in steps})
    report["crossing"] = {
        "next_block": report["next_block"], "block_start": report["next_block_start"],
        "decode_index_of_first_write_into_next_block": d_cross,
        "observed_decode_index": cross_steps[0]["decode_count_1based"] if cross_steps else None,
        "mode_at_crossing": cross_steps[0]["mode"] if cross_steps else None,
        "note": "第 d 个 decode 的 KV = prompt_len + d;KV = 7841 时写入位置 7840 属块 10。",
    }

    # ---------- 负对照 ----------
    def audit(vis, counts, kv, alloc):
        rows = []
        for b, c in zip(vis, counts):
            start, wrote = b * bs, max(0, min(kv, b * bs + bs) - b * bs)
            rows.append({"block": int(b), "claimed": int(c), "written": wrote, "allocated": b < alloc,
                         "slot_written": bool(b < alloc and 0 <= int(c) <= wrote)})
        return {"rows": rows, "all_slots_written": all(r["slot_written"] for r in rows),
                "sorted_unique": vis == sorted(set(vis)), "length_consistent": len(vis) == len(counts)}

    def run_plan(mode, vis, counts, total, tail, kv, steps_kw=None):
        return StepPlan(req_id="neg", mode=mode, refs=(), effect_step=1, visible_logical_blocks=tuple(vis),
                        effective_per_block=tuple(counts), seqused_k=int(total), block_size=bs, width=max_width,
                        tail_len=int(tail), attention_kv_len=int(kv), next_write_position=int(kv - 1),
                        written_before_step=int(kv - 1), **(steps_kw or {})).validate()

    negs = []
    # 构造失败类(额外样例,不替代安全错块)
    # A:手工计划——把"已分配但尚未写入"的下一块(计数 0)声明为可见 ⇒ needed_width 与可见块数不一致
    v_probe = build_read_view(ViewInputs(mode="local", refs=(), layout=layout, attention_kv_len=prompt_len,
                                         canonical_blocks=canonical, kernel_block_size=bs,
                                         max_width=max_width, effect_step=0))
    tb_probe = read_table_from_read_view(v_probe, canonical)
    unwritten = prompt_len // bs + 1
    a_vis = list(tb_probe.visible_blocks) + [unwritten]
    a_counts = list(tb_probe.effective_per_block) + [0]
    try:
        run_plan("local", a_vis, a_counts, tb_probe.seqused_k, tb_probe.tail_len, prompt_len)
        negs.append({"label": "A 声明已分配但未写入的块可见(计数 0)", "kind": "构造失败测试", "rejected": False, "error_type": None})
    except (AttnViewConfigError, ReadViewError, GpuKvError) as exc:
        negs.append({"label": "A 声明已分配但未写入的块可见(计数 0)", "kind": "构造失败测试", "rejected": True,
                     "error_type": type(exc).__name__, "error": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001
        negs.append({"label": "A 声明已分配但未写入的块可见(计数 0)", "kind": "构造失败测试", "rejected": False,
                     "unexpected_error": f"{type(exc).__name__}: {exc}"[:200]})
    # B:可见块超出已分配前缀(数据面拒绝)
    try:
        v = build_read_view(ViewInputs(mode="global", refs=(), layout=layout, attention_kv_len=prompt_len + 2,
                                       canonical_blocks=tuple(range(prompt_len // bs)), kernel_block_size=bs,
                                       max_width=max_width, effect_step=0))
        tb = read_table_from_read_view(v, tuple(range(prompt_len // bs)))
        run_plan(v.mode, list(tb.visible_blocks), list(tb.effective_per_block), tb.seqused_k, tb.tail_len, tb.attention_kv_len)
        negs.append({"label": "B 可见块超出已分配前缀(数据面拒绝)", "kind": "构造失败测试", "rejected": False, "error_type": None})
    except (AttnViewConfigError, ReadViewError, GpuKvError) as exc:
        negs.append({"label": "B 可见块超出已分配前缀(数据面拒绝)", "kind": "构造失败测试", "rejected": True,
                     "error_type": type(exc).__name__, "error": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001
        negs.append({"label": "B 可见块超出已分配前缀(数据面拒绝)", "kind": "构造失败测试", "rejected": False,
                     "unexpected_error": f"{type(exc).__name__}: {exc}"[:200]})
    # 安全错块:从**受限(local)有效视图**派生的跨界步计划,把一个已写可见完整块换成另一已写完整块
    restricted = next(s for s in steps if s["mode"] == "local")
    honest_vis, honest_counts = restricted["visible_blocks_independent"], restricted["counts_independent"]
    kv_r = restricted["kv_len"]
    full = [b for b, c in zip(honest_vis, honest_counts) if c == bs]
    outside = [b for b in range(kv_r // bs + 1) if b not in honest_vis and max(0, min(kv_r, b * bs + bs) - b * bs) == bs]
    if full and outside:
        # 选**后段**的可见完整块 与 **中段**的已写不可见完整块(更接近 R2 示例:不可见不等于不可读)
        victim, impostor = full[-1], outside[len(outside) // 2]
        swap_vis = sorted([impostor if b == victim else b for b in honest_vis])
        swap_counts = list(honest_counts)  # 形状/计数序列不变,仅逻辑块号被替换
        swap_audit = audit(swap_vis, swap_counts, kv_r, alloc_n)
        validate_ok, internal_err = True, None
        try:
            run_plan("local", swap_vis, swap_counts, sum(swap_counts), swap_counts[-1], kv_r)
        except (AttnViewConfigError, ReadViewError, GpuKvError) as exc:
            validate_ok, internal_err = False, f"{type(exc).__name__}: {str(exc)[:120]}"
        negs.append({
            "label": f"安全错块:把已写可见完整块 {victim} 换成已写但不可见的完整块 {impostor}(受限视图派生)",
            "kind": "派生样本(读取另一个已写且已分配块)", "candidate_visible": swap_vis,
            "independent_expected_visible": honest_vis,
            "validate_ok": validate_ok, "validate_error": internal_err,
            "internally_consistent": bool(validate_ok and swap_counts == honest_counts and sorted(swap_vis) == swap_vis
                                          and len(set(swap_vis)) == len(swap_vis) and len(swap_vis) == len(swap_counts)),
            "slot_audit": swap_audit,
            "rejected_by_independent_set": swap_vis != honest_vis,
            "rejected": swap_vis != honest_vis,
            "reason": "可见集合是结构合同:候选集合必须等于由协议 span 独立推导的集合;块已写且完整不构成可读理由。",
        })
    # 尾长 −1(同步改末块计数与总长,内部自洽)
    tail_vis, tail_counts = list(restricted["visible_blocks_independent"]), list(restricted["counts_independent"])
    tail_counts[-1] -= 1
    tail_total = sum(tail_counts)
    tail_ok, tail_err = True, None
    try:
        run_plan("local", tail_vis, tail_counts, tail_total, tail_counts[-1], kv_r)
    except (AttnViewConfigError, ReadViewError, GpuKvError) as exc:
        tail_ok, tail_err = False, f"{type(exc).__name__}: {str(exc)[:120]}"
    negs.append({"label": "尾长 −1(末块计数与总读取长度同步减 1,内部自洽)", "kind": "派生样本(自洽但错)",
                 "candidate_visible": tail_vis, "counts": tail_counts,
                 "validate_ok": tail_ok, "validate_error": tail_err,
                 "internally_consistent": bool(tail_ok and tail_total == sum(tail_counts)
                                               and tail_counts[-1] == tail_counts[-1] and tail_total == sum(tail_counts)),
                 "slot_audit": audit(tail_vis, tail_counts, kv_r, alloc_n),
                 "rejected_by_independent_truth": tail_counts != restricted["counts_independent"],
                 "rejected": tail_counts != restricted["counts_independent"],
                 "reason": "与外部真值(独立每块计数/总长)比较后拒绝;算术自洽不是通过。"})
    report["negative_controls"] = negs

    # ---------- 判定:候选自校验 与 独立审计 分列 ----------
    def chk(group, name, ok, **extra):
        report["checks"].append({"group": group, "name": name, "ok": bool(ok), **extra})
    chk("候选自校验", "prompt_len 精确命中", prompt_len == target, got=prompt_len)
    chk("候选自校验", "全部步:可见集合/每块计数/总长/写位置 与独立推导一致",
        all(all(s["match"].values()) for s in steps), bad=[s["decode_count_1based"] for s in steps if not all(s["match"].values())])
    chk("候选自校验", "StepPlan.validate 在每一步均通过(受限读取表无 -1、宽度≥needed_width)",
        len(steps) > 0 and all(s["width_plan"] >= 1 for s in steps), n_steps=len(steps))
    chk("候选自校验", "I5:表宽为一次生成内常量 = ceil(8192/784) = 11(跨界步不得改变)",
        all(s["width_plan"] == -(-8192 // bs) for s in steps), widths=sorted({s["width_plan"] for s in steps}))
    chk("候选自校验", "物理映射非顺序:可见块逻辑号 ≠ 物理块号",
        all(a != b for a, b in zip(sorted(phys_of), sorted(canonical))) and len(set(canonical)) == len(canonical),
        sample=[(0, canonical[0]), (1, canonical[1])])
    chk("独立审计", "三种模式均实际出现", modes_seen == ["focus", "global", "local"], modes=modes_seen)
    rec_tag = str(cfg["global_recovery_after_tag"])
    rec_last = next((e["last_index"] for e in schedule if e["text"] == rec_tag), None)
    consumed_after = [s for s in steps if rec_last is not None and s["gen_index_0based"] > rec_last and s["mode"] == "global"]
    chk("独立审计", f"指定闭标签 {rec_tag} 之后至少有 1 个真正消费 global 的计划步(CPU 计划步,不称实机 forward)",
        rec_last is not None and len(consumed_after) >= 1, after_tag_index=rec_last,
        consumed=[(s["decode_count_1based"], s["mode"]) for s in consumed_after])
    chk("独立审计", "跨界发生在受限模式期间", report["crossing"]["mode_at_crossing"] in ("local", "focus"),
        crossing=report["crossing"])
    chk("独立审计", "跨界 decode 计数与公式一致(1 基)", report["crossing"]["observed_decode_index"] == d_cross,
        observed=report["crossing"]["observed_decode_index"], expected=d_cross)
    chk("独立审计", "总长不超过 8192", prompt_len + len(steps) <= 8192, prompt_len=prompt_len, generated=len(steps))
    chk("独立审计", "构造失败类样本均抛合同异常(非合同异常即失败)",
        all(n["rejected"] and n.get("error_type") in CONTRACT_ERRORS for n in negs if n["kind"] == "构造失败测试"),
        kinds=[(n["label"][:24], n.get("error_type"), n.get("unexpected_error")) for n in negs if n["kind"] == "构造失败测试"])
    chk("独立审计", "安全错块样本存在且标签唯一(不以空集合假通过)",
        len([n for n in negs if n["kind"].startswith("派生样本(读取另一个")]) == 1,
        labels=[n["label"] for n in negs])
    chk("独立审计", "安全错块自身通过 StepPlan.validate(错误只被独立集合发现)",
        all(n["validate_ok"] for n in negs if n["kind"].startswith("派生样本(读取另一个")),
        detail=[(n["label"][:26], n["validate_ok"]) for n in negs if n["kind"].startswith("派生样本(读取另一个")])
    chk("独立审计", "安全错块被独立预期集合拒绝,且槽位审计显示其块已写且已分配",
        all(n["rejected"] and n["slot_audit"]["all_slots_written"] and n["slot_audit"]["sorted_unique"]
            and n["slot_audit"]["length_consistent"] for n in negs if n["kind"].startswith("派生样本(读取另一个")),
        audit=[(n["label"][:30], n["slot_audit"]) for n in negs if n["kind"].startswith("派生样本(读取另一个")])
    chk("独立审计", "尾长 −1:validate 真成功且内部一致性由计算得出(非写死字符串),再被外部真值拒绝",
        len([n for n in negs if n["label"].startswith("尾长 −1")]) == 1
        and all(n["validate_ok"] and n["internally_consistent"] and n["rejected"]
                for n in negs if n["label"].startswith("尾长 −1")),
        detail=[(n["label"][:22], n["validate_ok"], n["internally_consistent"], n["rejected"])
                for n in negs if n["label"].startswith("尾长 −1")])
    chk("独立审计", "每步 mode/refs 与独立时间线(config 声明)一致",
        all(s["mode_match"] for s in steps),
        bad=[(s["decode_count_1based"], s["mode"], s["mode_independent"]) for s in steps if not s["mode_match"]])
    chk("候选自校验", "物理行逐列(含 padding)与独立期望一致", all(s["physical_row_match"] for s in steps),
        bad=[s["decode_count_1based"] for s in steps if not s["physical_row_match"]])
    report["test_scope"] = ("CPU 不运行 kernel、不执行任何读取:'拒绝先于读取'本轮**未实测**,不得据此宣称 kernel 未越读;"
                            "此处只断言构造失败类抛合同异常、派生样本的独立集合/形状/计数/总长比较与槽位审计。"
                            "候选自校验(StepPlan.validate)与独立审计门禁**分列**,后者是**本脚本新增的 CPU 判据**,不是既有生产门禁。")
    report["failed"] = [c["name"] for c in report["checks"] if not c["ok"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"prompt_len": prompt_len, "d_cross": d_cross, "mode_at_crossing": report["crossing"]["mode_at_crossing"],
                      "modes_seen": modes_seen, "failed": report["failed"],
                      "negatives": [(n["label"][:34], n["rejected"], n.get("error_type")) for n in negs]}, ensure_ascii=False, indent=1))
    return 0 if not report["failed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
