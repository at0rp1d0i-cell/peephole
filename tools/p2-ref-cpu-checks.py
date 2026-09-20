#!/usr/bin/env python3
"""固定 pin 真实接口下的参考桥接 CPU 检查(不加载模型、不跑 GPU)。

真实接口:原生 `kv_cache=[num_blocks,num_kv_heads,block_size,2*head_size]`
(`kv_cache.transpose(1,2).split(head_size,-1)` 解包)、真实 `FlashAttentionMetadata`
(`block_table` 二维张量、`seq_lens` 等,无 mode/refs/spans)、真实 forward 签名
`(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)`。
模式/几何/独立时间线由 `TimelineBridge` 桥接(源自 config 声明,非候选状态)。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attnview.reference_bridge import (  # noqa: E402
    TimelineBridge,
    perform_reference_attention_native,
    unpack_native_kv,
)
from attnview.reference_dense import (  # noqa: E402
    ReferenceError,
    TestOnlyReferenceSwitch,
    independent_visible_positions,
)
from attnview.reference_hook import TestOnlyReferenceAttachment  # noqa: E402

ACCEPTED = REPO / "evidence/p3-calib/masked-prep/crossblock-note.json"


def load_timeline(cfg: dict, fixture: dict) -> tuple[TimelineBridge, list[dict]]:
    """由 config 段落 + 真实 tokenizer 独立累计上标,得到声明点(段末 token 触发、t+1 消费)。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(REPO / cfg["tokenizer_dir"]), trust_remote_code=False)
    timeline_cfg = json.loads((REPO / cfg["timeline_config"]).read_text())
    idx, decls = 0, []
    for piece in timeline_cfg["generation_script"]:
        n = len(tok.encode(piece["text"], add_special_tokens=False))
        idx += n
        decls.append({"parse_index_0based": idx - 1, "mode": piece["expected_mode_after"],
                      "refs": tuple(piece.get("refs", []))})
    prev = None
    transitions = []
    for d in decls:
        if prev is None or d["mode"] != prev:
            transitions.append((d["parse_index_0based"], d["mode"], d["refs"]))
        prev = d["mode"]
    bridge = TimelineBridge(prompt_len=int(fixture["prompt_len"]), kernel_block_size=int(cfg["block_size"]),
                            sink_span=tuple(fixture["sink_span"]),
                            local_window_span=tuple(fixture["local_window_span"]),
                            segment_spans=tuple(tuple(s) for s in fixture["segment_spans"]),
                            declarations=tuple(transitions))
    return bridge, decls


def make_real_metadata(block_table_2d: torch.Tensor, seq_lens_values, num_tokens: int, max_blocks: int):
    """构造**真实** `FlashAttentionMetadata`(按字段类型补齐必需项)。"""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    overrides = {
        "num_actual_tokens": int(num_tokens), "max_query_len": 1,
        "query_start_loc": torch.arange(len(seq_lens_values) + 1, dtype=torch.int32),
        "max_seq_len": int(max(seq_lens_values)),
        "seq_lens": torch.tensor(seq_lens_values, dtype=torch.int32),
        "block_table": block_table_2d,
        "slot_mapping": torch.zeros(num_tokens, dtype=torch.int64),
        "use_cascade": False,
        "num_decode_reqs": 1,
        "num_prefill_reqs": 0,
        "num_decode_tokens": int(num_tokens),
        "num_prefill_tokens": 0,
    }
    kw = {}
    for f in dataclasses.fields(FlashAttentionMetadata):
        if f.name in overrides:
            kw[f.name] = overrides[f.name]
            continue
        if f.default is not dataclasses.MISSING or getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:
            continue
        ann = str(f.type)
        kw[f.name] = (torch.zeros(1, dtype=torch.int32) if "Tensor" in ann
                      else 0 if ("int" in ann or "float" in ann and "Optional" not in ann)
                      else True if "bool" in ann else None)
    return FlashAttentionMetadata(**kw)


class StandInImpl:
    """原实现替身:计数并不改写缓冲(用于区分是否走了原实现);带真实 `scale`(门禁要求)。"""

    def __init__(self, name: str, scale: float = 0.0) -> None:
        self.name = name
        self.scale = scale
        self.calls = 0

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        self.calls += 1
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    accepted = REPO / cfg.get("accepted_fixture", "evidence/p3-calib/masked-prep/crossblock-note.json")
    fixture = json.loads(accepted.read_text())
    heads, kv_heads, d = int(cfg["heads"]), int(cfg["kv_heads"]), int(cfg["head_dim"])
    block_size = int(cfg["block_size"])
    scale = float(d) ** -0.5
    checks: list[dict] = []
    detail: dict = {"config": str(args.config), "limits": cfg["limits"], "accepted_fixture_sha256":
                    hashlib.sha256(accepted.read_bytes()).hexdigest(), "accepted_fixture": str(accepted)}

    def chk(name: str, ok: bool, **extra):
        checks.append({"name": name, "ok": bool(ok), **extra})

    bridge, decls = load_timeline(cfg, fixture)
    detail["declarations"] = [{"parse_index_0based": p, "mode": m, "refs": list(r)} for p, m, r in bridge.declarations]

    # ---------- 1. 位置:语义 vs 块外扩;与已验收夹具逐步一致 ----------
    idx_cross = 6          # 0 基下标 6 ⇒ 第 7 个 decode(local)
    mask = bridge.mask_at(idx_cross)
    lw, snk, pl = tuple(fixture["local_window_span"]), tuple(fixture["sink_span"]), int(fixture["prompt_len"])
    kv_expect = bridge.kv_len_at(idx_cross)
    sem_expect = (snk[1] - snk[0]) + (min(lw[1], pl) - lw[0]) + (kv_expect - pl)
    chk("语义位置 = sink ∪ local_window ∪ response(按夹具 span 推导)",
        len(mask.semantic_positions) == sem_expect, got=len(mask.semantic_positions), want=sem_expect)
    chk("读取位置 = 语义位置向块边界外扩后截断(块集与每块计数自洽,且被验收产物核对)",
        len(mask.read_positions) == sum(mask.effective_per_block)
        and set(mask.blocks) == {p // block_size for p in mask.read_positions}
        and mask.blocks == tuple(sorted(set(mask.blocks)))
        and max(mask.read_positions) == kv_expect - 1
        and len(mask.read_positions) > len(mask.semantic_positions),
        blocks=list(mask.blocks), per_block=list(mask.effective_per_block), read=len(mask.read_positions))
    bad = []
    for s_ in fixture["steps"]:
        mm = independent_visible_positions(
            mode=s_["mode"], refs=s_["refs"], kv_len=s_["kv_len"], prompt_len=fixture["prompt_len"],
            sink_span=tuple(fixture["sink_span"]), local_window_span=tuple(fixture["local_window_span"]),
            segment_spans=tuple(tuple(x) for x in fixture["segment_spans"]), block_size=block_size)
        if list(mm.blocks) != s_["visible_blocks_independent"] or list(mm.effective_per_block) != s_["counts_independent"] \
                or sum(mm.effective_per_block) != s_["total_independent"]:
            bad.append({"decode": s_["decode_count_1based"], "got": list(mm.blocks),
                        "want": s_["visible_blocks_independent"]})
    chk("与已验收 28 步夹具逐步一致(块集/每块计数/总长)", not bad and len(fixture["steps"]) == 28, bad=bad[:3])

    # ---------- 2. 原生 KV 解包 == pin 的写法 ----------
    n_blocks = 16
    native_kv = torch.randn(n_blocks, kv_heads, block_size, 2 * d, dtype=torch.bfloat16)
    kc, vc = unpack_native_kv(native_kv, d)
    ref_k, ref_v = native_kv.transpose(1, 2).split(d, dim=-1)
    chk("原生 KV 解包 == transpose(1,2).split(head_size,-1) 且形状为 [blocks,block,kv_heads,d]",
        torch.equal(kc, ref_k.contiguous()) and torch.equal(vc, ref_v.contiguous())
        and kc.shape == (n_blocks, block_size, kv_heads, d), shape=list(kc.shape))
    try:
        unpack_native_kv(torch.randn(n_blocks, kv_heads, block_size, d), d)
        chk("末维不等于 2*head_size 时拒绝", False)
    except ReferenceError as exc:
        chk("末维不等于 2*head_size 时拒绝", True, error=str(exc)[:80])

    # ---------- 3. 真实 metadata + 真实签名驱动 ----------
    block_row_a = [5, 12, 1, 9, 14, 3, 7, 10, 0, 13, 2]
    block_row_b = [4, 6, 8, 2, 11, 15, 1, 9, 3, 7, 12]
    impls = [StandInImpl(f"L{i}", scale) for i in range(2)]
    originals = {i: impls[i].forward for i in range(2)}
    sw = TestOnlyReferenceSwitch(enabled=True, request_id="req-A", layers=(0, 1), steps=(6, 7), scale=scale)
    at = TestOnlyReferenceAttachment(sw, bridge=bridge, request_idx=0, head_size=d, layers=(0, 1), steps=(6, 7))
    q = torch.randn(1, heads, d, dtype=torch.bfloat16)          # 真实三维单 token
    out_a = torch.zeros(1, heads, d, dtype=torch.bfloat16)
    out_b = torch.zeros(1, heads, d, dtype=torch.bfloat16)
    ptr_a = out_a.data_ptr()

    def meta_for(step: int):
        kv_len = bridge.kv_len_at(step - 1)
        return make_real_metadata(torch.tensor([block_row_a, block_row_b], dtype=torch.int32),
                                  [kv_len, kv_len], 1, len(block_row_a))

    try:
        at.wrap_all(impls)
        for step in (6, 7):
            md = meta_for(step)
            sw.current_request_id, sw.current_step = "req-A", step
            for layer, impl in enumerate(impls):
                impl.forward(layer, q, None, None, native_kv, md, out_a)
            sw.current_request_id, sw.current_step = "req-B", step
            for layer, impl in enumerate(impls):
                impl.forward(layer, q, None, None, native_kv, md, out_b)
    finally:
        at.restore()

    exp = {(l, s) for l in (0, 1) for s in (6, 7)}
    chk("真实签名下:目标请求的 (layer, step) 全部被参考覆盖", at.missing(exp) == set()
        and at.unexpected(exp) == set(), overrode=sorted(at.overrode()))
    chk("真实签名下:非目标请求经同一挂接全部直通", at.passthrough("req-B") == exp
        and len([e for e in at.ledger if e["request_id"] == "req-B"]) == 4)
    chk("非目标请求输出未被改写", float(out_b.abs().max()) == 0.0)
    chk("覆盖账本记录真实 metadata 来源(块表行取自二维张量)", all(
        len(e.get("block_table_row_head", [])) == 4 for e in at.ledger if e["action"] == "overrode"))
    chk("真实三维 query/output:写回对应视图且原缓冲身份不变",
        out_a.data_ptr() == ptr_a and float(out_a.abs().max()) > 0.0 and tuple(out_a.shape) == (1, heads, d),
        ptr_preserved=out_a.data_ptr() == ptr_a)
    m = at.metrics()[0]
    chk("记录真实三维 shape 与 num_tokens=1", m.get("query_shape") == [1, heads, d]
        and m.get("output_shape") == [1, heads, d] and m.get("num_tokens") == 1,
        query_shape=m.get("query_shape"), output_shape=m.get("output_shape"))
    chk("真实接口下记录真实指标(dtype=output.dtype、cast 差异非零、seq_lens 交叉核对一致)",
        m["output_dtype"] == "torch.bfloat16" and m["fp32_to_output_max_abs"] > 0.0
        and m["seq_lens_value"] == m["kv_len"] and m["non_finite_count"] == 0,
        metrics={k: m[k] for k in ("mode", "kv_len", "seq_lens_value", "read_positions", "blocks",
                                   "fp32_to_output_max_abs", "fp32_to_output_rms", "fp32_to_output_rel_l2",
                                   "ref_abs_max", "non_finite_count")})
    detail["metrics_sample"] = {k: m[k] for k in ("mode", "kv_len", "seq_lens_value", "semantic_positions",
                                                 "read_positions", "blocks", "effective_per_block",
                                                 "output_dtype", "fp32_to_output_max_abs",
                                                 "fp32_to_output_rms", "fp32_to_output_rel_l2",
                                                 "ref_abs_max", "non_finite_count")}
    chk("恢复后 forward 身份等于原方法", all(impls[i].forward == originals[i] for i in range(2))
        and at.restored == [0, 1], restored=at.restored)
    n_calls = [impls[i].calls for i in range(2)]
    out_a0 = out_a.clone()
    impls[0].forward(0, q, None, None, native_kv, meta_for(6), out_a)
    chk("恢复后调用走原实现(计数 +1、输出缓冲不被参考改写)",
        impls[0].calls == n_calls[0] + 1 and torch.equal(out_a, out_a0))

    # ---------- 4. metadata/时间线不一致必须拒绝 ----------
    # 多 token 必须拒绝(参考只支持单 token decode)
    try:
        perform_reference_attention_native(sw, layer_idx=0, step_index_0based=6, bridge=bridge, request_idx=0,
                                           query=torch.randn(3, heads, d, dtype=torch.bfloat16), kv_cache=native_kv,
                                           attn_metadata=meta_for(6), output=torch.zeros(3, heads, d, dtype=torch.bfloat16),
                                           head_size=d)
        chk("多 token 输入被拒绝(仅单 token decode)", False)
    except ReferenceError as exc:
        chk("多 token 输入被拒绝(仅单 token decode)", True, error=str(exc)[:80])
    # 门禁:scale 不一致 / 额外参数 / 非 BF16 / 几何不一致
    impl_bad = StandInImpl("bad", scale + 0.5)
    sw_bad = TestOnlyReferenceSwitch(enabled=True, request_id="req-S", layers=(0,), steps=(6,), scale=scale)
    at_bad = TestOnlyReferenceAttachment(sw_bad, bridge=bridge, request_idx=0, head_size=d, layers=(0,), steps=(6,))
    errs = {}
    try:
        at_bad.wrap_all([impl_bad])
        sw_bad.current_request_id, sw_bad.current_step = "req-S", 6
        try:
            impl_bad.forward(0, q, None, None, native_kv, meta_for(6), torch.zeros(1, heads, d, dtype=torch.bfloat16))
        except ReferenceError as exc:
            errs["scale_mismatch"] = str(exc)
        sw_bad.enabled, sw_bad.restore = True, None
    finally:
        at_bad.restore()
    chk("门禁:impl.scale 与开关 scale 不一致即拒绝(不从 head_dim 推导)", "不一致" in errs.get("scale_mismatch", ""),
        error=errs.get("scale_mismatch", "")[:90])
    impl_x = StandInImpl("x", scale)
    at_x = TestOnlyReferenceAttachment(TestOnlyReferenceSwitch(enabled=True, request_id="req-X", layers=(0,), steps=(6,), scale=scale),
                                       bridge=bridge, request_idx=0, head_size=d, layers=(0,), steps=(6,))
    extra_ok = False
    try:
        at_x.wrap_all([impl_x])
        at_x.switch.current_request_id, at_x.switch.current_step = "req-X", 6
        try:
            impl_x.forward(0, q, None, None, native_kv, meta_for(6),
                           torch.zeros(1, heads, d, dtype=torch.bfloat16), None, None)   # output_scale/output_block_scale
        except ReferenceError as exc:
            extra_ok = "额外参数" in str(exc)
    finally:
        at_x.restore()
    chk("门禁:量化/特殊特性(额外 output_scale/output_block_scale)不被静默丢弃", extra_ok)
    for label, kw, want in (("非 BF16 输出", {"output": torch.zeros(1, heads, d, dtype=torch.float16)}, "BF16"),
                            ("KV 块长与 kernel_block_size 不一致", {"kv_cache": torch.randn(16, kv_heads, 512, 2 * d, dtype=torch.bfloat16)}, "块长")):
        arg = dict(query=q, kv_cache=native_kv, attn_metadata=meta_for(6),
                   output=torch.zeros(1, heads, d, dtype=torch.bfloat16))
        arg.update(kw)
        try:
            perform_reference_attention_native(sw, layer_idx=0, step_index_0based=6, bridge=bridge, request_idx=0,
                                               head_size=d, impl_scale=scale, **arg)
            chk(f"门禁:{label} 被拒绝", False)
        except ReferenceError as exc:
            chk(f"门禁:{label} 被拒绝", want in str(exc), error=str(exc)[:90])
    bad_md = make_real_metadata(torch.tensor([block_row_a], dtype=torch.int32),
                                [bridge.kv_len_at(6) + 3], 1, len(block_row_a))
    try:
        perform_reference_attention_native(sw, layer_idx=0, step_index_0based=6, bridge=bridge, request_idx=0,
                                           query=q, kv_cache=native_kv, attn_metadata=bad_md, output=out_a, head_size=d)
        chk("metadata.seq_lens 与独立时间线不一致时拒绝", False)
    except ReferenceError as exc:
        chk("metadata.seq_lens 与独立时间线不一致时拒绝", True, error=str(exc)[:90])
    try:
        perform_reference_attention_native(sw, layer_idx=0, step_index_0based=6, bridge=bridge, request_idx=9,
                                           query=q, kv_cache=native_kv, attn_metadata=meta_for(6), output=out_a, head_size=d)
        chk("request_idx 超出块表行数时拒绝", False)
    except ReferenceError as exc:
        chk("request_idx 超出块表行数时拒绝", True, error=str(exc)[:90])

    # ---------- 5. 缺层/缺步、异常恢复、隔离 ----------
    impls2 = [StandInImpl("m0", scale), StandInImpl("m1", scale)]
    sw2 = TestOnlyReferenceSwitch(enabled=True, request_id="req-C", layers=(0, 1), steps=(6,), scale=scale)
    at2 = TestOnlyReferenceAttachment(sw2, bridge=bridge, request_idx=0, head_size=d, layers=(0, 1), steps=(6,))
    try:
        at2.wrap_all(impls2)
        sw2.current_request_id, sw2.current_step = "req-C", 6
        impls2[0].forward(0, q, None, None, native_kv, meta_for(6), torch.zeros(1, heads, d, dtype=torch.bfloat16))
    finally:
        at2.restore()
    chk("缺层/缺步被账本检出(missing 非空)", at2.missing({(0, 6), (1, 6)}) == {(1, 6)},
        missing=sorted(at2.missing({(0, 6), (1, 6)})))

    impl3 = StandInImpl("e0", scale)
    orig3 = impl3.forward
    sw3 = TestOnlyReferenceSwitch(enabled=True, request_id="req-D", layers=(0,), steps=(6,), scale=scale)
    at3 = TestOnlyReferenceAttachment(sw3, bridge=bridge, request_idx=0, head_size=d, layers=(0,), steps=(6,))
    raised = False
    try:
        at3.wrap_all([impl3])
        sw3.current_request_id, sw3.current_step = "req-D", 6
        impl3.forward(0, q, None, None, native_kv, meta_for(6), torch.zeros(1, heads + 1, d, dtype=torch.bfloat16))
    except ReferenceError:
        raised = True
    finally:
        at3.restore()
    chk("参考异常:抛错、开关关闭、forward 恢复为原方法", raised and sw3.enabled is False
        and impl3.forward == orig3 and bool(at3.errors()), errors=at3.errors()[:1])

    impls_iso = [StandInImpl("iso0", scale)]
    sw_iso = TestOnlyReferenceSwitch(enabled=True, request_id="req-B2", layers=(0,), steps=(7,), scale=scale)
    at_iso = TestOnlyReferenceAttachment(sw_iso, bridge=bridge, request_idx=1, head_size=d, layers=(0,), steps=(7,))
    try:
        at_iso.wrap_all(impls_iso)
        sw_iso.current_request_id, sw_iso.current_step = "req-B2", 7
        impls_iso[0].forward(0, q, None, None, native_kv, meta_for(7), torch.zeros(1, heads, d, dtype=torch.bfloat16))
    finally:
        at_iso.restore()
    chk("独立挂接(另一请求/另一 impl/另一块表行)与主挂接账本互不影响",
        at_iso.overrode() == {(0, 7)} and at.overrode() == exp and at_iso.request_idx == 1,
        iso=sorted(at_iso.overrode()), main=sorted(at.overrode()))

    # ---------- 6. 接线反例与边界 ----------
    stale = torch.zeros(1, heads, d, dtype=torch.bfloat16)
    from attnview.reference_dense import dense_attention_fp32, gather_positions
    _ = dense_attention_fp32(q.reshape(-1, d), *gather_positions(kc, vc, block_row_a, block_size, mask.read_positions),
                             scale=scale, dtype=torch.bfloat16)     # 只返回、不写缓冲
    chk("接线反例:只返回新张量、不写原缓冲 ⇒ 缓冲仍为初值(被检出)", float(stale.abs().max()) == 0.0)
    try:
        gather_positions(kc, vc, block_row_a, block_size, (bridge.kv_len_at(6) + block_size,))
        chk("越界读取位置被拒绝", False)
    except ReferenceError as exc:
        chk("越界读取位置被拒绝", True, error=str(exc)[:80])
    try:
        dense_attention_fp32(torch.randn(heads + 1, d, dtype=torch.bfloat16),
                             *gather_positions(kc, vc, block_row_a, block_size, mask.read_positions), scale=scale)
        chk("GQA 头数不整除被拒绝", False)
    except ReferenceError as exc:
        chk("GQA 头数不整除被拒绝", True, error=str(exc)[:80])

    failed = [c["name"] for c in checks if not c["ok"]]
    detail.update({"checks": checks, "failed": failed, "declarations_expected": decls,
                   "test_scope": ("CPU:真实 `FlashAttentionMetadata` 对象 + 原生 4 维 KV + 真实 forward 签名驱动挂接;"
                                  "impl 为桩件、不加载 27B、不跑 GPU;不据此宣称真实模型数值或 GPU 接线正确;"
                                  "不设数值通过阈值。")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps({"n_checks": len(checks), "ok": len(checks) - len(failed), "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
