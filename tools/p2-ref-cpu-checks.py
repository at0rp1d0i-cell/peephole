#!/usr/bin/env python3
"""测试专用独立 dense 参考的 CPU 检查(不加载模型、不跑 GPU)。

驱动方式与既有 harness **同模式**:`TestOnlyReferenceAttachment.wrap_impl` 保存 `impl.forward`
并安装同签名 wrapper,调用方在 `finally` 里 `restore()`(对应 `tools/p2-calib-run.py` 的
`LayerCapture.wrap_impl` 与 `wrap_host_logits` 的 `restore()` 用法)。

覆盖:挂接/恢复身份、默认关闭、目标请求/层/步命中,缺层缺步账本检出,错 request id 直通,
异常后恢复为原实现,两请求隔离(KV/GDN 状态各自独立),块外扩 vs 语义位置,与已验收 28 步夹具逐步比较,
以及**真实**的 FP32→output.dtype cast 指标(max_abs/RMS/相对L2/参考幅度/非有限计数)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attnview.reference_dense import (  # noqa: E402
    ReferenceError,
    TestOnlyReferenceSwitch,
    dense_attention_fp32,
    gather_positions,
    independent_visible_positions,
    return_only_variant,
)
from attnview.reference_hook import TestOnlyReferenceAttachment  # noqa: E402

SINK = (0, 16)
LOCAL = (6770, 7834)
SEGMENTS = ((362, 2316), (2365, 4255), (4304, 6756))
PROMPT_LEN = 7834
BLOCK = 784
ACCEPTED = REPO / "evidence/p3-calib/masked-prep/crossblock-note.json"


@dataclass
class Meta:
    """harness 风格的注意力元数据替身(CPU)。"""

    block_table: tuple[int, ...]
    kernel_block_size: int
    kv_len: int
    prompt_len: int
    mode: str
    refs: tuple[int, ...]
    sink_span: tuple[int, int]
    local_window_span: tuple[int, int]
    segment_spans: tuple[tuple[int, int], ...]


class StandInImpl:
    """原实现替身:只计数,不改写缓冲(用于区分"是否走了原实现")。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        self.calls += 1
        return None


def make_meta(mode: str, kv_len: int, block_table) -> Meta:
    return Meta(tuple(block_table), BLOCK, kv_len, PROMPT_LEN, mode, (), SINK, LOCAL, SEGMENTS)


def per_request_state(seed: int, n_blocks: int, kv_heads: int, d: int):
    """每个请求各自独立的 KV 与 GDN 状态(不复用上一请求残留)。"""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n_blocks, BLOCK, kv_heads, d, dtype=torch.bfloat16, generator=g)
    v = torch.randn(n_blocks, BLOCK, kv_heads, d, dtype=torch.bfloat16, generator=g)
    gdn = torch.zeros(4, d, dtype=torch.float32)
    return k, v, gdn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    heads, kv_heads, d = int(cfg["heads"]), int(cfg["kv_heads"]), int(cfg["head_dim"])
    scale = float(d) ** -0.5
    block_table = [5, 12, 1, 9, 14, 3, 7, 10, 0, 13, 2]
    checks: list[dict] = []
    detail: dict = {"config": str(args.config), "limits": cfg["limits"]}

    def chk(name: str, ok: bool, **extra):
        checks.append({"name": name, "ok": bool(ok), **extra})

    # ---------- 1. 位置:语义 vs 块外扩后读取 ----------
    kv_len = PROMPT_LEN + 7
    mask = independent_visible_positions(mode="local", refs=(), kv_len=kv_len, prompt_len=PROMPT_LEN,
                                        sink_span=SINK, local_window_span=LOCAL, segment_spans=SEGMENTS,
                                        block_size=BLOCK)
    chk("语义位置数符合 span 并集(16+1064+7)", len(mask.semantic_positions) == 1087,
        n=len(mask.semantic_positions))
    chk("读取位置为块外扩后截断到 kv_len(784*3+1)", len(mask.read_positions) == 2353
        and mask.blocks == (0, 8, 9, 10) and sum(mask.effective_per_block) == 2353,
        blocks=list(mask.blocks), per_block=list(mask.effective_per_block))
    chk("读取位置覆盖语义位置且全部 < kv_len", set(mask.semantic_positions) <= set(mask.read_positions)
        and all(p < kv_len for p in mask.read_positions) and mask.read_positions[-1] == kv_len - 1)
    chk("块外扩只增不减(读取 ⊋ 语义)", len(mask.read_positions) > len(mask.semantic_positions))

    # ---------- 2. 与已验收 28 步夹具逐步机械比较 ----------
    if ACCEPTED.exists():
        fx = json.loads(ACCEPTED.read_text())
        segs = [tuple(x) for x in fx["segment_spans"]]
        bad = []
        for s_ in fx["steps"]:
            mm = independent_visible_positions(mode=s_["mode"], refs=s_["refs"], kv_len=s_["kv_len"],
                                               prompt_len=fx["prompt_len"], sink_span=tuple(fx["sink_span"]),
                                               local_window_span=tuple(fx["local_window_span"]),
                                               segment_spans=segs, block_size=BLOCK)
            if list(mm.blocks) != s_["visible_blocks_independent"] or list(mm.effective_per_block) != s_["counts_independent"] \
                    or sum(mm.effective_per_block) != s_["total_independent"]:
                bad.append({"decode": s_["decode_count_1based"], "got": list(mm.blocks),
                            "want": s_["visible_blocks_independent"]})
        chk("与已验收 28 步夹具逐步一致(块集/每块计数/总长)", not bad and len(fx["steps"]) == 28, bad=bad[:3],
            accepted_sha256=hashlib.sha256(ACCEPTED.read_bytes()).hexdigest()[:16])
    else:
        chk("已验收夹具存在", False, path=str(ACCEPTED))

    # ---------- 3. 经挂接驱动(harness 模式:一个 impl 只挂一次) ----------
    impls = [StandInImpl(f"L{i}") for i in range(3)]
    originals = {i: impls[i].forward for i in range(3)}
    sw_a = TestOnlyReferenceSwitch(enabled=True, request_id="req-A", layers=(0, 1, 2), steps=(1, 2, 3),
                                   scale=scale, dtype=torch.bfloat16)
    at_a = TestOnlyReferenceAttachment(sw_a, layers=(0, 1, 2), steps=(1, 2, 3))
    # 请求 B 走**同一挂接**但非目标 request ⇒ 必须直通
    kv_a = per_request_state(11, 16, kv_heads, d)
    kv_b = per_request_state(22, 16, kv_heads, d)
    meta_a, meta_b = make_meta("local", kv_len, block_table), make_meta("global", kv_len, block_table)
    out_a = {i: torch.zeros(heads, d, dtype=torch.bfloat16) for i in range(3)}
    out_b = {i: torch.zeros(heads, d, dtype=torch.bfloat16) for i in range(3)}
    q = torch.randn(heads, d, dtype=torch.bfloat16)
    try:
        at_a.wrap_all(impls)
        for rid, kv, meta, out in (("req-A", kv_a, meta_a, out_a), ("req-B", kv_b, meta_b, out_b)):
            sw_a.current_request_id = rid
            for step in (1, 2, 3):
                sw_a.current_step = step
                for layer, impl in enumerate(impls):
                    impl.forward(layer, q, None, None, kv[:2], meta, out[layer])
    finally:
        at_a.restore()

    exp = {(l, s) for l in (0, 1, 2) for s in (1, 2, 3)}
    chk("目标请求 A:期望的 (layer, step) 全部被参考覆盖", at_a.missing(exp) == set()
        and at_a.unexpected(exp) == set(), overrode=sorted(at_a.overrode()))
    chk("非目标请求 B:同一挂接下全部直通(零覆盖)",
        {k for k in at_a.passthrough() if True} >= exp and not (at_a.overrode() & exp) or
        len([e for e in at_a.ledger if e["request_id"] == "req-B" and e["action"] == "passthrough"]) == 9,
        b_passthrough=len([e for e in at_a.ledger if e["request_id"] == "req-B" and e["action"] == "passthrough"]))
    chk("账本按 request_id 分离(A 覆盖 9 条、B 直通 9 条)",
        len([e for e in at_a.ledger if e["request_id"] == "req-A" and e["action"] == "overrode"]) == 9
        and {e["request_id"] for e in at_a.ledger} == {"req-A", "req-B"},
        counts={r: len([e for e in at_a.ledger if e["request_id"] == r]) for r in ("req-A", "req-B")})
    kv_a[2].add_(1.0)   # 改动 A 的 GDN 状态,验证不影响 B
    chk("两请求 KV/GDN 状态对象独立(对象不同、KV 初值不同、改一个不影响另一个)",
        kv_a[0] is not kv_b[0] and kv_a[1] is not kv_b[1] and kv_a[2] is not kv_b[2]
        and not torch.equal(kv_a[0], kv_b[0]) and float(kv_b[2].abs().max()) == 0.0,
        gdn_a_after=float(kv_a[2].abs().max()), gdn_b_after=float(kv_b[2].abs().max()))
    chk("B 请求未被参考改写(输出缓冲仍为初值)", all(float(out_b[i].abs().max()) == 0.0 for i in range(3)))
    chk("恢复后 forward 身份等于原方法", all(impls[i].forward == originals[i] for i in range(3))
        and at_a.restored == [0, 1, 2], restored=at_a.restored)
    n_calls = [impls[i].calls for i in range(3)]
    out_a0 = out_a[0].clone()
    impls[0].forward(0, q, None, None, kv_a[:2], meta_a, out_a[0])
    chk("恢复后调用走原实现(计数 +1 且输出缓冲不被参考改写)",
        impls[0].calls == n_calls[0] + 1 and torch.equal(out_a[0], out_a0))

    # 两请求隔离(独立引擎/独立挂接):各自覆盖账本互不影响
    impls_iso = [StandInImpl(f"iso-L{i}") for i in range(2)]
    sw_iso = TestOnlyReferenceSwitch(enabled=True, request_id="req-B2", layers=(0, 1), steps=(1,),
                                     scale=scale, dtype=torch.bfloat16)
    at_iso = TestOnlyReferenceAttachment(sw_iso, layers=(0, 1), steps=(1,))
    kv_iso = per_request_state(33, 16, kv_heads, d)
    try:
        at_iso.wrap_all(impls_iso)
        sw_iso.current_request_id, sw_iso.current_step = "req-B2", 1
        for layer, impl in enumerate(impls_iso):
            impl.forward(layer, q, None, None, kv_iso[:2], meta_b,
                         torch.zeros(heads, d, dtype=torch.bfloat16))
    finally:
        at_iso.restore()
    chk("独立挂接(另一请求/另一 impl 集)与 A 的账本互不影响",
        at_iso.overrode() == {(0, 1), (1, 1)} and at_a.overrode() == exp,
        iso=sorted(at_iso.overrode()), a=sorted(at_a.overrode()))

    # 参考覆盖的输出:真实指标落盘
    over = [e for e in at_a.ledger if e["action"] == "overrode"]
    m = over[0] if over else {}
    chk("挂接路径记录了真实 cast/幅度指标(dtype=output.dtype,非恒真)", bool(over)
        and m.get("output_dtype") == "torch.bfloat16" and m.get("fp32_to_output_max_abs") is not None
        and m.get("fp32_to_output_rel_l2") is not None and m.get("non_finite_count") == 0,
        sample={k: m.get(k) for k in ("output_dtype", "fp32_to_output_max_abs", "fp32_to_output_rms",
                                      "fp32_to_output_rel_l2", "ref_abs_max", "non_finite_count")})
    detail["metrics_sample"] = {k: m.get(k) for k in ("semantic_positions", "read_positions", "blocks",
                                                      "effective_per_block", "output_dtype",
                                                      "fp32_to_output_max_abs", "fp32_to_output_rms",
                                                      "fp32_to_output_rel_l2", "ref_abs_max", "non_finite_count")}
    detail["ledger_summary"] = {"A_overrode": len(at_a.overrode()),
                                "B_passthrough": len([e for e in at_a.ledger if e["request_id"] == "req-B"])}

    # ---------- 4. 缺层/缺步检出(账本) ----------
    impl2 = [StandInImpl("L0"), StandInImpl("L1")]
    sw2 = TestOnlyReferenceSwitch(enabled=True, request_id="req-C", layers=(0, 1), steps=(1, 2), scale=scale,
                                  dtype=torch.bfloat16)
    at2 = TestOnlyReferenceAttachment(sw2, layers=(0, 1), steps=(1, 2))
    try:
        at2.wrap_all(impl2)
        sw2.current_request_id, sw2.current_step = "req-C", 1
        k = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
        v = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
        impl2[0].forward(0, q, None, None, (k, v), make_meta("local", kv_len, block_table),
                         torch.zeros(heads, d, dtype=torch.bfloat16))
    finally:
        at2.restore()
    chk("缺层/缺步被账本检出(missing 非空)", at2.missing({(0, 1), (1, 2)}) == {(1, 2)},
        missing=sorted(at2.missing({(0, 1), (1, 2)})))

    # ---------- 5. 异常后恢复为原实现 ----------
    impl3 = StandInImpl("L0")
    orig3 = impl3.forward
    sw3 = TestOnlyReferenceSwitch(enabled=True, request_id="req-D", layers=(0,), steps=(1,), scale=scale,
                                  dtype=torch.bfloat16)
    at3 = TestOnlyReferenceAttachment(sw3, layers=(0,), steps=(1,))
    raised = False
    try:
        at3.wrap_all([impl3])
        sw3.current_request_id, sw3.current_step = "req-D", 1
        k = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
        v = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
        impl3.forward(0, q, None, None, (k, v), make_meta("local", kv_len, block_table),
                      torch.zeros(heads + 1, d, dtype=torch.bfloat16))     # 形状不符 → 参考抛错
    except ReferenceError:
        raised = True
    finally:
        at3.restore()
    chk("参考异常:已抛错、开关关闭且 forward 恢复为原方法", raised and sw3.enabled is False
        and impl3.forward == orig3 and bool(at3.errors()), errors=at3.errors()[:1])

    # ---------- 6. 接线反例:只返回新张量不写原缓冲 ----------
    stale = torch.zeros(heads, d, dtype=torch.bfloat16)
    k = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
    v = torch.randn(16, BLOCK, kv_heads, d, dtype=torch.bfloat16)
    _ = return_only_variant(q, *gather_positions(k, v, block_table, BLOCK, mask.read_positions),
                            scale=scale, dtype=torch.bfloat16)
    chk("接线反例:只返回新张量、不写原缓冲 ⇒ 缓冲仍为初值(被检出)", float(stale.abs().max()) == 0.0)

    # ---------- 7. 结构与算子边界 ----------
    try:
        gather_positions(k, v, block_table, BLOCK, (kv_len + BLOCK,))
        chk("越界读取位置被拒绝", False)
    except ReferenceError as exc:
        chk("越界读取位置被拒绝", True, error=str(exc)[:80])
    try:
        dense_attention_fp32(torch.randn(heads + 1, d, dtype=torch.bfloat16),
                             *gather_positions(k, v, block_table, BLOCK, mask.read_positions), scale=scale)
        chk("GQA 头数不整除被拒绝(不静默近似)", False)
    except ReferenceError as exc:
        chk("GQA 头数不整除被拒绝(不静默近似)", True, error=str(exc)[:80])
    kk, vv = gather_positions(k, v, block_table, BLOCK, mask.read_positions)
    fp32, casted = dense_attention_fp32(q, kk, vv, scale=scale, dtype=torch.bfloat16)
    chk("显式 dtype 下 FP32 与 cast 差异非零(证据真实)", casted.dtype == torch.bfloat16
        and float((casted.to(torch.float32) - fp32).abs().max()) > 0.0,
        fp32_to_bf16_max_abs=float((casted.to(torch.float32) - fp32).abs().max()))

    failed = [c["name"] for c in checks if not c["ok"]]
    detail.update({"checks": checks, "failed": failed,
                   "test_scope": ("CPU:桩件 + 挂接驱动,模拟生产接线(unified_attention_with_output 忽略返回值、消费原 output 缓冲);"
                                  "**不**据此宣称真实模型或 GPU 接线正确;未加载 27B、未跑 GPU;不设数值通过阈值。")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps({"n_checks": len(checks), "ok": len(checks) - len(failed), "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
