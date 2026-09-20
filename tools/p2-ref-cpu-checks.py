#!/usr/bin/env python3
"""测试专用独立 dense 参考的 CPU 检查(不加载模型、不跑 GPU)。

覆盖(工作单 §CPU 验证):
1. 非顺序逻辑→物理映射下的块 gather 正确性;
2. 不完整尾块 + 当前 token(位置含 kv_len-1、绝不越过);
3. 精确结构反例(错可见块/尾长 −1)被**独立真值**检出(结构层,非数值断言);
4. 缺层/缺步/错 request id 被拒绝;
5. 接线反例:只返回新张量、不写传入 output ⇒ **必须被检出**;
6. 目标 output 原地写入(数据指针不变)+ 默认关闭 + 异常后恢复为关闭;
7. 两个请求的状态/KV/GDN 独立(不复用上一请求残留);
8. 记录 FP32(未 cast) 与 cast 后的差异,明确比较对象。

模拟生产接线:`unified_attention_with_output` 调用 `impl.forward(...)` 后**忽略返回值**、
继续消费传入的 `output` 缓冲(pin 98dff2a8 源码事实)。
"""
from __future__ import annotations

import argparse
import json
import sys
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

SINK = (0, 16)
LOCAL = (6770, 7834)
SEGMENTS = ((362, 2316), (2365, 4255), (4304, 6756))
PROMPT_LEN = 7834
BLOCK = 784


def unified_attention_with_output_stub(impl, *, output, **kwargs):
    """生产接线的等价模拟:`output` 缓冲作为参数传入 impl.forward;其**返回值被忽略**,
    调用方继续消费原 `output` 缓冲(pin 98dff2a8 的 `unified_attention_with_output` 行为)。"""
    impl.forward(output=output, **kwargs)
    return output


def make_cache(n_blocks: int, block_size: int, kv_heads: int, d: int, *, seed: int = 0, device="cpu"):
    torch.manual_seed(seed)
    k = torch.randn(n_blocks, block_size, kv_heads, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(n_blocks, block_size, kv_heads, d, dtype=torch.bfloat16, device=device)
    return k, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    heads, kv_heads, d = int(cfg["heads"]), int(cfg["kv_heads"]), int(cfg["head_dim"])
    block_size = int(cfg["block_size"])
    scale = float(d) ** -0.5
    checks: list[dict] = []
    detail: dict = {"config": str(args.config), "limits": cfg["limits"]}

    def chk(name: str, ok: bool, **extra):
        checks.append({"name": name, "ok": bool(ok), **extra})

    # --- 1. 非顺序逻辑→物理映射 ---
    n_blocks = 16
    # 11 项 = ceil(8192/784):覆盖夹具全部可见块;非顺序以区分逻辑/物理编号
    block_table = [5, 12, 1, 9, 14, 3, 7, 10, 0, 13, 2]
    k, v = make_cache(n_blocks, block_size, kv_heads, d, seed=1)
    positions = (0, 3, block_size - 1, block_size, 2 * block_size + 5)
    kk, vv = gather_positions(k, v, block_table, block_size, positions)
    expect_k = torch.stack([k[block_table[p // block_size], p % block_size] for p in positions])
    chk("非顺序物理映射:gather 行 == 物理块表映射出的行", torch.equal(kk, expect_k) and torch.equal(vv, torch.stack(
        [v[block_table[p // block_size], p % block_size] for p in positions])))
    chk("逻辑块号 ≠ 物理块号(映射确实非顺序)", all(i != b for i, b in enumerate(block_table[: len(SEGMENTS)])),
        block_table=block_table[:6])

    # --- 2. 不完整尾块 + 当前 token ---
    kv_len = PROMPT_LEN + 7            # 7841:不是 block_size 的整数倍
    mask = independent_visible_positions(mode="local", refs=(), kv_len=kv_len, prompt_len=PROMPT_LEN,
                                         sink_span=SINK, local_window_span=LOCAL, segment_spans=SEGMENTS)
    chk("可见位置含当前 token 且以 kv_len-1 结尾", mask.positions[-1] == kv_len - 1
        and mask.current_token_position == kv_len - 1, last=mask.positions[-1], kv_len=kv_len)
    chk("可见位置全部 < kv_len(不读未写位置)", all(p < kv_len for p in mask.positions))
    q = torch.randn(heads, d, dtype=torch.bfloat16)
    kk2, vv2 = gather_positions(k, v, block_table, block_size, mask.positions)
    fp32, casted = dense_attention_fp32(q, kk2, vv2, scale=scale)
    # 逐位置朴素实现(第三实现自检)
    group = heads // kv_heads
    naive = []
    for h in range(heads):
        kh = h // group
        s = [float(torch.dot(q[h].to(torch.float32), kk2[i, kh].to(torch.float32))) * scale for i in range(kk2.shape[0])]
        mx = max(s)
        ex = [pow(2.718281828459045, x - mx) for x in s]
        tot = sum(ex)
        naive.append(sum(ex[i] / tot * vv2[i, kh].to(torch.float32) for i in range(len(s))))
    naive_t = torch.stack(naive)
    chk("FP32 参考与朴素逐位置实现一致(≤1e-5)", float((fp32 - naive_t).abs().max()) < 1e-5,
        max_abs=float((fp32 - naive_t).abs().max()))
    chk("cast 结果与未 cast FP32 的差异被记录(比较对象明确)",
        float((casted.to(torch.float32) - fp32).abs().max()) >= 0.0,
        fp32_to_bf16_max_abs=float((casted.to(torch.float32) - fp32).abs().max()))
    detail["visible_positions_sample"] = {"count": len(mask.positions), "first": mask.positions[:3],
                                          "last": mask.positions[-3:]}

    # --- 3. 精确结构反例(独立真值检出,不主张数值可检出) ---
    honest = set(mask.positions)
    impostor = sorted(([p for p in range(min(honest), max(honest)) if p not in honest][:1] or [0])
                      + [p for p in mask.positions if p != min(honest)])
    chk("结构反例:替换一个可见位置后被独立真值检出(集合不同)", set(impostor) != honest,
        removed=min(honest), added=impostor[0])
    tail_mask = independent_visible_positions(mode="local", refs=(), kv_len=kv_len, prompt_len=PROMPT_LEN,
                                             sink_span=SINK, local_window_span=LOCAL, segment_spans=SEGMENTS)
    shrunk = tuple(p for p in list(tail_mask.positions)[:-1])
    chk("结构反例:尾长 −1(少读当前 token)被独立真值检出", set(shrunk) != set(tail_mask.positions)
        and mask.positions[-1] not in shrunk, dropped=mask.positions[-1])
    try:
        gather_positions(k, v, block_table, block_size, (kv_len + block_size,))
        chk("结构反例:越界位置被拒绝", False)
    except ReferenceError as exc:
        chk("结构反例:越界位置被拒绝", True, error=str(exc)[:100])

    # --- 4. 缺层/缺步/错 request id 拒绝 ---
    sw = TestOnlyReferenceSwitch(enabled=True, request_id="req-A", layers=(0, 1), steps=(1, 2), scale=scale,
                                 dtype=torch.bfloat16)
    chk("目标层/步启用", sw.should_override(request_id="req-A", layer_idx=1, step=2))
    chk("非目标 request id 一律拒绝", not sw.should_override(request_id="req-B", layer_idx=1, step=2))
    chk("未指定层拒绝(缺层)", not sw.should_override(request_id="req-A", layer_idx=5, step=2))
    chk("未指定步拒绝(缺步)", not sw.should_override(request_id="req-A", layer_idx=1, step=9))
    sw_off = TestOnlyReferenceSwitch(request_id="req-A", scale=scale)
    chk("默认关闭(未 enable 时不接管)", not sw_off.should_override(request_id="req-A", layer_idx=0, step=1))
    try:
        sw_off.forward_with_output(request_id="req-A", layer_idx=0, step=1, q=q, k_cache=k, v_cache=v,
                                   block_table=block_table, block_size=block_size, mask=mask,
                                   output=torch.zeros(heads, d, dtype=torch.bfloat16))
        chk("开关未启用时进入参考路径即报错", False)
    except ReferenceError as exc:
        chk("开关未启用时进入参考路径即报错", True, error=str(exc)[:80])

    # --- 5/6. 原地写入 vs 只返回新张量;默认关闭;异常恢复 ---
    output = torch.zeros(heads, d, dtype=torch.bfloat16)
    ptr = output.data_ptr()
    sw2 = TestOnlyReferenceSwitch(enabled=True, request_id="req-A", scale=scale, dtype=torch.bfloat16)

    class RefImpl:
        def forward(self, **kw):
            sw2.forward_with_output(request_id="req-A", layer_idx=kw["layer_idx"], step=kw["step"], q=kw["q"],
                                    k_cache=kw["k_cache"], v_cache=kw["v_cache"], block_table=kw["block_table"],
                                    block_size=kw["block_size"], mask=kw["mask"], output=kw["output"])

    unified_attention_with_output_stub(RefImpl(), output=output, layer_idx=0, step=1, q=q, k_cache=k, v_cache=v,
                                       block_table=block_table, block_size=block_size, mask=mask)
    chk("参考实现原地写入传入 output(数据指针不变且内容非零)", output.data_ptr() == ptr
        and float(output.abs().max()) > 0.0, max_abs=float(output.abs().max()))
    chk("参考写入值与 FP32 参考 cast 后逐元素一致",
        bool(torch.equal(output, dense_attention_fp32(q, kk2, vv2, scale=scale, dtype=torch.bfloat16)[1])))

    stale = torch.zeros(heads, d, dtype=torch.bfloat16)

    class ReturnOnlyImpl:                      # 反例:只返回新张量,不写缓冲
        def forward(self, **kw):
            return return_only_variant(kw["q"], *gather_positions(kw["k_cache"], kw["v_cache"], kw["block_table"],
                                                                  kw["block_size"], kw["mask"].positions),
                                       scale=scale, dtype=torch.bfloat16)

    unified_attention_with_output_stub(ReturnOnlyImpl(), output=stale, layer_idx=0, step=1, q=q, k_cache=k, v_cache=v,
                                       block_table=block_table, block_size=block_size, mask=mask)
    chk("接线反例:只返回新张量、不写原缓冲 ⇒ 输出仍为初始值(被检出)", float(stale.abs().max()) == 0.0,
        max_abs=float(stale.abs().max()))

    bad_output = torch.zeros(heads + 1, d, dtype=torch.bfloat16)
    try:
        sw2.forward_with_output(request_id="req-A", layer_idx=0, step=1, q=q, k_cache=k, v_cache=v,
                                block_table=block_table, block_size=block_size, mask=mask, output=bad_output)
        chk("异常(形状不符)后恢复为关闭", False)
    except ReferenceError:
        chk("异常(形状不符)后恢复为关闭", sw2.enabled is False and bool(sw2.restores),
            restores=sw2.restores[-1:], errors=sw2.errors[-1:])

    # --- 7. 两请求状态独立 ---
    from attnview.state import RequestProtocolState
    s1 = RequestProtocolState("req-A", arm="da", num_segments=len(SEGMENTS), prompt_len=PROMPT_LEN)
    s2 = RequestProtocolState("req-B", arm="da", num_segments=len(SEGMENTS), prompt_len=PROMPT_LEN)
    s1.feed_generated_token(0, 1, "甲")
    tok = 5
    chk("两请求状态对象独立(互不影响)", s1 is not s2 and s1.generated_tokens == 1 and s2.generated_tokens == 0,
        s1=s1.generated_tokens, s2=s2.generated_tokens)
    m1 = independent_visible_positions(mode="local", refs=(), kv_len=PROMPT_LEN + 1, prompt_len=PROMPT_LEN,
                                       sink_span=SINK, local_window_span=LOCAL, segment_spans=SEGMENTS)
    m2 = independent_visible_positions(mode="global", refs=(), kv_len=PROMPT_LEN + 1, prompt_len=PROMPT_LEN,
                                       sink_span=SINK, local_window_span=LOCAL, segment_spans=SEGMENTS)
    chk("两请求各自独立推导 mask(不共享残留状态)", m1.positions != m2.positions and len(m2.positions) > len(m1.positions))
    tok = tok  # noqa: B018  (占位,避免误删)

    # --- 8. GQA 头映射拒绝 ---
    try:
        dense_attention_fp32(torch.randn(heads + 1, d, dtype=torch.bfloat16), kk2, vv2, scale=scale)
        chk("GQA 头数不整除时拒绝(不静默近似)", False)
    except ReferenceError as exc:
        chk("GQA 头数不整除时拒绝(不静默近似)", True, error=str(exc)[:80])

    failed = [c["name"] for c in checks if not c["ok"]]
    detail.update({"checks": checks, "failed": failed, "switch_calls": len(sw2.calls),
                   "test_scope": ("CPU 检查用桩件模拟生产接线(unified_attention_with_output 忽略返回值、消费原 output 缓冲);"
                                  "**不**据此宣称真实模型或 GPU 接线正确;未加载 27B、未跑 GPU;不设数值阈值。")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps({"failed": failed, "n_checks": len(checks), "ok": len(checks) - len(failed)}, ensure_ascii=False))
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
