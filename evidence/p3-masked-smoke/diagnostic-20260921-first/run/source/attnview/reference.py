"""读取视图的**独立参考**（读取视图规范 §6 的对照面）。

刻意的独立性：本模块不 import `readview`，也不使用它的 `_merge/_clip/_align_outward`；
它直接从"三种模式各自可见什么"的文字定义出发，逐 token 位置构造布尔可见集合，再把
"向外对齐到块边界"表述为"某一位置可见 ⇒ 其所在整块可见"。因此 `readview` 的期望值不是
由被测构造器生成的，两者对照才有意义（方案 §4「独立参考」要求）。

参考只证明索引与语义正确（哪些位置在集合里），不证明 attention/logits 的真实数值——后者
需要 GPU 上的 dense masked reference 对照（缺口 R03 保持打开）。
"""

from __future__ import annotations

from typing import Sequence

from .parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL
from .readview import TokenLayout


def reference_visible_mask(
    *,
    mode: str,
    refs: Sequence[int],
    layout: TokenLayout,
    attention_kv_len: int,
    kernel_block_size: int,
) -> list[bool]:
    """逐位置布尔 mask（长度 `attention_kv_len`），True = 本步可见。"""
    if kernel_block_size <= 0:
        raise ValueError(f"kernel_block_size 必须为正：{kernel_block_size}")
    if mode not in (MODE_GLOBAL, MODE_FOCUS, MODE_LOCAL):
        raise ValueError(f"未知模式：{mode}")

    selected = [False] * attention_kv_len

    def mark(start: int, end: int) -> None:
        for pos in range(max(0, start), min(end, attention_kv_len)):
            selected[pos] = True

    # 三区域恒可见（附录 B）：sink、question+instruction 本地窗口、已生成 response
    mark(*layout.sink_span)
    mark(*layout.local_window_span)
    mark(layout.prompt_len, attention_kv_len)

    if mode == MODE_GLOBAL:
        mark(0, attention_kv_len)
    elif mode == MODE_FOCUS:
        for ref in refs:
            mark(*layout.segment_span(ref))

    # 向外对齐：位置可见 ⇒ 其所在块整体可见（这正是 I1 的逐位置表述）
    visible = [False] * attention_kv_len
    for pos, is_selected in enumerate(selected):
        if not is_selected:
            continue
        block_start = (pos // kernel_block_size) * kernel_block_size
        for p in range(block_start, min(block_start + kernel_block_size, attention_kv_len)):
            visible[p] = True
    return visible


def reference_visible_positions(**kwargs) -> tuple[int, ...]:
    mask = reference_visible_mask(**kwargs)
    return tuple(i for i, v in enumerate(mask) if v)


def declared_positions(
    layout: TokenLayout, refs: Sequence[int], attention_kv_len: int
) -> tuple[int, ...]:
    """被声明点名的原始位置（未对齐），用于 I1「一个都不丢」的断言。"""
    out: list[int] = []
    for ref in refs:
        start, end = layout.segment_span(ref)
        out.extend(range(max(0, start), min(end, attention_kv_len)))
    return tuple(out)
