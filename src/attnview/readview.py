"""C4 读取视图构造（纯函数）与后端参数转换（读取视图规范 v0.1 §2–§4）。

链条（规范 §4）：`(mode, declared_spans, token_layout, canonical_block_table) → visible_spans →
visible_blocks → physical_block_ids → (physical_block_ids, valid_counts)`，全程无状态纯函数。

位置约定（与 `state.py` 一致）：

- `kv_len` = 本步 attention 可见的 KV 上界 = `prompt_len + s`（第 `s` 步 forward 消费生成流第 `s-1`
  个 token 并把它写入位置 `prompt_len + s - 1`；该位置的自身 KV 参与本步 attention）。
- 因此 `response_span = (prompt_len, kv_len)` 已包含本步写入的 token，而**写入位置**与读取视图无关。

不变量（规范 §3）：I1 只向外扩、I2 升序去重稳定、I3 canonical 映射与引用计数不受影响、
I4 `-1` 只在尾部且 `valid_counts` 一致、I5 `max_width` 一次生成内不变、I6 尾块按有效 KV 长度、
I7 三区域恒可见、I8 行列双向边界校验。

未采用/未证明项：`-1` 是本项目内部无效槽约定，**不能**据稀疏后端行为推断目标 dense
FlashAttention 会跳过 `-1`（规范 §8.3）——适配层必须保证填充槽不被解引用；本模块只产出对象，
不宣称目标后端可直接消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL

INVALID_SLOT = -1
DEFAULT_SINK_TOKENS = 16


class ReadViewError(RuntimeError):
    """视图构造/参数转换失败（越界、非法块大小等）。一律抛错，不静默截断。"""


@dataclass(frozen=True)
class TokenLayout:
    """请求的 token 位置布局（prompt 侧来自 `prompt.py` 的映射，response 侧逐步增长）。"""

    prompt_len: int
    segment_spans: tuple[tuple[int, int], ...]  # 下标 0 对应 segment 1
    local_window_span: tuple[int, int]  # question 起点 → prompt 末尾
    sink_span: tuple[int, int] = (0, DEFAULT_SINK_TOKENS)

    def segment_span(self, index: int) -> tuple[int, int]:
        if not 1 <= index <= len(self.segment_spans):
            raise ReadViewError(f"segment 编号越界：{index}（共 {len(self.segment_spans)}）")
        return self.segment_spans[index - 1]

    def response_span(self, kv_len: int) -> tuple[int, int]:
        return (self.prompt_len, kv_len)


@dataclass(frozen=True)
class ViewInputs:
    mode: str
    refs: tuple[int, ...]
    layout: TokenLayout
    kv_len: int
    canonical_blocks: tuple[int | None, ...]
    """逻辑块 → 物理块（canonical 映射；`None` = 未分配）。索引即逻辑块号。"""
    kernel_block_size: int
    max_width: int
    """一次生成内常量（I5）。适配层在请求开始时定一次，之后逐 step 复用。"""
    effect_step: int = 0
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ReadView:
    mode: str
    effect_step: int
    fallback_reason: str | None
    sink_span: tuple[int, int]
    local_window_span: tuple[int, int]
    response_span: tuple[int, int]
    declared_refs: tuple[int, ...]
    declared_spans: tuple[tuple[int, int], ...]
    raw_spans: tuple[tuple[int, int], ...]
    """语义可见区间（未做块对齐），已裁剪到 [0, kv_len)。"""
    visible_spans: tuple[tuple[int, int], ...]
    visible_blocks: tuple[int, ...]
    physical_block_ids: tuple[int, ...]
    valid_counts: int
    max_width: int
    kv_len: int
    kernel_block_size: int
    tail_block_valid_len: int
    next_write_position: int
    extras: dict = field(default_factory=dict)

    def positions(self) -> tuple[int, ...]:
        """视图覆盖的全部 token 位置（升序）。"""
        out: list[int] = []
        for start, end in self.visible_spans:
            out.extend(range(start, end))
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "effect_step": self.effect_step,
            "fallback_reason": self.fallback_reason,
            "sink_span": list(self.sink_span),
            "local_window_span": list(self.local_window_span),
            "response_span": list(self.response_span),
            "declared_refs": list(self.declared_refs),
            "declared_spans": [list(s) for s in self.declared_spans],
            "raw_spans": [list(s) for s in self.raw_spans],
            "visible_spans": [list(s) for s in self.visible_spans],
            "visible_blocks": list(self.visible_blocks),
            "physical_block_ids": list(self.physical_block_ids),
            "valid_counts": self.valid_counts,
            "max_width": self.max_width,
            "kv_len": self.kv_len,
            "kernel_block_size": self.kernel_block_size,
            "tail_block_valid_len": self.tail_block_valid_len,
            "next_write_position": self.next_write_position,
            "visible_tokens": len(self.positions()),
        }


def build_read_view(inputs: ViewInputs) -> ReadView:
    """纯函数：同一输入必得同一输出（规范 §4）。"""
    b = int(inputs.kernel_block_size)
    if b <= 0:
        raise ReadViewError(f"kernel_block_size 必须为正：{b}")
    if inputs.kv_len < 0:
        raise ReadViewError(f"kv_len 非法：{inputs.kv_len}")
    if inputs.mode not in (MODE_GLOBAL, MODE_FOCUS, MODE_LOCAL):
        raise ReadViewError(f"未知模式：{inputs.mode}")

    declared_spans: list[tuple[int, int]] = []
    if inputs.mode == MODE_FOCUS:
        for ref in inputs.refs:
            declared_spans.append(inputs.layout.segment_span(ref))

    semantic: list[tuple[int, int]] = [
        inputs.layout.sink_span,
        inputs.layout.local_window_span,
        inputs.layout.response_span(inputs.kv_len),
        *declared_spans,
    ]
    if inputs.mode == MODE_GLOBAL:
        semantic.append((0, inputs.kv_len))

    raw_spans = _merge(_clip(semantic, inputs.kv_len))
    # I1 先向外对齐到 kernel 块边界，再按有效 KV 长度裁回（I6：不读未写满的块外）
    visible_spans = _merge(_clip(_align_outward(raw_spans, b), inputs.kv_len))

    blocks: set[int] = set()
    for start, end in visible_spans:
        blocks.update(range(start // b, (end + b - 1) // b))
    visible_blocks = tuple(sorted(blocks))

    allocated = len(inputs.canonical_blocks)
    for blk in visible_blocks:
        if blk >= allocated:
            raise ReadViewError(
                f"可见块 {blk} 超出已分配块数 {allocated}（I3/I8：读取视图不得引用未分配块）"
            )

    if inputs.max_width < len(visible_blocks):
        raise ReadViewError(
            f"max_width={inputs.max_width} 小于本步有效块数 {len(visible_blocks)}（I5）"
        )

    physical: list[int] = []
    for blk in visible_blocks:
        pid = inputs.canonical_blocks[blk]
        if pid is None:
            raise ReadViewError(f"逻辑块 {blk} 尚未分配物理块（I3）")
        physical.append(int(pid))
    physical.extend([INVALID_SLOT] * (inputs.max_width - len(physical)))

    tail_len = _tail_block_valid_len(inputs.kv_len, b)
    return ReadView(
        mode=inputs.mode,
        effect_step=inputs.effect_step,
        fallback_reason=inputs.fallback_reason,
        sink_span=inputs.layout.sink_span,
        local_window_span=inputs.layout.local_window_span,
        response_span=inputs.layout.response_span(inputs.kv_len),
        declared_refs=tuple(inputs.refs),
        declared_spans=tuple(declared_spans),
        raw_spans=tuple(raw_spans),
        visible_spans=tuple(visible_spans),
        visible_blocks=visible_blocks,
        physical_block_ids=tuple(physical),
        valid_counts=len(visible_blocks),
        max_width=int(inputs.max_width),
        kv_len=int(inputs.kv_len),
        kernel_block_size=b,
        tail_block_valid_len=tail_len,
        next_write_position=int(inputs.kv_len),
    )


def to_kernel_args(view: ReadView, *, req_idx: int, block_table_rows: int, block_table_stride: int) -> dict:
    """适配层边界校验（I8）：行/列双向越界直接抛错，不静默通过。"""
    if not 0 <= req_idx < block_table_rows:
        raise ReadViewError(f"req_idx={req_idx} 越界（0..{block_table_rows - 1}）")
    if view.max_width > block_table_stride:
        raise ReadViewError(
            f"max_width={view.max_width} 超过 block_table stride={block_table_stride}"
        )
    if len(view.physical_block_ids) != view.max_width:
        raise ReadViewError("physical_block_ids 宽度与 max_width 不一致（I5）")
    valid_prefix = view.physical_block_ids[: view.valid_counts]
    if any(pid == INVALID_SLOT for pid in valid_prefix):
        raise ReadViewError("有效前缀里出现无效槽 -1（I4）")
    tail = view.physical_block_ids[view.valid_counts :]
    if not all(pid == INVALID_SLOT for pid in tail):
        raise ReadViewError("尾部填充区含非 -1 槽（I4）")
    return {
        "req_idx": req_idx,
        "physical_block_ids": list(view.physical_block_ids),
        "valid_counts": view.valid_counts,
        "max_width": view.max_width,
        "kv_len": view.kv_len,
        "tail_block_valid_len": view.tail_block_valid_len,
    }


# --- 内部工具 ---------------------------------------------------------------


def _clip(spans: Sequence[tuple[int, int]], kv_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in spans:
        start = max(0, min(start, kv_len))
        end = max(0, min(end, kv_len))
        if end > start:
            out.append((start, end))
    return out


def _align_outward(spans: Sequence[tuple[int, int]], block: int) -> list[tuple[int, int]]:
    """I1：只向外扩到 kernel 块边界；右端可越过 kv_len（尾块由有效长度界定，I6）。"""
    return [((s // block) * block, ((e + block - 1) // block) * block) for s, e in spans]


def _merge(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """I2：升序、去重、合并相邻/重叠（与输入顺序无关）。"""
    if not spans:
        return []
    ordered = sorted(set(spans))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _tail_block_valid_len(kv_len: int, block: int) -> int:
    """I6：最后一个已写块里的有效 token 数（1..block）；kv_len=0 时为 0。"""
    if kv_len <= 0:
        return 0
    return (kv_len - 1) % block + 1
