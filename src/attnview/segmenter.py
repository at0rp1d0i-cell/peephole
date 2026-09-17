"""C1 输入侧分段（依据 `docs/protocol-contract.md` §2，v1.0 已冻结）。

设计约束（合同条款号对应）：

- C1.1 目标 2048 token，硬上限 2560 token；**只切超过上限的单元**。
- C1.2 切割位置取该单元内最粗可用边界，层级由粗到细：空行段落 → 单换行 → 句末 → 从句末 → 词间空白。
- C1.3 无空白连续串原子：不切开，可作为自身的超限 segment。
- C1.4 全程在字符偏移空间工作：对 context 只做一次带 offset mapping 的 tokenize，
       用偏移按 token 数衡量候选 span，segment 以原文真实子串输出。
- C1.5 segment 必须是 context 的精确无损划分。
- C1.6 不超上限即单个 segment；空/纯空白 context 渲染为 `<empty_context>` 占位，编号 1..N。

本模块不依赖任何 tokenizer 实现：调用方传入 context 的 (start, end) 字符偏移序列
（`build_offsets_index`），token 计数完全通过这些偏移完成，便于 CPU 单测与独立参考对照。

合同未规定的部分（本项目默认值，报告中列为待决项，不静默改变合同）：
- 同一层有多个候选边界时，选「左段 token 数与目标 2048 最接近」的那个；仍然只在超过上限时切割。
- 层级集合不含 CJK 句末标点（`。！？；：，`）：合同 §C1.2 只列 `. ! ?` / `; : ,`，扩展需在合同层新增条目。
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

TARGET_TOKENS = 2048
CAP_TOKENS = 2560
EMPTY_CONTEXT_PLACEHOLDER = "<empty_context>"

#: 层级由粗到细；每层的匹配「结束位置」是切割点（分隔符归左段）。
DELIMITER_LEVELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("blank_line", re.compile(r"\n[ \t]*\n")),
    ("newline", re.compile(r"\n")),
    ("sentence", re.compile(r"[.!?](?=\s)")),
    ("clause", re.compile(r"[;:,](?=\s)")),
    ("whitespace", re.compile(r"\s+")),
)


@dataclass(frozen=True)
class Segment:
    """一个可寻址片段。`char_start/char_end` 是 context 原文的字符偏移（0 基半开）。"""

    index: int  # 1 基，对应协议的 magic chunk 编号
    text: str
    char_start: int
    char_end: int
    token_count: int
    is_placeholder: bool = False


class OffsetsIndex:
    """context 单次 tokenize 的字符偏移索引；token 计数一律走它（C1.4）。"""

    def __init__(self, offsets: Sequence[tuple[int, int]]) -> None:
        self._starts = [int(s) for s, _ in offsets]
        self._ends = [int(e) for _, e in offsets]
        if any(e < s for s, e in zip(self._starts, self._ends)):
            raise ValueError("offset 非法：存在 end < start")

    def __len__(self) -> int:
        return len(self._starts)

    def count(self, start: int, end: int) -> int:
        """完全落在 [start, end) 内的 token 数。"""
        if end <= start:
            return 0
        first = bisect_left(self._starts, start)
        last = bisect_right(self._ends, end)
        # token 的 end 必须 ≤ end 且 start ≥ start
        return max(0, last - first)

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(self._starts)

    @property
    def ends(self) -> tuple[int, ...]:
        return tuple(self._ends)


def build_offsets_index(offsets: Iterable[tuple[int, int]]) -> OffsetsIndex:
    """从 `tokenizer(..., return_offsets_mapping=True)` 的 offset 列表构造索引。"""
    return OffsetsIndex(list(offsets))


def _candidate_cuts(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[int]:
    return [m.end() for m in pattern.finditer(text, start, end) if start < m.end() < end]


def segment_context(
    text: str,
    offsets: OffsetsIndex,
    *,
    target: int = TARGET_TOKENS,
    cap: int = CAP_TOKENS,
) -> list[Segment]:
    """把 context 切成 1..N 个精确无损片段（C1.1–C1.6）。"""
    if text.strip() == "":
        return [
            Segment(
                index=1,
                text=EMPTY_CONTEXT_PLACEHOLDER,
                char_start=0,
                char_end=len(text),
                token_count=offsets.count(0, len(text)),
                is_placeholder=True,
            )
        ]

    spans: list[tuple[int, int]] = []
    _split_span(text, 0, len(text), offsets, target, cap, spans)
    return [
        Segment(
            index=i,
            text=text[a:b],
            char_start=a,
            char_end=b,
            token_count=offsets.count(a, b),
        )
        for i, (a, b) in enumerate(spans, start=1)
    ]


def _split_span(
    text: str,
    start: int,
    end: int,
    offsets: OffsetsIndex,
    target: int,
    cap: int,
    out: list[tuple[int, int]],
) -> None:
    size = offsets.count(start, end)
    if size <= cap:
        out.append((start, end))
        return

    for _, pattern in DELIMITER_LEVELS:
        left_best: tuple[int, int] | None = None  # (distance, cut)，左侧可独立成段（≤ cap）
        right_best: tuple[int, int] | None = None  # (distance, cut)，右侧可独立成段
        for cut in _candidate_cuts(text, start, end, pattern):
            left = offsets.count(start, cut)
            right = size - left
            if left <= 0 or right <= 0:
                continue
            if left <= cap:
                distance = abs(left - target)
                if left_best is None or (distance, cut) < left_best:
                    left_best = (distance, cut)
            elif right <= cap:
                # 左侧仍超限（例如无空白长串）：允许先切出右侧，左侧继续递归；
                # 这样原子超限串会成为**它自己**的 segment，而不是与后续内容合并
                distance = abs(right - target)
                if right_best is None or (distance, cut) < right_best:
                    right_best = (distance, cut)
        best = left_best or right_best
        if best is not None:
            cut = best[1]
            _split_span(text, start, cut, offsets, target, cap, out)
            _split_span(text, cut, end, offsets, target, cap, out)
            return

    # 每层都没有可用边界：原子超限 segment，绝不切开（C1.3）
    out.append((start, end))


def join_segments(segments: Sequence[Segment]) -> str:
    """C1.5 复原断言用：顺序拼接应等于原文（占位 segment 除外）。"""
    return "".join(s.text for s in segments if not s.is_placeholder)
