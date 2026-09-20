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
    """一个可寻址片段。`char_start/char_end` 是 context 原文的字符偏移（0 基半开）。

    `token_count` 用**重叠口径**（凡与片段文本有交集的 token 都算），与 `prompt.py` 把字符区间映射成
    最终 token span 的口径一致——这样"segment 不超上限"与"最终 span 不超上限"不会出现两套计数。
    `contained_token_count` 是完全落在片段内的 token 数，仅作参考；两者之差就是跨边界 token 数。
    """

    index: int  # 1 基，对应协议的 magic chunk 编号
    text: str
    char_start: int
    char_end: int
    token_count: int
    contained_token_count: int = 0
    is_placeholder: bool = False
    cut_mid_token: bool = False
    """本段的切点是否落在 token 内部。**不是豁免理由**，只作为可复核的元信息（上限判定只看覆盖口径）。"""
    cap_exceeded_by_covering: bool = False
    """重叠口径下超过硬上限。只有在 `cut_mid_token` 为真时允许出现，并必须在证据里计数。"""
    uncuttable_over_cap: bool = False
    """唯一的超限豁免：该单元在**任何层级**都没有边界（C1.3 的无空白连续串一类），不可能再切。"""


class OffsetsIndex:
    """context 单次 tokenize 的字符偏移索引；token 计数一律走它（C1.4）。"""

    def __init__(self, offsets: Sequence[tuple[int, int]]) -> None:
        self._starts = [int(s) for s, _ in offsets]
        self._ends = [int(e) for _, e in offsets]
        if any(e < s for s, e in zip(self._starts, self._ends)):
            raise ValueError("offset 非法：存在 end < start")
        if any(b < a for a, b in zip(self._ends, self._ends[1:])):
            raise ValueError("offset 非法：token 的结束位置不是单调不减（重叠口径的二分前提）")

    def __len__(self) -> int:
        return len(self._starts)

    def count(self, start: int, end: int) -> int:
        """**完全落在** [start, end) 内的 token 数（含边界 token 的判定见 `straddling`）。"""
        if end <= start:
            return 0
        first = bisect_left(self._starts, start)
        last = bisect_right(self._ends, end)
        return max(0, last - first)

    def count_overlapping(self, start: int, end: int) -> int:
        """与 [start, end) 有交集的 token 数（**重叠口径**，与最终 span 映射一致）。"""
        if end <= start:
            return 0
        before_end = bisect_left(self._starts, end)  # token 起点 < end
        ends_before_start = bisect_right(self._ends, start)  # token 终点 ≤ start
        return max(0, before_end - ends_before_start)

    def is_token_boundary(self, cut: int) -> bool:
        """cut 是否恰好落在某个 token 的边界上（用于优先在边界处切割）。"""
        return cut in set(self._starts) | set(self._ends)

    def straddling(self, cut: int) -> tuple[int, ...]:
        """跨越切割字符位置 `cut` 的 token 下标（start < cut < end）。

        这些 token 的 token 数记在**任一段之外**（`count` 是包含语义），但最终 prompt span 用
        重叠语义（`prompt.py` 的字符区间→token 区间）会把它们纳入相邻段，属于"只多不少"的安全方向。
        """
        out: list[int] = []
        first = bisect_left(self._starts, cut)
        for i in range(max(0, first - 2), min(len(self._starts), first + 2)):
            if self._starts[i] < cut < self._ends[i]:
                out.append(i)
        return tuple(out)

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(self._starts)

    @property
    def ends(self) -> tuple[int, ...]:
        return tuple(self._ends)


def build_offsets_index(offsets: Iterable[tuple[int, int]]) -> OffsetsIndex:
    """从 `tokenizer(..., return_offsets_mapping=True)` 的 offset 列表构造索引。"""
    return OffsetsIndex(list(offsets))


def _candidate_cuts(
    text: str, start: int, end: int, pattern: re.Pattern[str], *, both_edges: bool = False
) -> list[tuple[int, int]]:
    """候选切点 → [(cut, preference)]，preference 0 = 优先（分隔符**之后**，与其它层级语义一致），1 = 之前。

    只对词间空白层同时评估"空白之前/之后"两个切点：BPE 可能把空格并进相邻 token，
    只看"空白之后"会错过唯一可用的切点（见 `test_cap_check_uses_covering_token_count` 的反例）。
    """
    out: list[tuple[int, int]] = []
    for m in pattern.finditer(text, start, end):
        if start < m.end() < end:
            out.append((m.end(), 0))
        if both_edges and start < m.start() < end:
            out.append((m.start(), 1))
    return out


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

    spans: list[tuple[int, int, bool]] = []
    _split_span(text, 0, len(text), offsets, target, cap, spans)
    segments: list[Segment] = []
    for i, (a, b, uncuttable) in enumerate(spans, start=1):
        token_count = offsets.count_overlapping(a, b)
        mid_token = _cut_mid_token(offsets, [a, b], len(text))
        segments.append(
            Segment(
                index=i,
                text=text[a:b],
                char_start=a,
                char_end=b,
                token_count=token_count,
                contained_token_count=offsets.count(a, b),
                cut_mid_token=mid_token,
                cap_exceeded_by_covering=token_count > cap,
                uncuttable_over_cap=token_count > cap and uncuttable,
            )
        )
    _assert_cap_respected(segments, cap)
    return segments


def _cut_mid_token(offsets: OffsetsIndex, edges: Sequence[int], text_len: int) -> bool:
    """段的切点里是否含有落在 token 内部的（首尾不算切点）。"""
    return any(
        0 < edge < text_len and not offsets.is_token_boundary(edge) for edge in edges
    )


def _assert_cap_respected(segments: Sequence[Segment], cap: int) -> None:
    """硬上限用**重叠口径**判定；**唯一**豁免是 C1.3 的"单元内没有任何层级边界"（如无空白长串）。"""
    for seg in segments:
        if seg.token_count <= cap:
            continue
        if seg.uncuttable_over_cap:
            continue  # C1.3：该单元在任何层级都没有边界，不可能再切
        raise AssertionError(
            f"segment {seg.index} 重叠口径 {seg.token_count} token 超过上限 {cap}，但该单元存在层级边界"
            "——必须修切点选择，不接受任何豁免（含'切点在 token 内部'的说法）"
        )


def _split_span(
    text: str,
    start: int,
    end: int,
    offsets: OffsetsIndex,
    target: int,
    cap: int,
    out: list[tuple[int, int, bool]],
) -> None:
    size = offsets.count_overlapping(start, end)
    if size <= cap:
        out.append((start, end, False))
        return

    # 严格按 C1.2 的层级顺序（由粗到细）；不在层级之间插入"token 边界优先"的额外轮次。
    # 每层内部：优先"左侧可独立成段（≤ cap）"，其次"右侧可独立成段"，再退到"两侧都超也要切"，
    # 同一档内按 (与目标距离, 分隔符前后偏好, 切点位置) 取最优。
    for level, pattern in DELIMITER_LEVELS:
        candidates = _candidate_cuts(
            text, start, end, pattern, both_edges=(level == "whitespace")
        )
        if not candidates:
            continue
        left_best: tuple[int, int, int] | None = None
        right_best: tuple[int, int, int] | None = None
        any_best: tuple[int, int, int] | None = None
        for cut, preference in candidates:
            # 尺寸一律用**重叠口径**（与最终 span 映射、上限判定同一口径）
            left = offsets.count_overlapping(start, cut)
            right = offsets.count_overlapping(cut, end)
            if left <= 0 or right <= 0:
                continue
            score = (abs(left - target), preference, cut)
            if left <= cap:
                if left_best is None or score < left_best:
                    left_best = score
            elif right <= cap:
                # 左侧仍超限（例如无空白长串）：先切出右侧，左侧继续递归，
                # 使原子超限串成为**它自己**的 segment，而不是与后续内容合并
                right_score = (abs(right - target), preference, cut)
                if right_best is None or right_score < right_best:
                    right_best = right_score
            else:
                # 两侧都超 cap：仍然要切（单元超限且此处有边界），两侧各自继续递归
                if any_best is None or score < any_best:
                    any_best = score
        best = left_best or right_best or any_best
        if best is not None:
            cut = best[2]
            _split_span(text, start, cut, offsets, target, cap, out)
            _split_span(text, cut, end, offsets, target, cap, out)
            return

    # 走完所有层级都没有"能切出两个非空段"的切点：保留为超限段（C1.3：无空白连续串一类）。
    # 三档策略下只要存在可用切点就一定会切，因此能到这里就说明该单元确实不可再切。
    out.append((start, end, True))


def join_segments(segments: Sequence[Segment]) -> str:
    """C1.5 复原断言用：顺序拼接应等于原文（占位 segment 除外）。"""
    return "".join(s.text for s in segments if not s.is_placeholder)
