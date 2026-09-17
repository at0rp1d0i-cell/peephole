"""C1 分段验收（工作单 §4「分段与模板」面的 CPU 部分）。"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview import segmenter  # noqa: E402
from attnview.segmenter import (  # noqa: E402
    CAP_TOKENS,
    EMPTY_CONTEXT_PLACEHOLDER,
    build_offsets_index,
    join_segments,
    segment_context,
)


CHUNK = 4


def offsets_by_words(text: str) -> list[tuple[int, int]]:
    """离线分词器替代品：每个非空白串按 4 字符切成多个 token（模拟子词切分）。

    若整段当作 1 个 token，就测不出"无空白长串"的超限行为，因此必须切块。
    """
    out: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        run = m.group(0)
        for i in range(0, len(run), CHUNK):
            out.append((m.start() + i, m.start() + min(i + CHUNK, len(run))))
    return out


def word_index(text: str):
    return build_offsets_index(offsets_by_words(text))


def para(words: int, *, prefix: str = "p") -> str:
    """有区分度的词（词长可能 >4 字符 → 每个词可能被切成多个 token）。"""
    return " ".join(f"{prefix}{i}" for i in range(words))


def words_text(count: int, word: str = "a") -> str:
    """每个词恰好 1 token 的确定性文本，用于精确的 token 边界断言。"""
    return " ".join([word] * count)


class SegmenterTest(unittest.TestCase):
    def test_lossless_partition_and_cap(self) -> None:
        text = "\n\n".join([para(1500, prefix="a"), para(1500, prefix="b"), para(1500, prefix="c")])
        segs = segment_context(text, word_index(text))
        self.assertEqual(len(segs), 3)
        self.assertEqual(join_segments(segs), text)
        for seg in segs:
            self.assertLessEqual(seg.token_count, CAP_TOKENS)
        # 切割点落在最粗边界：空行段落
        self.assertTrue(segs[0].text.endswith("\n\n"))
        self.assertEqual([s.index for s in segs], [1, 2, 3])

    def test_single_segment_rules_and_boundaries(self) -> None:
        for words, expected in ((2000, 1), (CAP_TOKENS, 1), (CAP_TOKENS + 1, 2)):
            text = words_text(1000) + "\n\n" + words_text(words - 1000)
            idx = word_index(text)
            self.assertEqual(idx.count(0, len(text)), words)  # 夹具自检：每个词 1 token
            segs = segment_context(text, idx)
            self.assertEqual(len(segs), expected, f"words={words}")

    def test_whitespace_free_run_is_atomic(self) -> None:
        blob = "A" * 12000  # 4 字符/token → 3000 token，超过硬上限
        text = f"{para(600, prefix='h')}\n\n{blob}\n\n{para(600, prefix='t')}"
        segs = segment_context(text, word_index(text))
        atomic = [s for s in segs if blob in s.text]
        self.assertEqual(len(atomic), 1)
        self.assertGreater(atomic[0].token_count, CAP_TOKENS)
        self.assertEqual(atomic[0].text.strip(), blob)  # 无空白长串成为**自己**的 segment
        self.assertEqual(join_segments(segs), text)

    def test_coarsest_boundary_preferred_over_finer(self) -> None:
        text = "\n\n".join([para(1400, prefix="a"), para(1400, prefix="b")])
        segs = segment_context(text, word_index(text))
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].char_end, text.index("\n\n") + 2)

    def test_multibyte_and_emoji_intact(self) -> None:
        unit = "段落内容测试🙂，包含多字节字符；"
        text = ("\n\n".join(unit * 300 for _ in range(4)))
        segs = segment_context(text, word_index(text))
        self.assertEqual(join_segments(segs), text)
        for seg in segs:
            seg.text.encode("utf-8").decode("utf-8")
            self.assertIn(seg.text, text)

    def test_repeated_fragments_still_exact_partition(self) -> None:
        block = " ".join(["same"] * 900)
        text = "\n\n".join([block] * 4)
        segs = segment_context(text, word_index(text))
        self.assertEqual(join_segments(segs), text)
        self.assertGreaterEqual(len(segs), 2)

    def test_empty_and_whitespace_only(self) -> None:
        for text in ("", "   \n\t  "):
            segs = segment_context(text, word_index(text))
            self.assertEqual(len(segs), 1)
            self.assertTrue(segs[0].is_placeholder)
            self.assertEqual(segs[0].text, EMPTY_CONTEXT_PLACEHOLDER)
            self.assertEqual(join_segments(segs), "")

    def test_both_sides_over_cap_still_cuts_at_coarse_boundary(self) -> None:
        """两侧都超 cap 时仍要先沿最粗边界递归；不能当成"无边界原子串"。"""
        left, right = "a" * 12000, "b" * 12000  # 各 3000 token（4 字符/token）
        text = f"{left} {right}"
        idx = word_index(text)
        self.assertEqual(idx.count(0, len(text)), 6000)
        segs = segment_context(text, idx)
        self.assertEqual(len(segs), 2, "必须在中间空白处切开，两侧各自继续递归")
        self.assertEqual(segs[0].text.strip(), left)
        self.assertEqual(segs[1].text.strip(), right)
        self.assertEqual(join_segments(segs), text)
        self.assertGreater(segs[0].token_count, CAP_TOKENS)
        self.assertGreater(segs[1].token_count, CAP_TOKENS)

    def test_multi_level_recursion_with_mixed_units(self) -> None:
        """粗边界 + 长原子单元混合：先按段落切，段落内的超限原子单元保持完整。"""
        text = "\n\n".join([
            words_text(1500),
            "c" * 12000,
            words_text(1500),
            "d" * 12000,
        ])
        idx = word_index(text)
        segs = segment_context(text, idx)
        self.assertEqual(join_segments(segs), text)
        for marker in ("c" * 12000, "d" * 12000):
            holders = [s for s in segs if marker in s.text]
            self.assertEqual(len(holders), 1)
            self.assertEqual(holders[0].text.strip(), marker, "原子单元必须保持完整")
        for seg in segs:
            self.assertIn(seg.text, text)

    def test_cap_holds_by_covering_count_on_adversarial_fixture(self) -> None:
        """advisor 反例：`'ab cd ef'` + offsets [(0,1),(1,4),(4,5),(5,8)] + cap=2。

        要求：用**覆盖口径**判定上限、**所有段 ≤ cap**、精确复原；不接受任何豁免
        （空白层要同时评估空白前后的切点：切在 3 与 5 得 ['ab ','cd',' ef'] = 2/2/1）。
        """
        idx = build_offsets_index([(0, 1), (1, 4), (4, 5), (5, 8)])
        segs = segment_context("ab cd ef", idx, target=2, cap=2)
        self.assertTrue(all(s.token_count <= 2 for s in segs), [s.token_count for s in segs])
        self.assertFalse(any(s.cap_exceeded_by_covering for s in segs))
        self.assertFalse(any(s.uncuttable_over_cap for s in segs))
        self.assertEqual(join_segments(segs), "ab cd ef")
        self.assertEqual("".join(s.text for s in segs), "ab cd ef")
        # 不丢不重：段内含 token + 跨切点 token = 全部 token
        self.assertEqual(
            sum(s.contained_token_count for s in segs)
            + len(idx.straddling(3)) + len(idx.straddling(5)),
            len(idx),
        )

    def test_cap_verified_by_independent_overlap_count(self) -> None:
        """R2：上限必须由**独立 overlap 计数**复核（不依赖被测对象的字段）。"""
        offsets = [(0, 1), (1, 4), (4, 5), (5, 8), (8, 9)]
        idx = build_offsets_index(offsets)
        segs = segment_context("ab cd efg", idx, target=2, cap=2)

        def covering(start: int, end: int) -> int:  # 独立实现：直接扫原始 offsets
            return sum(1 for a, b in offsets if a < end and b > start)

        for seg in segs:
            if seg.uncuttable_over_cap:
                continue
            self.assertLessEqual(
                covering(seg.char_start, seg.char_end), 2,
                f"segment {seg.index} 文本={seg.text!r} 独立计数超限",
            )
        self.assertEqual(join_segments(segs), "ab cd efg")

    def test_coarse_boundary_through_token_is_not_demoted(self) -> None:
        """R2：最粗边界（空行）即使穿过 token 也必须用它，不得跳到更细层级。"""
        text = "ab\n\ncd ef"
        offsets = [(0, 1), (1, 5), (5, 6), (6, 7), (7, 9)]  # token (1,5) 覆盖了空行末尾
        idx = build_offsets_index(offsets)
        self.assertFalse(idx.is_token_boundary(4), "夹具要求：空行切点 4 落在 token 内部")
        segs = segment_context(text, idx, target=2, cap=2)
        self.assertEqual(segs[0].char_end, 4, "必须切在最粗边界（空行之后），不能跳过它")
        self.assertEqual(join_segments(segs), text)
        for seg in segs:
            if not seg.uncuttable_over_cap:
                self.assertLessEqual(seg.token_count, 2, seg.text)

    def test_boundary_cut_is_preferred_and_cap_holds(self) -> None:
        """存在 token 边界切点时必须选它：任何一段都不得超限。"""
        idx = build_offsets_index([(0, 4), (4, 5), (5, 9), (9, 10), (10, 14)])
        segs = segment_context("aaaa bbbb cccc dddd", idx, target=2, cap=2)
        self.assertTrue(all(s.token_count <= 2 for s in segs), [s.token_count for s in segs])
        self.assertFalse(any(s.cut_mid_token for s in segs))
        self.assertFalse(any(s.cap_exceeded_by_covering for s in segs))
        self.assertEqual(join_segments(segs), "aaaa bbbb cccc dddd")

    def test_cap_exceedance_only_for_units_without_boundary(self) -> None:
        """每字符一个 token、cap=1 时 `a*3000+空格+b*3000`：两个长原子串是 C1.3 的合法超限段。"""
        text = "a" * 3000 + " " + "b" * 3000
        idx = build_offsets_index([(i, i + 1) for i in range(len(text))])
        segs = segment_context(text, idx, target=1, cap=1)
        self.assertEqual([s.token_count for s in segs], [3000, 1, 3000])
        self.assertEqual([s.uncuttable_over_cap for s in segs], [True, False, True])
        self.assertEqual(join_segments(segs), text)
        self.assertTrue(all(s.token_count <= 1 or s.uncuttable_over_cap for s in segs))

    def test_crossing_tokens_are_accounted_and_reported(self) -> None:
        """跨切割位置的 token 既不计入任何一段的 token 数，也不丢：可查询且最终 span 用重叠语义。"""
        text = "aa bb cc dd"
        offsets = [(0, 2), (2, 5), (5, 8), (8, 12)]  # (2,5) 跨过 cut=3
        idx = build_offsets_index(offsets)
        self.assertEqual(idx.count(0, 3), 1)
        self.assertEqual(idx.count(3, 12), 2)
        self.assertEqual(idx.straddling(3), (1,))
        self.assertEqual(
            idx.count(0, 3) + idx.count(3, 12) + len(idx.straddling(3)),
            len(offsets),
            "包含语义 + 跨边界 token = 全部 token，不丢不重",
        )

    def test_no_whitespace_at_all_is_atomic(self) -> None:
        text = "B" * 12000
        segs = segment_context(text, word_index(text))
        self.assertEqual(len(segs), 1)
        self.assertGreater(segs[0].token_count, CAP_TOKENS)
        self.assertEqual(segments_text(segs), text)

    def test_segment_offsets_are_原文子串(self) -> None:
        text = "\n\n".join([para(1200, prefix=f"a{i}") for i in range(4)])
        segs = segment_context(text, word_index(text))
        for seg in segs:
            self.assertEqual(text[seg.char_start : seg.char_end], seg.text)


def segments_text(segs) -> str:
    return "".join(s.text for s in segs)


if __name__ == "__main__":
    unittest.main()
