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
