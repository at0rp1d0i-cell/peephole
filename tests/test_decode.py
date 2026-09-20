"""增量解码与真实 tokenizer 的混排文本验收（真实 tokenizer，仅 CPU）。"""

from __future__ import annotations

import unittest

from _support import REPO
from attnview.decode import (  # noqa: E402
    IncrementalDetokenizer,
    bytes_to_unicode,
    reference_decode,
    token_bytes,
)

TOKENIZER_DIR = (
    REPO
    / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)

MIXED = (
    "第一段：平台评审 2026 年完成 🙂 <global>look</global> "
    '<focus magic_chunks="1">费用 41 万元</focus><local>确认</local><answer>41 万元</answer>'
)


def _expected_continuations(pending: bytes) -> set[int]:
    """给定未完成的 UTF-8 前导字节，返回合法的下一个续字节集合。"""
    first = pending[0]
    if 0xC2 <= first <= 0xDF:
        return set(range(0x80, 0xC0))
    if 0xE0 <= first <= 0xEF:
        return set(range(0x80, 0xC0))
    if 0xF0 <= first <= 0xF4:
        return set(range(0x80, 0xC0))
    return set(range(0x80, 0xC0))


class IncrementalDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
        cls.decoder = IncrementalDetokenizer(cls.tokenizer.convert_ids_to_tokens)

    def test_mixed_text_round_trip(self) -> None:
        ids = self.tokenizer(MIXED, add_special_tokens=False)["input_ids"]
        self.assertEqual(self.decoder.feed_all(iter(ids)), MIXED)
        self.assertGreater(len(ids), 5)

    def test_naive_per_token_decode_is_unsafe_in_general(self) -> None:
        """结论：naive 逐 token 解码在一般情形不成立（用 pin 的固定反例证明，不扫全词表）。

        全词表扫描结果作为一次性证据记录在 `evidence/p1-cpu/prompt-facts.json`
        （953/248077 个 token 单独解码含 U+FFFD）。
        """


    #: 固定反例（Qwen3.8-27B 词表，已实测）：lead 字节 `eb 94`（3 字节 UTF-8 的前两字节），
    #: 由 follow 的续字节 `bd` 补全为 U+B53D；单 token 解码两者都是 U+FFFD。
    PINNED_SPLIT_PAIR = (64253, 121)

    def test_pinned_split_pair_naive_vs_incremental(self) -> None:
        lead, follow = self.PINNED_SPLIT_PAIR
        self.assertEqual(token_bytes(self.tokenizer.convert_ids_to_tokens(lead)), b"\xeb\x94")
        self.assertEqual(token_bytes(self.tokenizer.convert_ids_to_tokens(follow)), b"\xbd")
        self.assertIn("\ufffd", self.tokenizer.decode([lead]))
        naive = self.tokenizer.decode([lead]) + self.tokenizer.decode([follow])
        incremental = IncrementalDetokenizer(self.tokenizer.convert_ids_to_tokens).feed_all(
            iter([lead, follow])
        )
        joint = self.tokenizer.decode([lead, follow])
        self.assertEqual(naive, "\ufffd\ufffd", "naive 逐 token 拼接的确定性反例")
        self.assertEqual(incremental, joint)
        self.assertNotIn("\ufffd", incremental)

    def test_invalid_prefix_does_not_swallow_later_valid_sequence(self) -> None:
        """反例（R1 第二轮）：字节 `e2 41 e2 82 ac` 按单字节 token 喂入。

        手写"只看前导字节长度"的实现会给 `"\ufffdA\ufffd\ufffd\ufffd"`（错误前缀提前吃掉后面的合法尾段）；
        标准库增量解码给 `"\ufffdA€"`，与整体解码一致。
        """
        by = bytes_to_unicode()

        def one_byte_token(byte: int) -> str:
            return by[byte]

        payload = bytes([0xE2, 0x41, 0xE2, 0x82, 0xAC])
        decoder = IncrementalDetokenizer(lambda i: one_byte_token(i))
        got = "".join(decoder.feed(i) for i in payload) + decoder.flush()
        self.assertEqual(got, reference_decode(payload))
        self.assertEqual(got, "\ufffdA€")

    def test_incomplete_tail_is_flushed_not_dropped(self) -> None:
        """结尾未完成的 UTF-8 序列必须在 flush 时按 replace 吐出，不得静默丢弃。"""
        by = bytes_to_unicode()
        payload = bytes([0x41, 0xE2, 0x82])
        decoder = IncrementalDetokenizer(lambda i: by[i])
        streamed = "".join(decoder.feed(i) for i in payload)
        tail = decoder.flush()
        self.assertEqual(streamed, "A")
        self.assertEqual(tail, "\ufffd")
        self.assertEqual(streamed + tail, reference_decode(payload))

    def test_incremental_decoder_repairs_real_split_sequence(self) -> None:
        """在真实混排文本里，凡是被跨 token 拆开的字符，增量解码都能还原。"""
        ids = self.tokenizer(MIXED, add_special_tokens=False)["input_ids"]
        decoder = IncrementalDetokenizer(self.tokenizer.convert_ids_to_tokens)
        pieces = [decoder.feed(i) for i in ids]
        pieces.append(decoder.flush())
        self.assertEqual("".join(pieces), MIXED)
        self.assertGreaterEqual(sum(len(p) for p in pieces if p), len(MIXED) - 1)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
