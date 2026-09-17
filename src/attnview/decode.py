"""字节级增量解码（worker 侧解析的前置能力）。

为什么需要它：`tokenizer.decode([id])` 对**单个** token 解码，在 UTF-8 字节跨 token 断开时会产出 U+FFFD。
本阶段实测 Qwen3.8-27B 词表：248,077 个 token 中有 **953** 个**单独**解码含 U+FFFD
（证据 `evidence/p1-cpu/prompt-facts.json:mixed_text_incremental_decode`）——即"按 token 逐条 decode
再字符串拼接"在一般情形**不安全**；一个 token 的字节可能同时含正文与协议字符，也可能与相邻 token
合成/断开 UTF-8 序列。

实现方式：把 token 还原成原始字节（GPT-2/Qwen 的 byte-level BPE 映射），交给标准库的
**增量 UTF-8 解码器**（`codecs.getincrementaldecoder("utf-8")(errors="replace")`）管理缓冲。
不自己写"扫前导字节长度"的判定——手写版本不校验续字节，错误前缀会提前吞掉后面合法的 UTF-8 尾段
（反例：字节 `e2 41 e2 82 ac`，手写实现给 `"\\ufffdA\\ufffd\\ufffd\\ufffd"`，标准库给 `"\\ufffdA€"`）。
"""

from __future__ import annotations

import codecs
from typing import Callable, Iterator


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2/Qwen 的 byte→unicode 映射（标准表，用于把 token 字符串还原成原始字节）。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def token_bytes(token_text: str) -> bytes:
    """把 byte-level BPE 的 token 字符串还原为原始字节。"""
    return bytes(UNICODE_TO_BYTE.get(ch, ord(ch)) for ch in token_text)


class IncrementalDetokenizer:
    """喂 token id，吐出**完整**文本片段；未完成的 UTF-8 尾字节由标准库解码器缓冲。"""

    def __init__(self, token_text_of: Callable[[int], str]) -> None:
        self._token_text_of = token_text_of
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        return self._decoder.decode(token_bytes(self._token_text_of(token_id)), final=False)

    def flush(self) -> str:
        """生成结束：把缓冲里未完成的序列按 replace 吐出（不能静默丢弃）。"""
        return self._decoder.decode(b"", final=True)

    def feed_all(self, token_ids: Iterator[int]) -> str:
        return "".join(self.feed(i) for i in token_ids) + self.flush()


def reference_decode(data: bytes) -> str:
    """对照口径：同字节流的整体解码（`errors="replace"`）。"""
    return data.decode("utf-8", errors="replace")
