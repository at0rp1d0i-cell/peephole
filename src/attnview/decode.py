"""字节级增量解码（worker 侧解析的前置能力）。

为什么需要它：`tokenizer.decode([id])` 对**单个** token 解码，在 UTF-8 字节跨 token 断开时会产出
U+FFFD。本阶段实测 Qwen3.8-27B 词表：248,077 个 token 中有 **953** 个**单独**解码含 U+FFFD
（证据 `evidence/p1-cpu/prompt-facts.json:mixed_text_incremental_decode`）——即"按 token 逐条 decode
再字符串拼接"在一般情形下**不安全**，不能用"只含控制 ASCII 的小表"来绕过：一个 token 的字节可能
同时含正文与协议字符，也可能与相邻 token 合成/断开 UTF-8 序列。

本模块用**字节级缓冲**解决：把 token 还原成原始字节（GPT-2 风格的 byte-level BPE 映射），
只吐出"已经能确定 UTF-8 边界"的部分，未完成的尾字节留在缓冲里等下一个 token。
控制解析只关心 ASCII 标签，但正文可能任意；缓冲策略对两者都成立。
"""

from __future__ import annotations

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


def split_complete_utf8(data: bytes) -> tuple[bytes, bytes]:
    """→ (可确定的完整字节前缀, 未完成的尾字节)。不合法字节按单字节吐出，避免永久卡住。"""
    index = 0
    while index < len(data):
        byte = data[index]
        if byte < 0x80:
            length = 1
        elif 0xC0 <= byte < 0xE0:
            length = 2
        elif 0xE0 <= byte < 0xF0:
            length = 3
        elif 0xF0 <= byte < 0xF8:
            length = 4
        else:
            length = 1  # 非法首字节：当单字节处理
        if index + length > len(data):
            return data[:index], data[index:]
        index += length
    return data, b""


class IncrementalDetokenizer:
    """喂 token id，吐出**完整**文本片段；未完成的 UTF-8 尾字节留在缓冲。"""

    def __init__(self, token_text_of: Callable[[int], str]) -> None:
        self._token_text_of = token_text_of
        self._pending = b""
        self.pending_bytes = 0

    def feed(self, token_id: int) -> str:
        self._pending += token_bytes(self._token_text_of(token_id))
        complete, self._pending = split_complete_utf8(self._pending)
        self.pending_bytes = len(self._pending)
        return complete.decode("utf-8", errors="replace")

    def flush(self) -> str:
        text = self._pending.decode("utf-8", errors="replace")
        self._pending = b""
        self.pending_bytes = 0
        return text

    def feed_all(self, token_ids: Iterator[int]) -> str:
        return "".join(self.feed(i) for i in token_ids) + self.flush()
