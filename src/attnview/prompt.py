"""C2 prompt 组装与最终 token-span 映射（SegmentMap 契约）。

分工：`segmenter.py` 负责原文→片段（字符偏移空间）；本模块负责片段→**最终 chat template 渲染后**的
token 半开区间，因为合同 C1.7 要求 span 以最终渲染与 tokenization 结果确定，字符偏移不能直接当 token 下标。

定位方法（不使用"第一次字符串匹配"，避免重复文本误定位）：
1. 逐前缀渲染 `apply_chat_template(messages[:k+1], add_generation_prompt=False)`，并用
   `full.startswith(prefix_k)` 做结构性断言；每个 message 的块区间 = 相邻前缀长度之差。
2. 在**块内部**定位内容文本，要求块内出现次数恰为 1，否则抛错（不静默取第一个）。
3. 字符区间 → token 区间用 offset 重叠规则（完全包含边界字符的 token 都算入），保证不丢 token。

模板会 `|trim` 消息内容：tool response 里正文首尾空白被裁掉，因此定位时用 `rstrip()` 后的文本，
并把这一变换记录在证据里（不改写 segment 本身）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .prompts import (
    BOOTSTRAP_USER_TURN,
    DA_INSTRUCTION_PROMPT,
    GET_MAGIC_CHUNK_TOOL,
    MAGIC_CHUNK_HEADER,
    SYSTEM_INSTRUCTION,
    VANILLA_INSTRUCTION_PROMPT,
)
from .segmenter import Segment

ARMS = ("vanilla", "da_no_mask", "da")
SINK_TOKENS = 16


class SpanMappingError(RuntimeError):
    """定位/映射失败。永远抛错，不退化成近似定位。"""


@dataclass(frozen=True)
class Scaffold:
    """prompt 侧固定区域在最终 token 序列中的位置（C4.2 的 scaffold 语义）。"""

    sink_span: tuple[int, int]
    question_span: tuple[int, int]
    """question 起点到 prompt 末尾（含 DA 指令）——local window 的 prompt 侧部分。"""
    local_window_span: tuple[int, int]
    system_content_span: tuple[int, int]
    sink_decoded: tuple[str, ...]
    sink_message_role: str

    @property
    def sink_inside_system_content(self) -> bool:
        """前 16 token 是否**完全**落在固定的 system 指令正文上（C4.2 的字面要求）。"""
        return (
            self.sink_span[0] >= self.system_content_span[0]
            and self.sink_span[1] <= self.system_content_span[1]
        )


@dataclass(frozen=True)
class ArmPrompt:
    arm: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    rendered: str
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    segment_spans: tuple[tuple[int, int], ...]  # 下标 0 对应 segment 1
    question_span: tuple[int, int]
    scaffold: Scaffold
    tokenizer_hash: str
    template_hash: str
    enable_thinking: bool | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids)

    def segment_span(self, index: int) -> tuple[int, int]:
        if not 1 <= index <= len(self.segment_spans):
            raise SpanMappingError(f"segment 编号越界：{index}（共 {len(self.segment_spans)}）")
        return self.segment_spans[index - 1]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_messages(
    arm: str,
    segments: Sequence[Segment],
    question: str,
    context_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """三臂共享同一 question；DA/no-mask 用同一份 DA prompt，vanilla 用 Vanilla prompt + 内联 context。"""
    if arm not in ARMS:
        raise ValueError(f"未知臂：{arm}")

    if arm == "vanilla":
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": VANILLA_INSTRUCTION_PROMPT.format(context=context_text, question=question),
            },
        ]
        return messages, None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": BOOTSTRAP_USER_TURN},
    ]
    for seg in segments:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_magic_chunk", "arguments": {"chunk": seg.index}},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": MAGIC_CHUNK_HEADER.format(n=seg.index) + "\n" + seg.text,
            }
        )
    messages.append({"role": "user", "content": DA_INSTRUCTION_PROMPT.format(question=question)})
    return messages, [GET_MAGIC_CHUNK_TOOL]


def render_arm(
    arm: str,
    segments: Sequence[Segment],
    question: str,
    context_text: str,
    tokenizer: Any,
    *,
    tokenizer_hash: str = "",
    template_hash: str = "",
    enable_thinking: bool | None = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> ArmPrompt:
    """渲染一个臂并映射出 segment / scaffold 的最终 token 区间。"""
    messages, tools = build_messages(arm, segments, question, context_text)
    extra = dict(chat_template_kwargs or {})
    if enable_thinking is not None:
        extra["enable_thinking"] = enable_thinking

    def apply(msgs: list[dict[str, Any]], generation_prompt: bool) -> str:
        # 注意：transformers 5.17 的 apply_chat_template 把额外 kwargs **直接**转给模板；
        # 传 `chat_template_kwargs={...}` 会被静默忽略（实测：thinking 仍开着）。
        # 证据见 evidence/p1-cpu/template-kwargs-check.txt。
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=generation_prompt,
            tools=tools,
            **extra,
        )
        return tokenizer.apply_chat_template(msgs, **kwargs)

    full = apply(messages, True)

    def prefix_for(k: int) -> str:
        """第 0 条消息（system）的块单独构造：模板在"没有任何 user 消息"时会 raise
        （`No user query found in messages`）；且 tools 分支无论首条是否为 system 都会输出 system 块，
        所以不能靠 [m1] 的渲染做差。改用第 1 条消息的已知包装构成反推。"""
        if k == 0 and len(messages) > 1:
            second = messages[1]
            if second["role"] != "user":
                raise SpanMappingError("首块构造假设第 2 条消息是 user（协议 prompt 结构）")
            block = "<|im_start|>user\n" + str(second["content"]).strip() + "<|im_end|>\n"
            after_first = apply(messages[:2], False)
            if not after_first.endswith(block):
                raise SpanMappingError("第 1 条消息的渲染与预期包装不一致，拒绝近似构造")
            return after_first[: len(after_first) - len(block)]
        return apply(messages[: k + 1], False)

    prefixes = [prefix_for(k) for k in range(len(messages))]
    for k, prefix in enumerate(prefixes):
        if not full.startswith(prefix):
            raise SpanMappingError(
                f"模板渲染不是前缀可组合的（消息 {k}）；拒绝用字符串近似定位"
            )

    notes: list[str] = []
    block_bounds: list[tuple[int, int]] = []
    prev = 0
    for prefix in prefixes:
        block_bounds.append((prev, len(prefix)))
        prev = len(prefix)

    # 逐条消息在块内定位内容
    located: list[tuple[int, int] | None] = []
    for idx, msg in enumerate(messages):
        start, end = block_bounds[idx]
        block = full[start:end]
        if msg["role"] == "tool":
            seg_i = (idx - 3) // 2
            needle = MAGIC_CHUNK_HEADER.format(n=segments[seg_i].index)
            header_at = _locate_unique(block, needle, what=f"tool response 头 {needle!r}")
            text = msg["content"][len(needle) + 1 :]
            stripped = text.rstrip()
            if stripped != text:
                notes.append(
                    f"模板 |trim 裁掉了 segment {seg_i + 1} 的尾部空白 "
                    f"{len(text) - len(stripped)} 字符（定位按裁剪后文本，segment 本体不改）"
                )
            seg_start = header_at + len(needle) + 1
            if block[seg_start : seg_start + len(stripped)] != stripped:
                raise SpanMappingError(f"segment {seg_i + 1} 正文在渲染结果中不连续")
            located.append((seg_start + start, seg_start + len(stripped) + start))
        elif msg["role"] == "system":
            body = msg["content"].strip()
            at = _locate_unique(block, body, what="system 内容")
            located.append((at + start, at + len(body) + start))
        elif msg["role"] == "user" and idx == len(messages) - 1:
            # 末轮 user = DA prompt：question 与 local window 起点都按该块定位
            q_start = _locate_unique(block, question, what="DA prompt 中的 question")
            located.append((q_start + start, len(block) + start))
        else:
            located.append(None)

    token_ids, offsets = _tokenize_with_offsets(tokenizer, full)

    segment_spans: list[tuple[int, int]] = []
    if arm != "vanilla":
        for seg_i in range(len(segments)):
            msg_idx = 3 + 2 * seg_i
            span = located[msg_idx]
            if span is None:
                raise SpanMappingError(f"未能定位 segment {seg_i + 1} 的 tool response")
            segment_spans.append(_char_range_to_token_range(offsets, *span))

    if arm == "vanilla":
        question_span = (0, 0)
        for idx, msg in enumerate(messages):
            if msg["role"] == "user":
                at = _locate_unique(full[block_bounds[idx][0] : block_bounds[idx][1]], question, what="question")
                question_start = at + block_bounds[idx][0]
                question_span = _char_range_to_token_range(offsets, question_start, question_start + len(question))
        local_window_span = question_span
    else:
        span = located[-1]
        if span is None:
            raise SpanMappingError("未能定位 DA prompt")
        question_span = _char_range_to_token_range(offsets, span[0], span[0] + len(question))
        local_window_span = _char_range_to_token_range(offsets, *span)

    system_span = _char_range_to_token_range(offsets, *(located[0] or (0, 0)))
    sink_span = (0, min(SINK_TOKENS, len(token_ids)))
    sink_message_role = _role_of_token_span(block_bounds, offsets, sink_span, messages)

    scaffold = Scaffold(
        sink_span=sink_span,
        question_span=question_span,
        local_window_span=local_window_span,
        system_content_span=system_span,
        sink_decoded=tuple(tokenizer.decode([t]) for t in token_ids[:SINK_TOKENS]),
        sink_message_role=sink_message_role,
    )

    return ArmPrompt(
        arm=arm,
        messages=messages,
        tools=tools,
        rendered=full,
        token_ids=tuple(token_ids),
        offsets=tuple(offsets),
        segment_spans=tuple(segment_spans),
        question_span=question_span,
        scaffold=scaffold,
        tokenizer_hash=tokenizer_hash,
        template_hash=template_hash,
        enable_thinking=enable_thinking,
        notes=tuple(notes),
    )


def _locate_unique(haystack: str, needle: str, *, what: str) -> int:
    if not needle:
        raise SpanMappingError(f"{what}：空 needle")
    count = haystack.count(needle)
    if count != 1:
        raise SpanMappingError(f"{what}：块内出现 {count} 次（要求恰为 1 次），拒绝近似定位")
    return haystack.index(needle)


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise SpanMappingError("input_ids 与 offset_mapping 长度不一致")
    return ids, offsets


def _char_range_to_token_range(
    offsets: Sequence[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """重叠规则：与字符区间有交集的 token 全部算入（不丢 token，向外的选择）。"""
    if char_end <= char_start:
        raise SpanMappingError(f"非法字符区间 [{char_start}, {char_end})")
    lo = None
    hi = None
    for i, (s, e) in enumerate(offsets):
        if e <= char_start or s >= char_end:
            continue
        lo = i if lo is None else lo
        hi = i
    if lo is None or hi is None:
        raise SpanMappingError(f"字符区间 [{char_start}, {char_end}) 不覆盖任何 token")
    return (lo, hi + 1)


def _role_of_token_span(
    block_bounds: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
    token_span: tuple[int, int],
    messages: Sequence[dict[str, Any]],
) -> str:
    """判断某个 token 区间落在哪条消息的块内（用于 sink 合同复核）。"""
    t0 = token_span[0]
    if t0 >= len(offsets):
        return "none"
    char_at = offsets[t0][0]
    for idx, (start, end) in enumerate(block_bounds):
        if start <= char_at < end:
            return str(messages[idx]["role"])
    return "none"
