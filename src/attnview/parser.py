"""C3 增量输出流解析（协议合同 v1.0 §4）。

只解析**本请求生成的 token 流**：本类没有任何接口接收输入侧（prompt/文档）文本（C3.7），
调用方只能喂生成流。

关键语义：

- C3.2/C3.4 转移发生在开标签的闭合 `>` 上；标签可跨多个 token，`>` 到达前不得提前转移。
- C3.3 `<global>` 不产生转移；论文 Figure 1 的输出里出现 `</global>`，故 `</global>` 同样按"无转移"处理
  （合同未逐字列举闭标签全表，此项在报告中列为已记录的项目决定）。
- C3.6 多引用按集合去重、稳定排序；越界/不存在的编号 = 无效引用 → 整个声明无效、保持当前模式（C6.2）。
- C6.2 异常一律**保持当前模式**，不做隐式 global 回退，并记录原因码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MODE_GLOBAL = "global"
MODE_FOCUS = "focus"
MODE_LOCAL = "local"

DEFAULT_MAX_TAG_BUFFER = 256

_TAG_RE = re.compile(r"^(?P<closing>/?)(?P<name>[a-z][a-z0-9_]*)(?P<attrs>(?:\s+[a-z_]+=\"[^\"]*\")*)\s*$")
_ATTR_RE = re.compile(r"([a-z_]+)=\"([^\"]*)\"")
_REFS_RE = re.compile(r"^\d+(?:,\d+)*$")
_LOOSE_TAG_RE = re.compile(r"^(?P<closing>/?)(?P<name>[a-z][a-z0-9_]*)(?:\s+.*)?$")

#: 协议定义的标签名（用于把"名称可识别但属性非法"与"未知标签"区分开）
KNOWN_TAGS = frozenset({"global", "local", "focus", "answer"})

FOCUS_ATTR = "magic_chunks"

#: 已知但**不产生控制转移**的标签（答案包装；C3 只管模式，答案区不改变模式）
MARKER_TAGS = frozenset({"answer"})


@dataclass(frozen=True)
class ParseEvent:
    kind: str  # transition | noop | marker | anomaly
    name: str
    mode_before: str
    mode_after: str
    refs: tuple[int, ...]
    closed_at_token_index: int
    effect_step: int
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "refs": list(self.refs),
            "closed_at_token_index": self.closed_at_token_index,
            "effect_step": self.effect_step,
            "reason": self.reason,
            "detail": self.detail,
        }


class TagParser:
    """字符级增量扫描器：喂入生成文本片段，返回在该片段内闭合的解析事件。"""

    def __init__(self, num_segments: int, *, max_tag_buffer: int = DEFAULT_MAX_TAG_BUFFER) -> None:
        self.num_segments = int(num_segments)
        self.max_tag_buffer = int(max_tag_buffer)
        self.mode = MODE_GLOBAL
        self.refs: tuple[int, ...] = ()
        self._buffer: list[str] = []
        self._buffer_token_index: int | None = None
        self._buffer_started = False
        self.plain_text: list[str] = []
        self.anomalies: list[ParseEvent] = []

    # --- 输入 ---------------------------------------------------------------
    def feed(self, text: str, token_index: int) -> list[ParseEvent]:
        events: list[ParseEvent] = []
        for ch in text:
            events.extend(self._feed_char(ch, token_index))
        return events

    def flush(self, token_index: int) -> list[ParseEvent]:
        """生成结束：未闭合的标签缓冲按异常记录（C6.1 标签未闭合）。"""
        events: list[ParseEvent] = []
        if self._buffer_started:
            buffered = "".join(self._buffer)
            events.append(
                self._anomaly(
                    "incomplete_tag",
                    token_index,
                    detail=f"未闭合标签缓冲 {len(buffered)} 字符",
                )
            )
            self.plain_text.append(buffered)
            self._reset_buffer()
        return events

    # --- 内部 ---------------------------------------------------------------
    def _feed_char(self, ch: str, token_index: int) -> list[ParseEvent]:
        if not self._buffer_started:
            if ch == "<":
                self._buffer_started = True
                self._buffer = ["<"]
                self._buffer_token_index = token_index
            else:
                self.plain_text.append(ch)
            return []

        if len(self._buffer) == 1:
            # `<` 之后第一个字符决定它是否可能是标签
            if ch == "/" or ch.isascii() and ch.isalpha():
                self._buffer.append(ch)
                return []
            if ch == "<":
                # 前一个 `<` 不是标签：作为正文吐出，重新从这个 `<` 开始
                self.plain_text.append("<")
                self._buffer = ["<"]
                self._buffer_token_index = token_index
                return []
            self.plain_text.append("<")
            self.plain_text.append(ch)
            self._reset_buffer()
            return []

        self._buffer.append(ch)
        if ch == ">":
            buffered = "".join(self._buffer)
            self._reset_buffer()
            return [self._close(buffered, token_index)]
        if len(self._buffer) > self.max_tag_buffer:
            buffered = "".join(self._buffer)
            self.plain_text.append(buffered)
            start = self._buffer_token_index
            self._reset_buffer()
            return [
                self._anomaly(
                    "tag_buffer_overflow",
                    token_index,
                    detail=f"标签缓冲超过 {self.max_tag_buffer} 字符（自 token {start} 起，按正文处理）",
                )
            ]
        return []

    def _reset_buffer(self) -> None:
        self._buffer = []
        self._buffer_started = False
        self._buffer_token_index = None

    def _close(self, buffered: str, token_index: int) -> ParseEvent:
        inner = buffered[1:-1]
        match = _TAG_RE.match(inner)
        if match is None:
            # 名称可识别但属性语法非法时，报 malformed_attribute（比 unknown_tag 更有信息量）
            loose = _LOOSE_TAG_RE.match(inner)
            if loose is not None and loose.group("name") in KNOWN_TAGS:
                return self._anomaly(
                    "malformed_attribute",
                    token_index,
                    detail=f"标签 {loose.group('name')} 的属性语法非法：{buffered!r}",
                    tag=loose.group("name"),
                )
            return self._anomaly("unknown_tag", token_index, detail=f"无法解析的标签 {buffered!r}")
        closing = bool(match.group("closing"))
        name = match.group("name")
        attrs = dict(_ATTR_RE.findall(match.group("attrs")))
        raw_attrs = match.group("attrs").strip()

        if name in MARKER_TAGS:
            if attrs:
                return self._anomaly("unexpected_attribute", token_index, detail=buffered, tag=name)
            return self._event("marker", name, token_index, self.mode, self.mode, self.refs)

        if name == "global":
            if attrs:
                return self._anomaly("unexpected_attribute", token_index, detail=buffered, tag=name)
            return self._event("noop", name, token_index, self.mode, self.mode, self.refs)

        if name == "local":
            if attrs:
                return self._anomaly("unexpected_attribute", token_index, detail=buffered, tag=name)
            if closing:
                return self._transition(name, MODE_GLOBAL, (), token_index)
            return self._transition(name, MODE_LOCAL, (), token_index)

        if name == "focus":
            if closing:
                if attrs:
                    return self._anomaly("unexpected_attribute", token_index, detail=buffered, tag=name)
                return self._transition(name, MODE_GLOBAL, (), token_index)
            if FOCUS_ATTR not in attrs:
                return self._anomaly(
                    "missing_attribute",
                    token_index,
                    detail=f"focus 缺少 {FOCUS_ATTR}；原文 {buffered!r}",
                    tag=name,
                )
            value = attrs[FOCUS_ATTR]
            if not _REFS_RE.match(value.strip()):
                return self._anomaly(
                    "malformed_attribute",
                    token_index,
                    detail=f"{FOCUS_ATTR}={value!r} 不符合 K[,M[,N]] 整数列表",
                    tag=name,
                )
            refs = tuple(sorted({int(x) for x in value.split(",")}))
            invalid = tuple(r for r in refs if not 1 <= r <= self.num_segments)
            if invalid:
                return self._anomaly(
                    "invalid_reference",
                    token_index,
                    detail=(
                        f"越界引用 {list(invalid)}（可用 1..{self.num_segments}）；"
                        "整条声明作废，保持当前模式"
                    ),
                    refs=refs,
                    tag=name,
                )
            return self._transition(name, MODE_FOCUS, refs, token_index)

        return self._anomaly("unknown_tag", token_index, detail=f"未知标签 {buffered!r}", tag=name)

    def _transition(
        self, name: str, mode: str, refs: tuple[int, ...], token_index: int
    ) -> ParseEvent:
        before = self.mode
        self.mode = mode
        self.refs = refs if mode == MODE_FOCUS else ()
        return self._event("transition", name, token_index, before, mode, self.refs)

    def _anomaly(
        self,
        reason: str,
        token_index: int,
        *,
        detail: str,
        refs: tuple[int, ...] = (),
        tag: str = "",
    ) -> ParseEvent:
        event = self._event(
            "anomaly", tag, token_index, self.mode, self.mode, refs, reason=reason, detail=detail
        )
        self.anomalies.append(event)
        return event

    def _event(
        self,
        kind: str,
        name: str,
        token_index: int,
        mode_before: str,
        mode_after: str,
        refs: tuple[int, ...],
        *,
        reason: str = "",
        detail: str = "",
    ) -> ParseEvent:
        return ParseEvent(
            kind=kind,
            name=name,
            mode_before=mode_before,
            mode_after=mode_after,
            refs=refs,
            closed_at_token_index=token_index,
            effect_step=token_index + 1,  # C3.5：第 t 步解析、第 t+1 步生效
            reason=reason,
            detail=detail,
        )
