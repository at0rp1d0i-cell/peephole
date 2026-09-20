"""方案 §4.1 的公开输出提取与规定标签机械过滤（E4）。

边界（用户已确认的首版语义）：

- 原始生成流先由引擎按协议解析控制，公开响应**只**承载最终答案区域；
- 只去除**规定的注意力标签**（`<global>`/`</global>`/`<local>`/`</local>`/`<focus magic_chunks="...">`/`</focus>`）
  与 `<answer>` 包装；不删除普通单词 `focus`/`answer`，不泛删 XML/HTML，不改正文空白与代码缩进；
- 答案缺失/截断时返回内部失败结果，**不回传原始流**，错误消息也不拼入内部文本；
- 纯函数：不修改模型历史/KV/协议状态，调用间无残留缓冲。

已知限制（沿用已接受的范围假设，不冒充正文保真通过）：正文中与控制标签**完全同名**的字面串会被
一并过滤；测试必须把这类用例单列为限制，而不是当作保真通过。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

#: 只匹配**规定的合法控制标签形式**（focus 必须带 magic_chunks 属性）。
CONTROL_TAG_RE = re.compile(r"</?global\s*>|</?local\s*>|</?focus\s+magic_chunks=\"[^\"]*\"\s*>|</focus\s*>")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

ERROR_MESSAGES = {
    "answer_missing": "no final answer region was produced",
    "answer_unterminated": "final answer region was not closed",
    "answer_empty": "final answer region was empty",
    "answer_ambiguous": "final answer region could not be classified unambiguously",
}


@dataclass(frozen=True)
class ExtractionResult:
    ok: bool
    answer: str | None
    error_code: str | None
    error_message: str | None
    answer_char_span: tuple[int, int] | None
    stripped_tags: tuple[str, ...]
    answer_open_count: int
    answer_close_count: int

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "answer": self.answer,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "answer_char_span": list(self.answer_char_span) if self.answer_char_span else None,
            "stripped_tags": list(self.stripped_tags),
            "answer_open_count": self.answer_open_count,
            "answer_close_count": self.answer_close_count,
        }


def extract_public_output(raw_stream: str) -> ExtractionResult:
    """从原始生成流提取可公开的最终答案区域。

    规则（本阶段修正后）：

    - **保留正文首尾空白**：只按标签机械切分，不对答案正文做 `strip()`（代码缩进与空白属正文保真范围）。
    - 恰好一对 `<answer>…</answer>` 且顺序正确 → 返回两者之间的内容（去掉其中的规定控制标签）。
    - 多个答案区、嵌套答案、多余的闭标签 → `answer_ambiguous`（**内部失败**），不自行选择"最后一个完整答案"；
      该语义列为待决，实现不得先冻结。
    - 有开标签无对应闭标签 → `answer_unterminated`（内部失败）。
    - 内容为空字符串 → `answer_empty`。
    - 失败一律不返回原始流，错误消息也不含内部文本。
    """
    opens = raw_stream.count(ANSWER_OPEN)
    closes = raw_stream.count(ANSWER_CLOSE)
    if opens == 0 and closes == 0:
        return _error("answer_missing", opens, closes)
    if opens == 1 and closes == 1:
        start = raw_stream.index(ANSWER_OPEN)
        end = raw_stream.index(ANSWER_CLOSE)
        if end < start:  # `</answer>` 先出现
            return _error("answer_ambiguous", opens, closes)
        inner = raw_stream[start + len(ANSWER_OPEN) : end]
        stripped: list[str] = []
        cleaned = CONTROL_TAG_RE.sub(lambda m: (stripped.append(m.group(0)), "")[1], inner)
        if cleaned == "":
            return ExtractionResult(
                ok=False,
                answer=None,
                error_code="answer_empty",
                error_message=ERROR_MESSAGES["answer_empty"],
                answer_char_span=(start + len(ANSWER_OPEN), end),
                stripped_tags=tuple(stripped),
                answer_open_count=opens,
                answer_close_count=closes,
            )
        return ExtractionResult(
            ok=True,
            answer=cleaned,
            error_code=None,
            error_message=None,
            answer_char_span=(start + len(ANSWER_OPEN), end),
            stripped_tags=tuple(stripped),
            answer_open_count=opens,
            answer_close_count=closes,
        )
    if opens > closes and opens == 1:
        return _error("answer_unterminated", opens, closes)
    return _error("answer_ambiguous", opens, closes)


def _error(code: str, opens: int = 0, closes: int = 0) -> ExtractionResult:
    return ExtractionResult(
        ok=False,
        answer=None,
        error_code=code,
        error_message=ERROR_MESSAGES[code],
        answer_char_span=None,
        stripped_tags=(),
        answer_open_count=opens,
        answer_close_count=closes,
    )


def public_stream_accounting(raw_stream: str) -> dict:
    """§4.1 记账：内部生成量与公开内容量分开（隐藏输出不虚增速度）。"""
    result = extract_public_output(raw_stream)
    tags: Sequence[str] = CONTROL_TAG_RE.findall(raw_stream)
    tag_chars = sum(len(t) for t in tags)
    return {
        "raw_chars": len(raw_stream),
        "answer_chars": len(result.answer) if result.answer else 0,
        "protocol_tag_count": len(tags),
        "protocol_tag_chars": tag_chars,
        "analysis_chars": len(raw_stream) - tag_chars - (len(result.answer) if result.answer else 0),
        "ok": result.ok,
        "error_code": result.error_code,
    }
