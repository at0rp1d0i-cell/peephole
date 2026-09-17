"""请求级协议状态与生命周期（协议合同 C3/C6；方案 §4「三个独立状态」中的控制状态）。

本模块**只管控制状态**（模式、声明引用、逐步事件、异常），不持有块表、不构造读取视图；
读取视图构造见 `readview.py`，写入位置由调用方按「decode step s 写入位置 = prompt_len + s - 1」
换算（见 `readview.py` 顶部约定），两者互相独立（读取视图不改变写入位置）。

位置约定（与 `readview.py`/`tools/p1cpu-demo.py` 一致，写入设计文档）：

- `s`（1 基）为 decode step 序号：第 `s` 步 forward 消费生成流第 `s-1` 个 token；
- 该 token 的写入位置 = `prompt_len + s - 1`；该步 attention 的 KV 上界（含刚写入的自身）= `prompt_len + s`；
- 生成流第 `t` 个 token（0 基）内闭合的声明，`effect_step = t + 1`，即**消费该 token 的那一步 forward**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .parser import (
    MODE_GLOBAL,
    ParseEvent,
    TagParser,
)


@dataclass(frozen=True)
class StepRecord:
    """生成流第 `token_index` 个 token 处理后的状态快照。"""

    token_index: int
    token_id: int
    text: str
    mode_after: str
    refs: tuple[int, ...]
    events: tuple[ParseEvent, ...]
    generated_tokens: int
    kv_len_after: int  # prompt_len + generated_tokens
    next_write_position: int  # == kv_len_after

    @property
    def effect_step(self) -> int:
        return self.token_index + 1


class RequestProtocolState:
    """每个请求独立持有；由 `ProtocolRegistry` 按 request_id 管理。"""

    def __init__(
        self,
        request_id: str,
        *,
        arm: str,
        num_segments: int,
        prompt_len: int,
        max_tag_buffer: int = 256,
    ) -> None:
        self.request_id = request_id
        self.arm = arm
        self.num_segments = int(num_segments)
        self.prompt_len = int(prompt_len)
        self.parser = TagParser(num_segments, max_tag_buffer=max_tag_buffer)
        self.steps: list[StepRecord] = []
        self.closed = False

    # --- 生成流输入 ---------------------------------------------------------
    def feed_generated_token(self, token_index: int, token_id: int, text: str) -> StepRecord:
        """喂入**本请求生成流**的 token。输入侧文本永远不走这个接口（C3.7）。"""
        if self.closed:
            raise RuntimeError("请求状态已结束，不能再喂生成 token")
        if token_index != len(self.steps):
            raise ValueError(
                f"token_index 必须连续递增：期望 {len(self.steps)}，得到 {token_index}"
            )
        events = tuple(self.parser.feed(text, token_index))
        generated = len(self.steps) + 1
        record = StepRecord(
            token_index=token_index,
            token_id=token_id,
            text=text,
            mode_after=self.parser.mode,
            refs=self.parser.refs,
            events=events,
            generated_tokens=generated,
            kv_len_after=self.prompt_len + generated,
            next_write_position=self.prompt_len + generated,
        )
        self.steps.append(record)
        return record

    def finish(self) -> tuple[ParseEvent, ...]:
        """生成结束：处理未闭合标签缓冲，冻结状态。"""
        events = tuple(self.parser.flush(len(self.steps)))
        self.closed = True
        return events

    # --- 当前控制状态 -------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.parser.mode

    @property
    def refs(self) -> tuple[int, ...]:
        return self.parser.refs

    @property
    def generated_tokens(self) -> int:
        return len(self.steps)

    @property
    def kv_len(self) -> int:
        return self.prompt_len + len(self.steps)

    @property
    def anomalies(self) -> tuple[ParseEvent, ...]:
        return tuple(self.parser.anomalies)

    def view_for_next_step(self, *, mode: str | None = None, refs: tuple[int, ...] | None = None) -> tuple[str, tuple[int, ...]]:
        """第 `generated_tokens + 1` 步 forward 应使用的模式与引用。

        默认取"解析完已生成 token 之后"的模式——即 C3.5 的 t 解析、t+1 生效。
        """
        return (mode or self.mode, refs if refs is not None else self.refs)

    # --- trace（C6.3 的协议侧字段） ----------------------------------------
    _FOCUS_FAILURES = frozenset(
        {"missing_attribute", "malformed_attribute", "invalid_reference", "unexpected_attribute"}
    )

    def focus_stats(self) -> dict:
        """focus 尝试/成功（论文 §6.2 口径）：只统计**调用** `<focus ...>` 的事件。

        闭合标签 `</focus>`（回到 global）不是一次调用，不计入分母；属性非法/越界引用计为尝试失败。
        """
        successes = 0
        failures = 0
        for step in self.steps:
            for event in step.events:
                if event.name != "focus":
                    continue
                if event.kind == "transition" and event.mode_after == "focus":
                    successes += 1
                elif event.kind == "anomaly" and event.reason in self._FOCUS_FAILURES:
                    failures += 1
        return {"focus_attempts": successes + failures, "focus_successes": successes}

    def trace(self) -> dict:
        stats = self.focus_stats()
        return {
            "request_id": self.request_id,
            "arm": self.arm,
            "num_segments": self.num_segments,
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "kv_len": self.kv_len,
            "final_mode": self.mode,
            "focus_attempts": stats["focus_attempts"],
            "focus_successes": stats["focus_successes"],
            "anomalies": [e.as_dict() for e in self.anomalies],
            "steps": [
                {
                    "token_index": s.token_index,
                    "token_id": s.token_id,
                    "mode_after": s.mode_after,
                    "refs": list(s.refs),
                    "generated_tokens": s.generated_tokens,
                    "kv_len_after": s.kv_len_after,
                    "events": [e.as_dict() for e in s.events],
                }
                for s in self.steps
            ],
        }


class ProtocolRegistry:
    """按 request_id 隔离的请求状态容器；结束/取消后必须显式释放（C6.4）。"""

    def __init__(self) -> None:
        self._states: dict[str, RequestProtocolState] = {}

    def create(self, request_id: str, **kwargs) -> RequestProtocolState:
        if request_id in self._states:
            raise KeyError(f"request_id 已存在：{request_id}")
        state = RequestProtocolState(request_id, **kwargs)
        self._states[request_id] = state
        return state

    def get(self, request_id: str) -> RequestProtocolState:
        return self._states[request_id]

    def release(self, request_id: str) -> RequestProtocolState:
        return self._states.pop(request_id)

    def active_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._states))
