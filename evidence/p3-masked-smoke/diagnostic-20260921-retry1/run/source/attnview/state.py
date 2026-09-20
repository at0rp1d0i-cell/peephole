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
    MODE_FOCUS,
    ParseEvent,
    TagParser,
)


@dataclass(frozen=True)
class StepRecord:
    """生成流第 `token_index` 个 token **被采样出来后**的状态快照。

    时序表（本项目的固定约定，见 `readview.py` 顶部）：

    - prefill 写完 prompt 的 `prompt_len` 个位置，并采样出第 0 个生成 token；
    - 采样出生成流第 `t` 个 token 之后，**已写 KV 长度仍是 `prompt_len + t`**
      （该 token 自己还没被写进去）；
    - 消费该 token 的那次 forward（decode step `t+1`）才把它写到位置 `prompt_len + t`，
      因此那一步 attention 的 KV 上界 = `prompt_len + t + 1`。
    """

    token_index: int
    token_id: int
    text: str
    mode_after: str
    refs: tuple[int, ...]
    events: tuple[ParseEvent, ...]
    sampled_tokens: int  # 含本 token
    written_kv_len: int  # prompt_len + token_index（本 token 尚未写入）
    attention_kv_len_next: int  # 消费本 token 的那次 forward 的 KV 上界 = written_kv_len + 1

    @property
    def next_write_position(self) -> int:
        """下一个 forward 将写入的位置 = 已写长度（顺序追加）。"""
        return self.written_kv_len

    @property
    def effect_step(self) -> int:
        """本 token 内闭合的声明的生效步 = 消费本 token 的那次 forward。"""
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
        self.flush_events: tuple[ParseEvent, ...] = ()
        self.stop_token_index: int | None = None
        """触发停止条件的 token 下标：它被采样出来，但**不会有 forward 再消费它**。"""
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
        written = self.prompt_len + token_index
        record = StepRecord(
            token_index=token_index,
            token_id=token_id,
            text=text,
            mode_after=self.parser.mode,
            refs=self.parser.refs,
            events=events,
            sampled_tokens=len(self.steps) + 1,
            written_kv_len=written,
            attention_kv_len_next=written + 1,
        )
        self.steps.append(record)
        return record

    def mark_stop_token(self, token_index: int) -> None:
        """标记触发停止的 token（例如 `</answer>` 闭合、EOS、长度上限）。

        时序含义：该 token 被采样 → 生成停止 → 消费它的那次 forward **不会发生**，
        因此它对应的"下一步视图"从不被使用。`trace` 里据此外推出少一行。
        """
        if not 0 <= token_index < len(self.steps):
            raise ValueError(f"stop token 下标越界：{token_index}（已采样 {len(self.steps)} 个）")
        self.stop_token_index = int(token_index)

    def finish(self) -> tuple[ParseEvent, ...]:
        """生成结束：处理未闭合标签缓冲，冻结状态。

        结束阶段的异常（如未闭合的 `<focus`）同样进入统计与 trace，不得只留在 parser 里。
        """
        events = tuple(self.parser.flush(len(self.steps)))
        self.flush_events = events
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
    def written_kv_len(self) -> int:
        """当前**已写**的 KV 长度。

        prefill 写完 `prompt_len` 个位置；此后每采样一个 token，它在**下一次 forward** 才被写入，
        所以采样了 k 个 token 时已写长度是 `prompt_len + (k-1)`（k ≥ 1）。
        """
        return self.prompt_len + max(0, len(self.steps) - 1)

    @property
    def attention_kv_len_next(self) -> int:
        """下一次 forward 的 attention KV 上界 = 已写长度 + 1（含该步自身写入）。"""
        return self.written_kv_len + 1

    @property
    def anomalies(self) -> tuple[ParseEvent, ...]:
        return tuple(self.parser.anomalies)

    def view_for_next_step(self, *, mode: str | None = None, refs: tuple[int, ...] | None = None) -> tuple[str, tuple[int, ...]]:
        """第 `generated_tokens + 1` 步 forward 应使用的模式与引用。

        默认取"解析完已生成 token 之后"的模式——即 C3.5 的 t 解析、t+1 生效。
        """
        return (mode or self.mode, refs if refs is not None else self.refs)

    # --- trace（C6.3 的协议侧字段） ----------------------------------------
    def _all_events(self) -> tuple[ParseEvent, ...]:
        out: list[ParseEvent] = []
        for step in self.steps:
            out.extend(step.events)
        out.extend(self.flush_events)
        return tuple(out)

    def focus_stats(self) -> dict:
        """focus 尝试/成功（论文 §6.2 口径）。

        分母 = 所有 `is_focus_attempt` 的事件（开标签转移 + 属性/引用异常 + 结束时的未闭合 focus 缓冲）；
        闭标签与不匹配闭标签**不计入**。分子 = 合法 chunk 引用并进入 focus 的转移数。
        """
        attempts = 0
        successes = 0
        for event in self._all_events():
            if not event.is_focus_attempt:
                continue
            attempts += 1
            if event.kind == "transition" and event.mode_after == MODE_FOCUS:
                successes += 1
        return {"focus_attempts": attempts, "focus_successes": successes}

    def trace(self) -> dict:
        stats = self.focus_stats()
        return {
            "request_id": self.request_id,
            "arm": self.arm,
            "num_segments": self.num_segments,
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "written_kv_len": self.written_kv_len,
            "attention_kv_len_next": self.attention_kv_len_next,
            "stop_token_index": self.stop_token_index,
            "final_mode": self.mode,
            "focus_attempts": stats["focus_attempts"],
            "focus_successes": stats["focus_successes"],
            "anomalies": [e.as_dict() for e in self.anomalies],
            "flush_events": [e.as_dict() for e in self.flush_events],
            "steps": [
                {
                    "token_index": s.token_index,
                    "token_id": s.token_id,
                    "mode_after": s.mode_after,
                    "refs": list(s.refs),
                    "sampled_tokens": s.sampled_tokens,
                    "written_kv_len": s.written_kv_len,
                    "attention_kv_len_next": s.attention_kv_len_next,
                    "next_write_position": s.next_write_position,
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
