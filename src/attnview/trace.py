"""逐 step 轨迹：把协议事件、读取视图、写入位置放进同一张表（阶段 03 的 E3 交付物）。

表的两条关键读法（工作单要求用户能直接看懂）：

1. **t 解析、t+1 生效**：某条声明的 `closed_at_token_index = t`（生成流第 t 个 token 内闭合），
   它出现在 `effect_step = t + 1` 那一行——即**消费该 token 的那次 forward 就用新视图**。
2. **读清单变化、写位置不变**：`visible_blocks` 随模式/声明变化，而 `write_position` 与
   `next_write_position` 只由生成进度决定，与视图无关（I3）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .parser import ParseEvent
from .readview import ReadView, TokenLayout, ViewInputs, build_read_view
from .state import RequestProtocolState


@dataclass(frozen=True)
class StepTraceRow:
    decode_step: int  # 0 = prefill；s ≥ 1 消费生成流第 s-1 个 token
    input_token_index: int | None
    input_token_id: int | None
    input_token_text: str | None
    write_position: int | None
    kv_len: int
    mode: str
    refs: tuple[int, ...]
    events: tuple[ParseEvent, ...]
    view: ReadView

    def as_dict(self) -> dict:
        return {
            "decode_step": self.decode_step,
            "input_token_index": self.input_token_index,
            "input_token_id": self.input_token_id,
            "input_token_text": self.input_token_text,
            "write_position": self.write_position,
            "kv_len": self.kv_len,
            "mode": self.mode,
            "refs": list(self.refs),
            "events": [e.as_dict() for e in self.events],
            "view": self.view.as_dict(),
        }


def build_step_trace(
    state: RequestProtocolState,
    layout: TokenLayout,
    *,
    canonical_blocks: tuple[int | None, ...],
    kernel_block_size: int,
    max_width: int,
    include_prefill: bool = True,
) -> tuple[StepTraceRow, ...]:
    """按状态里已喂入的生成 token 生成整条轨迹。"""
    rows: list[StepTraceRow] = []

    if include_prefill:
        prefill_view = build_read_view(
            ViewInputs(
                mode="global",
                refs=(),
                layout=layout,
                kv_len=layout.prompt_len,
                canonical_blocks=canonical_blocks,
                kernel_block_size=kernel_block_size,
                max_width=max_width,
                effect_step=0,
            )
        )
        rows.append(
            StepTraceRow(
                decode_step=0,
                input_token_index=None,
                input_token_id=None,
                input_token_text=None,
                write_position=None,
                kv_len=layout.prompt_len,
                mode="global",
                refs=(),
                events=(),
                view=prefill_view,
            )
        )

    for record in state.steps:
        step = record.token_index + 1  # 消费该 token 的 forward 序号
        kv_len = layout.prompt_len + step
        view = build_read_view(
            ViewInputs(
                mode=record.mode_after,
                refs=record.refs,
                layout=layout,
                kv_len=kv_len,
                canonical_blocks=canonical_blocks,
                kernel_block_size=kernel_block_size,
                max_width=max_width,
                effect_step=step,
            )
        )
        rows.append(
            StepTraceRow(
                decode_step=step,
                input_token_index=record.token_index,
                input_token_id=record.token_id,
                input_token_text=record.text,
                write_position=layout.prompt_len + record.token_index,
                kv_len=kv_len,
                mode=record.mode_after,
                refs=record.refs,
                events=record.events,
                view=view,
            )
        )
    return tuple(rows)


def independence_summary(rows: tuple[StepTraceRow, ...]) -> dict:
    """读/写分离的可核对摘要：写位置只随步数递增，读清单随模式变化。"""
    read_changes = 0
    write_monotonic = True
    prev_blocks: tuple[int, ...] | None = None
    prev_write: int | None = None
    for row in rows:
        if prev_blocks is not None and row.view.visible_blocks != prev_blocks:
            read_changes += 1
        if prev_write is not None and row.write_position is not None:
            if prev_write is not None and row.write_position != prev_write + 1:
                write_monotonic = False
        prev_blocks = row.view.visible_blocks
        if row.write_position is not None:
            prev_write = row.write_position
    steps = [r for r in rows if r.decode_step >= 1]
    return {
        "steps": len(rows),
        "generation_steps": len(steps),
        "read_list_changes": read_changes,
        "write_position_step1": steps[0].write_position if steps else None,
        "write_position_last": steps[-1].write_position if steps else None,
        "write_monotonic_plus_one": write_monotonic,
        "read_list_changes_while_write_advances": read_changes,
    }
