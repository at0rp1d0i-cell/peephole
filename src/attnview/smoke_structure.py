"""masked smoke 的**结构检查草稿**（纯 CPU，独立预期来自 config 声明 + 原始 spans）。

**状态：草稿，不是门禁（NATIVE-053 认定不可验收，尚未接入 `_run`）。** 已知缺陷：
- 逐步快照在**运行结束后**读 runner 的最后一份可变表，并非在本步 `model.forward` 内采集；
- canonical 行用 `rows[0]`（会整行转 int，且硬选 GDN 组，pin `gpu/block_table.py:73-75` 是 `list[Tensor]`）；
- trace 门禁未 fail-closed：空/空壳 trace 被当作"无覆写"的证明，且未区分"DA 受限覆写 trace"与"enforce_global 标记"。
重做要求见 `outbox/SUP-004-masked-smoke-closeout.md` §2b（逐步比**值**：seq_lens/物理行/slot/位置；FA 组由 FullAttentionSpec 确定；缺字段/缺 trace/缺受限覆写即失败）。

要点（NATIVE-050/052）：
- 独立预期**不调用候选筛选 helper**：可见位置用 `reference_dense.independent_visible_positions`
  （原始 sink/local_window/segment spans + config 声明的模式/引用），再块外扩成读取集合；
- 实际值从 **capture / 真实 runner 字段**读取，**字段缺失一律明确失败**（不猜、不静默跳过）；
- 步号换算显式：生成 token 序号 `decode_index`（1 基）与内部 forward 步 `forward = decode_index + 1`
  （prefill = 1）；`override.step` 按 `decode_index`，capture 按 forward；
- 覆盖全部步；`global` 恢复必须被检查；缺步/缺层/无 trace 失败。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .reference_dense import independent_visible_positions

__all__ = ["StructureError", "StepExpectation", "expectations_from_config", "check_capture_structure"]


class StructureError(RuntimeError):
    """结构证据缺失或与独立预期不符（fail closed，不猜测）。"""


@dataclass(frozen=True)
class StepExpectation:
    forward: int                 # 内部 forward 步号（prefill = 1）
    decode_index: int | None     # 生成 token 序号（1 基）；prefill 为 None
    mode: str
    refs: tuple[int, ...]
    kv_len: int
    blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]

    @property
    def total_read(self) -> int:
        return sum(self.effective_per_block)


def _transitions(script: Sequence[dict], tokenizer) -> list[tuple[int, str, tuple[int, ...]]]:
    """config 段落 → 声明点（段末 token 下标触发、t+1 消费）。"""
    idx, decls, prev = 0, [], None
    for piece in script:
        idx += len(tokenizer.encode(piece["text"], add_special_tokens=False))
        mode = piece["expected_mode_after"]
        if prev is None or mode != prev:
            decls.append((idx - 1, mode, tuple(piece.get("refs", []))))
        prev = mode
    return decls


def expectations_from_config(*, timeline_config: Path, doc_fixture: Path, tokenizer, prompt_len: int,
                             sink_span, local_window_span, segment_spans, block_size: int,
                             total_steps: int) -> dict[int, StepExpectation]:
    """生成**逐步独立预期**（forward 1..total_steps）。"""
    cfg = json.loads(Path(timeline_config).read_text())
    decls = _transitions(cfg["generation_script"], tokenizer)
    out: dict[int, StepExpectation] = {}
    for forward in range(1, total_steps + 1):
        decode_index = None if forward == 1 else forward - 1
        t = 0 if forward == 1 else forward - 2          # 0 基生成 token 下标
        mode, refs = "global", ()
        for parse_index, decl_mode, decl_refs in decls:
            if parse_index <= t:
                mode, refs = decl_mode, tuple(decl_refs)
        kv_len = prompt_len if forward == 1 else prompt_len + t + 1
        mask = independent_visible_positions(
            mode=mode, refs=refs, kv_len=kv_len, prompt_len=prompt_len,
            sink_span=tuple(sink_span), local_window_span=tuple(local_window_span),
            segment_spans=tuple(tuple(s) for s in segment_spans), block_size=block_size)
        out[forward] = StepExpectation(forward=forward, decode_index=decode_index, mode=mode, refs=refs,
                                       kv_len=kv_len, blocks=tuple(mask.blocks),
                                       effective_per_block=tuple(mask.effective_per_block))
    return out


def _require(obj: Any, name: str, *, where: str):
    """读取真实字段;**缺失即失败**(支持对象属性与 dict 记录两种真实形态)。"""
    if isinstance(obj, dict):
        if name not in obj:
            raise StructureError(f"{where}: 缺少真实字段 {name!r}（不得猜测，字段缺失即失败）")
        return obj[name]
    if obj is None or not hasattr(obj, name):
        raise StructureError(f"{where}: 缺少真实字段 {name!r}（不得猜测，字段缺失即失败）")
    return getattr(obj, name)


def _physical_row_from_runner(runner: Any, *, decode_index: int) -> list[int]:
    """从**固定 pin runner** 读本步 canonical 完整行（FA metadata 之外、未被覆写）。"""
    tables = _require(runner, "block_tables", where="runner")
    rows = _require(tables, "input_block_tables", where="runner.block_tables")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise StructureError("runner.block_tables.input_block_tables 为空:无法核对 canonical 行")
    return [int(x) for x in rows[0]]


def check_capture_structure(*, capture: Any, expectations: dict[int, StepExpectation], runner: Any,
                            trace: dict | None, prompt_len: int, block_size: int,
                            representative_decodes: Iterable[int],
                            expected_layers: int | None = None) -> tuple[list[dict], dict]:
    """逐步核对 capture 与独立预期；返回 (`checks`, `evidence`)。任何缺失/不符即抛 `StructureError`。"""
    positions = _require(capture, "positions", where="capture")
    records = _require(capture, "records", where="capture")
    rep = set(int(x) for x in representative_decodes)
    checks: list[dict] = []
    evidence: dict = {"steps": {}, "trace": None}

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # 层数要求:显式传入(真实运行 = 16 个 FA 层)或由**prefill 步**自洽推导;不得凭空假设。
    prefill_layers = len(records.get(1) or {})
    if prefill_layers == 0:
        raise StructureError("capture 的 prefill 步没有任何层记录(缺层即失败)")
    want_layers = int(expected_layers) if expected_layers is not None else prefill_layers

    missing_steps = [f for f in expectations if f not in positions]
    if missing_steps:
        raise StructureError(f"capture 缺少步 {missing_steps[:8]}（缺步即失败）")
    canon_row = _physical_row_from_runner(runner, decode_index=1)

    def expect_block_row(exp: StepExpectation) -> list[int]:
        if max(exp.blocks) >= len(canon_row):
            raise StructureError(f"预期逻辑块 {max(exp.blocks)} 超出 canonical 行长度 {len(canon_row)}")
        return [canon_row[b] for b in exp.blocks]

    for forward, exp in sorted(expectations.items()):
        layers = records.get(forward) or {}
        if not layers:
            raise StructureError(f"第 {forward} 步没有任何层记录（缺层即失败）")
        if len(layers) != want_layers:
            raise StructureError(
                f"第 {forward} 步记录 {len(layers)} 层，与要求的 {want_layers} 层不一致（缺层/漏层即失败）")
        meta = positions[forward]
        rec_meta = (capture.__dict__.get("record_meta", {}) or {}).get(forward) or {}
        got_len = int(next(iter(rec_meta.values()))["q_len"]) if rec_meta else int(meta.get("q_len", -1))
        # 1) 相位(真实 `phase`,由 positions+is_prefilling_np 判定)
        want_phase = "prefill" if forward == 1 else "decode"
        got_phase = str(meta.get("phase", "<missing>"))
        if got_phase != want_phase:
            raise StructureError(f"第 {forward} 步 phase={got_phase!r} 与预期 {want_phase!r} 不符")
        # 2) q_len 与真实绝对位置(逐值比较)
        want_len = prompt_len if forward == 1 else 1
        if got_len != want_len:
            raise StructureError(f"第 {forward} 步 q_len={got_len} 与预期 {want_len} 不符")
        got_pos = [int(x) for x in meta.get("positions", [])]
        want_pos = list(range(prompt_len)) if forward == 1 else [prompt_len + (forward - 2)]
        if got_pos != want_pos:
            raise StructureError(f"第 {forward} 步 positions={got_pos[:6]} 与预期 {want_pos[:6]} 不符")
        # 3) 读取长度:受限模式必须等于独立预期的总读取长度;global 等于 canonical 上界
        view = meta.get("view") or {}
        if not view.get("observable"):
            raise StructureError(f"第 {forward} 步 view 不可观察({view.get('reason')}):无证据即失败")
        want_read = exp.kv_len if exp.mode == "global" else exp.total_read
        if int(view.get("seq_lens_first", -1)) != int(want_read):
            raise StructureError(
                f"第 {forward} 步 seq_lens_first={view.get('seq_lens_first')} 与预期读取长度 {want_read} 不符"
                f"(mode={exp.mode}, canonical 上界={exp.kv_len})")
        # 4) 物理块行逐列比较(期望 = canonical 映射后的预期逻辑块)
        got_row = [int(x) for x in view.get("block_table_head", [])][: len(exp.blocks)]
        want_row = expect_block_row(exp)
        if got_row != want_row:
            raise StructureError(f"第 {forward} 步物理行 {got_row} 与预期 {want_row} 不符(错块即失败)")
        evidence["steps"][forward] = {"phase": got_phase, "mode_expected": exp.mode, "q_len": got_len,
                                      "positions": got_pos, "read_len": int(view.get("seq_lens_first")),
                                      "read_len_expected": int(want_read), "blocks_expected": list(exp.blocks),
                                      "physical_row": got_row, "physical_row_expected": want_row,
                                      "total_expected": exp.total_read}
        chk(f"step{forward} 相位/位置/q_len/读取长度/物理行 与独立预期一致", True,
            f"mode={exp.mode} read={view.get('seq_lens_first')} blocks={list(exp.blocks)}")
    # canonical 行与 slot_mappings 的真实观测（缺失即失败，不猜）
    canon = _physical_row_from_runner(runner, decode_index=1)
    # 真实 slot 映射(pin `model_runner.py:771`):`execute_model_state.slot_mappings_by_layer`
    state = _require(runner, "execute_model_state", where="runner")
    _require(state, "slot_mappings_by_layer", where="runner.execute_model_state")
    chk("runner canonical 行可读(长度为逻辑块数上限)", len(canon) >= 1, f"len={len(canon)}")
    chk("runner slot_mappings 可读", True)
    # trace 步号换算:override.step = decode_index;capture forward = decode_index + 1
    if trace is None:
        raise StructureError("缺少 trace(engine 侧逐步记录)⇒ 无 trace 即失败")
    overrides = trace.get("override_steps") or []
    if overrides:
        for rec in overrides:
            di = int(_require(rec, "step", where="trace.override"))
            chk(f"override.step={di} ⇒ capture forward={di + 1} 存在", (di + 1) in positions,
                f"forward={di + 1}")
        evidence["trace"] = {"override_steps": overrides, "mapping": "override.step = decode_index; forward = decode_index + 1"}
    else:
        chk("masked 臂 trace 明确记录无强制全局覆写", True, "override_steps 为空(masked 不得经 enforce_global)")
    # global 恢复必须被检查
    globals_after = [f for f, e in expectations.items()
                     if e.mode == "global" and e.decode_index is not None and e.decode_index >= 12]
    if not globals_after:
        raise StructureError("独立预期里没有 global 恢复步:轨迹不满足要求")
    chk("global 恢复步存在且被核对", all(f in positions for f in globals_after),
        f"recovery forwards={globals_after[:6]}")
    return checks, evidence
