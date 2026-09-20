"""阶段 04 验收判据汇总（纯函数，无 torch 依赖，可在 CPU 上做失败注入测试）。

设计要点：**每条判据都参与顶层 PASS/失败**（数值、参考自检、读取表有效性、数据面交叉核对、
KV 内容不变、追加写 slot 纪律、常驻性、行隔离、敏感度）。任何一项 False 都会让整体失败并
带上"用例/种子/行 · 判据名 · 完整细节"，不允许"输出 False 但进程返回 0"。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Criterion:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Summary:
    ok: bool
    total: int
    failed: tuple[Criterion, ...]
    reasons: tuple[str, ...]


def _crit(name: str, ok: Any, detail: str = "") -> Criterion:
    return Criterion(name=name, ok=bool(ok), detail=detail)


REQUIRED_MEASUREMENT_KEYS = (
    "label", "mode", "kv_len", "numeric", "oracle_selfcheck", "read_table",
    "data_plane", "kv_integrity", "residency", "appended", "expectation", "append",
)


def _measurement_criteria(label: str, m: Mapping[str, Any]) -> list[Criterion]:
    out: list[Criterion] = []
    missing = [key for key in REQUIRED_MEASUREMENT_KEYS if key not in m]
    out.append(
        _crit(
            f"{label}/schema_complete",
            not missing,
            f"缺少字段：{missing}" if missing else "字段齐全",
        )
    )
    numeric = m.get("numeric", {})
    out.append(
        _crit(
            f"{label}/numeric",
            numeric.get("within_tolerance"),
            str(numeric.get("failure") or f"max_abs={numeric.get('max_abs')} rms={numeric.get('rms')}"),
        )
    )
    selfcheck = m.get("oracle_selfcheck", {})
    out.append(
        _crit(
            f"{label}/oracle_selfcheck",
            float(selfcheck.get("max_diff", float("inf"))) <= float(selfcheck.get("limit", 1e-5)),
            f"max_diff={selfcheck.get('max_diff')} limit={selfcheck.get('limit')}",
        )
    )
    table = m.get("read_table", {})
    out.append(_crit(f"{label}/read_table_no_minus_one", table.get("has_minus_one") is False, str(table)))
    out.append(
        _crit(
            f"{label}/read_table_width",
            table.get("width_ok") is True and table.get("physical_nonneg") is True,
            f"width={table.get('width')} seqused_k={table.get('seqused_k')} width_ok={table.get('width_ok')} "
            f"nonneg={table.get('physical_nonneg')}",
        )
    )
    plane = m.get("data_plane", {})
    for key, name in (
        ("blocks_match", "data_plane_blocks"),
        ("physical_match", "data_plane_physical"),
        ("seqused_match", "data_plane_seqused"),
        ("write_slot_match", "data_plane_write_slot"),
        ("current_block_retained", "data_plane_current_block"),
    ):
        out.append(_crit(f"{label}/{name}", plane.get(key) is True, str(plane.get("details", ""))))
    # 每次测量**必需** K 与 V 两项内容判据都为 True；None/缺字段一律失败，
    # 不允许"K 为 None 就跳过、连 V=False 也不查"这种绕过。
    integrity = m.get("kv_integrity")
    if not isinstance(integrity, Mapping):
        out.append(
            _crit(
                f"{label}/kv_k_unchanged",
                False,
                f"缺少 kv_integrity（实际 {integrity!r}）",
            )
        )
        out.append(_crit(f"{label}/kv_v_unchanged", False, f"缺少 kv_integrity（实际 {integrity!r}）"))
    else:
        for key, name in (("k_unchanged", "kv_k_unchanged"), ("v_unchanged", "kv_v_unchanged")):
            out.append(
                _crit(
                    f"{label}/{name}",
                    integrity.get(key) is True,
                    f"{key}={integrity.get(key)!r}（必须为 True）",
                )
            )

    # 预期块三方一致：配置预冻结预期 == 独立 oracle == 候选实际；缺预期即失败
    expectation = m.get("expectation")
    if not isinstance(expectation, Mapping):
        out.append(_crit(f"{label}/expectation_match", False, f"缺少 expectation（实际 {expectation!r}）"))
    else:
        config_blocks = expectation.get("config")
        oracle_blocks = expectation.get("oracle")
        actual_blocks = expectation.get("actual")
        present = isinstance(config_blocks, list) and bool(config_blocks)
        agree = (
            present
            and list(config_blocks) == list(oracle_blocks or [])
            and list(config_blocks) == list(actual_blocks or [])
        )
        out.append(
            _crit(
                f"{label}/expectation_match",
                agree,
                f"配置={config_blocks} oracle={oracle_blocks} 候选={actual_blocks}"
                + ("" if present else "（配置缺少 expect_blocks：预冻结预期必须参与判据）"),
            )
        )

    # 追加判据是**附加**判据；期望槽必须来自独立原位置公式，而不是被测 helper
    append = m.get("append")
    if not isinstance(append, Mapping):
        out.append(_crit(f"{label}/append_declared", False, f"缺少 append 段（实际 {append!r}）"))
        append = {"declared": None}
    declared = append.get("declared")
    appended = m.get("appended")
    out.append(
        _crit(
            f"{label}/append_declared",
            declared in (True, False) and declared == appended,
            f"append.declared={declared!r} appended={appended!r}",
        )
    )
    if declared:
        independent = append.get("slot_independent")
        helper = append.get("slot_helper")
        out.append(
            _crit(
                f"{label}/append_slot_source_match",
                isinstance(independent, int) and isinstance(helper, int) and independent == helper,
                f"独立原位置公式 slot={independent} 被测 helper slot={helper}",
            )
        )
        for key, name in (("k_changed_slots", "append_slot_k"), ("v_changed_slots", "append_slot_v")):
            changed = append.get(key)
            out.append(
                _crit(
                    f"{label}/{name}",
                    isinstance(independent, int) and changed == [independent],
                    f"{key}={changed} 期望（独立公式）=[{independent}]",
                )
            )
    else:
        leaked = [append.get(k) for k in ("k_changed_slots", "v_changed_slots") if append.get(k)]
        out.append(
            _crit(f"{label}/append_not_declared_is_clean", not leaked, f"未声明追加却记录了变化槽：{leaked}")
        )
    residency = m.get("residency", {})
    out.append(
        _crit(
            f"{label}/resident_cache",
            residency.get("data_ptr_stable") is True,
            f"data_ptr_stable={residency.get('data_ptr_stable')}",
        )
    )
    return out


def evaluate_case(case: Mapping[str, Any]) -> list[Criterion]:
    """把一个用例记录展开成判据列表（每行/每步一条，外加用例级判据）。"""
    out: list[Criterion] = []
    prefix = f"{case.get('id')}#seed{case.get('seed')}"
    measurements = list(case.get("measurements", []))
    if not measurements and case.get("control_only"):
        for extra in case.get("extra_criteria", []):
            out.append(_crit(f"{prefix}/{extra.get('name')}", extra.get("ok"), str(extra.get("detail", ""))))
        return out
    if not measurements:
        out.append(_crit(f"{prefix}/has_measurements", False, "用例没有产生任何测量记录"))
    for m in measurements:
        out.extend(_measurement_criteria(f"{prefix}:{m.get('label', '?')}", m))
    for extra in case.get("extra_criteria", []):
        out.append(_crit(f"{prefix}/{extra.get('name')}", extra.get("ok"), str(extra.get("detail", ""))))
    if case.get("error"):
        out.append(_crit(f"{prefix}/case_executed", False, str(case["error"])))
    return out


def summarize(cases: Iterable[Mapping[str, Any]]) -> Summary:
    cases = list(cases)
    criteria: list[Criterion] = []
    for case in cases:
        criteria.extend(evaluate_case(case))
    if not cases:
        return Summary(ok=False, total=0, failed=(), reasons=("没有用例记录",))
    failed = tuple(c for c in criteria if not c.ok)
    return Summary(
        ok=not failed,
        total=len(criteria),
        failed=failed,
        reasons=tuple(f"{c.name} — {c.detail}" for c in failed),
    )
