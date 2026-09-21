"""纯函数数值判定；正式入口必须先从两份原始 run 现场重算可信摘要。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, TypedDict, cast

__all__ = [
    "NumericAcceptanceError",
    "canonical_input_identity",
    "evaluate_numeric_acceptance",
]

CONTRACT_SCHEMA = "attnview.p3-masked-numeric-contract/v1"
DECISION_SCHEMA = "attnview.p3-masked-numeric-decision/v1"


class _RequiredContract(TypedDict):
    summary_schema: str
    decodes: list[int]
    reference_layers: int
    forward_steps: int
    reference_overrides: int
    gdn_layers: int
    attention_arrays_per_kind: int
    logits_steps: int


class NumericAcceptanceError(ValueError):
    """合同或摘要结构无效，无法作出验收决定。"""


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, where: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise NumericAcceptanceError(f"{where} 缺失 {missing} / 多余 {unknown}")


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NumericAcceptanceError(f"{where} 必须是对象")
    return value


def _require_positive_finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericAcceptanceError(f"{where} 必须是有限正数")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise NumericAcceptanceError(f"{where} 必须是有限正数")
    return result

def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def canonical_input_identity(input_hashes: Mapping[str, str]) -> str:
    """把摘要输入哈希集合归一化为稳定身份；路径和值都参与。"""
    if not input_hashes or any(not isinstance(k, str) or not isinstance(v, str) for k, v in input_hashes.items()):
        raise NumericAcceptanceError("provenance.input_hashes 必须是非空字符串映射")
    payload = json.dumps(dict(input_hashes), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_contract(
    contract: Mapping[str, Any],
) -> tuple[_RequiredContract, dict[str, float], dict[str, str]]:
    _require_exact_keys(
        contract,
        {
            "schema",
            "frozen_on",
            "scope",
            "calibration_input_identity_sha256",
            "heldout_input_identity_sha256",
            "run_identity",
            "required",
            "relative_l2_max",
        },
        where="contract",
    )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise NumericAcceptanceError(f"未知 contract schema：{contract['schema']!r}")
    if contract["scope"] != "held-out-only":
        raise NumericAcceptanceError("contract.scope 必须是 'held-out-only'")
    calibration_identity = contract["calibration_input_identity_sha256"]
    heldout_identity = contract["heldout_input_identity_sha256"]
    for name, identity in (
        ("calibration_input_identity_sha256", calibration_identity),
        ("heldout_input_identity_sha256", heldout_identity),
    ):
        if not isinstance(identity, str) or len(identity) != 64:
            raise NumericAcceptanceError(f"{name} 必须是 64 位 SHA256")
    if calibration_identity == heldout_identity:
        raise NumericAcceptanceError("calibration 与 held-out 输入身份不得相同")
    run_identity = _require_mapping(contract["run_identity"], where="contract.run_identity")
    _require_exact_keys(run_identity, {"head", "model_revision", "vllm_revision"}, where="contract.run_identity")
    validated_identity: dict[str, str] = {}
    for name, value in run_identity.items():
        if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise NumericAcceptanceError(f"contract.run_identity.{name} 必须是 40 位小写十六进制 revision")
        validated_identity[name] = value

    required = _require_mapping(contract["required"], where="contract.required")
    required_keys = {
        "summary_schema",
        "decodes",
        "reference_layers",
        "forward_steps",
        "reference_overrides",
        "gdn_layers",
        "attention_arrays_per_kind",
        "logits_steps",
    }
    _require_exact_keys(required, required_keys, where="contract.required")
    decodes = required["decodes"]
    if not isinstance(decodes, list) or not decodes or any(isinstance(v, bool) or not isinstance(v, int) for v in decodes):
        raise NumericAcceptanceError("contract.required.decodes 必须是非空整数数组")
    if len(decodes) != len(set(decodes)):
        raise NumericAcceptanceError("contract.required.decodes 不得重复")
    for key in required_keys - {"summary_schema", "decodes"}:
        value = required[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NumericAcceptanceError(f"contract.required.{key} 必须是正整数")

    limits = _require_mapping(contract["relative_l2_max"], where="contract.relative_l2_max")
    _require_exact_keys(limits, {"decode_q", "decode_out", "logits"}, where="contract.relative_l2_max")
    validated = _RequiredContract(
        summary_schema=cast(str, required["summary_schema"]),
        decodes=cast(list[int], required["decodes"]),
        reference_layers=cast(int, required["reference_layers"]),
        forward_steps=cast(int, required["forward_steps"]),
        reference_overrides=cast(int, required["reference_overrides"]),
        gdn_layers=cast(int, required["gdn_layers"]),
        attention_arrays_per_kind=cast(int, required["attention_arrays_per_kind"]),
        logits_steps=cast(int, required["logits_steps"]),
    )
    validated_limits = {
        key: _require_positive_finite(value, where=f"contract.relative_l2_max.{key}")
        for key, value in limits.items()
    }
    return validated, validated_limits, validated_identity


def evaluate_numeric_acceptance(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """返回可序列化的验收决定；合同/摘要损坏抛 ``NumericAcceptanceError``。"""
    required, limits, run_identity = _validated_contract(_require_mapping(contract, where="contract"))
    failures: list[dict[str, Any]] = []

    def check(code: str, actual: Any, expected: Any, ok: bool) -> None:
        if not ok:
            failures.append({"code": code, "actual": actual, "expected": expected})

    check("summary_schema", summary.get("schema"), required["summary_schema"], summary.get("schema") == required["summary_schema"])
    for field in ("head", "model_revision", "vllm_revision"):
        check(
            f"run_identity_{field}",
            summary.get(field),
            run_identity[field],
            summary.get(field) == run_identity[field],
        )
    check("decodes", summary.get("decodes"), required["decodes"], summary.get("decodes") == required["decodes"])
    check("reference_exit_code", summary.get("reference_exit_code"), 0, summary.get("reference_exit_code") == 0)
    check("reference_manifest_failures", summary.get("reference_manifest_failures"), [], summary.get("reference_manifest_failures") == [])
    expected_forwards = list(range(1, int(required["forward_steps"]) + 1))
    check(
        "reference_completed_forwards",
        summary.get("reference_completed_forwards"),
        expected_forwards,
        summary.get("reference_completed_forwards") == expected_forwards,
    )
    check(
        "reference_overrides",
        summary.get("reference_overrides"),
        required["reference_overrides"],
        summary.get("reference_overrides") == required["reference_overrides"],
    )
    check(
        "reference_layers",
        summary.get("reference_layers"),
        required["reference_layers"],
        summary.get("reference_layers") == required["reference_layers"],
    )
    check("reference_nonfinite", summary.get("reference_nonfinite"), 0, summary.get("reference_nonfinite") == 0)

    gdn = _require_mapping(summary.get("gdn_state"), where="summary.gdn_state")
    check("gdn_enabled", gdn.get("enabled"), True, gdn.get("enabled") is True)
    check("gdn_required", gdn.get("required"), True, gdn.get("required") is True)
    check("gdn_layers", gdn.get("layer_count"), required["gdn_layers"], gdn.get("layer_count") == required["gdn_layers"])
    check("gdn_forward_steps", gdn.get("forward_steps"), required["forward_steps"], gdn.get("forward_steps") == required["forward_steps"])
    check("gdn_missing_steps", gdn.get("missing_steps"), [], gdn.get("missing_steps") == [])
    check("gdn_state_nonfinite", gdn.get("state_nonfinite"), 0, gdn.get("state_nonfinite") == 0)
    digest_counts = gdn.get("unique_digest_sample")
    digest_ok = isinstance(digest_counts, Mapping) and bool(digest_counts) and all(
        value == required["forward_steps"] for value in digest_counts.values()
    )
    check("gdn_unique_digest_sample", digest_counts, required["forward_steps"], digest_ok)

    provenance = _require_mapping(summary.get("provenance"), where="summary.provenance")
    input_hashes = _require_mapping(provenance.get("input_hashes"), where="summary.provenance.input_hashes")
    input_identity = canonical_input_identity(input_hashes)
    calibration_identity = contract["calibration_input_identity_sha256"]
    heldout_identity = contract["heldout_input_identity_sha256"]
    check("not_calibration_input_identity", input_identity, f"not {calibration_identity}", input_identity != calibration_identity)
    check("held_out_input_identity", input_identity, heldout_identity, input_identity == heldout_identity)

    attention = _require_mapping(
        summary.get("attention_candidate_vs_reference"),
        where="summary.attention_candidate_vs_reference",
    )
    rows = attention.get("per_array")
    if not isinstance(rows, list):
        raise NumericAcceptanceError("summary.attention_candidate_vs_reference.per_array 必须是数组")
    expected_decodes = set(required["decodes"])
    layer_values = {row.get("layer") for row in rows if isinstance(row, Mapping)}
    check("attention_layer_count", len(layer_values), required["reference_layers"], len(layer_values) == required["reference_layers"])
    expected_identities = {
        (layer, decode, kind)
        for layer in layer_values
        for decode in expected_decodes
        for kind in ("q", "out")
    }
    identities: list[tuple[Any, Any, Any]] = []
    observed_attention: dict[str, list[float]] = {"q": [], "out": []}
    attention_nonfinite = {"q": 0, "out": 0}
    kind_limit = {"q": limits["decode_q"], "out": limits["decode_out"]}
    for index, row_value in enumerate(rows):
        row = _require_mapping(row_value, where=f"summary.attention.per_array[{index}]")
        kind = row.get("kind")
        identity = (row.get("layer"), row.get("decode"), kind)
        identities.append(identity)
        if kind not in kind_limit:
            failures.append({"code": "attention_kind", "actual": kind, "expected": ["q", "out"]})
            continue
        relative_l2 = row.get("relative_l2")
        metric = _finite_number(relative_l2)
        if metric is not None:
            observed_attention[kind].append(metric)
            check(f"attention_{kind}_relative_l2[{index}]", metric, f"<= {kind_limit[kind]}", metric <= kind_limit[kind])
        else:
            failures.append({"code": f"attention_{kind}_relative_l2[{index}]", "actual": relative_l2, "expected": "finite"})
        nonfinite = row.get("nonfinite")
        if isinstance(nonfinite, int) and not isinstance(nonfinite, bool):
            attention_nonfinite[kind] += nonfinite
        check(f"attention_{kind}_nonfinite[{index}]", nonfinite, 0, nonfinite == 0)
    check("attention_identities", sorted(identities, key=str), "full unique layer×decode×kind product", set(identities) == expected_identities and len(identities) == len(expected_identities))

    expected_per_kind = int(required["attention_arrays_per_kind"])
    for kind, aggregate_name in (("q", "decode_q_step"), ("out", "decode_out_step")):
        aggregate = _require_mapping(attention.get(aggregate_name), where=f"summary.attention.{aggregate_name}")
        values = observed_attention[kind]
        observed_max = max(values) if values else None
        check(f"{aggregate_name}_count", aggregate.get("count"), expected_per_kind, aggregate.get("count") == expected_per_kind and len(values) == expected_per_kind)
        check(f"{aggregate_name}_nonfinite", aggregate.get("nonfinite"), attention_nonfinite[kind], aggregate.get("nonfinite") == attention_nonfinite[kind])
        check(f"{aggregate_name}_max_relative_l2", aggregate.get("max_relative_l2"), observed_max, aggregate.get("max_relative_l2") == observed_max)

    logits = _require_mapping(summary.get("logits_candidate_vs_reference"), where="summary.logits_candidate_vs_reference")
    logit_rows = summary.get("logits_steps")
    if not isinstance(logit_rows, list):
        raise NumericAcceptanceError("summary.logits_steps 必须是数组")
    expected_logit_steps = list(range(1, int(required["logits_steps"]) + 1))
    actual_logit_steps: list[Any] = []
    observed_logits: list[float] = []
    logits_nonfinite = 0
    argmax_mismatch = 0
    for index, row_value in enumerate(logit_rows):
        row = _require_mapping(row_value, where=f"summary.logits_steps[{index}]")
        actual_logit_steps.append(row.get("step"))
        relative_l2 = row.get("relative_l2")
        metric = _finite_number(relative_l2)
        if metric is not None:
            observed_logits.append(metric)
            check(f"logits_relative_l2[{index}]", metric, f"<= {limits['logits']}", metric <= limits["logits"])
        else:
            failures.append({"code": f"logits_relative_l2[{index}]", "actual": relative_l2, "expected": "finite"})
        nonfinite = row.get("nonfinite")
        if isinstance(nonfinite, int) and not isinstance(nonfinite, bool):
            logits_nonfinite += nonfinite
        check(f"logits_nonfinite[{index}]", nonfinite, 0, nonfinite == 0)
        argmax_equal = row.get("argmax_equal")
        if argmax_equal is not True:
            argmax_mismatch += 1
        check(f"logits_argmax_equal[{index}]", argmax_equal, True, argmax_equal is True)
    check("logits_steps", actual_logit_steps, expected_logit_steps, actual_logit_steps == expected_logit_steps)
    observed_logits_max = max(observed_logits) if observed_logits else None
    check("logits_count", logits.get("count"), required["logits_steps"], logits.get("count") == required["logits_steps"] and len(observed_logits) == required["logits_steps"])
    check("logits_nonfinite_total", logits.get("nonfinite"), logits_nonfinite, logits.get("nonfinite") == logits_nonfinite)
    check("logits_argmax_mismatch", logits.get("argmax_mismatch"), argmax_mismatch, logits.get("argmax_mismatch") == argmax_mismatch and argmax_mismatch == 0)
    check("logits_max_relative_l2", logits.get("max_relative_l2"), observed_logits_max, logits.get("max_relative_l2") == observed_logits_max)

    return {
        "schema": DECISION_SCHEMA,
        "passed": not failures,
        "contract": {
            "schema": contract["schema"],
            "frozen_on": contract["frozen_on"],
            "scope": contract["scope"],
            "calibration_input_identity_sha256": calibration_identity,
            "heldout_input_identity_sha256": heldout_identity,
            "run_identity": run_identity,
            "relative_l2_max": limits,
        },
        "summary": {
            "head": summary.get("head"),
            "model_revision": summary.get("model_revision"),
            "vllm_revision": summary.get("vllm_revision"),
            "reference_run": summary.get("reference_run"),
            "masked_run": summary.get("masked_run"),
            "input_identity_sha256": input_identity,
            "input_hashes": dict(input_hashes),
        },
        "observed": {
            "decode_q_max_relative_l2": max(observed_attention["q"]) if observed_attention["q"] else None,
            "decode_out_max_relative_l2": max(observed_attention["out"]) if observed_attention["out"] else None,
            "logits_max_relative_l2": observed_logits_max,
            "logits_argmax_mismatch": argmax_mismatch,
            "nonfinite_total": sum(attention_nonfinite.values()) + logits_nonfinite,
        },
        "failures": failures,
    }
