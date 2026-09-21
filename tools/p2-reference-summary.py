#!/usr/bin/env python3
"""从两次真实 GPU run 的证据目录重算 candidate-vs-reference 诊断摘要。

只读 CPU 工具：输入是 masked 候选 run 与独立 reference run 的 `run/` 目录，
输出 `summary.json`（schema `attnview.p3-reference-summary/v1`）。不设通过线；
所有比较对象、指标口径与输入哈希都写进产物，便于第三方复算。

fail-closed 条件（任一不满足即拒绝出摘要）：
  - 两 run 的 arm 分别是 `patched-masked` / `patched-reference`；
  - HEAD、模型 revision、payload、输入哈希、目标层名映射、产出 token 轨迹一致；
  - 候选结构门禁为 ok、reference manifest 无 failures 且 exit_code 为 0；
  - 两 run 的 `capture/layers.npz` 与 manifest 记录的 SHA256 一致；
  - 两 run 的 logits 记录数与逐步 step 一一对应。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DECODES = (6, 7, 20, 25)
GDN_SAMPLE_LAYERS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run: Path, *, arm: str) -> dict:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺失 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("arm") != arm:
        raise SystemExit(f"{run} 的 arm 是 {manifest.get('arm')!r}，期望 {arm!r}")
    return {
        "run": run,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
    }


def require_identity(reference: dict, candidate: dict) -> None:
    ref, cand = reference["manifest"], candidate["manifest"]
    for field in ("head", "pin_commit"):
        if ref[field] != cand[field]:
            raise SystemExit(f"两 run 的 {field} 不一致：{ref[field]!r} vs {cand[field]!r}")
    if ref["model"]["revision"] != cand["model"]["revision"]:
        raise SystemExit("两 run 的模型 revision 不一致：拒绝比较")
    if ref["payload"] != cand["payload"]:
        raise SystemExit("两 run 的协议载荷不一致：拒绝比较")
    if ref["sources"]["input_hashes"] != cand["sources"]["input_hashes"]:
        raise SystemExit("两 run 的输入哈希集合不一致：拒绝比较")
    if ref["capture"]["expected_layers"] != cand["capture"]["expected_layers"]:
        raise SystemExit("两 run 的目标 FA 层名映射不一致：拒绝比较")
    ref_tokens = ref["trajectory_comparison"]["produced"]
    cand_tokens = cand["trajectory_comparison"]["produced"]
    if ref_tokens != cand_tokens:
        raise SystemExit("两 run 的产出 token 轨迹不一致：拒绝比较")
    if not cand["masked_structure"].get("ok"):
        raise SystemExit("候选 run 的结构门禁未通过：拒绝出数值摘要")
    if not ref["reference"].get("ok"):
        raise SystemExit(f"reference run 未 ok：{ref['reference'].get('error')!r}")
    if ref["failures"] or ref["exit_code"] != 0:
        raise SystemExit(f"reference run 有 failures/非零退出：{ref['failures']} / {ref['exit_code']}")
    if cand["failures"] or cand["exit_code"] != 0:
        raise SystemExit(f"候选 run 有 failures/非零退出：{cand['failures']} / {cand['exit_code']}")


def verify_capture(run: dict) -> Path:
    path = REPO / run["manifest"]["capture"]["npz"]
    recorded = run["manifest"]["capture"]["npz_sha256"]
    actual = sha256_file(path)
    if actual != recorded:
        raise SystemExit(f"{path} 的 SHA256 与 manifest 不符：{actual} != {recorded}")
    return path


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = actual.float() - reference.float()
    return {
        "max_abs": float(diff.abs().max()),
        "rms": float(diff.square().mean().sqrt()),
        "relative_l2": float(diff.norm() / reference.float().norm().clamp_min(1e-30)),
        "nonfinite": int((~torch.isfinite(actual)).sum()),
    }


def compare_attention(ref_npz: Path, cand_npz: Path, *, decodes, layers: list[int]) -> dict:
    rows = []
    with np.load(ref_npz) as ref_arrays, np.load(cand_npz) as cand_arrays:
        for layer in layers:
            for key_suffix in ("layer_name", "scale", "capture_dtype"):
                ref_value = ref_arrays[f"{key_suffix}_L{layer}"]
                cand_value = cand_arrays[f"{key_suffix}_L{layer}"]
                if not np.array_equal(ref_value, cand_value):
                    raise SystemExit(f"L{layer} 的 {key_suffix} 在两 run 间不一致：拒绝比较")
        for decode in decodes:
            for layer in layers:
                for kind in ("q", "out"):
                    key = f"decode_{kind}_step{decode}_L{layer}"
                    ref = torch.from_numpy(ref_arrays[key])
                    cand = torch.from_numpy(cand_arrays[key])
                    if ref.shape != cand.shape:
                        raise SystemExit(f"{key} 形状不一致：{ref.shape} vs {cand.shape}")
                    rows.append({"layer": layer, "decode": decode, "kind": kind,
                                 **error_metrics(cand, ref)})
    summary = {}
    for kind, name in (("q", "decode_q_step"), ("out", "decode_out_step")):
        subset = [row for row in rows if row["kind"] == kind]
        summary[name] = {
            "count": len(subset),
            "max_abs": max(row["max_abs"] for row in subset),
            "max_rms": max(row["rms"] for row in subset),
            "max_relative_l2": max(row["relative_l2"] for row in subset),
            "nonfinite": sum(row["nonfinite"] for row in subset),
        }
    summary["per_array"] = rows
    return summary


def compare_logits(ref_run: Path, cand_run: Path) -> tuple[dict, list[dict]]:
    ref_records = torch.load(ref_run / "logits.pt", weights_only=False)
    cand_records = torch.load(cand_run / "logits.pt", weights_only=False)
    if len(ref_records) != len(cand_records):
        raise SystemExit(f"logits 记录数不一致：{len(ref_records)} vs {len(cand_records)}")
    steps = []
    for ref_record, cand_record in zip(ref_records, cand_records, strict=True):
        if ref_record["step"] != cand_record["step"]:
            raise SystemExit(f"logits step 不对应：{ref_record['step']} vs {cand_record['step']}")
        metrics = error_metrics(cand_record["logits"], ref_record["logits"])
        steps.append({
            "step": int(ref_record["step"]),
            "max_abs": metrics["max_abs"],
            "rms": metrics["rms"],
            "relative_l2": metrics["relative_l2"],
            "argmax_equal": bool(int(cand_record["logits"].argmax()) == int(ref_record["logits"].argmax())),
            "nonfinite": metrics["nonfinite"],
        })
    summary = {
        "count": len(steps),
        "max_abs": max(step["max_abs"] for step in steps),
        "max_rms": max(step["rms"] for step in steps),
        "max_relative_l2": max(step["relative_l2"] for step in steps),
        "argmax_mismatch": sum(0 if step["argmax_equal"] else 1 for step in steps),
        "nonfinite": sum(step["nonfinite"] for step in steps),
    }
    return summary, steps


def summarize_gdn(reference: dict) -> dict:
    """从 reference manifest 的逐步 GDN 快照复算状态观测统计（不读候选压缩表）。"""
    state = reference["manifest"]["capture"].get("gdn_state") or {}
    if not state.get("enabled"):
        raise SystemExit("reference run 没有 GDN 状态快照：拒绝出摘要")
    snapshots = state["snapshots"]
    # 每层每步的"状态"是各 cache tensor 摘要的组合：样本层摘要数 = 该层不同步数，
    # 单看某一个 tensor 会把两个 tensor 的摘要混成一个集合而高估步数。
    digests: dict[str, set[tuple[str, ...]]] = {}
    nonfinite = 0
    for record in snapshots.values():
        for name, entry in record.items():
            nonfinite += sum(int(part["nonfinite"]) for part in entry["parts"])
            digests.setdefault(name, set()).add(
                tuple(part["digest"] for part in entry["parts"]))
    # 旧 schema 记录逐层 alias 对，新 schema 直接给去重后的重叠对数；两者都可审计。
    if "alias_pair_count" in state:
        alias_pair_count = int(state["alias_pair_count"])
    else:
        alias_pair_count = len(state.get("alias_pairs", []))
    return {
        "enabled": bool(state["enabled"]),
        "required": bool(state["required"]),
        "layer_count": len(state["gdn_layers"]),
        "missing_steps": state["missing_steps"],
        "forward_steps": len(snapshots),
        "state_nonfinite": nonfinite,
        "unique_digest_sample": {
            name: len(digests[name]) for name in sorted(digests)[:GDN_SAMPLE_LAYERS]
        },
        "shared_backing_expected": bool(state.get("shared_backing_expected")),
        "alias_pair_count": alias_pair_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="reference run 的 run/ 目录")
    parser.add_argument("--candidate", type=Path, required=True, help="masked 候选 run 的 run/ 目录")
    parser.add_argument("--out", type=Path, required=True, help="summary.json 目标路径（已存在即拒绝）")
    parser.add_argument("--decodes", default=",".join(str(d) for d in DEFAULT_DECODES),
                        help="比较的代表性 decode 步（逗号分隔）")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在：拒绝覆盖旧摘要")
    decodes = tuple(int(part) for part in args.decodes.split(","))

    reference = load_run(args.reference, arm="patched-reference")
    candidate = load_run(args.candidate, arm="patched-masked")
    require_identity(reference, candidate)
    ref_npz = verify_capture(reference)
    cand_npz = verify_capture(candidate)
    layers = sorted(int(key) for key in reference["manifest"]["capture"]["expected_layers"])
    attention = compare_attention(ref_npz, cand_npz, decodes=decodes, layers=layers)
    logits, logits_steps = compare_logits(reference["run"], candidate["run"])
    ref_manifest = reference["manifest"]

    summary = {
        "schema": "attnview.p3-reference-summary/v1",
        "reference_run": str(args.reference),
        "masked_run": str(args.candidate),
        "head": ref_manifest["head"],
        "model_revision": ref_manifest["model"]["revision"],
        "vllm_revision": ref_manifest["pin_commit"],
        "decodes": list(decodes),
        "reference_exit_code": ref_manifest["exit_code"],
        "reference_manifest_failures": ref_manifest["failures"],
        "reference_completed_forwards": ref_manifest["reference"]["completed_forwards"],
        "reference_overrides": sum(
            1 for entry in ref_manifest["reference"]["ledger"]
            if entry.get("action") != "passthrough"),
        "reference_layers": len(ref_manifest["capture"]["expected_layers"]),
        "reference_nonfinite": sum(int(value) for value in ref_manifest["nonfinite"].values()),
        "gdn_state": summarize_gdn(reference),
        "attention_candidate_vs_reference": attention,
        "logits_candidate_vs_reference": logits,
        "logits_steps": logits_steps,
        "provenance": {
            "generator": {
                "script": "tools/p2-reference-summary.py",
                "sha256": sha256_file(Path(__file__)),
                "summary_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            },
            "reference_manifest_sha256": reference["manifest_sha256"],
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "reference_capture_npz_sha256": ref_manifest["capture"]["npz_sha256"],
            "candidate_capture_npz_sha256": candidate["manifest"]["capture"]["npz_sha256"],
            "input_hashes": ref_manifest["sources"]["input_hashes"],
            "metric": "diff = candidate - reference；relative_l2 = ||diff|| / ||reference||",
        },
        "boundary": "Diagnostic evidence only. No formal numeric threshold, quality acceptance, or performance claim.",
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("attention_candidate_vs_reference", "logits_candidate_vs_reference")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
