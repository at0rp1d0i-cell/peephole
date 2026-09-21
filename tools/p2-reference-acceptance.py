#!/usr/bin/env python3
"""从两份原始 run 重算摘要，再按预冻结合同判定 held-out masked/reference 数值验收。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from _lib import sha256_file  # noqa: E402
from attnview.numeric_acceptance import NumericAcceptanceError, evaluate_numeric_acceptance  # noqa: E402


def recompute_summary(reference_run: Path, candidate_run: Path) -> tuple[dict, dict]:
    """调用唯一摘要生成器现场读取 manifest/capture/logits；不接受外部 summary。"""
    with tempfile.TemporaryDirectory(prefix="attnview-acceptance-") as directory:
        summary_path = Path(directory) / "summary.json"
        command = [
            sys.executable,
            str(REPO / "tools/p2-reference-summary.py"),
            "--reference",
            str(reference_run),
            "--candidate",
            str(candidate_run),
            "--out",
            str(summary_path),
        ]
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise NumericAcceptanceError(
                f"原始 run 摘要重算失败(exit={result.returncode})：{detail}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = {
            "command": command,
            "summary_sha256": sha256_file(summary_path),
            "generator": summary.get("provenance", {}).get("generator"),
            "reference_manifest_sha256": summary.get("provenance", {}).get("reference_manifest_sha256"),
            "candidate_manifest_sha256": summary.get("provenance", {}).get("candidate_manifest_sha256"),
            "reference_capture_npz_sha256": summary.get("provenance", {}).get("reference_capture_npz_sha256"),
            "candidate_capture_npz_sha256": summary.get("provenance", {}).get("candidate_capture_npz_sha256"),
        }
    return summary, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True, help="patched-reference 的原始 run/ 目录")
    parser.add_argument("--candidate-run", type=Path, required=True, help="patched-masked 的原始 run/ 目录")
    parser.add_argument("--contract", type=Path, required=True, help="预冻结数值合同 JSON")
    parser.add_argument("--out", type=Path, required=True, help="验收决定 JSON；已存在即拒绝覆盖")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在：拒绝覆盖旧验收决定")
    try:
        summary, recomputation = recompute_summary(args.reference_run, args.candidate_run)
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        decision = evaluate_numeric_acceptance(summary, contract)
        decision["provenance"] = {
            "acceptance_script": str(Path(__file__).relative_to(REPO)),
            "acceptance_script_sha256": sha256_file(Path(__file__)),
            "contract": str(args.contract),
            "contract_sha256": sha256_file(args.contract),
            "recomputed_summary": recomputation,
        }
    except (OSError, json.JSONDecodeError, NumericAcceptanceError) as exc:
        raise SystemExit(f"无法作出数值验收决定：{type(exc).__name__}: {exc}") from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: decision[key] for key in ("passed", "observed", "failures")}, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
