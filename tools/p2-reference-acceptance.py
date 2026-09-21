#!/usr/bin/env python3
"""按预冻结合同判定 held-out masked/reference 摘要；通过返回 0，拒绝返回 3。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attnview.numeric_acceptance import NumericAcceptanceError, evaluate_numeric_acceptance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="p2-reference-summary.py 生成的 summary.json")
    parser.add_argument("--contract", type=Path, required=True, help="预冻结数值合同 JSON")
    parser.add_argument("--out", type=Path, required=True, help="验收决定 JSON；已存在即拒绝覆盖")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在：拒绝覆盖旧验收决定")
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        decision = evaluate_numeric_acceptance(summary, contract)
    except (OSError, json.JSONDecodeError, NumericAcceptanceError) as exc:
        raise SystemExit(f"无法作出数值验收决定：{type(exc).__name__}: {exc}") from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: decision[key] for key in ("passed", "observed", "failures")}, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
