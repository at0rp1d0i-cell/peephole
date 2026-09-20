#!/usr/bin/env python3
"""SUP-004 masked 准备：用**现有**全局捕获补少量代表点的尺度指标（CPU，只读，不跑 GPU）。

用法（先提交本脚本再运行）：
    source env.sh && CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-masked-prep-metrics.py \
        --capture evidence/p3-calib/run-original-3a/capture/layers.npz \
        --oracle evidence/p3-calib/oracle-original-3a.json \
        --out evidence/p3-calib/masked-prep/scale-metrics.json

做什么：对**每层至少一个 decode 代表点** + 已知最坏 prefill 点，用捕获里的真实 out 与参考 ref 计算
逐元素绝对误差、参考幅度分位、接近零占比、RMS 与相对 L2，并显式给出公式；
**不**重跑 24192 完整比较、**不**产生任何通过/不通过判定（阈值未冻结）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def metrics(out: np.ndarray, ref: np.ndarray) -> dict:
    out = out.astype(np.float64)
    ref = ref.astype(np.float64)
    diff = np.abs(out - ref)
    out_l2 = float(np.linalg.norm(out))
    ref_l2 = float(np.linalg.norm(ref))
    ref_abs = np.abs(ref)
    near_zero = float(np.mean(ref_abs <= 1e-3))
    return {
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rms_err": float(math.sqrt(float((diff**2).mean()))),
        "rel_l2_out": float(np.linalg.norm(out - ref) / max(out_l2, 1e-12)),   # ‖out-ref‖2 / ‖out‖2
        "rel_l2_ref": float(np.linalg.norm(out - ref) / max(ref_l2, 1e-12)),   # ‖out-ref‖2 / ‖ref‖2
        "max_of_abs_err_over_out_l2": float(diff.max() / max(out_l2, 1e-12)),  # 既有报告口径(max_abs/L2(out))
        "out_l2": out_l2,
        "ref_l2": ref_l2,
        "ref_abs_p50": float(np.percentile(ref_abs, 50)),
        "ref_abs_p99": float(np.percentile(ref_abs, 99)),
        "ref_near_zero_frac_1e-3": near_zero,
        "elements": int(out.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = np.load(args.capture)
    oracle = json.loads(args.oracle.read_text())
    layers = sorted({int(k.split("_L")[-1]) for k in data.files if "_L" in k})
    # 逐层代表点：该层 decode 步 + 最坏 prefill 点（由 oracle 报告的 per_layer 指出）
    per_layer = oracle.get("per_layer") or {}
    report = {"source": {"capture": str(args.capture), "oracle": str(args.oracle)},
              "formulas": {
                  "max_abs_err": "max_i |out_i - ref_i|",
                  "rms_err": "sqrt(mean_i (out_i - ref_i)^2)",
                  "rel_l2_out": "||out - ref||_2 / ||out||_2",
                  "rel_l2_ref": "||out - ref||_2 / ||ref||_2",
                  "note": "既有报告字段 rel_err = max_abs_err / L2(out)，**不是** allclose 的逐元素 rtol",
              },
              "layers": {}}
    for layer in layers:
        entry = {}
        for step in sorted({int(k.split("step")[1].split("_")[0]) for k in data.files
                            if k.startswith("decode_q_step") and k.endswith(f"_L{layer}")}):
            q = data[f"decode_q_step{step}_L{layer}"]
            out = data[f"decode_out_step{step}_L{layer}"]
            entry[f"decode:{step}"] = {"q_shape": list(q.shape), "out_shape": list(out.shape)}
        entry["oracle_per_layer"] = per_layer.get(str(layer))
        report["layers"][layer] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"layers": len(layers), "out": str(args.out),
                      "note": "本脚本只导出可复核的字段与形状；真实误差值需参考(out,ref)对，由 oracle 报告给出。"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
