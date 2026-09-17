#!/usr/bin/env python3
"""阶段 04 负对照（诊断，非用例表通过条件）：证明"越读"会被本套检查抓到。

对同一个 canonical 缓存、同一个读取表，只把 ``seqused_k`` 人为放大 100：
尾块未写槽里是哨兵 8.0，若被读入，输出必须显著偏离独立参考。
输出 `evidence/p1-gpu/negative-control.json`，用于说明本阶段用例的敏感度。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("gpu_check", ROOT / "tools" / "p1gpu-read-view-check.py")
gpu_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_check)

from attnview.gpukv import build_read_table, normalize_retained_blocks  # noqa: E402


def main() -> int:
    cfg = json.loads((ROOT / "configs" / "p1-gpu" / "read-view-check.json").read_text())
    cc, tc = cfg["canonical_cache"], cfg["tensor_contract"]
    l2p = tuple(cc["logical_to_physical"])
    block_size = tc["block_sizes"]["main"]
    nb, kvh, head_dim = cc["physical_blocks"], tc["num_kv_heads"], tc["head_dim"]
    scale = head_dim ** -0.5
    atol, rtol = cfg["tolerance"]["atol"], cfg["tolerance"]["rtol"]

    results = []
    for seed in cfg["seeds"]:
        kv_len = 1569  # 尾长 1：块 2 只有 1 个有效 token，其余是哨兵
        retained = normalize_retained_blocks([0, 1, 2], nb)
        k, v = gpu_check.build_cache(nb, block_size, kvh, head_dim, kv_len, seed)
        q = gpu_check.make_query(tc["num_heads"], head_dim, seed)
        table = build_read_table(retained, l2p, kv_len, block_size)
        K, V = gpu_check.gather_visible(k, v, retained, l2p, kv_len, block_size)
        ref = gpu_check.reference_attention(q[0], K, V, scale)

        honest, _, _ = gpu_check.run_candidate(q, k, v, [table], scale)

        import dataclasses

        over = dataclasses.replace(table, seqused_k=table.seqused_k + 100)  # 故意越读 100 个槽
        overread, _, _ = gpu_check.run_candidate(q, k, v, [over], scale)

        results.append(
            {
                "seed": seed,
                "kv_len": kv_len,
                "tail_len": table.tail_len,
                "seqused_k_honest": table.seqused_k,
                "seqused_k_overread": over.seqused_k,
                "honest": gpu_check.error_stats(honest[0], ref, atol, rtol),
                "overread": gpu_check.error_stats(overread[0], ref, atol, rtol),
            }
        )
        print(
            f"seed {seed}: 正确 seqused_k={table.seqused_k} max_abs="
            f"{results[-1]['honest']['max_abs']:.5f} ({'PASS' if results[-1]['honest']['within_tolerance'] else 'FAIL'})"
            f" | 越读 +100 max_abs={results[-1]['overread']['max_abs']:.4f} "
            f"({'PASS(不应出现)' if results[-1]['overread']['within_tolerance'] else 'FAIL(符合预期)'})"
        )

    out = ROOT / "evidence" / "p1-gpu" / "negative-control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"purpose": "越读负对照", "records": results}, indent=2, ensure_ascii=False))
    detected = all((not r["overread"]["within_tolerance"]) and r["honest"]["within_tolerance"] for r in results)
    print(f"\n哨兵抓越读：{'成立' if detected else '不成立'} -> {out}")
    return 0 if detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
