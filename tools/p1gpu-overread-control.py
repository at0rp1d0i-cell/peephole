#!/usr/bin/env python3
"""阶段 04 返工负对照（诊断，非用例表通过条件）：证明"越读"会被本套检查抓到。

对同一份常驻 GPU 缓存、同一读取表，只把 ``seqused_k`` 人为放大 100（读入尾块之后的哨兵槽）：
输出必须显著偏离独立参考。用于说明本阶段用例的敏感度，并反证正确调用下后端确实不读
``seqused_k`` 之后的槽。
"""

from __future__ import annotations

import argparse
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

from attnview.gpuoracle import attention_fp32, build_oracle_view, gather_positions  # noqa: E402
from attnview.gpukv import read_table_from_read_view  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1-gpu/read-view-check-v3.json")
    parser.add_argument("--evidence", default="evidence/p1-gpu-v3")
    args = parser.parse_args()
    out_dir = ROOT / args.evidence
    manifest = gpu_check.write_manifest(out_dir, ROOT / args.config, start=gpu_check.now_cst(),
                                        extra={"script": "p1gpu-overread-control.py"})
    print(f"run manifest: HEAD={manifest['head_commit'][:12]} clean={manifest['worktree_clean']} "
          f"config_sha256={manifest['config_sha256'][:16]} start={manifest['start_time_cst']}")
    cfg = json.loads((ROOT / args.config).read_text())
    layout = cfg["layouts"]["tail1"]
    cc, tc = cfg["canonical_cache"], cfg["tensor_contract"]
    l2p = tuple(cc["logical_to_physical"])
    block_size, nb = tc["block_sizes"]["main"], cc["physical_blocks"]
    kvh, heads, dim = tc["num_kv_heads"], tc["num_heads"], tc["head_dim"]
    scale = dim ** -0.5
    atol, rtol = cfg["tolerance"]["atol"], cfg["tolerance"]["rtol"]
    kv_len = int(layout["prompt_len"])

    results = []
    for seed in cfg["fixture_truth"]["seeds"]:
        runner = gpu_check.CaseRunner(cfg, {"id": "negative_control", "kind": "single", "layout": "tail1",
                                            "mode": "local", "refs": [], "kv_len": kv_len}, seed)
        k_true, v_true = gpu_check.make_truth(kv_len, kvh, dim, seed)
        k_cpu, v_cpu, verified = gpu_check.scatter_and_verify(k_true, v_true, kv_len, nb, block_size, l2p)
        k_gpu, v_gpu = k_cpu.to("cuda"), v_cpu.to("cuda")
        q_gpu = torch.randn(1, heads, dim, generator=gpu_check.make_generator(seed, 2),
                            dtype=torch.float32).to(torch.bfloat16).to("cuda")
        view = runner.read_view("local", (), kv_len, kv_len)
        table = read_table_from_read_view(view, l2p)
        oracle = build_oracle_view(mode="local", refs=(), segments=tuple(tuple(s) for s in layout["segments"]),
                                   local_window=(layout["question_start"], kv_len), prompt_len=kv_len,
                                   attention_kv_len=kv_len, kernel_block_size=block_size,
                                   logical_to_physical=l2p, sink=tuple(layout["sink"]))
        k_sel, v_sel = gather_positions(k_true, v_true, oracle.positions)
        ref = attention_fp32(q_gpu[0].float().cpu(), k_sel, v_sel, scale)

        honest, _, _ = gpu_check.run_candidate(q_gpu, k_gpu, v_gpu, [table], scale)
        import dataclasses
        over = dataclasses.replace(table, seqused_k=table.seqused_k + 100)
        overread, _, _ = gpu_check.run_candidate(q_gpu, k_gpu, v_gpu, [over], scale)

        results.append({
            "seed": seed, "kv_len": kv_len, "tail_len": table.tail_len,
            "fill_verified_positions": verified,
            "seqused_k_honest": table.seqused_k, "seqused_k_overread": over.seqused_k,
            "honest": gpu_check.error_stats(honest[0], ref, atol, rtol),
            "overread": gpu_check.error_stats(overread[0], ref, atol, rtol),
        })
        h, o = results[-1]["honest"], results[-1]["overread"]
        print(f"seed {seed}: 正确 seqused_k={table.seqused_k} max_abs={h['max_abs']:.5f} "
              f"({'PASS' if h['within_tolerance'] else 'FAIL'}) | 越读 +100 max_abs={o['max_abs']:.4f} "
              f"({'PASS(不应出现)' if o['within_tolerance'] else 'FAIL(符合预期)'})")

    out = ROOT / args.evidence / "negative-control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"purpose": "越读负对照（v2 夹具）", "records": results}, indent=2, ensure_ascii=False))
    detected = all((not r["overread"]["within_tolerance"]) and r["honest"]["within_tolerance"] for r in results)
    gpu_check.write_manifest(out_dir, ROOT / args.config, start=manifest["start_time_cst"], end=gpu_check.now_cst(),
                             exit_code=0 if detected else 1, extra={"script": "p1gpu-overread-control.py",
                                                                   "detected": detected})
    print(f"\n哨兵抓越读：{'成立' if detected else '不成立'} -> {out}")
    print(f"结束时间 {gpu_check.now_cst()}")
    return 0 if detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
