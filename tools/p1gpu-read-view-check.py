#!/usr/bin/env python3
"""阶段 04：GPU 读取视图的独立数值验证入口。

候选路径：**完整 KV 常驻**，只构造分页读取表（block_table）+ 有效长度（seqused_k），
走安装版本的 vendored FA2（`vllm.vllm_flash_attn.flash_attn_varlen_func`，fa_version=2）。

参考路径：从 canonical KV 缓存按**原始逻辑位置**独立 gather 出可见集合，FP32 显式计算
softmax(q·Kᵀ·scale)·V；不使用候选读表反推可见集合，也不使用 SDPA。

用法：
    source env.sh && python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnview.gpukv import (  # noqa: E402
    build_read_table,
    canonical_slot,
    next_write_slot,
    normalize_retained_blocks,
    valid_count,
)

SENTINEL = 8.0  # 未写槽 / 尾块填充：被读到就会显著放大误差
DOCUMENT_BLOCKS = (1, 2, 3, 4, 5)  # local 模式排除的"文档"块
DOCUMENT_AMPLIFY = 5.0


# ---------------------------------------------------------------- 基础构件


def make_generator(seed: int, tag: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed * 100000 + tag)


def build_cache(nb: int, block_size: int, kvh: int, d: int, kv_len: int, seed: int):
    """构造 canonical KV：有效位置写真实值，其余保持哨兵。"""
    gen = make_generator(seed, 1)
    k = torch.full((nb, block_size, kvh, d), SENTINEL, dtype=torch.bfloat16)
    v = torch.full((nb, block_size, kvh, d), SENTINEL, dtype=torch.bfloat16)
    for i in range(nb):
        count = valid_count(i, kv_len, block_size)
        if count == 0:
            continue
        amp = DOCUMENT_AMPLIFY if i in DOCUMENT_BLOCKS else 1.0
        k[i, :count] = (torch.randn(count, kvh, d, generator=gen, dtype=torch.float32) * amp).to(torch.bfloat16)
        v[i, :count] = (torch.randn(count, kvh, d, generator=gen, dtype=torch.float32) * amp).to(torch.bfloat16)
    return k, v


def make_query(num_heads: int, head_dim: int, seed: int, batch: int = 1):
    gen = make_generator(seed, 2)
    q = torch.randn(batch, num_heads, head_dim, generator=gen, dtype=torch.float32)
    return q.to(torch.bfloat16)


def gather_visible(k, v, blocks, l2p, kv_len, block_size):
    """按原始逻辑位置 gather 出可见集合（参考 oracle 专用，候选路径不得这么做）。"""
    ks, vs = [], []
    for i in blocks:
        count = valid_count(i, kv_len, block_size)
        ks.append(k[l2p[i], :count].float())
        vs.append(v[l2p[i], :count].float())
    return torch.cat(ks), torch.cat(vs)


def reference_attention(q, K, V, scale: float):
    """FP32 显式缩放点积 + softmax + V 加权和。

    GQA：q head h 使用 kv head ``h // (H / KVH)``（连续分组）。每个 q head 只对自己的
    kv head 做加权和；把 kv head 维留在 einsum 输入里会被隐式求和，必须显式展开成 [S, H, D]。
    """
    H = q.shape[0]
    KVH = K.shape[1]
    if H % KVH:
        raise SystemExit(f"q head 数 {H} 不是 kv head 数 {KVH} 的整数倍")
    kvh_of_head = torch.arange(H, device=K.device) // (H // KVH)
    K_expanded = K[:, kvh_of_head, :]
    V_expanded = V[:, kvh_of_head, :]
    scores = torch.einsum("hd,shd->hs", q.float(), K_expanded) * scale
    probs = torch.softmax(scores, dim=1)
    return torch.einsum("hs,shd->hd", probs, V_expanded)


def reference_attention_per_head(q, K, V, scale: float):
    """逐 head 逐 kv head 的参考实现（独立写法，用于自检批量参考的 GQA 处理）。"""
    H, KVH = q.shape[0], K.shape[1]
    rows = []
    for h in range(H):
        kvh = h // (H // KVH)
        probs = torch.softmax((q[h].float() @ K[:, kvh].T) * scale, dim=0)
        rows.append(probs @ V[:, kvh])
    return torch.stack(rows)


def run_candidate(q, k, v, tables, scale: float, fa_version: int = 2):
    """走真实分页 FA2；读取表不含 -1（短行用自身最后一块补齐）。"""
    batch = len(tables)
    width = max(t.width for t in tables)
    block_table = torch.empty((batch, width), dtype=torch.int32)
    for row, table in enumerate(tables):
        ids = list(table.physical_blocks)
        ids += [ids[-1]] * (width - len(ids))
        block_table[row] = torch.tensor(ids, dtype=torch.int32)
    seqused_k = torch.tensor([t.seqused_k for t in tables], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    out = torch.empty((batch, q.shape[1], q.shape[2]), dtype=torch.bfloat16, device="cuda")
    result = flash_attn_varlen_func(
        q=q.to("cuda"),
        k=k.to("cuda"),
        v=v.to("cuda"),
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=int(seqused_k.max().item()),
        softmax_scale=scale,
        causal=True,
        window_size=None,
        block_table=block_table.to("cuda"),
        softcap=0.0,
        fa_version=fa_version,
    )
    output = result if torch.is_tensor(result) else result[0]
    return output, block_table, seqused_k


def error_stats(candidate: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    diff = candidate.float().detach().cpu() - ref.float().detach().cpu()
    max_abs_ref = float(ref.abs().max().item())
    stats = {
        "max_abs": float(diff.abs().max().item()),
        "rms": float(math.sqrt(float((diff ** 2).mean().item()))),
        "max_abs_ref": max_abs_ref,
        "relative_max": float(diff.abs().max().item()) / max_abs_ref if max_abs_ref else 0.0,
    }
    try:
        torch.testing.assert_close(candidate.float().cpu(), ref.float().cpu(), atol=atol, rtol=rtol)
        stats["within_tolerance"] = True
        stats["failure"] = None
    except AssertionError as exc:  # 保留反例，不放宽容差
        stats["within_tolerance"] = False
        stats["failure"] = str(exc).splitlines()[0]
    return stats


def cache_digest(tensor: torch.Tensor) -> str:
    """缓存内容指纹（bf16 先无损升到 fp32 再取字节）。"""
    payload = tensor.float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:32]


def table_record(table) -> dict:
    return {
        "physical_blocks": list(table.physical_blocks),
        "width": table.width,
        "seqused_k": table.seqused_k,
        "effective_per_block": list(table.effective_per_block),
        "tail_len": table.tail_len,
        "dropped_empty_blocks": list(table.dropped_empty_blocks),
    }


# ---------------------------------------------------------------- 用例


def run_case(case: dict, cfg: dict, seed: int) -> dict:
    cc = cfg["canonical_cache"]
    tc = cfg["tensor_contract"]
    nb = cc["physical_blocks"]
    l2p = tuple(cc["logical_to_physical"])
    block_size = int(case.get("block_size", tc["block_sizes"]["main"]))
    num_heads = tc["num_heads"]
    kvh = tc["num_kv_heads"]
    head_dim = tc["head_dim"]
    scale = head_dim ** -0.5
    atol, rtol = cfg["tolerance"]["atol"], cfg["tolerance"]["rtol"]
    case_id = case["id"]
    record: dict = {"id": case_id, "seed": seed, "block_size": block_size, "kv_len": case.get("kv_len")}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    kv_len = int(case.get("kv_len") or max(r["kv_len"] for r in case.get("rows", [])))
    k, v = build_cache(nb, block_size, kvh, head_dim, kv_len, seed)
    q = make_query(num_heads, head_dim, seed, batch=int(case.get("batch", 1)))
    digest_before = (cache_digest(k), cache_digest(v))

    if case_id == "batch2_rows":
        rows = case["rows"]
        tables = []
        for row in rows:
            blocks = normalize_retained_blocks(row["retained_logical_blocks"], nb)
            tables.append(build_read_table(blocks, l2p, row["kv_len"], block_size))
        out, block_table, seqused_k = run_candidate(q, k, v, tables, scale)
        record["rows"] = []
        for index, (row, table) in enumerate(zip(rows, tables)):
            blocks = normalize_retained_blocks(row["retained_logical_blocks"], nb)
            K, V = gather_visible(k, v, blocks, l2p, row["kv_len"], block_size)
            ref = reference_attention(q[index], K, V, scale)
            stats = error_stats(out[index], ref, atol, rtol)
            single, _, _ = run_candidate(q[index: index + 1], k, v, [table], scale)
            stats["row_isolation_max_abs"] = float(
                (single[0].float() - out[index].float()).abs().max().item()
            )
            record["rows"].append({**table_record(table), **stats, "retained": list(blocks)})
        record["block_table_no_minus_one"] = bool((block_table >= 0).all().item())
        record["seqused_k"] = [int(x) for x in seqused_k.tolist()]
        record["within_tolerance"] = all(r["within_tolerance"] for r in record["rows"])

    elif case_id in ("append_cross_block", "original_write_position"):
        steps = int(case.get("append", 1))
        retained = normalize_retained_blocks(
            case.get("retained_logical_blocks", [0, 1, 2]), nb
        )
        base_len = kv_len - steps if case_id == "original_write_position" else kv_len - 1
        k, v = build_cache(nb, block_size, kvh, head_dim, base_len, seed)
        digest_before = None
        gen = make_generator(seed, 3)
        record["steps"] = []
        designated = []
        for step in range(steps):
            logical, offset, slot = next_write_slot(base_len + step, block_size, l2p)
            designated.append({"logical_block": logical, "offset": offset, "physical_slot": slot})
            cur_len = base_len + step
            table = build_read_table(retained, l2p, cur_len, block_size)
            flat_k = k.view(-1, kvh, head_dim)
            flat_v = v.view(-1, kvh, head_dim)
            out, block_table, seqused_k = run_candidate(q, k, v, [table], scale)
            K, V = gather_visible(k, v, retained, l2p, cur_len, block_size)
            ref = reference_attention(q[0], K, V, scale)
            stats = error_stats(out[0], ref, atol, rtol)
            record["steps"].append(
                {
                    "step": step,
                    "kv_len": cur_len,
                    **table_record(table),
                    "write_slot": {"logical_block": logical, "offset": offset, "physical_slot": slot},
                    **stats,
                }
            )
            # 追加一个 token：只写预定 canonical slot
            before = flat_k.clone()
            new_k = torch.randn(1, kvh, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
            new_v = torch.randn(1, kvh, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
            flat_k[slot] = new_k[0]
            flat_v[slot] = new_v[0]
            changed = (
                (flat_k != before).any(dim=(1, 2)).nonzero().flatten().tolist()
            )
            record["steps"][-1]["cache_slots_changed_by_append"] = changed
            record["steps"][-1]["append_wrote_only_designated_slot"] = changed == [slot]
        # 追加循环只验证了"每次写入前"的读取；这里补一次**全部追加完成后**的读取，
        # 覆盖"追加 token 跨块、尾长随之变化"的要求
        final_len = base_len + steps
        final_table = build_read_table(retained, l2p, final_len, block_size)
        final_out, _, final_seqused = run_candidate(q, k, v, [final_table], scale)
        K, V = gather_visible(k, v, retained, l2p, final_len, block_size)
        final_stats = error_stats(final_out[0], reference_attention(q[0], K, V, scale), atol, rtol)
        record["final_read"] = {
            "kv_len": final_len,
            **table_record(final_table),
            "backend_seqused_k": [int(x) for x in final_seqused.tolist()],
            **final_stats,
        }
        record["designated_write_slots"] = designated
        record["all_writes_used_original_positions"] = all(
            s["physical_slot"] == canonical_slot(base_len + i, block_size, l2p)
            for i, s in enumerate(designated)
        )
        record["within_tolerance"] = bool(
            all(s["within_tolerance"] and s["append_wrote_only_designated_slot"] for s in record["steps"])
            and record["final_read"]["within_tolerance"]
            and record["all_writes_used_original_positions"]
        )

    else:
        raw_blocks = case["retained_logical_blocks_raw"] if "retained_logical_blocks_raw" in case else case["retained_logical_blocks"]
        retained = normalize_retained_blocks(raw_blocks, nb)
        table = build_read_table(retained, l2p, kv_len, block_size)
        block_index = block_size
        record["retained"] = list(retained)
        record.update(table_record(table))
        out, block_table, seqused_k = run_candidate(q, k, v, [table], scale)
        record["block_table"] = block_table.tolist()
        record["block_table_no_minus_one"] = bool((block_table >= 0).all().item())
        record["backend_seqused_k"] = [int(x) for x in seqused_k.tolist()]
        K, V = gather_visible(k, v, retained, l2p, kv_len, block_size)
        ref = reference_attention(q[0], K, V, scale)
        record["reference_selfcheck_max_abs"] = float(
            (ref - reference_attention_per_head(q[0], K, V, scale)).abs().max().item()
        )
        record.update(error_stats(out[0], ref, atol, rtol))
        if case.get("sensitive"):
            full_blocks = tuple(range(nb))
            full_table = build_read_table(normalize_retained_blocks(full_blocks, nb), l2p, kv_len, block_size)
            full_out, _, _ = run_candidate(q, k, v, [full_table], scale)
            record["full_vs_masked_max_abs"] = float(
                (full_out[0].float() - out[0].float()).abs().max().item()
            )
            record["mask_has_measurable_effect"] = record["full_vs_masked_max_abs"] > 10 * atol
            record["within_tolerance"] = record["within_tolerance"] and record["mask_has_measurable_effect"]
        if case_id == "global_recovery":
            full_table = build_read_table(normalize_retained_blocks(range(nb), nb), l2p, kv_len, block_size)
            full_out, _, _ = run_candidate(q, k, v, [full_table], scale)
            K, V = gather_visible(k, v, tuple(range(nb)), l2p, kv_len, block_size)
            ref_full = reference_attention(q[0], K, V, scale)
            record["recovery"] = error_stats(full_out[0], ref_full, atol, rtol)
            record["within_tolerance"] = record["within_tolerance"] and record["recovery"]["within_tolerance"]

    record["cache_unchanged_by_read"] = (
        None if digest_before is None else (cache_digest(k), cache_digest(v)) == digest_before
    )
    record["peak_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 2)
    record["seconds"] = round(time.time() - started, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1-gpu/read-view-check.json")
    parser.add_argument("--evidence", default="evidence/p1-gpu")
    parser.add_argument("--only", default=None, help="逗号分隔的用例 id")
    args = parser.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    cases = cfg["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    seeds = cfg["seeds"]

    print(f"torch {torch.__version__} | vllm entry vllm.vllm_flash_attn.flash_attn_varlen_func | fa_version=2")
    print(f"device {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    print(f"cases {len(cases)} × seeds {seeds} | atol={cfg['tolerance']['atol']} rtol={cfg['tolerance']['rtol']}\n")

    header = f"{'case':32s} {'seed':>4s} {'seqused_k':>9s} {'tail':>4s} {'max_abs':>9s} {'rms':>9s} 结果"
    print(header)
    print("-" * len(header))
    per_seed: dict[int, list[dict]] = {}
    failures = 0
    for seed in seeds:
        per_seed[seed] = []
        for case in cases:
            record = run_case(case, cfg, seed)
            per_seed[seed].append(record)
            ok = record.get("within_tolerance")
            failures += 0 if ok else 1
            seqused = record.get("seqused_k")
            if isinstance(seqused, list):
                seqused = max(seqused)
            print(
                f"{record['id']:32s} {seed:>4d} {str(seqused):>9s} {str(record.get('tail_len', '-')):>4s} "
                f"{record.get('max_abs', float('nan')):>9.5f} {record.get('rms', float('nan')):>9.6f} "
                f"{'PASS' if ok else 'FAIL'}"
            )
            out_dir = ROOT / args.evidence
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"cases-seed{seed}.json").write_text(
                json.dumps(per_seed[seed], indent=2, ensure_ascii=False, default=str)
            )

    summary = {
        "config": args.config,
        "config_sha256": hashlib.sha256((ROOT / args.config).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "vllm_entry": "vllm.vllm_flash_attn.flash_attn_varlen_func",
        "fa_version": 2,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "platform": platform.platform(),
        "seeds": seeds,
        "cases_run": sum(len(v) for v in per_seed.values()),
        "failures": failures,
        "tolerance": cfg["tolerance"],
        "records": {str(seed): per_seed[seed] for seed in seeds},
    }
    out_dir = ROOT / args.evidence
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n{summary['cases_run']} case-runs, {failures} failures -> {out_dir}/summary.json")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
