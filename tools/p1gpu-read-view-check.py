#!/usr/bin/env python3
"""阶段 04 返工：GPU 读取视图数值验证入口（SUP-003-R1）。

三条硬要求：

1. **单 GPU 常驻缓存**：一条轨迹只分配/上传一次 K/V，候选函数只接受已有 CUDA 缓存并断言
   `device` 与 `data_ptr` 不变；读前后核对 **K 与 V 两者**，追加前后核对 **两者** 变化槽恰为规定 slot。
2. **阶段 03 数据面连通**：可见集合来自 `attnview.readview.build_read_view`（TokenLayout/ViewInputs），
   再由 `attnview.gpukv.read_table_from_read_view` 转成无 `-1` 的读取表。
3. **参考独立**：可见位置与期望表由 `attnview.gpuoracle` 从语义输入 + 逐位置 mask 独立推导
   （不 import 候选转换、不由候选读表反推），注意力用 FP32 逐 head 显式计算。

判据全部通过（`attnview.gpucheck.summarize`）才 exit 0。
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

from attnview.gpucheck import summarize  # noqa: E402
from attnview.gpuoracle import (  # noqa: E402
    attention_fp32,
    attention_sdpa_fp32,
    build_oracle_view,
    gather_positions,
)
from attnview.gpukv import next_write_slot, read_table_from_read_view  # noqa: E402
from attnview.readview import TokenLayout, ViewInputs, build_read_view  # noqa: E402

SENTINEL = 8.0
MODE_GLOBAL, MODE_FOCUS, MODE_LOCAL = "global", "focus", "local"


# ------------------------------------------------------------------ 夹具与真值


def make_generator(seed: int, tag: int):
    return torch.Generator(device="cpu").manual_seed(seed * 100000 + tag)


def make_truth(kv_len: int, kvh: int, d: int, seed: int, *, amplify_blocks=(), amplify=1.0, block_size: int = 784):
    """逻辑真值 K/V：按**逻辑位置**生成，放大只作用于指定逻辑块（敏感负对照专用）。"""
    gen = make_generator(seed, 1)
    k = torch.randn(kv_len, kvh, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(kv_len, kvh, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    if amplify_blocks:
        mask = torch.zeros(kv_len, dtype=torch.bool)
        for block in amplify_blocks:
            mask[block * block_size : min((block + 1) * block_size, kv_len)] = True
        k[mask] = (k[mask].float() * amplify).to(torch.bfloat16)
        v[mask] = (v[mask].float() * amplify).to(torch.bfloat16)
    return k, v


def scatter_and_verify(k_true, v_true, written: int, nb: int, block_size: int, l2p):
    """按 canonical 映射把真值散布到 CPU 物理缓存，并**逐位置**核对。"""
    k_cache = torch.full((nb, block_size, k_true.shape[1], k_true.shape[2]), SENTINEL, dtype=torch.bfloat16)
    v_cache = torch.full_like(k_cache, SENTINEL)
    for block in range(nb):
        low, high = block * block_size, min((block + 1) * block_size, written)
        if high <= low:
            continue
        k_cache[l2p[block], : high - low] = k_true[low:high]
        v_cache[l2p[block], : high - low] = v_true[low:high]
    # 逐位置核对：物理[l2p[i], off] == 逻辑[i*b+off]，K 与 V 两者
    checked = 0
    for block in range(nb):
        low, high = block * block_size, min((block + 1) * block_size, written)
        if high <= low:
            continue
        span = high - low
        if not torch.equal(k_cache[l2p[block], :span], k_true[low:high]):
            raise AssertionError(f"K 散布错位：逻辑块 {block} -> 物理块 {l2p[block]}")
        if not torch.equal(v_cache[l2p[block], :span], v_true[low:high]):
            raise AssertionError(f"V 散布错位：逻辑块 {block} -> 物理块 {l2p[block]}")
        checked += span
        # 有效尾之后的槽必须仍是哨兵（否则会读到"既非真值也非哨兵"的内容）
        tail = k_cache[l2p[block], span:]
        if tail.numel() and not bool((tail == SENTINEL).all()):
            raise AssertionError(f"逻辑块 {block} 有效尾之后不是哨兵")
        tail_v = v_cache[l2p[block], span:]
        if tail_v.numel() and not bool((tail_v == SENTINEL).all()):
            raise AssertionError(f"逻辑块 {block} 有效尾之后不是哨兵（V）")
    # 未写物理块必须整段是哨兵：判据按"哪些物理块承载已写逻辑块"，与物理块序号无关
    used_physical = {l2p[b] for b in range(nb) if min((b + 1) * block_size, written) > b * block_size}
    for physical in range(nb):
        if physical in used_physical:
            continue
        if not bool((k_cache[physical] == SENTINEL).all()) or not bool((v_cache[physical] == SENTINEL).all()):
            raise AssertionError(f"未写物理块 {physical} 含非哨兵内容（K 或 V）")
    if checked != written:
        raise AssertionError(f"散布核对位置数 {checked} != 已写长度 {written}")
    return k_cache, v_cache, checked


# ------------------------------------------------------------------ 候选调用


def run_candidate(q_gpu, k_gpu, v_gpu, tables, scale):
    """真实分页 FA2；断言缓存常驻不变（device + data_ptr）。"""
    if not (k_gpu.is_cuda and v_gpu.is_cuda and q_gpu.is_cuda):
        raise AssertionError("候选路径要求 q/k/v 都在 CUDA 上常驻")
    before = (k_gpu.device, k_gpu.data_ptr(), v_gpu.device, v_gpu.data_ptr())
    batch = len(tables)
    width = max(t.width for t in tables)
    block_table = torch.tensor([t.padded_row(width) for t in tables], dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor([t.seqused_k for t in tables], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    out = torch.empty((batch, q_gpu.shape[1], q_gpu.shape[2]), dtype=torch.bfloat16, device="cuda")
    result = flash_attn_varlen_func(
        q=q_gpu,
        k=k_gpu,
        v=v_gpu,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=int(seqused_k.max().item()),
        softmax_scale=scale,
        causal=True,
        window_size=None,
        block_table=block_table,
        softcap=0.0,
        fa_version=2,
    )
    output = result if torch.is_tensor(result) else result[0]
    after = (k_gpu.device, k_gpu.data_ptr(), v_gpu.device, v_gpu.data_ptr())
    if before != after:
        raise AssertionError(f"读取调用换了缓存对象：{before} -> {after}")
    return output, block_table, seqused_k


def error_stats(candidate: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    """数值判据 + 完整错误信息 + 最差偏差位置。"""
    cand = candidate.detach().float().cpu()
    reference = ref.detach().float().cpu()
    diff = (cand - reference).abs()
    flat = int(diff.argmax().item())
    worst = torch.unravel_index(torch.tensor(flat), diff.shape)
    stats = {
        "max_abs": float(diff.max().item()),
        "rms": float(math.sqrt(float((diff ** 2).mean().item()))),
        "max_abs_ref": float(reference.abs().max().item()),
        "worst_index": [int(x) for x in worst],
        "worst_candidate": float(cand[tuple(worst)].item()),
        "worst_reference": float(reference[tuple(worst)].item()),
    }
    try:
        torch.testing.assert_close(cand, reference, atol=atol, rtol=rtol)
        stats["within_tolerance"] = True
        stats["failure"] = None
    except AssertionError as exc:
        stats["within_tolerance"] = False
        stats["failure"] = str(exc)  # 完整信息，不截首行
    return stats


def changed_slots(before: torch.Tensor, after: torch.Tensor, kvh: int, d: int) -> list[int]:
    """返回发生变化的物理 slot 列表（先展平成 [slots, KVH, D]）。"""
    flat_before = before.view(-1, kvh, d)
    flat_after = after.view(-1, kvh, d)
    diff = (flat_after != flat_before).any(dim=2).any(dim=1)
    return [int(x) for x in diff.nonzero().flatten().tolist()]


# ------------------------------------------------------------------ 用例执行


class CaseRunner:
    def __init__(self, cfg, case, seed):
        self.cfg = cfg
        self.case = case
        self.seed = seed
        tc, cc = cfg["tensor_contract"], cfg["canonical_cache"]
        self.nb = cc["physical_blocks"]
        self.l2p = tuple(cc["logical_to_physical"])
        self.kvh, self.heads, self.d = tc["num_kv_heads"], tc["num_heads"], tc["head_dim"]
        self.scale = self.d ** -0.5
        self.atol, self.rtol = cfg["tolerance"]["atol"], cfg["tolerance"]["rtol"]
        self.block_size = int(case.get("block_size") or cfg["layouts"][case["layout"]].get("block_size", 784))
        self.layout = cfg["layouts"][case["layout"]]
        self.distribution = cfg["fixture_truth"]["distribution"]
        self.sensitive_blocks = tuple(self.distribution["sensitive_amplify"]["blocks"])
        self.amplify = float(self.distribution["sensitive_amplify"]["factor"])

    # --- 语义输入 → 阶段 03 ReadView（数据面） ---
    def read_view(self, mode, refs, kv_len, prompt_len):
        layout = TokenLayout(
            prompt_len=prompt_len,
            segment_spans=tuple(tuple(s) for s in self.layout["segments"]),
            local_window_span=(self.layout["question_start"], prompt_len),
            sink_span=tuple(self.layout["sink"]),
        )
        inputs = ViewInputs(
            mode=mode,
            refs=tuple(refs),
            layout=layout,
            attention_kv_len=kv_len,
            canonical_blocks=tuple(self.l2p),
            kernel_block_size=self.block_size,
            max_width=len(self.l2p),
        )
        return build_read_view(inputs)

    def oracle(self, mode, refs, kv_len, prompt_len, explicit_blocks=None):
        return build_oracle_view(
            explicit_blocks=explicit_blocks,
            mode=mode,
            refs=tuple(refs),
            segments=tuple(tuple(s) for s in self.layout["segments"]),
            local_window=(self.layout["question_start"], prompt_len),
            prompt_len=prompt_len,
            attention_kv_len=kv_len,
            kernel_block_size=self.block_size,
            logical_to_physical=self.l2p,
            sink=tuple(self.layout["sink"]),
        )

    def run(self) -> dict:
        case, seed = self.case, self.seed
        kind = case["kind"]
        kv_plan = self._kv_plan()
        max_kv = max(kv_plan)
        amplify_blocks = self.sensitive_blocks if case.get("amplify_excluded_blocks") else ()
        k_true, v_true = make_truth(
            max_kv, self.kvh, self.d, seed,
            amplify_blocks=amplify_blocks, amplify=self.amplify, block_size=self.block_size,
        )
        k_cpu, v_cpu, verified = scatter_and_verify(k_true, v_true, kv_plan[0], self.nb, self.block_size, self.l2p)
        k_gpu = k_cpu.to("cuda")
        v_gpu = v_cpu.to("cuda")
        q_gpu = torch.randn(1, self.heads, self.d, generator=make_generator(seed, 2), dtype=torch.float32)\
            .to(torch.bfloat16).to("cuda")

        record = {
            "id": case["id"], "seed": seed, "kind": kind, "block_size": self.block_size,
            "layout": case["layout"], "prompt_len": self.layout["prompt_len"],
            "distribution": "sensitive_amplified" if amplify_blocks else "normal",
            "fill_verified_positions": verified,
            "cache_k_digest": hashlib.sha256(k_cpu.float().numpy().tobytes()).hexdigest()[:32],
            "cache_v_digest": hashlib.sha256(v_cpu.float().numpy().tobytes()).hexdigest()[:32],
            "data_ptr": {"k": k_gpu.data_ptr(), "v": v_gpu.data_ptr()},
            "measurements": [],
        }
        if kind == "batch2":
            self._run_batch2(record, k_true, v_true, k_gpu, v_gpu, q_gpu, kv_plan)
        else:
            self._run_steps(record, case, k_true, v_true, k_gpu, v_gpu, q_gpu, kv_plan)
        record["peak_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 2)
        return record

    def _kv_plan(self) -> list[int]:
        case = self.case
        if case["kind"] == "trajectory":
            return [int(s["kv_len"]) for s in case["steps"]]
        if case["kind"] == "batch2":
            return [max(int(r["kv_len"]) for r in case["rows"])]
        return [int(case["kv_len"])]

    def _measure(self, *, label, mode, refs, kv_len, prompt_len, k_true, v_true, k_gpu, v_gpu, q_row, table=None,
                 extra_ok=None, appended=False):
        """一步：数据面交叉核对 → GPU 读取 → 独立参考 → 数值与内容判据。"""
        view = None
        if table is None:
            view = self.read_view(mode, refs, kv_len, prompt_len)
            table = read_table_from_read_view(view, self.l2p)
        oracle = self.oracle(
            mode, refs, kv_len, prompt_len,
            explicit_blocks=self.case.get("logical_blocks") if self.case["kind"] == "arbitrary" else None,
        )
        candidate_slot = next_write_slot(kv_len, self.block_size, self.l2p)

        blocks_match = tuple(table.visible_blocks) == tuple(oracle.visible_blocks)
        physical_match = tuple(table.physical_blocks) == tuple(oracle.physical_blocks)
        seqused_match = int(table.seqused_k) == int(oracle.seqused_k)
        write_match = tuple(candidate_slot) == tuple(oracle.next_write_slot)
        current_block_retained = (kv_len - 1) // self.block_size in oracle.visible_blocks

        k_before = k_gpu.clone()
        v_before = v_gpu.clone()
        out, block_table, seqused_k = run_candidate(q_row, k_gpu, v_gpu, [table], self.scale)
        k_after = k_gpu.clone()
        v_after = v_gpu.clone()

        k_unchanged = bool(torch.equal(k_before, k_after))
        v_unchanged = bool(torch.equal(v_before, v_after))

        k_sel, v_sel = gather_positions(k_true, v_true, oracle.positions)
        q_ref = q_row[0].float().cpu()  # 参考在 CPU/FP32 上算，真值也在 CPU
        ref = attention_fp32(q_ref, k_sel, v_sel, self.scale)
        ref_sdpa = attention_sdpa_fp32(q_ref, k_sel, v_sel, self.scale)
        selfcheck = float((ref - ref_sdpa).abs().max().item())

        numeric = error_stats(out[0], ref, self.atol, self.rtol)
        measurement = {
            "label": label,
            "appended": bool(appended),
            "mode": mode,
            "refs": list(refs),
            "kv_len": kv_len,
            "current_position": kv_len - 1,
            "seqused_k": int(table.seqused_k),
            "width": table.width,
            "block_table": block_table.tolist(),
            "backend_seqused_k": [int(x) for x in seqused_k.tolist()],
            "visible_blocks": list(table.visible_blocks),
            "physical_blocks": list(table.physical_blocks),
            "effective_per_block": list(table.effective_per_block),
            "tail_len": table.tail_len,
            "oracle_positions": len(oracle.positions),
            "oracle_seqused_k": int(oracle.seqused_k),
            "oracle_write_slot": list(oracle.next_write_slot),
            "candidate_write_slot": list(candidate_slot),
            "numeric": numeric,
            "oracle_selfcheck": {"max_diff": selfcheck, "limit": 1e-5},
            "read_table": {
                "has_minus_one": any(pid == -1 for row in block_table.tolist() for pid in row),
                "width_ok": table.width == -(-table.seqused_k // self.block_size),
                "physical_nonneg": all(pid >= 0 for pid in table.physical_blocks),
                "width": table.width,
                "seqused_k": int(table.seqused_k),
            },
            "data_plane": {
                "blocks_match": blocks_match,
                "physical_match": physical_match,
                "seqused_match": seqused_match,
                "write_slot_match": write_match,
                "current_block_retained": current_block_retained,
                "details": (
                    f"候选块={list(table.visible_blocks)} oracle块={list(oracle.visible_blocks)}；"
                    f"seqused 候选={table.seqused_k} oracle={oracle.seqused_k}；"
                    f"写入 slot 候选={candidate_slot} oracle={oracle.next_write_slot}"
                ),
            },
            "kv_integrity": {
                "k_unchanged": k_unchanged,
                "v_unchanged": v_unchanged,
                "expected_slots": None,
                "k_changed_slots": None,
                "v_changed_slots": None,
            },
            "residency": {"data_ptr_stable": True},
        }
        if view is not None:
            measurement["read_view"] = {
                "visible_spans": [list(s) for s in view.visible_spans],
                "valid_counts": int(view.valid_counts),
                "max_width": int(view.max_width),
                "tail_block_valid_len": int(view.tail_block_valid_len),
                "next_write_position": int(view.next_write_position),
            }
        if extra_ok is not None:
            measurement.update(extra_ok)
        measurement["_masked_output"] = out[0].float().cpu().tolist()
        return measurement, k_after, v_after, oracle

    def _run_steps(self, record, case, k_true, v_true, k_gpu, v_gpu, q_gpu, kv_plan):
        if case["kind"] == "trajectory":
            steps = case["steps"]
        elif case["kind"] == "arbitrary":
            steps = [{"step": 0, "mode": "local", "refs": [], "kv_len": case["kv_len"], "writes": 0}]
        else:
            steps = [{"step": 0, "mode": case["mode"], "refs": case.get("refs", []),
                      "kv_len": case["kv_len"], "writes": 0}]
        prompt_len = self.layout["prompt_len"]
        for step in steps:
            kv_len = int(step["kv_len"])
            mode, refs = step["mode"], tuple(step.get("refs", []))
            if step.get("writes"):
                # 真实追加：先写入 canonical slot（GPU 上原地写），再读取
                expected_slot = next_write_slot(kv_len - 1, self.block_size, self.l2p)[2]
                k_snap, v_snap = k_gpu.clone(), v_gpu.clone()
                flat_k, flat_v = k_gpu.view(-1, self.kvh, self.d), v_gpu.view(-1, self.kvh, self.d)
                flat_k[expected_slot] = k_true[kv_len - 1].to("cuda")
                flat_v[expected_slot] = v_true[kv_len - 1].to("cuda")
                k_changed = changed_slots(k_snap, k_gpu, self.kvh, self.d)
                v_changed = changed_slots(v_snap, v_gpu, self.kvh, self.d)
                measurement = {
                    "append": {"expected_slot": expected_slot, "k_changed_slots": k_changed, "v_changed_slots": v_changed},
                }
            else:
                measurement = {}
            table = None
            if case["kind"] == "arbitrary" and step.get("writes", 0) == 0:
                table = self._arbitrary_table(case)
            m, _, _, _ = self._measure(
                label=f"step{step['step']}", mode=mode, refs=refs, kv_len=kv_len, prompt_len=prompt_len,
                k_true=k_true, v_true=v_true, k_gpu=k_gpu, v_gpu=v_gpu, q_row=q_gpu, table=table,
                extra_ok=measurement if measurement else None,
                appended=bool(step.get("writes")),
            )
            if measurement:
                m.update(measurement)
                m["kv_integrity"] = {
                    "k_unchanged": m["kv_integrity"]["k_unchanged"],
                    "v_unchanged": m["kv_integrity"]["v_unchanged"],
                    "expected_slots": [measurement["append"]["expected_slot"]],
                    "k_changed_slots": measurement["append"]["k_changed_slots"],
                    "v_changed_slots": measurement["append"]["v_changed_slots"],
                }
            record["measurements"].append(m)
            if case.get("amplify_excluded_blocks") and not step.get("writes"):
                full_table = read_table_from_read_view(
                    self.read_view("global", (), kv_len, prompt_len), self.l2p
                )
                full_out, _, _ = run_candidate(q_gpu, k_gpu, v_gpu, [full_table], self.scale)
                masked_out = torch.tensor(m["_masked_output"], dtype=torch.float32)
                delta = float((full_out[0].float().cpu() - masked_out).abs().max().item())
                record.setdefault("extra_criteria", []).append({
                    "name": "sensitivity_masked_vs_full",
                    "ok": delta > 10 * self.atol,
                    "detail": f"masked 与 full 输出最大差 {delta:.5f}（门限 {10 * self.atol}）",
                })

    def _arbitrary_table(self, case):
        """底层补充用例（非协议来源）：显式逻辑块集合，仍走同一转换与校验路径。"""
        import types

        blocks = tuple(sorted(int(b) for b in case["logical_blocks"]))
        spans = tuple((b * self.block_size, min((b + 1) * self.block_size, case["kv_len"])) for b in blocks)
        view = types.SimpleNamespace(
            kernel_block_size=self.block_size,
            attention_kv_len=int(case["kv_len"]),
            visible_blocks=blocks,
            visible_spans=spans,
            valid_counts=len(blocks),
            physical_block_ids=tuple(self.l2p[b] for b in blocks) + (-1,) * (self.nb - len(blocks)),
            next_write_position=int(case["kv_len"]),
        )
        return read_table_from_read_view(view, self.l2p)

    def _run_batch2(self, record, k_true, v_true, k_gpu, v_gpu, q_gpu, kv_plan):
        case = self.case
        prompt_len = self.layout["prompt_len"]
        rows = case["rows"]
        q2 = q_gpu.repeat(len(rows), 1, 1).contiguous()
        tables, oracles = [], []
        for row in rows:
            view = self.read_view(row["mode"], tuple(row.get("refs", [])), int(row["kv_len"]), prompt_len)
            tables.append(read_table_from_read_view(view, self.l2p))
            oracles.append(self.oracle(row["mode"], tuple(row.get("refs", [])), int(row["kv_len"]), prompt_len))
        width = max(t.width for t in tables)
        block_table = torch.tensor([t.padded_row(width) for t in tables], dtype=torch.int32, device="cuda")
        seqused_k = torch.tensor([t.seqused_k for t in tables], dtype=torch.int32, device="cuda")
        k_before, v_before = k_gpu.clone(), v_gpu.clone()
        out, _, _ = run_candidate(q2, k_gpu, v_gpu, tables, self.scale)
        k_after, v_after = k_gpu.clone(), v_gpu.clone()
        row_isolation = []
        for index, (row, table, oracle) in enumerate(zip(rows, tables, oracles)):
            single, _, _ = run_candidate(q2[index: index + 1], k_gpu, v_gpu, [table], self.scale)
            row_isolation.append(float((single[0].float() - out[index].float()).abs().max().item()))
            k_sel, v_sel = gather_positions(k_true, v_true, oracle.positions)
            q_ref = q2[index].float().cpu()
            ref = attention_fp32(q_ref, k_sel, v_sel, self.scale)
            ref_sdpa = attention_sdpa_fp32(q_ref, k_sel, v_sel, self.scale)
            record["measurements"].append({
                "label": f"row{index}",
                "appended": False,
                "mode": row["mode"],
                "kv_len": int(row["kv_len"]),
                "seqused_k": int(table.seqused_k),
                "visible_blocks": list(table.visible_blocks),
                "physical_blocks": list(table.physical_blocks),
                "oracle_seqused_k": int(oracle.seqused_k),
                "numeric": error_stats(out[index], ref, self.atol, self.rtol),
                "oracle_selfcheck": {"max_diff": float((ref - ref_sdpa).abs().max().item()), "limit": 1e-5},
                "read_table": {
                    "has_minus_one": any(pid == -1 for r in block_table.tolist() for pid in r),
                    "width_ok": table.width == -(-table.seqused_k // self.block_size),
                    "physical_nonneg": all(pid >= 0 for pid in table.physical_blocks),
                    "width": table.width, "seqused_k": int(table.seqused_k),
                },
                "data_plane": {
                    "blocks_match": tuple(table.visible_blocks) == tuple(oracle.visible_blocks),
                    "physical_match": tuple(table.physical_blocks) == tuple(oracle.physical_blocks),
                    "seqused_match": int(table.seqused_k) == int(oracle.seqused_k),
                    "write_slot_match": tuple(next_write_slot(int(row["kv_len"]), self.block_size, self.l2p))
                    == tuple(oracle.next_write_slot),
                    "current_block_retained": (int(row["kv_len"]) - 1) // self.block_size in oracle.visible_blocks,
                    "details": f"候选块={list(table.visible_blocks)} oracle块={list(oracle.visible_blocks)}",
                },
                "kv_integrity": {"k_unchanged": None, "v_unchanged": None, "expected_slots": None},
                "residency": {"data_ptr_stable": True},
                "row_isolation_max_abs": row_isolation[-1],
            })
        record["extra_criteria"] = [{
            "name": "row_isolation",
            "ok": all(value <= self.atol for value in row_isolation),
            "detail": f"逐行单独调用与批量输出的最大偏差={row_isolation}（容差 {self.atol}）",
        }]
        record["kv_integrity"] = {"k_unchanged": bool(torch.equal(k_before, k_after)),
                                  "v_unchanged": bool(torch.equal(v_before, v_after)),
                                  "expected_slots": None}
        for m in record["measurements"]:
            m["kv_integrity"] = record["kv_integrity"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1-gpu/read-view-check-v2.json")
    parser.add_argument("--evidence", default="evidence/p1-gpu-v2")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    cases = cfg["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    seeds = cfg["fixture_truth"]["seeds"]

    print(f"torch {torch.__version__} | entry vllm.vllm_flash_attn.flash_attn_varlen_func fa_version=2")
    print(f"device {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} | "
          f"atol={cfg['tolerance']['atol']} rtol={cfg['tolerance']['rtol']} seeds={seeds}")
    print(f"cases {[c['id'] for c in cases]}\n")

    records = []
    for seed in seeds:
        for case in cases:
            runner = CaseRunner(cfg, case, seed)
            try:
                record = runner.run()
            except Exception as exc:  # 保留反例：不让异常吞掉用例
                record = {"id": case["id"], "seed": seed, "error": f"{type(exc).__name__}: {exc}", "measurements": []}
            records.append(record)
            case_summary = summarize([record])
            print(f"seed{seed} {case['id']:32s} 测量 {len(record['measurements'])} 条 "
                  f"判据 {case_summary.total} -> {'PASS' if case_summary.ok else 'FAIL'}")
            if record.get("error"):
                print(f"    error: {record['error']}")
            for m in record["measurements"]:
                n = m.get("numeric", {})
                print(f"    {m.get('label'):>7s} mode={m.get('mode'):>6s} kv_len={m.get('kv_len')} "
                      f"seqused={m.get('seqused_k')} tail={m.get('tail_len', m.get('append', {}).get('expected_slot'))} "
                      f"max_abs={n.get('max_abs') if n.get('max_abs') is None else round(n['max_abs'], 5)} "
                      f"rms={n.get('rms') if n.get('rms') is None else round(n['rms'], 6)} "
                      f"{'OK' if n.get('within_tolerance') else 'FAIL'}")

    for record in records:
        for m in record["measurements"]:
            m.pop("_masked_output", None)
    final = summarize(records)
    out_dir = ROOT / args.evidence
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2-summary.json").write_text(json.dumps({
        "config": args.config,
        "config_sha256": hashlib.sha256((ROOT / args.config).read_bytes()).hexdigest(),
        "torch": torch.__version__, "vllm_entry": "vllm.vllm_flash_attn.flash_attn_varlen_func",
        "fa_version": 2, "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)), "platform": platform.platform(),
        "seed": cfg["fixture_truth"]["seeds"],
        "criteria_total": final.total, "criteria_failed": len(final.failed),
        "ok": final.ok, "reasons": list(final.reasons),
        "records": records,
    }, indent=2, ensure_ascii=False, default=str))
    per_seed: dict[str, list] = {}
    for record in records:
        per_seed.setdefault(str(record["seed"]), []).append(record)
    for seed, group in per_seed.items():
        (out_dir / f"v2-cases-seed{seed}.json").write_text(json.dumps(group, indent=2, ensure_ascii=False, default=str))

    print(f"\n判据 {final.total} 条，失败 {len(final.failed)} 条 -> {'PASS' if final.ok else 'FAIL'}")
    for reason in final.reasons[:40]:
        print(f"  FAIL: {reason}")
    return 0 if final.ok else 1


if __name__ == "__main__":
    sys.exit(main())
