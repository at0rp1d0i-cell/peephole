#!/usr/bin/env python3
"""SUP-004 校准步骤②：**不加载 27B** 的小 GPU metadata 探针。

用**真实 CUDA 张量**调用 worker 覆写入口 `attnview_adapter.fa_override_for_step`，核对：

1. canonical/非 FA 输入**未被就地改写**（按 `data_ptr`/`_version` 与值相等双向核对）；
2. 索引（可见逻辑块 → 物理块）、长度（`seqused_k`）、当前尾块与右侧填充正确；
3. `max_seq_len` = 覆写后 CPU 长度向量的最大值；
4. **恢复 global**：无计划的一步返回 `None`，且上一步的覆写缓冲不会泄漏到调用方张量；
5. 真实 stream 与 H2D 行为（记录 stream id/是否图捕获/审计计数，不做性能结论）。

用法（需 GPU）：
    source env.sh && "$ATTNVIEW_PYTHON" tools/p2-calib-metadata-probe.py --out evidence/p3-calib/metadata-probe.json
退出码：0 = 全部检查通过；3 = 存在失败检查（报告仍写出）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

ADAPTER = REPO / "vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py"

# 目标配置的实测值（来自 evidence/p0-model/e4-kernel-block-probe.json）：manager=kernel=784、ratio=1
BLOCK = 784
MAX_MODEL_LEN = 8192
PROMPT_LEN = 6272  # 8 个满块
FA_VISIBLE = [0, 5, 6, 7, 8]  # 计划里可见的逻辑块（含当前尾块）
FA_PHYSICAL = [41, 7, 90, 12, 63, 28, 55, 88, 102]


def load_adapter():
    spec = importlib.util.spec_from_file_location("attnview_adapter_probe", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["attnview_adapter_probe"] = module
    spec.loader.exec_module(module)
    return module


def make_specs():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    mamba = [MambaSpec(block_size=BLOCK, shapes=(), dtypes=()) for _ in range(3)]
    fa = FullAttentionSpec(block_size=BLOCK, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    return [SimpleNamespace(kv_cache_spec=s) for s in (*mamba, fa)]


class _Backend:
    """镜像 pin 的真实契约：`AttentionGroup.backend` 是**类**，`get_name` 是 `@staticmethod`
    （`vllm/v1/attention/backends/flash_attn.py:128-130`）。"""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"


def make_runner(device: torch.device):
    groups = make_specs()
    table = torch.zeros((1, 11), dtype=torch.int32, device=device)
    table[0, : len(FA_PHYSICAL)] = torch.tensor(FA_PHYSICAL, dtype=torch.int32, device=device)
    group_tables = tuple(table.clone() for _ in range(4))
    num_blocks = torch.zeros((4, 1), dtype=torch.int32, device=device)
    num_blocks[3, 0] = len(FA_PHYSICAL)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        attention_config=SimpleNamespace(flash_attn_version=2),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        max_concurrent_batches=1,
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
    )
    runner = SimpleNamespace(
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        kernel_block_sizes=[BLOCK] * 4,
        block_tables=SimpleNamespace(
            block_sizes=[BLOCK] * 4,
            kernel_block_sizes=[BLOCK] * 4,
            blocks_per_kv_block=[1] * 4,
            num_blocks=SimpleNamespace(np=num_blocks.to("cpu").numpy()),
        ),
        attn_groups=[[SimpleNamespace(backend=_Backend, kv_cache_spec=g.kv_cache_spec)] for g in groups],
    )
    return runner, group_tables


def plan_payload():
    """用**真实计划构造器**生成载荷（避免手写字段时间不一致；与适配层测试同一入口）。"""
    from attnview.step_plan import DaRequestConfig, Geometry, build_step_plan

    config = DaRequestConfig.from_extra_args(
        {
            "attnview": {
                "protocol": "v1.0",
                "prompt_len": PROMPT_LEN,
                "segment_spans": [[784, 2352], [2352, 3920], [3920, 5488]],
                "local_window_span": [4000, 6272],
                "sink_span": [0, 16],
            }
        }
    )
    plan = build_step_plan(
        req_id="calib-r1",
        mode="local",
        refs=(),
        config=config,
        geometry=Geometry(
            kernel_block_size=BLOCK,
            num_kv_groups=4,
            fa_group_index=3,
            blocks_per_kv_block=1,
            max_model_len=MAX_MODEL_LEN,
        ),
        canonical_blocks=FA_PHYSICAL,
        attention_kv_len=PROMPT_LEN + 1,  # post：含本步写入的当前 token
        effect_step=1,
    )
    return plan.as_payload()


def fingerprint(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "data_ptr": int(t.data_ptr()),
        "version": int(getattr(t, "_version", -1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mod = load_adapter()
    mod.reset_metering()
    if not torch.cuda.is_available():
        print("需要 GPU：本探针必须用真实 CUDA 张量运行", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    runner, group_tables = make_runner(device)
    before = {f"group{i}_table": fingerprint(t) for i, t in enumerate(group_tables)}
    geometry, manager_sizes = mod.derive_geometry(runner)
    check("几何取自 runner 实际对象", geometry.kernel_block_size == BLOCK and manager_sizes == [BLOCK] * 4,
          {"kernel_block_size": geometry.kernel_block_size, "blocks_per_kv_block": geometry.blocks_per_kv_block})

    # --- 覆写步 ---
    stripped = SimpleNamespace(seq_lens=torch.tensor([PROMPT_LEN + 1], dtype=torch.int32, device=device))
    batch = SimpleNamespace(
        req_ids=["calib-r1"],
        idx_mapping_np=torch.zeros(1, dtype=torch.int64).numpy(),
        seq_lens=stripped.seq_lens,
        seq_lens_cpu_upper_bound=torch.tensor([PROMPT_LEN + 1], dtype=torch.int32),
        is_prefilling_np=torch.zeros(1, dtype=torch.bool).numpy(),
    )
    sched = SimpleNamespace(da_step_plans={"calib-r1": plan_payload()}, num_scheduled_tokens={"calib-r1": 1})
    stream_before = mod._stream_fingerprint()
    override = mod.fa_override_for_step(runner, sched, batch, group_tables)
    check("返回覆写对象", override is not None, None)
    plan = plan_payload()
    visible = [int(b) for b in plan["visible_logical_blocks"]]
    row = override.block_table[0, : len(visible)].to("cpu").tolist()
    expected = [FA_PHYSICAL[b] for b in visible]
    check("索引：可见逻辑块→物理块", row == expected, {"got": row, "expected": expected})
    check("当前尾块确实进入可见集合（含最后一块）", visible[-1] == len(FA_PHYSICAL) - 1,
          {"visible": visible})
    tail = override.block_table[0, len(visible):].to("cpu").tolist()
    check("当前尾块之后的填充复用尾块（绝不放 -1）", set(tail) == {expected[-1]} and -1 not in tail,
          {"fill": sorted(set(tail))})
    seqused_k = int(plan["seqused_k"])
    check("长度：seqused_k 与计划一致", int(override.seq_lens[0]) == seqused_k,
          {"got": int(override.seq_lens[0]), "plan": seqused_k})
    check("max_seq_len=覆写后 CPU 向量最大值（单请求 ⇒ 等于 seqused_k）",
          int(override.max_seq_len) == seqused_k, int(override.max_seq_len))
    check("覆写缓冲覆盖 FA 行", tuple(override.rows_with_override) == (0,), list(override.rows_with_override))

    # --- 输入未被就地改写（FA 与非 FA 组都要核对）---
    after = {f"group{i}_table": fingerprint(t) for i, t in enumerate(group_tables)}
    unchanged = {k: (before[k] == after[k]) for k in before}
    check("canonical/非 FA 输入未被就地改写（data_ptr/_version 不变）", all(unchanged.values()), unchanged)
    check("调用方 seq_lens 未被就地改写", bool(torch.equal(stripped.seq_lens, torch.tensor([PROMPT_LEN + 1], dtype=torch.int32, device=device))),
          fingerprint(stripped.seq_lens))
    check("覆写缓冲与输入张量不是同一对象",
          int(override.block_table.data_ptr()) != int(group_tables[3].data_ptr()),
          {"override_ptr": int(override.block_table.data_ptr()), "input_ptr": int(group_tables[3].data_ptr())})

    # --- 持久缓冲（I5）---
    ptr_before = int(override.block_table.data_ptr())
    override2 = mod.fa_override_for_step(runner, sched, batch, group_tables)
    check("二次调用沿用同一缓冲（地址/形状稳定）", int(override2.block_table.data_ptr()) == ptr_before,
          {"ptr": ptr_before})

    # --- 恢复 global ---
    sched_global = SimpleNamespace(da_step_plans=None, num_scheduled_tokens={"calib-r1": 1})
    none_override = mod.fa_override_for_step(runner, sched_global, batch, group_tables)
    check("无计划的一步返回 None（调用方走 canonical/global）", none_override is None, None)
    check("global 步后输入张量仍未变",
          all(before[k] == fingerprint(t) for k, t in zip(before, group_tables)), None)

    # --- 运行期 H2D/同步实测（calibration #2 要求；插桩计数不能替代本项）-------------
    import warnings

    from torch.profiler import ProfilerActivity, profile

    trace_path = args.out.with_suffix(".chrome.json")
    schema_path = args.out.with_suffix(".events.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")  # 任何隐式同步都会以告警形式出现
            try:
                mod.fa_override_for_step(runner, sched, batch, group_tables)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    prof.export_chrome_trace(str(trace_path))
    # 事件 schema 落盘（本轮 bytes/stream 解析仍为空，先固化真实字段名，下一步按它取值）
    schema_path.write_text(json.dumps(
        [
            {
                "name": str(getattr(event, "name", ""))[:60],
                "attrs": sorted(a for a in dir(event) if not a.startswith("_"))[:24],
                "args": sorted((getattr(event, "args", None) or {}).keys()),
                "device_type": str(getattr(event, "device_type", "")),
            }
            for event in list(prof.events())[:8]
        ],
        ensure_ascii=False,
        indent=2,
    ))
    memcpy = {"HtoD": {"count": 0, "bytes": 0}, "DtoH": {"count": 0, "bytes": 0}}
    streams: set[int] = set()
    sync_events: list[dict] = []
    for event in prof.events():
        name = str(getattr(event, "name", ""))
        ev_args = getattr(event, "args", None) or {}
        stream = ev_args.get("stream", getattr(event, "stream", None))
        if "Memcpy HtoD" in name or "Memcpy DtoH" in name:
            kind = "HtoD" if "HtoD" in name else "DtoH"
            memcpy[kind]["count"] += 1
            memcpy[kind]["bytes"] += int(ev_args.get("bytes", 0) or 0)
            if stream is not None:
                streams.add(int(stream))
        if "Synchronize" in name or "synchronize" in name:
            sync_events.append({"name": name, "stream": stream,
                                "correlation": int(ev_args.get("correlation", -1) or -1)})
    for event in prof.events():
        ev_args2 = getattr(event, "args", None) or {}
        external_id = ev_args2.get("External id")
        if external_id is None:
            continue
        for sync_event in sync_events:
            if sync_event["correlation"] == int(external_id):
                sync_event["after"] = str(getattr(event, "name", ""))
    sync_warnings = [str(w.message) for w in caught if "synchron" in str(w.message).lower()]
    real_syncs = [w for w in sync_warnings if "synchronizing CUDA operation" in w]
    check("运行期未触发隐式同步告警（torch sync_debug warn；候选段内）", not real_syncs, real_syncs[:3])
    check("运行期确有 H2D 但无 DtoH（候选只把索引送上去）",
          memcpy["HtoD"]["count"] > 0 and memcpy["DtoH"]["count"] == 0, memcpy)

    metering = mod.metering_snapshot()
    report = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "stream_before": stream_before,
        "stream_after": mod._stream_fingerprint(),
        "metering": metering,
        "runtime": {
            "memcpy": memcpy,
            "profiler_stream_ids": sorted(streams),
            "stream_synchronize_events": sync_events,
            "sync_warnings": sync_warnings,
            "chrome_trace": str(trace_path),
            "event_schema": str(schema_path),
            "caveat": "sync 判定只看候选段内的隐式同步告警；profiler 自身收尾的 "
                      "cudaDeviceSynchronize（无 stream、correlation=-1）不计入。"
                      "bytes/stream 解析仍待按 events schema 修正（当前为 0/[]）。",
            "note": "运行期实测：profiler CUDA activity（按 args.bytes/args.stream 解析）+ "
                    "torch sync-debug；stream_synchronize_events 用 External id ↔ correlation 对齐到"
                    "触发它的 H2D，用于定位同步来源；探针自身的 .to('cpu')/item() 属诊断读取，与候选调用分开。",
        },
        "checks": checks,
        "failed": [c["name"] for c in checks if not c["ok"]],
        "notes": [
            "本探针只调用 metadata 覆写入口，不加载模型、不做性能结论。",
            "metering 是**代码插桩计数**；运行期同步/H2D 证据在 `runtime` 段"
            "（profiler CUDA activity + sync-debug 告警 + 原始 chrome trace），两者口径不同、不可互相替代。",
            "探针内出现的 .to('cpu')/item() 只用于把结果写进报告，属诊断读取；"
            "chrome trace 里它们与候选调用可分开辨认。",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"failed": report["failed"], "checks": len(checks),
                      "metering": metering}, ensure_ascii=False))
    return 0 if not report["failed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
