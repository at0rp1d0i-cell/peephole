"""attnview 声明式读取：vLLM **worker 侧** 薄适配层（阶段 05，版本锁定、可撤销）。

职责**且仅此**：
1. 从运行期 KV 缓存配置读几何（kernel 块大小、组划分、FA 组下标）并拒绝不支持的配置；
2. 把 engine 下推的每步计划落位到**全注意力组**的 metadata 输入（构造**新**张量，不改共享对象、
   不做 host 侧 GPU 读取）；
3. 计量 H2D 拷贝与设备操作次数（供后续成本测量，**不作性能结论**）。

协议/视图算术不在这里：见 `attnview.step_plan`（阶段 03 已验收的 `readview`/`gpukv`）。
engine 侧解析见 `vllm/v1/engine/attnview_engine.py`（跨进程边界：本模块只在 worker 进程使用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from attnview.step_plan import Geometry, StepPlan, check_supported_config

__all__ = [
    "DA_PLAN_FIELD",
    "DaFaOverride",
    "derive_geometry",
    "assert_supported_config",
    "fa_override_for_step",
    "metering_snapshot",
    "reset_metering",
]

#: `SchedulerOutput` 上承载每步计划的可选字段名（唯一入口）。
DA_PLAN_FIELD = "da_step_plans"

_METERING: dict[str, int] = {
    "h2d_calls": 0,
    "h2d_bytes": 0,
    "device_copy_calls": 0,
    "device_index_select_calls": 0,
    "device_fill_calls": 0,
    "host_reads_of_device_tensors": 0,
}


def reset_metering() -> None:
    for key in _METERING:
        _METERING[key] = 0


def metering_snapshot() -> dict[str, int]:
    return dict(_METERING)


# --------------------------------------------------------------------------- #
# 几何与支持范围
# --------------------------------------------------------------------------- #


def _is_mamba_spec(spec: Any) -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec

    return isinstance(spec, MambaSpec)


def derive_geometry(vllm_config: Any, kv_cache_config: Any) -> tuple[Geometry, list[int]]:
    """从**运行期** KV 缓存配置推导几何，返回 `(Geometry, 各组块大小)`。

    块大小一律取自各组 `kv_cache_spec.block_size`；不接受调用者传入的值，也不默认 784。
    """
    groups = list(kv_cache_config.kv_cache_groups)
    if not groups:
        raise RuntimeError("attnview: kv_cache_config 没有任何 KV 组")
    fa_indices = [i for i, g in enumerate(groups) if not _is_mamba_spec(g.kv_cache_spec)]
    if len(fa_indices) != 1:
        raise RuntimeError(
            f"attnview: 期望恰好 1 个全注意力组，实际 {len(fa_indices)} 个"
        )
    fa_index = fa_indices[0]
    group_block_sizes = [int(getattr(g.kv_cache_spec, "block_size", 0)) for g in groups]
    kernel_block_size = group_block_sizes[fa_index]
    if kernel_block_size <= 0:
        raise RuntimeError("attnview: 无法从全注意力组读出 kernel 块大小")
    # `blocks_per_kv_block` 在这里是**保守代理**：仅当各组块大小一致时认为是 1，否则置 0
    # 交给 `Geometry.validate()` 显式拒绝。这不是对 vLLM 内部布局的断言。
    blocks_per_kv_block = 1 if len(set(group_block_sizes)) == 1 else 0
    geometry = Geometry(
        kernel_block_size=kernel_block_size,
        num_kv_groups=len(groups),
        fa_group_index=fa_index,
        blocks_per_kv_block=blocks_per_kv_block,
        max_model_len=int(vllm_config.model_config.max_model_len),
    ).validate()
    return geometry, group_block_sizes


def assert_supported_config(vllm_config: Any, *, _cache: dict = {}) -> None:
    """拒绝本阶段不支持的运行配置（fail-fast，不静默降级）。

    首次调用即校验；后续复用缓存结果（配置在一次运行内不变）。
    """
    if _cache.get("checked"):
        return
    scheduler_config = vllm_config.scheduler_config
    compilation = getattr(vllm_config, "compilation_config", None)
    check_supported_config(
        async_scheduling=bool(getattr(scheduler_config, "async_scheduling", False)),
        max_concurrent_batches=int(vllm_config.max_concurrent_batches),
        cudagraph_mode=getattr(compilation, "cudagraph_mode", None),
        enable_prefix_caching=bool(
            getattr(vllm_config.cache_config, "enable_prefix_caching", False)
        ),
        speculative_config=getattr(vllm_config, "speculative_config", None),
    )
    _cache["checked"] = True


# --------------------------------------------------------------------------- #
# 每步落位
# --------------------------------------------------------------------------- #


@dataclass
class DaFaOverride:
    """本步全注意力组的读取覆写（**新张量**，不改共享对象）。"""

    group_index: int
    block_table: torch.Tensor  # [rows, width] int32，kernel 块号；宽度一次生成内常量
    seq_lens: torch.Tensor  # [rows] int32：各行的**压缩可见长度**（seqused_k）
    max_seq_len: int
    seq_lens_cpu_upper_bound: torch.Tensor  # [rows] int32（CPU），与 seq_lens 一致
    rows_with_override: tuple[int, ...]
    plans: Mapping[int, StepPlan] = field(default_factory=dict)


def _buffers_for(runner: Any, width: int, rows: int, device: torch.device) -> dict[str, torch.Tensor]:
    """请求级持久缓冲：地址与形状在生成期固定（I5；不得在步间重建）。"""
    cached = getattr(runner, "_attnview_buffers", None)
    if cached is not None:
        if int(cached["block_table"].shape[1]) != width or int(cached["block_table"].shape[0]) != rows:
            raise RuntimeError("attnview: 持久缓冲形状在一次运行内发生变化（违反 I5）")
        return cached
    buffers = {
        "block_table": torch.zeros((rows, width), dtype=torch.int32, device=device),
        "seq_lens": torch.zeros((rows,), dtype=torch.int32, device=device),
        "seq_lens_cpu": torch.zeros((rows,), dtype=torch.int32),
    }
    runner._attnview_buffers = buffers
    return buffers


def _geometry_for(runner: Any) -> Geometry:
    cached = getattr(runner, "_attnview_geometry", None)
    if cached is not None:
        return cached
    geometry, group_block_sizes = derive_geometry(runner.vllm_config, runner.kv_cache_config)
    runner._attnview_geometry = geometry
    runner._attnview_group_block_sizes = group_block_sizes
    return geometry


def _plan_from_payload(raw: Mapping[str, Any], geometry: Geometry) -> StepPlan:
    """从下推载荷还原 `StepPlan`，用**运行期几何**补全块大小/表宽后校验。

    载荷本身不含几何（见 `step_plan.FORBIDDEN_PAYLOAD_KEYS`）。
    """
    return StepPlan(
        req_id=str(raw["req_id"]),
        mode=str(raw["mode"]),
        refs=tuple(int(r) for r in raw.get("refs", ())),
        effect_step=int(raw["effect_step"]),
        visible_logical_blocks=tuple(int(b) for b in raw["visible_logical_blocks"]),
        effective_per_block=tuple(int(c) for c in raw["effective_per_block"]),
        seqused_k=int(raw["seqused_k"]),
        block_size=geometry.kernel_block_size,
        width=geometry.max_width,
        tail_len=int(raw["tail_len"]),
        attention_kv_len=int(raw["attention_kv_len"]),
        next_write_position=int(raw["next_write_position"]),
        written_before_step=int(raw["written_before_step"]),
    ).validate()


def fa_override_for_step(
    runner: Any,
    scheduler_output: Any,
    input_batch: Any,
    block_tables: Sequence[torch.Tensor],
) -> DaFaOverride | None:
    """把本步计划落位成全注意力组的读取覆写；无计划时返回 `None`（原版路径）。"""
    plans = getattr(scheduler_output, DA_PLAN_FIELD, None)
    if not plans:
        return None
    # 门禁：在任何覆写缓冲分配/落位**之前**校验运行配置（同步调度/eager/关前缀缓存/关投机）。
    assert_supported_config(runner.vllm_config)
    geometry = _geometry_for(runner)
    fa = geometry.fa_group_index
    group_table = block_tables[fa]
    if int(group_table.shape[0]) != len(input_batch.req_ids):
        raise RuntimeError(
            f"attnview: 全注意力组的块表行数 {int(group_table.shape[0])} 与批内请求数 "
            f"{len(input_batch.req_ids)} 不一致（拒绝按错误行数落位）"
        )
    if int(group_table.shape[1]) < geometry.max_width:
        raise RuntimeError(
            f"attnview: 全注意力组块表宽度 {int(group_table.shape[1])} < 运行期 max_width "
            f"{geometry.max_width}（无法表达完整历史视图，拒绝落位）"
        )
    device = group_table.device
    rows = int(group_table.shape[0])
    buffers = _buffers_for(runner, geometry.max_width, rows, device)

    # 非 DA 行必须保持原语义：先整组镜像 canonical 读表（按 max_width 切片），再覆写 DA 行。
    buffers["block_table"].copy_(group_table[:, : geometry.max_width], non_blocking=True)
    _METERING["device_copy_calls"] += 1
    buffers["seq_lens"].copy_(input_batch.seq_lens[:rows], non_blocking=True)
    if input_batch.seq_lens_cpu_upper_bound is not None:
        buffers["seq_lens_cpu"].copy_(input_batch.seq_lens_cpu_upper_bound[:rows])
    _METERING["device_copy_calls"] += 1

    canonical_row_numbers = runner.block_tables.num_blocks.np[fa]
    row_plans: dict[int, StepPlan] = {}
    overridden: list[int] = []
    for row, req_id in enumerate(input_batch.req_ids):
        raw = plans.get(req_id)
        if raw is None:
            continue
        plan = _plan_from_payload(raw, geometry)
        allocated = int(canonical_row_numbers[int(input_batch.idx_mapping_np[row])])
        last_visible = plan.visible_logical_blocks[-1]
        if last_visible >= allocated:
            raise RuntimeError(
                f"attnview: {req_id} 可见块 {last_visible} 超出该请求已分配块数 {allocated}"
            )
        indices = torch.tensor(
            list(plan.visible_logical_blocks), dtype=torch.long, device=device
        )
        _METERING["h2d_calls"] += 1
        _METERING["h2d_bytes"] += int(indices.numel() * indices.element_size())
        gathered = group_table[row, : geometry.max_width].index_select(0, indices)
        _METERING["device_index_select_calls"] += 1
        n = int(gathered.numel())
        if n < geometry.max_width:
            # 右侧填充复用最后一块（合法块号，绝不放 -1）：后端只索引
            # ceil(seqused_k/block_size) 列，填充列不会被解引用。
            gathered = torch.cat([gathered, gathered[-1:].expand(geometry.max_width - n)], dim=0)
            _METERING["device_fill_calls"] += 1
        buffers["block_table"][row].copy_(gathered, non_blocking=True)
        buffers["seq_lens"][row] = plan.seqused_k
        buffers["seq_lens_cpu"][row] = plan.seqused_k
        _METERING["device_copy_calls"] += 1
        row_plans[row] = plan
        overridden.append(row)

    if not overridden:
        return None
    # FA 组的 max_seq_len 是**整批**标量：必须并入非 DA 行的 canonical 上界。
    canonical_upper = int(
        input_batch.seq_lens_cpu_upper_bound[:rows].max().item()
        if input_batch.seq_lens_cpu_upper_bound is not None
        else max(int(buffers["seq_lens_cpu"][r]) for r in range(rows))
    )
    max_seq_len = max([canonical_upper] + [row_plans[r].seqused_k for r in overridden])
    return DaFaOverride(
        group_index=fa,
        block_table=buffers["block_table"],
        seq_lens=buffers["seq_lens"],
        max_seq_len=max_seq_len,
        seq_lens_cpu_upper_bound=buffers["seq_lens_cpu"],
        rows_with_override=tuple(overridden),
        plans=row_plans,
    )
