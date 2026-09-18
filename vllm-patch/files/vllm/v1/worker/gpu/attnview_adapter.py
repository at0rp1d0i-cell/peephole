"""attnview 声明式读取：vLLM **worker 侧** 薄适配层（阶段 05，版本锁定、可撤销）。

职责**且仅此**：
1. 从**运行期 runner 的实际对象**读 KV 几何（manager/kernel 块大小、每组比值、FA 组下标）并**互相核对**；
   只接受单卡 TP1、真实 `FullAttentionSpec` 目标组与已支持的 FA2 后端；不一致一律显式拒绝，不做推断兜底；
2. 把 engine 下推的每步计划落位到**全注意力组**的 metadata 输入（构造**新**张量，不改共享对象、
   不做 host 侧设备读取）；只支持"批内单活跃请求 + 单 query(decode)"；
3. 计量本模块**自身**的代码插桩计数（供后续 GPU 侧成本观测做对照，**不作性能结论**）。

协议/视图算术不在这里：见 `attnview.step_plan`（阶段 03 已验收的 `readview`/`gpukv`）。
engine 侧解析见 `vllm/v1/engine/attnview_engine.py`（跨进程边界：本模块只在 worker 进程使用）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from attnview.step_plan import Geometry, StepPlan, UnsupportedConfig, check_supported_config

__all__ = [
    "DA_PLAN_FIELD",
    "DaFaOverride",
    "calibration_force_tokens",
    "calibration_capture_logits",
    "calibration_note_override",
    "derive_geometry",
    "assert_supported_config",
    "fa_override_for_step",
    "metering_snapshot",
    "reset_metering",
]

# --------------------------------------------------------------------------- #
# 校准专用钩子（仅当环境变量设置时生效；普通运行**零影响**，不引入分支成本以外的行为）
#
# - `ATTNVIEW_CALIB_FORCE`：JSON 文件 `{"tokens": [[t], ...]}`，按步给强制轨迹；
# - `ATTNVIEW_CALIB_FORCE_LOG`：把每步的**原始采样**与**强制值**逐行写成 JSONL；
# - `ATTNVIEW_CALIB_TRACE`：把每步覆写后的 FA metadata 与 canonical/非 FA 输入不变证据写成 JSONL。
# --------------------------------------------------------------------------- #

_CALIB_LOGITS = "ATTNVIEW_CALIB_LOGITS"
_CALIB_FORCE = "ATTNVIEW_CALIB_FORCE"
_CALIB_FORCE_LOG = "ATTNVIEW_CALIB_FORCE_LOG"
_CALIB_TRACE = "ATTNVIEW_CALIB_TRACE"

_CALIB_STATE: dict[str, Any] = {"tokens": None, "step": 0, "bound": None}


def _append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def calibration_capture_logits(logits: torch.Tensor, input_batch: Any) -> bool:
    """测试专用：保存**完整** logits（不是 top-k 归一化后的近似）。

    调用点必须在 worker `GPUModelRunner.sample` 里 `self.model.compute_logits(...)` **之后**、
    grammar/sampler **就地改写之前**（pin `model_runner.py:1421-1432`）—— 归一化或截断后的 top-k
    无法用于全词表最大绝对误差/RMS，也会漏掉尾部的非有限值。

    落盘 `ATTNVIEW_CALIB_LOGITS` 指向的 `.pt`（`torch.save`，追加成 dict 列表）：
    `{"step": i, "req_ids": [...], "logits": fp32 CPU 张量[num_rows, vocab]}`。
    未设置该环境变量时完全不介入（普通请求零影响）。本钩子含显式 D2H 同步，属校准专用。
    """
    path = os.environ.get(_CALIB_LOGITS)
    if not path:
        return False
    _CALIB_STATE["step"] = int(_CALIB_STATE.get("step", 0))
    record = {
        "step": int(_CALIB_STATE["step"]) + 1,
        "req_ids": list(getattr(input_batch, "req_ids", ()) or ()),
        "logits": logits.detach().to("cpu", dtype=torch.float32),
        "shape": list(logits.shape),
        "dtype_on_device": str(logits.dtype),
        "sync_note": "本钩子含显式 D2H 同步（保存完整 logits），属校准专用，非稳态行为",
    }
    target = Path(path)
    existing = torch.load(target, weights_only=False) if target.exists() else []
    existing.append(record)
    torch.save(existing, target)
    return True


def calibration_force_tokens(
    sampler_output: Any, req_ids: Sequence[str], num_sampled: Any
) -> bool:
    """测试专用 token 强制：**原地**替换 `sampler_output.sampled_token_ids` 中被消费的行。

    调用点必须紧跟 worker 的 `self.sample(...)` 之后、PP broadcast / `AsyncOutput` /
    `postprocess_sampled` **之前**（pin `model_runner.py:1861-1920`）—— 这样 worker 历史与经
    `AsyncOutput` 送往宿主的 token 是**同一个值**，不会分叉。

    边界（本地复核指出后收紧）：

    - **只在真正消费采样时推进轨迹**：未完成 prefill 的步 `num_sampled == 0`（`input_batch.py:506-512`），
      不是生成一步 ⇒ 不消耗轨迹、不改写、不记日志；只强制 `num_sampled > 0` 的**行**。
    - **按 `req_id` 绑定**：首个在消费步出现的请求被绑定为被校准请求；其它 `req_id`（含其后的
      普通请求与清理请求）**一律不命中**，也不会消耗轨迹。
    - 轨迹用尽只对**被绑定的那个请求**报错；形状不符即拒绝且不写一半。

    返回是否发生了替换；未设置 `ATTNVIEW_CALIB_FORCE` 时完全不介入（普通请求无此钩子）。
    """
    force_path = os.environ.get(_CALIB_FORCE)
    if not force_path:
        return False
    counts = num_sampled.detach().to("cpu").tolist() if hasattr(num_sampled, "detach") else list(num_sampled)
    if isinstance(counts, int):
        counts = [counts]
    consuming = [i for i, n in enumerate(counts) if int(n) > 0]
    if not consuming:
        return False  # 未完成 prefill 等丢弃采样的情况：不是生成一步

    if _CALIB_STATE["tokens"] is None:
        payload = json.loads(Path(force_path).read_text())
        raw_tokens = payload["tokens"]
        parsed: list[list[list[int]]] = []
        for step_index, rows in enumerate(raw_tokens):
            if not isinstance(rows, (list, tuple)) or any(
                isinstance(row, int) or not isinstance(row, (list, tuple)) for row in rows
            ):
                raise RuntimeError(
                    f"attnview 校准: 强制轨迹第 {step_index + 1} 步格式不对 —— 需要"
                    " tokens[step][row] = [token_ids...]（每步 × 每个消费请求行的两级列表）"
                )
            parsed.append([[int(t) for t in row] for row in rows])
        _CALIB_STATE["tokens"] = parsed
        _CALIB_STATE["bound"] = None
        _CALIB_STATE["step"] = 0

    bound = _CALIB_STATE.get("bound")
    reqs = list(req_ids)
    if bound is None:
        if len(consuming) != 1:
            raise RuntimeError(
                f"attnview 校准: 首次消费步有 {len(consuming)} 个请求被采样，"
                "校准只支持批内单活跃请求"
            )
        bound = reqs[consuming[0]]
        _CALIB_STATE["bound"] = bound
    if bound not in reqs:
        return False  # 被绑定请求已结束（或本步不含它）：后续普通/清理请求不受影响
    row = reqs.index(bound)
    if row not in consuming:
        return False
    tokens = _CALIB_STATE["tokens"]
    step = int(_CALIB_STATE["step"])
    if step >= len(tokens):
        raise RuntimeError(
            f"attnview 校准: 被绑定请求 {bound} 需要第 {step + 1} 步，但强制轨迹只有 {len(tokens)} 步"
        )
    forced_rows = tokens[step]
    if len(forced_rows) != 1:
        raise RuntimeError(
            f"attnview 校准: 第 {step + 1} 步轨迹有 {len(forced_rows)} 行，"
            "而本步只有 1 个被绑定的消费请求"
        )
    forced = forced_rows[0]
    sampled = sampler_output.sampled_token_ids
    raw = [[int(v) for v in r] for r in sampled.detach().to("cpu").tolist()]
    if len(forced) != len(raw[row]):
        raise RuntimeError(
            f"attnview 校准: 第 {step + 1} 步强制 token 数 {len(forced)} 与采样数 {len(raw[row])} 不一致"
        )
    replacement = torch.tensor([forced], dtype=sampled.dtype, device=sampled.device)
    sampled[row : row + 1].copy_(replacement)  # 原地写回同一块内存
    _CALIB_STATE["step"] = step + 1
    log_path = os.environ.get(_CALIB_FORCE_LOG)
    if log_path:
        _append_jsonl(
            log_path,
            {
                "kind": "force_tokens",
                "step": step + 1,
                "req_id": bound,
                "req_ids": reqs,
                "consuming_rows": consuming,
                "raw_sampled": raw,
                "forced": forced,
                "sync_note": "本钩子含 D2H(读原始采样)+H2D(写强制 token)，属校准专用同步，非稳态行为",
            },
        )
    return True


def calibration_note_override(
    override: DaFaOverride | None,
    *,
    req_ids: Sequence[str],
    scheduler_output: Any,
    inputs: Mapping[str, Any],
) -> None:
    """把本步 FA 覆写与其"未改写入参"证据写成 JSONL（仅在 `ATTNVIEW_CALIB_TRACE` 设置时）。

    `inputs` 传本步的 canonical 块表/长度张量等；这里记录 `data_ptr()` 与 `_version`
    （都是 **CPU 侧**元数据，不读设备内存、不触发同步），用于事后核对"canonical/非 FA 输入未被就地改写"。
    """
    trace_path = os.environ.get(_CALIB_TRACE)
    if not trace_path:
        return
    record: dict[str, Any] = {
        "kind": "override_step",
        "step": int(_CALIB_STATE["step"]),
        "req_ids": list(req_ids),
        "num_scheduled_tokens": dict(getattr(scheduler_output, "num_scheduled_tokens", None) or {}),
        "stream": _stream_fingerprint(),
        "inputs": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "data_ptr": int(t.data_ptr()),
                "version": int(getattr(t, "_version", -1)),
            }
            for name, t in inputs.items()
            if hasattr(t, "data_ptr")
        },
    }
    if override is None:
        record["override"] = None
    else:
        record["override"] = {
            "group_index": int(override.group_index),
            "rows_with_override": list(override.rows_with_override),
            "max_seq_len": int(override.max_seq_len),
            "seqused_k": [int(v) for v in override.seq_lens.detach().to("cpu").tolist()],
            "visible_logical_blocks": {
                str(row): list(plan.visible_logical_blocks) for row, plan in override.plans.items()
            },
            "block_table": {
                "shape": list(override.block_table.shape),
                "dtype": str(override.block_table.dtype),
                "data_ptr": int(override.block_table.data_ptr()),
            },
            "metering": metering_snapshot(),
        }
    _append_jsonl(trace_path, record)


def _stream_fingerprint() -> dict[str, Any]:
    """记录当前 CUDA stream 与图捕获状态（只读元数据；不读设备内容、不触发同步）。"""
    if not torch.cuda.is_available():
        return {"cuda": False}
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device)
    return {
        "cuda": True,
        "device": int(device),
        "stream_id": int(getattr(stream, "cuda_stream", 0)),
        "capturing": bool(torch.cuda.is_current_stream_capturing()),
    }

#: `SchedulerOutput` 上承载每步计划的可选字段名（唯一入口）。
DA_PLAN_FIELD = "da_step_plans"

#: 本阶段支持的注意力后端名（FA2 路径；见 pin `v1/attention/backends/flash_attn.py:129`）。
SUPPORTED_ATTN_BACKENDS = frozenset({"FLASH_ATTN"})

#: 计量口径（**代码插桩范围**，不是运行时成本证据）：
#: 只统计**本模块内显式写出**的拷贝/索引/填充/标量赋值/CPU 暂存读取次数，
#: 用来说明"我们主动做了多少次操作"。设备侧真实同步、耗时必须由 GPU 观测（检查点 2/3）；
#: 任何计数为 0 都**不得**被解释为"零成本"，本模块也不读设备张量（无 `.item()`/`.cpu()` on device）。
_METERING: dict[str, int] = {
    "h2d_calls": 0,
    "h2d_bytes": 0,
    "device_copy_calls": 0,
    "device_index_select_calls": 0,
    "device_fill_calls": 0,
    "scalar_assignments": 0,
    "host_reads_of_cpu_staging": 0,
}


def reset_metering() -> None:
    for key in _METERING:
        _METERING[key] = 0


def metering_snapshot() -> dict[str, int]:
    return dict(_METERING)


# --------------------------------------------------------------------------- #
# 几何与支持范围
# --------------------------------------------------------------------------- #


def _fa_spec_types() -> tuple[type, type]:
    """返回 `(FullAttentionSpec, 已知非全注意力 spec 的类型元组)`。"""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    return FullAttentionSpec, MambaSpec


def _int_list(value: Any, name: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError(f"attnview: {name} 不是序列：{type(value).__name__}")
    return [int(v) for v in value]


def _backend_name(group: Any) -> str:
    backend = getattr(group, "backend", None)
    getter = getattr(backend, "get_name", None)
    name = getter() if callable(getter) else getattr(backend, "__name__", "")
    return str(name or "")


def derive_geometry(runner: Any) -> tuple[Geometry, list[int]]:
    """从**运行期 runner 的实际对象**读几何，返回 `(Geometry, 各组 manager 块大小)`。

    来源与互核（任一处不一致即拒绝，不用"相等即推断"的代理值）：

    - manager 块大小：`runner.block_tables.block_sizes` ↔ 各组 `kv_cache_spec.block_size`；
    - kernel 块大小：`runner.kernel_block_sizes`（`init_attn_backend` 落定）↔
      `runner.block_tables.kernel_block_sizes`；
    - 每组比值：`runner.block_tables.blocks_per_kv_block` ↔ `block_sizes // kernel_block_sizes`；
    - 目标组：`runner.attn_groups` 里**恰好一个** `FullAttentionSpec` 组，且其后端在本阶段支持集内；
    - 并行度：单卡 TP1（`tensor_parallel_size == 1` 且 `world_size == 1`）。

    例（必须拒绝，不得报告 784/1）：各 manager 块大小 784、kernel 块大小 16、比值 49 ——
    此时 `Geometry.validate()` 会以 `UnsupportedConfig` 拒绝（逻辑块↔kernel 列换算会改变
    `seqused_k` 的计量单位）。
    """
    vllm_config = getattr(runner, "vllm_config", None)
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    tables = getattr(runner, "block_tables", None)
    if vllm_config is None or kv_cache_config is None or tables is None:
        raise RuntimeError(
            "attnview: runner 缺少 vllm_config/kv_cache_config/block_tables，无法读实际块几何"
        )

    groups = list(kv_cache_config.kv_cache_groups)
    if not groups:
        raise RuntimeError("attnview: kv_cache_config 没有任何 KV 组")

    manager_block_sizes = _int_list(getattr(tables, "block_sizes", ()), "block_tables.block_sizes")
    table_kernel_sizes = _int_list(
        getattr(tables, "kernel_block_sizes", ()), "block_tables.kernel_block_sizes"
    )
    ratios = _int_list(
        getattr(tables, "blocks_per_kv_block", ()), "block_tables.blocks_per_kv_block"
    )
    runner_kernel_sizes = _int_list(
        getattr(runner, "kernel_block_sizes", ()), "runner.kernel_block_sizes"
    )
    n = len(groups)
    if not (len(manager_block_sizes) == len(table_kernel_sizes) == len(ratios) == len(runner_kernel_sizes) == n):
        raise RuntimeError(
            "attnview: KV 组数与块尺寸向量长度不一致："
            f"groups={n} manager={manager_block_sizes} table_kernel={table_kernel_sizes} "
            f"ratios={ratios} runner_kernel={runner_kernel_sizes}"
        )

    spec_types = _fa_spec_types()
    fa_indices: list[int] = []
    for i, group in enumerate(groups):
        spec = group.kv_cache_spec
        spec_block = int(getattr(spec, "block_size", -1))
        if spec_block != manager_block_sizes[i]:
            raise RuntimeError(
                f"attnview: 第 {i} 组 manager 块大小不一致：kv_cache_spec={spec_block} "
                f"block_tables={manager_block_sizes[i]}"
            )
        if table_kernel_sizes[i] != runner_kernel_sizes[i]:
            raise RuntimeError(
                f"attnview: 第 {i} 组 kernel 块大小不一致：block_tables={table_kernel_sizes[i]} "
                f"runner={runner_kernel_sizes[i]}"
            )
        if table_kernel_sizes[i] <= 0 or manager_block_sizes[i] <= 0:
            raise RuntimeError(f"attnview: 第 {i} 组块大小非正，无法推导几何")
        if manager_block_sizes[i] % table_kernel_sizes[i] != 0:
            raise RuntimeError(
                f"attnview: 第 {i} 组 manager/kernel 块大小不可整除："
                f"{manager_block_sizes[i]} / {table_kernel_sizes[i]}"
            )
        expected_ratio = manager_block_sizes[i] // table_kernel_sizes[i]
        if ratios[i] != expected_ratio:
            raise RuntimeError(
                f"attnview: 第 {i} 组 blocks_per_kv_block 不一致：{ratios[i]} != "
                f"{manager_block_sizes[i]}//{table_kernel_sizes[i]}"
            )
        if isinstance(spec, spec_types[0]):
            fa_indices.append(i)
        elif not isinstance(spec, spec_types[1]):
            raise UnsupportedConfig(
                f"attnview: 第 {i} 组的 KV spec 类型 {type(spec).__name__} 不在支持范围"
                "（只支持 FullAttentionSpec 目标组）"
            )
    if len(fa_indices) != 1:
        raise UnsupportedConfig(
            f"attnview: 期望恰好 1 个 FullAttentionSpec 全注意力组，实际 {len(fa_indices)} 个"
        )
    fa_index = fa_indices[0]

    # 目标组的后端名必须从 **runner.attn_groups** 取：那是初始化时按 KV 组切好的
    # `AttentionGroup`（带 `.backend`/`.kv_cache_spec`），而 `kv_cache_config.kv_cache_groups`
    # 是 `KVCacheGroupSpec`（只有 `layer_names`/`kv_cache_spec`，**没有** backend 字段）。
    attn_groups = getattr(runner, "attn_groups", None)
    if not isinstance(attn_groups, (list, tuple)) or len(attn_groups) != n:
        raise RuntimeError(
            "attnview: runner.attn_groups 层数与 KV 组数不一致："
            f"{len(attn_groups) if attn_groups is not None else None} != {n}"
        )
    for i, layer_groups in enumerate(attn_groups):
        if not layer_groups:
            raise RuntimeError(f"attnview: 第 {i} 个 KV 组在 runner.attn_groups 里为空")
        spec_block = int(getattr(groups[i].kv_cache_spec, "block_size", -1))
        for group in layer_groups:
            backend_name = _backend_name(group)
            if not backend_name:
                raise RuntimeError(
                    f"attnview: 第 {i} 个 KV 组里的 AttentionGroup 没有可读后端名（拒绝按空名放行）"
                )
            group_spec = getattr(group, "kv_cache_spec", None)
            if group_spec is None or int(getattr(group_spec, "block_size", -1)) != spec_block:
                raise RuntimeError(
                    f"attnview: 第 {i} 个 KV 组的两层 spec 不一致"
                    f"（AttentionGroup={type(group_spec).__name__} block="
                    f"{int(getattr(group_spec, 'block_size', -1))} vs 组 spec block={spec_block}）"
                )
    fa_backends = sorted({_backend_name(g) for g in attn_groups[fa_index]})
    unsupported_backends = [b for b in fa_backends if b not in SUPPORTED_ATTN_BACKENDS]
    if unsupported_backends:
        raise UnsupportedConfig(
            f"attnview: 全注意力组后端 {unsupported_backends} 不在本阶段支持集 "
            f"{sorted(SUPPORTED_ATTN_BACKENDS)}（只支持 FA2 路径）"
        )
    # 后端名 `FLASH_ATTN` **不等于** FA2：pin 里同一实现按 `impl.vllm_flash_attn_version` 选择 FA2/3/4
    # （`flash_attn.py:879-899`）。本 pin 的选择规则是：`major == 9` 且支持则 FA3、`major == 10` 且支持则 FA4、
    # 其余回退 **FA2**（`fa_utils.py:96-117`），且 FA4 仅在 9.x/10.x/11.x 可用（`flash_attn_interface.py:72-84`）——
    # 本机 SM120 的默认本来就是 FA2。仍然要求**显式**声明 FA2：为了复跑时不受平台/配置漂移影响、并可审计
    # （`None` 表示交给平台默认 ⇒ 拒绝，而不是因为本机默认是 FA4）。
    attn_config = getattr(vllm_config, "attention_config", None)
    fa_version = getattr(attn_config, "flash_attn_version", None)
    if "FLASH_ATTN" in fa_backends and int(fa_version or 0) != 2:
        raise UnsupportedConfig(
            f"attnview: 全注意力组后端为 FLASH_ATTN，但 flash_attn_version={fa_version!r}"
            "（None = 交给平台默认决定）：本阶段只支持 FA2，"
            "请显式设置 --attention-config.flash_attn_version=2（固定版本以便复跑与审计）"
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
    world_size = int(getattr(parallel_config, "world_size", 1))
    if tp_size != 1 or world_size != 1:
        raise UnsupportedConfig(
            f"attnview: 只支持单卡 TP1，实际 tensor_parallel_size={tp_size} world_size={world_size}"
        )

    geometry = Geometry(
        kernel_block_size=table_kernel_sizes[fa_index],
        num_kv_groups=n,
        fa_group_index=fa_index,
        blocks_per_kv_block=ratios[fa_index],
        max_model_len=int(vllm_config.model_config.max_model_len),
    ).validate()
    return geometry, manager_block_sizes


def assert_supported_config(runner: Any) -> None:
    """拒绝本阶段不支持的运行配置（fail-fast，不静默降级）。

    结果标记**记在该 runner 实例上**（不是模块级共享缓存）：同一进程里的第二个 runner
    必须各自重新校验，避免"第一个 runner 校验通过后，后续 runner 一律免检"。
    """
    if getattr(runner, "_attnview_config_checked", False):
        return
    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError("attnview: runner 上没有 vllm_config，无法校验运行配置")
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
    runner._attnview_config_checked = True


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


def _staging(runner: Any, *, length: int, dtype: torch.dtype) -> torch.Tensor:
    """请求级**pinned** CPU 暂存（异步 H2D 的前提）：小 H2D 若不是从 pinned 源发出，
    会在 stream 上触发同步（实测：40B/4B 的 H2D 后紧跟 cudaStreamSynchronize）。"""
    key = f"_attnview_staging_{length}_{str(dtype)}"
    cached = getattr(runner, key, None)
    if cached is not None:
        return cached
    tensor = torch.empty(length, dtype=dtype, pin_memory=torch.cuda.is_available())
    setattr(runner, key, tensor)
    return tensor


def _geometry_for(runner: Any) -> Geometry:
    cached = getattr(runner, "_attnview_geometry", None)
    if cached is not None:
        return cached
    geometry, manager_block_sizes = derive_geometry(runner)
    runner._attnview_geometry = geometry
    runner._attnview_group_block_sizes = manager_block_sizes
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
    """把本步计划落位成全注意力组的读取覆写；无计划时返回 `None`（原版路径）。

    消费边界（显式核对，不假定上层正确）：

    - 批内**恰好一个**请求（本阶段单活跃请求）；
    - 该请求本步**恰好 1 个 query token**（decode 步）——prefill 步不出计划，出现即拒绝；
    - `seq_lens_cpu_upper_bound` 必须存在（FA2 路径具备）；缺失时拒绝，绝不回读设备张量。
    """
    plans = getattr(scheduler_output, DA_PLAN_FIELD, None)
    if not plans:
        return None
    # 门禁：在任何覆写缓冲分配/落位**之前**校验运行配置（同步调度/eager/关前缀缓存/关投机）。
    assert_supported_config(runner)
    geometry = _geometry_for(runner)
    fa = geometry.fa_group_index
    group_table = block_tables[fa]

    req_ids = list(getattr(input_batch, "req_ids", ()) or ())
    if len(req_ids) != 1:
        raise UnsupportedConfig(
            f"attnview: 只支持批内单活跃请求，实际批内 {len(req_ids)} 个请求"
        )
    da_req_ids = [r for r in req_ids if r in plans]
    if len(da_req_ids) != 1:
        raise UnsupportedConfig(
            f"attnview: 批内携带读取计划的请求数 {len(da_req_ids)} != 1（拒绝按错误行数落位）"
        )
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    scheduled = int(num_scheduled.get(da_req_ids[0], 0))
    if scheduled != 1:
        raise UnsupportedConfig(
            f"attnview: 只支持单 query(decode) 步，本步 {da_req_ids[0]} 调度 {scheduled} 个 token"
            "（prefill 步不出计划；出现即为协议/时序冲突）"
        )
    if int(group_table.shape[0]) != len(req_ids):
        raise RuntimeError(
            f"attnview: 全注意力组的块表行数 {int(group_table.shape[0])} 与批内请求数 "
            f"{len(req_ids)} 不一致（拒绝按错误行数落位）"
        )
    if int(group_table.shape[1]) < geometry.max_width:
        raise RuntimeError(
            f"attnview: 全注意力组块表宽度 {int(group_table.shape[1])} < 运行期 max_width "
            f"{geometry.max_width}（无法表达完整历史视图，拒绝落位）"
        )
    if getattr(input_batch, "seq_lens_cpu_upper_bound", None) is None:
        raise UnsupportedConfig(
            "attnview: 缺少 seq_lens_cpu_upper_bound（FA2 路径应具备）：拒绝为取上界而回读设备张量"
        )
    # `scheduled == 1` **不能**证明是 decode：最后一个 prefill chunk 也可能只调度 1 个 token。
    # 必须读批内每行的真实 prefill 标记（`InputBatch.is_prefilling_np`）。
    is_prefilling = getattr(input_batch, "is_prefilling_np", None)
    if is_prefilling is None:
        raise UnsupportedConfig("attnview: 缺少 is_prefilling_np，无法区分 decode/prefill（拒绝放行）")
    da_row = req_ids.index(da_req_ids[0])
    if bool(is_prefilling[da_row]):
        raise UnsupportedConfig(
            f"attnview: {da_req_ids[0]} 本步是 prefill（is_prefilling=True，query 长度可恰为 1）："
            "本阶段只支持单 query(decode) 步"
        )
    device = group_table.device
    rows = int(group_table.shape[0])
    buffers = _buffers_for(runner, geometry.max_width, rows, device)

    # 非 DA 行必须保持原语义：先整组镜像 canonical 读表（按 max_width 切片），再覆写 DA 行。
    buffers["block_table"].copy_(group_table[:, : geometry.max_width], non_blocking=True)
    _METERING["device_copy_calls"] += 1
    buffers["seq_lens"].copy_(input_batch.seq_lens[:rows], non_blocking=True)
    buffers["seq_lens_cpu"].copy_(input_batch.seq_lens_cpu_upper_bound[:rows])
    _METERING["device_copy_calls"] += 1

    canonical_row_numbers = runner.block_tables.num_blocks.np[fa]
    _METERING["host_reads_of_cpu_staging"] += 1
    row_plans: dict[int, StepPlan] = {}
    overridden: list[int] = []
    for row, req_id in enumerate(req_ids):
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
        # 索引走 pinned 暂存 + 异步 H2D：避免小 H2D 在 stream 上触发同步
        visible = list(plan.visible_logical_blocks)
        host_index = _staging(runner, length=geometry.max_width, dtype=torch.long)[: len(visible)]
        host_index.copy_(torch.tensor(visible, dtype=torch.long))
        indices = host_index.to(device=device, non_blocking=True)
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
        # 标量也用 pinned 暂存 + 异步拷贝：直接 `= int` 会触发同步的小 H2D
        host_scalar = _staging(runner, length=1, dtype=torch.int32)
        host_scalar[0] = plan.seqused_k
        buffers["seq_lens"][row : row + 1].copy_(host_scalar, non_blocking=True)
        buffers["seq_lens_cpu"][row] = plan.seqused_k
        _METERING["device_copy_calls"] += 2
        _METERING["scalar_assignments"] += 2
        row_plans[row] = plan
        overridden.append(row)

    if not overridden:
        return None
    # FA 组的 max_seq_len 是**整批**标量：取**覆写后 CPU 长度向量**的最大值 ——
    # 非 DA 行已镜像 canonical 长度，DA 行是 seqused_k，因此这个最大值同时覆盖两类行，
    # 不需要再并入 DA 行原来的 canonical 上界（那会让单请求场景永不缩短，妨碍后续耗时归因）。
    max_seq_len = int(buffers["seq_lens_cpu"][:rows].max().item())
    _METERING["host_reads_of_cpu_staging"] += 1
    override = DaFaOverride(
        group_index=fa,
        block_table=buffers["block_table"],
        seq_lens=buffers["seq_lens"],
        max_seq_len=max_seq_len,
        seq_lens_cpu_upper_bound=buffers["seq_lens_cpu"],
        rows_with_override=tuple(overridden),
        plans=row_plans,
    )
    if os.environ.get(_CALIB_TRACE):
        calibration_note_override(
            override,
            req_ids=req_ids,
            scheduler_output=scheduler_output,
            inputs={
                "fa_group_table": group_table,
                "input_seq_lens": input_batch.seq_lens,
                "input_seq_lens_cpu": input_batch.seq_lens_cpu_upper_bound,
                "group0_table": block_tables[0],
            },
        )
    return override
