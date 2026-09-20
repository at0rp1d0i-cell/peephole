"""固定 pin 的真实接线桥接(CPU 可测):FlashAttentionMetadata + 原生 KV → 独立参考。

真实接口(源码依据,pin 98dff2a8):
- `vllm/v1/attention/backends/flash_attn.py` 的 `forward(query, key, value, kv_cache, attn_metadata, layer, output, ...)`;
- `kv_cache` 形状 `[num_blocks, num_kv_heads, block_size, 2*head_size]`,K/V 解包方式
  `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` → 各 `[num_blocks, block_size, num_kv_heads, head_size]`;
- `FlashAttentionMetadata` 只有张量字段(`block_table` 为 2 维、`seq_lens`、`query_start_loc` 等),
  **没有** mode/refs/prompt_len/spans ⇒ 这些必须由既有 harness 的**独立时间线 + 真实 canonical 状态**桥接进来。

本模块只做桥接与参考执行,**默认关闭**;生产路径不引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .reference_dense import (
    ReferenceError,
    TestOnlyReferenceSwitch,
    dense_attention_fp32,
    gather_positions,
    independent_visible_positions,
)

__all__ = ["TimelineBridge", "unpack_native_kv", "resolve_geometry", "perform_reference_attention_native"]


def unpack_native_kv(kv_cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按固定 pin 的方式解包原生 KV:`transpose(1,2).split(head_size, -1)`。

    输入 `[num_blocks, num_kv_heads, block_size, 2*head_size]` → 各 `[num_blocks, block_size, num_kv_heads, head_size]`。
    """
    if kv_cache.dim() != 4:
        raise ReferenceError(f"kv_cache 必须是 4 维,实际 {tuple(kv_cache.shape)}")
    blocks, kv_heads, block_size, two_d = kv_cache.shape
    if two_d != 2 * head_size:
        raise ReferenceError(f"kv_cache 末维应为 2*head_size={2 * head_size},实际 {two_d}")
    # **不做 contiguous**:保留 stride 视图,避免每层每步复制整个 KV 缓存;
    # 后续按可见位置逐点 gather,只触碰被选中的已写位置。
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    if key_cache.shape != (blocks, block_size, kv_heads, head_size):
        raise ReferenceError(f"解包后形状异常:{tuple(key_cache.shape)}")
    return key_cache, value_cache


@dataclass
class TimelineBridge:
    """独立时间线 + 几何:由 config 声明(段末 token 触发、t+1 消费),不读取候选状态。

    `declarations` 形如 `[(parse_index_0based, mode, refs), ...]`,按 parse_index 升序;
    第 t 步(0 基生成 token 下标)消费的模式 = 最近一个 `parse_index <= t` 的声明。
    """

    prompt_len: int
    kernel_block_size: int
    sink_span: tuple[int, int]
    local_window_span: tuple[int, int]
    segment_spans: tuple[tuple[int, int], ...]
    declarations: tuple[tuple[int, str, tuple[int, ...]], ...] = ()
    current_request_id: str | None = None
    #: 每步相位(由 harness 的 `InputBatch.is_prefilling_np[目标行]` 提供);缺失即 fail closed。
    is_prefilling_by_step: dict[int, bool] = field(default_factory=dict)

    def mode_at(self, step_index_0based: int) -> tuple[str, tuple[int, ...]]:
        mode, refs = "global", ()
        for parse_index, decl_mode, decl_refs in self.declarations:
            if parse_index <= step_index_0based:
                mode, refs = decl_mode, tuple(decl_refs)
            else:
                break
        return mode, refs

    def kv_len_at(self, step_index_0based: int) -> int:
        """独立 KV 上界:生成流第 t 个 token 被消费的 forward 的 KV 长度 = prompt_len + t + 1。"""
        return self.prompt_len + step_index_0based + 1

    def is_prefilling_at(self, step_index_0based: int) -> bool | None:
        """相位(**唯一权威证据**来自 harness 的 `is_prefilling_np[目标行]`);未提供返回 None。"""
        return self.is_prefilling_by_step.get(step_index_0based)

    def mask_at(self, step_index_0based: int) -> Any:
        mode, refs = self.mode_at(step_index_0based)
        return independent_visible_positions(
            mode=mode, refs=refs, kv_len=self.kv_len_at(step_index_0based), prompt_len=self.prompt_len,
            sink_span=self.sink_span, local_window_span=self.local_window_span,
            segment_spans=self.segment_spans, block_size=self.kernel_block_size,
        )


def resolve_geometry(
    *,
    bridge: TimelineBridge,
    attn_metadata: Any,
    request_idx: int,
    step_index_0based: int,
) -> dict[str, Any]:
    """把**真实 metadata** 与**独立时间线**对齐:块表取自 metadata,模式/几何取自时间线。

    `block_table` 必须是二维张量(每请求一行);`seq_lens` 存在时与独立 KV 上界**交叉核对**。
    """
    bt = getattr(attn_metadata, "block_table", None)
    if bt is None or not torch.is_tensor(bt) or bt.dim() != 2:
        raise ReferenceError("metadata.block_table 必须是二维张量(每请求一行)")
    if not 0 <= request_idx < bt.shape[0]:
        raise ReferenceError(f"request_idx={request_idx} 超出 block_table 行数 {bt.shape[0]}")
    row = [int(x) for x in bt[request_idx].tolist()]
    if any(b < 0 for b in row):
        raise ReferenceError("block_table 行含负值(未分配槽不能进入参考)")
    kv_len = bridge.kv_len_at(step_index_0based)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if not torch.is_tensor(seq_lens) or seq_lens.dim() != 1 or not 0 <= request_idx < seq_lens.shape[0]:
        raise ReferenceError("metadata.seq_lens 缺失或形状不符:参考拒绝在缺证据下继续(fail closed)")
    seq_lens_value = int(seq_lens[request_idx])
    if seq_lens_value != kv_len:
        raise ReferenceError(
            f"metadata.seq_lens[{request_idx}]={seq_lens_value} 与独立时间线 KV 上界 {kv_len} 不一致")
    # 单请求:由 query_start_loc 长度判定(n+1 个边界 = n 个请求)
    qsl = getattr(attn_metadata, "query_start_loc", None)
    if not torch.is_tensor(qsl) or qsl.dim() != 1 or qsl.numel() - 1 != 1:
        raise ReferenceError(f"仅支持单请求:query_start_loc 给出 {None if not torch.is_tensor(qsl) else qsl.numel() - 1} 个请求")
    # 计数器(pin 可能全 0,不作相位证据):仅在已填充时交叉核对
    nd, nt = getattr(attn_metadata, "num_decode_reqs", None), getattr(attn_metadata, "num_decode_tokens", None)
    npf = getattr(attn_metadata, "num_prefill_tokens", None)
    counters_populated = any(int(x or 0) for x in (nd, nt, npf))
    if counters_populated and int(nd or 0) != 1:
        raise ReferenceError(f"计数器已填充但与单请求 decode 冲突:num_decode_reqs={nd}")
    # 相位:必须由 harness 的 is_prefilling_np 明示为 decode(false);缺失或 prefill 即拒绝
    phase = bridge.is_prefilling_at(step_index_0based)
    if phase is None:
        raise ReferenceError("缺少相位证据(harness 的 is_prefilling_np[目标行]):拒绝用 q_len/单 token 代替")
    if phase:
        raise ReferenceError("本步相位为 prefill(可能是末尾单 token prefill chunk):参考不接管")
    return {"block_table_row": row, "kv_len": kv_len, "seq_lens_value": seq_lens_value,
            "mode": bridge.mode_at(step_index_0based)[0], "refs": bridge.mode_at(step_index_0based)[1]}


def perform_reference_attention_native(
    switch: TestOnlyReferenceSwitch,
    *,
    layer_idx: int,
    step_index_0based: int,
    bridge: TimelineBridge,
    request_idx: int,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
    head_size: int,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    impl_scale: float | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    """真实签名下的参考执行:解包原生 KV → 独立 mask → gather → FP32 → cast → 原地写 output。

    `key`/`value` 是真实 forward 传入的**本步** K/V(写入前的形状);dense 参考读取的是
    canonical KV 缓存(含本步已写入位置),因此这两个参数只做存在性记录,不参与计算。
    """
    geo = resolve_geometry(bridge=bridge, attn_metadata=attn_metadata, request_idx=request_idx,
                           step_index_0based=step_index_0based)
    # --- fail-fast 门禁(全部从真实 impl/metadata/张量提取) ---
    if output_scale is not None or output_block_scale is not None:
        raise ReferenceError("出现量化输出缩放(output_scale/output_block_scale 非 None):参考拒绝在未支持特性下继续")
    if impl_scale is None:
        raise ReferenceError("缺少 impl.scale:参考必须使用真实 impl 的 scale")
    if float(switch.scale) and float(switch.scale) != float(impl_scale):
        raise ReferenceError(f"switch.scale={switch.scale} 与 impl.scale={impl_scale} 不一致")
    if kv_cache.shape[2] != bridge.kernel_block_size:
        raise ReferenceError(
            f"native KV 块长 {kv_cache.shape[2]} 与 kernel_block_size {bridge.kernel_block_size} 不一致")
    if output.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
        raise ReferenceError(f"仅支持 BF16:query={query.dtype} output={output.dtype}")
    causal = getattr(attn_metadata, "causal", None)
    if causal is not None and bool(causal) is not True:
        raise ReferenceError(f"仅支持 causal=True,实际 {causal}")
    n_decode = getattr(attn_metadata, "num_decode_reqs", None)
    num_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    if num_tokens is not None and int(num_tokens) != 1:
        raise ReferenceError(f"仅支持单 token decode,num_actual_tokens={num_tokens}")
    mask = bridge.mask_at(step_index_0based)
    key_cache, value_cache = unpack_native_kv(kv_cache, head_size)
    # 真实 pin 传入的是 `[num_tokens, num_heads, head_dim]`(attention.py:524-525);本参考只支持单 token decode。
    if query.dim() == 3:
        if query.shape[0] != 1:
            raise ReferenceError(f"仅支持单 token decode,实际 num_tokens={query.shape[0]}")
        if output.shape != query.shape:
            raise ReferenceError(f"output 形状 {tuple(output.shape)} 与 query {tuple(query.shape)} 不一致")
    elif query.dim() == 2:
        if output.shape != query.shape:
            raise ReferenceError(f"output 形状 {tuple(output.shape)} 与 query {tuple(query.shape)} 不一致")
    else:
        raise ReferenceError(f"query 维数非法:{query.dim()}")
    q = query.reshape(-1, query.shape[-1])
    k, v = gather_positions(key_cache, value_cache, geo["block_table_row"], bridge.kernel_block_size,
                            mask.read_positions)
    fp32, casted = dense_attention_fp32(q, k, v, scale=float(impl_scale), dtype=output.dtype)
    if casted.dtype != output.dtype:
        raise ReferenceError(f"cast 目标 dtype 与 output 不一致:{casted.dtype} vs {output.dtype}")
    ptr_before = output.data_ptr()
    target = output if output.dim() == 2 else output[0]        # 写回**对应视图**(缓冲本身不变)
    if tuple(target.shape) != tuple(casted.shape):
        raise ReferenceError(f"写回视图形状 {tuple(target.shape)} 与参考 {tuple(casted.shape)} 不一致")
    target.copy_(casted)
    if output.data_ptr() != ptr_before:
        raise ReferenceError("写回后 output 缓冲身份发生变化(不得替换缓冲)")
    diff = casted.to(torch.float32) - fp32
    ref_l2 = float(torch.linalg.vector_norm(fp32))
    non_finite = int((~torch.isfinite(fp32)).sum().item())
    if non_finite:
        raise ReferenceError(f"参考输出含非有限值:{non_finite}")
    return {
        "layer_idx": layer_idx, "step_index_0based": step_index_0based, "mode": geo["mode"],
        "refs": list(geo["refs"]), "kv_len": geo["kv_len"], "seq_lens_value": geo["seq_lens_value"],
        "semantic_positions": len(mask.semantic_positions), "read_positions": len(mask.read_positions),
        "blocks": list(mask.blocks), "effective_per_block": list(mask.effective_per_block),
        "block_table_row_head": geo["block_table_row"][:4], "output_dtype": str(output.dtype),
        "fp32_to_output_max_abs": float(diff.abs().max()),
        "fp32_to_output_rms": float(torch.sqrt((diff ** 2).mean())),
        "fp32_to_output_rel_l2": (float(torch.linalg.vector_norm(diff)) / ref_l2) if ref_l2 > 0 else None,
        "ref_abs_max": float(fp32.abs().max()), "non_finite_count": non_finite, "wrote_in_place": True,
        "query_shape": list(query.shape), "output_shape": list(output.shape),
        "num_tokens": int(query.shape[0]) if query.dim() == 3 else None,
        "buffer_ptr_preserved": True,
        "got_key_arg": key is not None, "got_value_arg": value is not None,
        "impl_scale": float(impl_scale), "kv_block_len": int(kv_cache.shape[2]),
        "kv_strides": [int(s) for s in k_cache.stride()],
        "causal": bool(causal) if causal is not None else None,
        "num_decode_reqs": int(n_decode) if n_decode is not None else None,
    }
