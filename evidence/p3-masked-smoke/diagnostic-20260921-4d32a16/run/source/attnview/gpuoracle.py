"""阶段 04 独立参考 oracle（**不 import 候选转换模块**，不读候选读表）。

独立性的具体含义：

* 可见集合从**语义输入**独立推出：mode、refs、layout 的 sink/segment/local_window spans、
  `prompt_len`、`attention_kv_len`。用逐位置布尔 mask 表达，再做块外扩与因果上界，
  **不复用** `readview` 的 span 代数，也不调用 `gpukv` 的任何函数。
* 注意力从**逻辑真值** K_true/V_true 按 mask 位置 gather 后 FP32 显式计算（逐 head 实现）。
* oracle 另外给出它对"期望物理表/每块有效计数/seqused_k/下一写入位置"的**独立预期**，
  供与候选数据面逐项比对（方向是 oracle→期望，不是由候选表反推集合）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class OracleError(ValueError):
    """用例/语义输入不合法（例如当前 token 不可见）。"""


@dataclass(frozen=True)
class OracleView:
    mode: str
    visible_blocks: tuple[int, ...]
    physical_blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]
    seqused_k: int
    positions: tuple[int, ...]
    next_write_slot: tuple[int, int, int]
    attention_kv_len: int
    kernel_block_size: int


def _segments_refs(segments, refs):
    out = []
    for ref in refs:
        if not 1 <= ref <= len(segments):
            raise OracleError(f"segment 编号越界：{ref}（共 {len(segments)}）")
        out.append(tuple(segments[ref - 1]))
    return out


def build_oracle_view(
    *,
    mode: str,
    refs,
    segments,
    local_window,
    prompt_len: int,
    attention_kv_len: int,
    kernel_block_size: int,
    logical_to_physical,
    sink=(0, 16),
    explicit_blocks=None,
) -> OracleView:
    """独立构造可见集合、期望物理表与写入位置。

    `explicit_blocks` 仅供底层**补充用例**（可见集合直接给定、不来自 ReadView）使用：
    oracle 仍自行推导位置、每块有效计数、seqused_k 与物理表，不读候选读表。
    """
    b = int(kernel_block_size)
    kv_len = int(attention_kv_len)
    if b <= 0 or kv_len <= 0:
        raise OracleError(f"block_size/attention_kv_len 非法：{b}/{kv_len}")
    if mode not in ("global", "focus", "local"):
        raise OracleError(f"未知模式：{mode}")

    if explicit_blocks is not None:
        semantic_spans = [
            (int(block) * b, min((int(block) + 1) * b, kv_len)) for block in explicit_blocks
        ]
    else:
        semantic_spans = [tuple(sink), tuple(local_window), (int(prompt_len), kv_len)]
        if mode == "focus":
            semantic_spans.extend(_segments_refs(segments, refs))
        if mode == "global":
            semantic_spans.append((0, kv_len))

    # 逐位置布尔 mask（不看 span 排序、不做合并，直接按位置判定）
    semantic = bytearray(kv_len)
    for start, end in semantic_spans:
        for p in range(max(0, int(start)), min(int(end), kv_len)):
            semantic[p] = 1

    visible_blocks = sorted({p // b for p, flag in enumerate(semantic) if flag})
    if not visible_blocks:
        raise OracleError("可见集合为空")

    # 块外扩：整块可见（右端被 kv_len 截断）
    aligned = bytearray(kv_len)
    for block in visible_blocks:
        for p in range(block * b, min((block + 1) * b, kv_len)):
            aligned[p] = 1

    current_position = kv_len - 1
    if not aligned[current_position]:
        raise OracleError(
            f"当前 token 位置 {current_position} 不在可见集合内：阶段 04 只覆盖保留当前 token 的 decode"
        )

    positions = tuple(p for p, flag in enumerate(aligned) if flag)
    counts = tuple(
        sum(1 for p in positions if p // b == block) for block in visible_blocks
    )
    mapping = tuple(int(x) for x in logical_to_physical)
    for block in visible_blocks:
        if block >= len(mapping):
            raise OracleError(f"可见块 {block} 超出逻辑块数 {len(mapping)}")
        if mapping[block] < 0:
            raise OracleError(f"逻辑块 {block} 的物理 ID 为负：{mapping[block]}")

    logical, offset = divmod(kv_len, b)
    if logical >= len(mapping):
        raise OracleError(f"写入位置 {kv_len} 超出缓存容量")
    return OracleView(
        mode=mode,
        visible_blocks=tuple(visible_blocks),
        physical_blocks=tuple(mapping[block] for block in visible_blocks),
        effective_per_block=counts,
        seqused_k=len(positions),
        positions=positions,
        next_write_slot=(logical, offset, mapping[logical] * b + offset),
        attention_kv_len=kv_len,
        kernel_block_size=b,
    )


def gather_positions(k_true: torch.Tensor, v_true: torch.Tensor, positions):
    """按**逻辑位置**从真值取 K/V（oracle 专属；候选路径不得这样 gather）。"""
    index = torch.tensor(positions, dtype=torch.long)
    return k_true[index], v_true[index]


def attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """逐 head FP32 显式缩放点积 + softmax + V 加权和（GQA：h → h // (H/KVH)）。"""
    heads, kv_heads = q.shape[0], k.shape[1]
    if heads % kv_heads:
        raise OracleError(f"q head 数 {heads} 不是 kv head 数 {kv_heads} 的整数倍")
    rows = []
    for h in range(heads):
        kvh = h // (heads // kv_heads)
        scores = (q[h].float() @ k[:, kvh].float().T) * scale
        probs = torch.softmax(scores, dim=0)
        rows.append(probs @ v[:, kvh].float())
    return torch.stack(rows)


def attention_sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """第三种实现（SDPA）用于 oracle 自检：同一可见集合、同一真值。"""
    heads, kv_heads = q.shape[0], k.shape[1]
    repeat = heads // kv_heads
    q4 = q.float().view(1, heads, 1, -1)                        # [1, H, 1, D]
    k4 = k.float().repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
    v4 = v.float().repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    return out.view(heads, -1)
