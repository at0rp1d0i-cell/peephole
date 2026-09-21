"""阶段 04 候选侧：读取视图的 metadata 转换与校验（纯函数，不依赖 torch/vLLM）。

输入是阶段 03 已验收的 `ReadView`（`attnview.readview`），输出是 FA2 分页调用需要的
**无 `-1`** 读取表与有效长度。候选执行路径只做这一件事：把视图的可见块映射成物理块表，
不 gather、不复制 KV。

不变量（全部硬校验，违反即抛错，不静默降级）：

* 视图给出的有效前缀里不得出现 `-1`（`-1` 只允许出现在阶段 03 的尾部填充区）。
* 物理块 ID 非负且互不重复；逻辑可见块升序唯一。
* 每个可见块必须**含至少一个已写位置**（否则读取会读入未写槽）——误用输入在转换边界报错，
  不用"静默丢弃"掩盖坏 span。
* 当前 token 位置（`attention_kv_len - 1`）必须在可见集合内：本阶段只承诺
  query_len=1 且保留当前 token 的因果 decode。
* 前缀语义：可见块中除最大块外必须整块有效，否则 kernel 会把块内未写槽读进来。
"""

from __future__ import annotations

from dataclasses import dataclass


class GpuKvError(ValueError):
    """读取视图 metadata 不合法（调用方应修输入，不得放行）。"""


@dataclass(frozen=True)
class ReadTable:
    """一次分页读取的候选侧描述。"""

    physical_blocks: tuple[int, ...]  # 表宽 = len()；不含 -1
    visible_blocks: tuple[int, ...]  # 对应逻辑块
    effective_per_block: tuple[int, ...]
    seqused_k: int
    tail_len: int
    attention_kv_len: int
    next_write_position: int
    block_size: int

    @property
    def width(self) -> int:
        return len(self.physical_blocks)

    @property
    def effective_len(self) -> int:
        return self.seqused_k

    def padded_row(self, width: int) -> list[int]:
        """补到指定宽度：用自身最后一块重复填充，**不使用 -1**。"""
        if width < self.width:
            raise GpuKvError(f"目标宽度 {width} 小于表宽 {self.width}")
        return list(self.physical_blocks) + [self.physical_blocks[-1]] * (width - self.width)


def validate_mapping(logical_to_physical) -> tuple[int, ...]:
    """校验逻辑→物理块映射：非负、不重复。"""
    mapping = tuple(int(x) for x in logical_to_physical)
    if not mapping:
        raise GpuKvError("逻辑→物理映射为空")
    for logical, physical in enumerate(mapping):
        if physical < 0:
            raise GpuKvError(f"逻辑块 {logical} 的物理 ID 为负：{physical}")
    if len(set(mapping)) != len(mapping):
        raise GpuKvError("逻辑→物理映射存在重复物理块")
    return mapping


def canonical_slot(position: int, block_size: int, logical_to_physical) -> int:
    """原始逻辑位置在物理缓存中的 slot（写入路径用）。"""
    mapping = validate_mapping(logical_to_physical)
    if position < 0:
        raise GpuKvError(f"位置为负：{position}")
    logical, offset = divmod(position, block_size)
    if logical >= len(mapping):
        raise GpuKvError(f"位置 {position} 超出缓存容量 {len(mapping) * block_size}")
    return mapping[logical] * block_size + offset


def next_write_slot(seq_len: int, block_size: int, logical_to_physical):
    """下一 token 的 canonical 写入位置：(逻辑块, 块内偏移, 物理 slot)。"""
    mapping = validate_mapping(logical_to_physical)
    if seq_len < 0:
        raise GpuKvError(f"序列长度为负：{seq_len}")
    logical, offset = divmod(seq_len, block_size)
    if logical >= len(mapping):
        raise GpuKvError(f"序列长度 {seq_len} 已超出缓存容量 {len(mapping) * block_size}")
    return logical, offset, mapping[logical] * block_size + offset


def read_table_from_read_view(view, logical_to_physical) -> ReadTable:
    """阶段 03 `ReadView` → 无 `-1` 的读取表与有效长度。

    只用视图已经算好的可见块与可见 span（数据面），并对前缀语义、当前 token 可见性做硬校验。
    """
    mapping = validate_mapping(logical_to_physical)
    block_size = int(view.kernel_block_size)
    kv_len = int(view.attention_kv_len)
    if block_size <= 0 or kv_len <= 0:
        raise GpuKvError(f"block_size/kv_len 非法：{block_size}/{kv_len}")

    visible_blocks = tuple(int(b) for b in view.visible_blocks)
    if not visible_blocks:
        raise GpuKvError("可见块集合为空")
    if tuple(sorted(set(visible_blocks))) != visible_blocks:
        raise GpuKvError("可见块必须升序且唯一")
    valid_prefix = tuple(view.physical_block_ids[: int(view.valid_counts)])
    if len(valid_prefix) != len(visible_blocks):
        raise GpuKvError(
            f"可见块数 {len(visible_blocks)} 与物理表有效前缀 {len(valid_prefix)} 不一致"
        )
    if any(pid == -1 for pid in valid_prefix):
        raise GpuKvError("物理表有效前缀出现 -1（-1 只能出现在尾部填充）")
    if len(set(valid_prefix)) != len(valid_prefix):
        raise GpuKvError("物理表有效前缀存在重复物理块")
    for block, pid in zip(visible_blocks, valid_prefix, strict=True):
        if block >= len(mapping):
            raise GpuKvError(f"可见块 {block} 超出逻辑块数 {len(mapping)}")
        if pid != mapping[block]:
            raise GpuKvError(
                f"可见块 {block} 的物理 ID {pid} 与 canonical 映射 {mapping[block]} 不一致"
            )
        if block * block_size >= kv_len:
            raise GpuKvError(
                f"可见块 {block} 不含任何已写位置（kv_len={kv_len}）：误用 span 必须在转换边界报错"
            )

    counts = []
    for block in visible_blocks:
        low, high = block * block_size, min((block + 1) * block_size, kv_len)
        covered = 0
        for start, end in view.visible_spans:
            lo, hi = max(start, low), min(end, high)
            if hi > lo:
                covered += hi - lo
        counts.append(covered)
    tail_block = visible_blocks[-1]
    for block, count in zip(visible_blocks[:-1], counts[:-1], strict=True):
        if count != block_size:
            raise GpuKvError(
                f"可见块 {block} 只有 {count}/{block_size} 个已覆盖位置且不是最大可见块："
                "分页前缀语义会读入未写槽"
            )
    if counts[-1] == 0:
        raise GpuKvError(f"最大可见块 {tail_block} 无覆盖位置")

    current_position = kv_len - 1
    if current_position // block_size not in visible_blocks:
        raise GpuKvError(
            f"当前 token 位置 {current_position}（块 {current_position // block_size}）不可见："
            "阶段 04 只承诺 query_len=1 且保留当前 token 的因果 decode"
        )

    seqused_k = sum(counts)
    needed_width = -(-seqused_k // block_size)
    if needed_width != len(valid_prefix):
        raise GpuKvError(
            f"表宽 {len(valid_prefix)} 与 ceil(seqused_k/block_size)={needed_width} 不一致："
            "后端会按前缀语义索引第 needed_width 列，列数不足即越界"
        )
    return ReadTable(
        physical_blocks=tuple(valid_prefix),
        visible_blocks=visible_blocks,
        effective_per_block=tuple(counts),
        seqused_k=seqused_k,
        tail_len=counts[-1],
        attention_kv_len=kv_len,
        next_write_position=int(view.next_write_position),
        block_size=block_size,
    )
