"""阶段 04：读取视图的 CPU 侧 metadata 转换与校验（纯函数，不依赖 torch/vLLM）。

职责边界：把"模式给出的可见范围（逻辑块集合）"转成 FA2 分页调用需要的 `block_table`
与 `seqused_k`，并给出尾长、下一写入 slot 等可核对字段。不做数值计算，也不接触 GPU。

关键语义（与阶段 04 工作单一致）：

* 读取按**块粒度**生效：模式 span 向外对齐到整块。
* `seqused_k` 是**前缀语义**：kernel 依次读表内各物理块，读满 `seqused_k` 个 token 即停。
  由于有效计数随块号前缀单调，保留集合里"部分有效且不是最后一块"不可能出现：块 i 不满整块
  意味着序列在块 i 内结束，其后所有块计数为 0，会被 ``dropped_empty_blocks`` 丢弃并如实报告。
  因此表内除最后一块外必为整块，前缀读取与保留集合逐 slot 一致。
* 有效读取表中不得出现 `-1`；表宽 = ``ceil(seqused_k / block_size)``。
* 下一 token 的写入 slot 只由**原始序列位置**决定，与读取视图无关。
"""

from __future__ import annotations

from dataclasses import dataclass


class GpuKvError(ValueError):
    """读取视图 metadata 不合法（调用方应修输入，不得放行）。"""


@dataclass(frozen=True)
class ReadTable:
    """一次分页读取的 CPU 侧描述。"""

    physical_blocks: tuple[int, ...]  # 表宽 = len()；不含 -1
    seqused_k: int  # 后端实际应看到的有效 K 长度
    effective_per_block: tuple[int, ...]  # 每个保留块的有效 token 数
    tail_len: int  # 最后保留块的有效尾长
    dropped_empty_blocks: tuple[int, ...]  # 无有效 token 因而未进表的保留块

    @property
    def width(self) -> int:
        return len(self.physical_blocks)

    @property
    def effective_len(self) -> int:
        return self.seqused_k


def validate_mapping(logical_to_physical: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """校验逻辑→物理块映射：非负、不重复。返回不可变副本。"""
    mapping = tuple(int(x) for x in logical_to_physical)
    if not mapping:
        raise GpuKvError("逻辑→物理映射为空")
    for logical, physical in enumerate(mapping):
        if physical < 0:
            raise GpuKvError(f"逻辑块 {logical} 的物理 ID 为负：{physical}")
    if len(set(mapping)) != len(mapping):
        raise GpuKvError("逻辑→物理映射存在重复物理块（同一物理块被两个逻辑块引用）")
    return mapping


def valid_count(block_index: int, seq_len: int, block_size: int) -> int:
    """逻辑块 ``block_index`` 中已写入的真实 token 数（0 表示尚未写入）。"""
    if block_index < 0:
        raise GpuKvError(f"逻辑块索引为负：{block_index}")
    if block_size <= 0:
        raise GpuKvError(f"块大小非法：{block_size}")
    if seq_len < 0:
        raise GpuKvError(f"序列长度为负：{seq_len}")
    return max(0, min(block_size, seq_len - block_index * block_size))


def normalize_retained_blocks(raw, num_logical_blocks: int) -> tuple[int, ...]:
    """规范化模式给出的保留块：去重 + 按逻辑升序（乱序/重复引用交给这里处理）。"""
    if num_logical_blocks <= 0:
        raise GpuKvError("逻辑块数量非法")
    seen: set[int] = set()
    for item in raw:
        index = int(item)
        if index < 0:
            raise GpuKvError(f"保留块索引为负：{index}")
        if index >= num_logical_blocks:
            raise GpuKvError(f"保留块索引越界：{index} >= {num_logical_blocks}")
        seen.add(index)
    if not seen:
        raise GpuKvError("保留块集合为空：读取视图至少要保留当前块")
    return tuple(sorted(seen))


def build_read_table(
    retained_blocks,
    logical_to_physical,
    seq_len: int,
    block_size: int,
) -> ReadTable:
    """由规范化保留块构造分页读取表与有效长度。"""
    mapping = validate_mapping(logical_to_physical)
    blocks = tuple(int(x) for x in retained_blocks)
    if not blocks:
        raise GpuKvError("保留块集合为空")
    if tuple(sorted(set(blocks))) != blocks:
        raise GpuKvError("保留块必须先经 normalize_retained_blocks 规范化（去重升序）")

    counts: list[int] = []
    dropped: list[int] = []
    keep: list[int] = []
    for index in blocks:
        if index >= len(mapping):
            raise GpuKvError(f"保留块 {index} 超出逻辑块数 {len(mapping)}")
        count = valid_count(index, seq_len, block_size)
        if count == 0:
            dropped.append(index)
            continue
        keep.append(index)
        counts.append(count)
    if not keep:
        raise GpuKvError("所有保留块都没有有效 token：读取视图为空")

    seqused_k = sum(counts)
    width = -(-seqused_k // block_size)  # ceil
    if width > len(keep):
        raise GpuKvError(f"表宽 {width} 超过保留块数 {len(keep)}：有效计数与块数不一致")
    return ReadTable(
        physical_blocks=tuple(mapping[index] for index in keep),
        seqused_k=seqused_k,
        effective_per_block=tuple(counts),
        tail_len=counts[-1],
        dropped_empty_blocks=tuple(dropped),
    )


def canonical_slot(position: int, block_size: int, logical_to_physical) -> int:
    """原始逻辑位置在物理缓存中的唯一 slot。"""
    mapping = validate_mapping(logical_to_physical)
    if position < 0:
        raise GpuKvError(f"位置为负：{position}")
    logical, offset = divmod(position, block_size)
    if logical >= len(mapping):
        raise GpuKvError(f"位置 {position} 超出缓存容量 {len(mapping) * block_size}")
    return mapping[logical] * block_size + offset


def next_write_slot(seq_len: int, block_size: int, logical_to_physical):
    """下一 token 的 canonical 写入位置：返回 (逻辑块, 块内偏移, 物理 slot)。

    只取决于原始序列长度，不随读取视图改变。
    """
    mapping = validate_mapping(logical_to_physical)
    if seq_len < 0:
        raise GpuKvError(f"序列长度为负：{seq_len}")
    logical, offset = divmod(seq_len, block_size)
    if logical >= len(mapping):
        raise GpuKvError(f"序列长度 {seq_len} 已超出缓存容量 {len(mapping) * block_size}")
    return logical, offset, mapping[logical] * block_size + offset
