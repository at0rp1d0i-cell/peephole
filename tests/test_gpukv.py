"""阶段 04 CPU 侧：读取视图 metadata 转换与校验（不含 GPU）。"""

from __future__ import annotations

import unittest

from attnview.gpukv import (
    GpuKvError,
    build_read_table,
    canonical_slot,
    next_write_slot,
    normalize_retained_blocks,
    valid_count,
)

# 与 configs/p1-gpu/read-view-check.json 冻结值一致
B = 784
L2P = (7, 2, 11, 0, 5, 9, 1, 8, 3, 10, 4, 6)


class MappingTest(unittest.TestCase):
    def test_canonical_slot_uses_original_logical_position(self) -> None:
        self.assertEqual(canonical_slot(0, B, L2P), 7 * B)
        self.assertEqual(canonical_slot(B, B, L2P), 2 * B)
        self.assertEqual(canonical_slot(B + 3, B, L2P), 2 * B + 3)
        self.assertEqual(canonical_slot(8 * B - 1, B, L2P), 8 * B + (B - 1))  # 逻辑块 7 -> 物理 8

    def test_negative_physical_id_is_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            canonical_slot(0, B, (0, -1, 2))

    def test_duplicate_physical_block_is_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            build_read_table((0, 1), (0, 0), seq_len=B, block_size=B)

    def test_position_beyond_capacity_is_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            canonical_slot(12 * B, B, L2P)


class NormalizeTest(unittest.TestCase):
    def test_unordered_and_duplicate_blocks_become_sorted_unique(self) -> None:
        self.assertEqual(normalize_retained_blocks((5, 4, 1, 0, 4, 0), 12), (0, 1, 4, 5))

    def test_negative_or_out_of_range_index_is_rejected(self) -> None:
        for raw in ((-1,), (12,), (0, 99)):
            with self.subTest(raw=raw), self.assertRaises(GpuKvError):
                normalize_retained_blocks(raw, 12)

    def test_empty_set_is_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            normalize_retained_blocks((), 12)


class EffectiveCountTest(unittest.TestCase):
    def test_counts_follow_sequence_length(self) -> None:
        self.assertEqual(valid_count(0, 2587, B), B)
        self.assertEqual(valid_count(3, 2587, B), 2587 - 3 * B)
        self.assertEqual(valid_count(4, 2587, B), 0)
        self.assertEqual(valid_count(2, 1569, B), 1)
        self.assertEqual(valid_count(2, 2352, B), B)


class ReadTableTest(unittest.TestCase):
    def test_full_read_spanning_tail_block(self) -> None:
        table = build_read_table((0, 1, 2, 3), L2P, seq_len=2587, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 2, 11, 0))
        self.assertEqual(table.seqused_k, 2587)
        self.assertEqual(table.width, 4)
        self.assertEqual(table.effective_per_block, (B, B, B, 235))
        self.assertEqual(table.tail_len, 235)
        self.assertEqual(table.dropped_empty_blocks, ())

    def test_non_adjacent_focus_skips_middle_blocks(self) -> None:
        table = build_read_table((0, 1, 4, 5), L2P, seq_len=6272, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 2, 5, 9))
        self.assertEqual(table.seqused_k, 4 * B)
        self.assertEqual(table.width, 4)
        self.assertEqual(table.tail_len, B)

    def test_local_view_excludes_document_blocks(self) -> None:
        table = build_read_table((0, 6, 7), L2P, seq_len=6272, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 1, 8))
        self.assertEqual(table.seqused_k, 3 * B)
        self.assertEqual(table.effective_per_block, (B, B, B))

    def test_tail_lengths_one_and_b_minus_one(self) -> None:
        one = build_read_table((0, 1, 2), L2P, seq_len=2 * B + 1, block_size=B)
        self.assertEqual((one.seqused_k, one.tail_len, one.width), (2 * B + 1, 1, 3))
        near = build_read_table((0, 1, 2), L2P, seq_len=2 * B + (B - 1), block_size=B)
        self.assertEqual((near.seqused_k, near.tail_len, near.width), (3 * B - 1, B - 1, 3))
        full = build_read_table((0, 1, 2), L2P, seq_len=3 * B, block_size=B)
        self.assertEqual((full.seqused_k, full.tail_len, full.width), (3 * B, B, 3))

    def test_table_has_no_minus_one_and_width_matches_seqused(self) -> None:
        table = build_read_table((0, 2), L2P, seq_len=2 * B + 1, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 11))
        self.assertEqual(table.seqused_k, B + 1)
        self.assertEqual(table.width, 2)
        self.assertTrue(all(p >= 0 for p in table.physical_blocks))

    def test_dropped_blocks_are_reported_not_silently_short(self) -> None:
        table = build_read_table((0, 1, 5), L2P, seq_len=2 * B, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 2))
        self.assertEqual(table.seqused_k, 2 * B)
        self.assertEqual(table.dropped_empty_blocks, (5,))

    def test_partial_block_in_the_middle_is_impossible_by_construction(self) -> None:
        # seq_len = B+5：块 1 只有 5 个有效 token，块 2 无有效 token -> 块 2 被丢弃，
        # 块 1 成为最后一块（尾长 5），不会再出现"非最后块部分有效"的读法
        table = build_read_table((0, 1, 2), L2P, seq_len=B + 5, block_size=B)
        self.assertEqual(table.physical_blocks, (7, 2))
        self.assertEqual(table.seqused_k, B + 5)
        self.assertEqual(table.tail_len, 5)
        self.assertEqual(table.dropped_empty_blocks, (2,))
        self.assertEqual(table.width, 2)

    def test_unnormalized_blocks_are_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            build_read_table((1, 0), L2P, seq_len=2 * B, block_size=B)


class WriteSlotTest(unittest.TestCase):
    def test_write_slot_follows_original_position_not_read_view(self) -> None:
        # 完整读取（块 0..2）与 local 视图（块 0、2）下的下一写入位置必须相同
        self.assertEqual(next_write_slot(1569, B, L2P), (2, 1, 11 * B + 1))
        self.assertEqual(next_write_slot(1568, B, L2P), (2, 0, 11 * B))
        self.assertEqual(next_write_slot(3 * B, B, L2P), (3, 0, 0))

    def test_write_beyond_capacity_is_rejected(self) -> None:
        with self.assertRaises(GpuKvError):
            next_write_slot(12 * B, B, L2P)


if __name__ == "__main__":
    unittest.main()
