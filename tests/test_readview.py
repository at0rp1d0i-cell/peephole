"""C4 读取视图验收（工作单 §4「视图」面）：独立参考逐位置对照 + I1–I8 不变量。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview.parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL  # noqa: E402
from attnview.readview import (  # noqa: E402
    INVALID_SLOT,
    ReadViewError,
    TokenLayout,
    ViewInputs,
    build_read_view,
    to_kernel_args,
)
from attnview.reference import declared_positions, reference_visible_positions  # noqa: E402

PROMPT_LEN = 2000
SEGMENTS = ((100, 300), (900, 1200), (1600, 1900))
LOCAL_WINDOW = (1950, 2000)
PHYSICAL = (77, 12, 305, 41, 9, 88, 23, 61, 150, 4)  # 故意非连续、且与逻辑块号不同


def canonical(count: int) -> tuple[int, ...]:
    """确定性、唯一、非连续的物理块映射（逻辑块 i → 物理块不被打乱后可验证）。"""
    ids = list(range(count + 500, count + 500 + count))
    ids = ids[::2] + ids[1::2][::-1]  # 固定置换，且逻辑块号 != 物理块号
    return tuple(ids)


def layout() -> TokenLayout:
    return TokenLayout(
        prompt_len=PROMPT_LEN,
        segment_spans=SEGMENTS,
        local_window_span=LOCAL_WINDOW,
    )


def view(mode: str, refs=(), attention_kv_len: int = PROMPT_LEN, block: int = 16) -> object:
    table = canonical(attention_kv_len // block + 2)
    return build_read_view(
        ViewInputs(
            mode=mode,
            refs=tuple(refs),
            layout=layout(),
            attention_kv_len=attention_kv_len,
            canonical_blocks=table,
            kernel_block_size=block,
            max_width=len(table),
        )
    )


class ReadViewTest(unittest.TestCase):
    def test_global_equals_full_read(self) -> None:
        v = view(MODE_GLOBAL)
        self.assertEqual(v.positions(), tuple(range(PROMPT_LEN)))
        self.assertEqual(v.visible_blocks, tuple(range((PROMPT_LEN + 15) // 16)))
        self.assertEqual(v.valid_counts, len(v.visible_blocks))

    def test_independent_reference_agreement(self) -> None:
        """被测构造器与**独立**逐位置参考在所有组合上一致。"""
        for block in (16, 32, 784):
            for mode, refs in (
                (MODE_GLOBAL, ()),
                (MODE_LOCAL, ()),
                (MODE_FOCUS, (1,)),
                (MODE_FOCUS, (2, 3)),
                (MODE_FOCUS, (1, 2, 3)),
            ):
                for attention_kv_len in (PROMPT_LEN, PROMPT_LEN + 1, PROMPT_LEN + 37, 2200):
                    v = view(mode, refs, attention_kv_len=attention_kv_len, block=block)
                    expected = reference_visible_positions(
                        mode=mode,
                        refs=refs,
                        layout=layout(),
                        attention_kv_len=attention_kv_len,
                        kernel_block_size=block,
                    )
                    self.assertEqual(
                        v.positions(), expected,
                        f"{mode}{refs} kv={attention_kv_len} b={block}",
                    )

    def test_outward_alignment_keeps_every_declared_position(self) -> None:
        for block in (16, 32, 784):
            v = view(MODE_FOCUS, (1, 3), block=block)
            declared = declared_positions(layout(), (1, 3), PROMPT_LEN)
            positions = set(v.positions())
            self.assertTrue(set(declared) <= positions, "声明过的位置一个都不能丢（I1）")
            semantic = (
                set(range(0, 16))
                | set(range(*LOCAL_WINDOW))
                | set(range(PROMPT_LEN, v.attention_kv_len))
            )
            for ref in (1, 3):
                semantic |= set(range(*SEGMENTS[ref - 1]))
            extra = positions - semantic
            # 多读的每个位置都必须与某个语义位置同块（即：恰好是块外扩的产物）
            for pos in extra:
                block_span = set(range((pos // block) * block, (pos // block) * block + block))
                self.assertTrue(
                    block_span & semantic, f"pos={pos} 不属于任何语义位置的块（多读越界）"
                )
            self.assertLessEqual(
                len(extra), 2 * (block - 1) * 6, "每边最多多读 b-1（附录 B）"
            )

    def test_refs_order_independence_and_dedupe(self) -> None:
        a = view(MODE_FOCUS, (3, 1, 2))
        b = view(MODE_FOCUS, (2, 3, 1, 2))
        self.assertEqual(a.visible_blocks, b.visible_blocks)
        self.assertEqual(a.visible_spans, b.visible_spans)
        self.assertEqual(list(a.visible_blocks), sorted(set(a.visible_blocks)))

    def test_adjacent_and_overlapping_segments_merge(self) -> None:
        v = view(MODE_FOCUS, (1, 2, 3), block=64)
        # 三段合并后应是单块区域的并集，且不重复计数
        spans = v.visible_spans
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertLess(e1, s2, "相邻/重叠区间必须已合并")

    def test_timing_table_fields(self) -> None:
        """读/写时序字段：写入前已写 = attention 上界 - 1；下一步写位置 = attention 上界。"""
        v = view(MODE_GLOBAL, attention_kv_len=PROMPT_LEN)
        self.assertEqual(v.written_before_step, PROMPT_LEN - 1)
        self.assertEqual(v.next_write_position, PROMPT_LEN)
        self.assertEqual(v.response_span, (PROMPT_LEN, PROMPT_LEN))

    def test_partial_tail_block_and_valid_len(self) -> None:
        block = 16
        attention_kv_len = PROMPT_LEN + 5  # 尾块只写了 5 个 token
        v = view(MODE_LOCAL, attention_kv_len=attention_kv_len, block=block)
        self.assertEqual(v.tail_block_valid_len, (attention_kv_len - 1) % block + 1)
        self.assertEqual(v.next_write_position, attention_kv_len)
        self.assertEqual(v.written_before_step, attention_kv_len - 1)
        self.assertLessEqual(
            max(v.positions()), attention_kv_len - 1, "不得读到未写满的块外（I6）"
        )

    def test_shape_stability_and_no_invalid_slot_in_prefix(self) -> None:
        widths = set()
        table = canonical(600)  # 适配层在请求开始时定一次，之后逐 step 复用（I5）
        for extra in range(0, 200, 37):
            v = build_read_view(
                ViewInputs(
                    mode=MODE_FOCUS,
                    refs=(2,),
                    layout=layout(),
                    attention_kv_len=PROMPT_LEN + extra,
                    canonical_blocks=table,
                    kernel_block_size=16,
                    max_width=len(table),
                )
            )
            widths.add(v.max_width)
            self.assertEqual(len(v.physical_block_ids), v.max_width)
            valid = v.physical_block_ids[: v.valid_counts]
            tail = v.physical_block_ids[v.valid_counts :]
            self.assertNotIn(INVALID_SLOT, valid, "有效前缀内不得有 -1（I4）")
            self.assertTrue(all(p == INVALID_SLOT for p in tail), "尾部必须是 -1 填充（I4）")
        self.assertEqual(len(widths), 1, "max_width 在一次生成内必须恒定（I5）")

    def test_logical_block_is_not_used_as_physical_id(self) -> None:
        v = view(MODE_FOCUS, (2,))
        for logical, physical in zip(v.visible_blocks, v.physical_block_ids[: v.valid_counts]):
            self.assertNotEqual(logical, physical, "物理块号必须来自 canonical 映射")

    def test_rejects_unallocated_visible_block(self) -> None:
        with self.assertRaises(ReadViewError):
            build_read_view(
                ViewInputs(
                    mode=MODE_FOCUS,
                    refs=(3,),
                    layout=layout(),
                    attention_kv_len=PROMPT_LEN,
                    canonical_blocks=(5, 6, 7),  # 不足以覆盖逻辑块 6+
                    kernel_block_size=16,
                    max_width=3,
                )
            )

    def test_negative_physical_id_in_mapping_is_rejected(self) -> None:
        table = canonical(20)
        bad = (-2,) + table[1:]
        with self.assertRaises(ReadViewError):
            build_read_view(
                ViewInputs(MODE_GLOBAL, (), layout(), PROMPT_LEN, bad, 16, len(bad))
            )

    def test_valid_counts_and_width_ranges_are_checked(self) -> None:
        v = view(MODE_LOCAL)
        to_kernel_args(v, req_idx=0, block_table_rows=1, block_table_stride=v.max_width)  # 合法
        with self.assertRaises(ReadViewError):
            to_kernel_args(v, req_idx=0, block_table_rows=1, block_table_stride=v.max_width - 1)
        with self.assertRaises(ReadViewError):
            to_kernel_args(v, req_idx=1, block_table_rows=1, block_table_stride=v.max_width)

        class Tampered:
            max_width = 5
            valid_counts = 6
            physical_block_ids = (1, 2, 3, 4, 5)
            attention_kv_len = 100
            tail_block_valid_len = 4

        with self.assertRaises(ReadViewError):
            to_kernel_args(Tampered, req_idx=0, block_table_rows=1, block_table_stride=5)

    def test_to_kernel_args_bounds_checks(self) -> None:
        v = view(MODE_LOCAL)
        args = to_kernel_args(v, req_idx=0, block_table_rows=2, block_table_stride=v.max_width)
        self.assertEqual(args["valid_counts"], v.valid_counts)
        with self.assertRaises(ReadViewError):
            to_kernel_args(v, req_idx=2, block_table_rows=2, block_table_stride=v.max_width)
        with self.assertRaises(ReadViewError):
            to_kernel_args(v, req_idx=0, block_table_rows=2, block_table_stride=v.max_width - 1)

    def test_global_recovery_reads_everything_without_changing_mapping(self) -> None:
        table = canonical(PROMPT_LEN // 784 + 2)
        focus = build_read_view(
            ViewInputs(MODE_FOCUS, (1,), layout(), PROMPT_LEN, table, 784, len(table))
        )
        back = build_read_view(
            ViewInputs(MODE_GLOBAL, (), layout(), PROMPT_LEN, table, 784, len(table))
        )
        self.assertLess(focus.valid_counts, back.valid_counts)
        self.assertEqual(back.positions(), tuple(range(PROMPT_LEN)))
        self.assertEqual(
            back.physical_block_ids[: back.valid_counts],
            tuple(table[i] for i in back.visible_blocks),
            "canonical 映射不得被读取视图改写（I3）",
        )

    def test_pure_function_same_input_same_output(self) -> None:
        a = view(MODE_FOCUS, (1, 3), attention_kv_len=PROMPT_LEN + 11)
        b = view(MODE_FOCUS, (1, 3), attention_kv_len=PROMPT_LEN + 11)
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_local_mode_hides_every_segment(self) -> None:
        v = view(MODE_LOCAL, attention_kv_len=PROMPT_LEN + 3)
        positions = set(v.positions())
        for seg_start, seg_end in SEGMENTS:
            self.assertEqual(
                positions & set(range(seg_start, seg_end)),
                set(),
                f"local 模式不得看见 segment 区间 [{seg_start},{seg_end})",
            )
        self.assertTrue(set(range(0, 16)) <= positions, "sink 恒可见（I7）")
        self.assertTrue(set(range(*LOCAL_WINDOW)) <= positions, "本地窗口恒可见（I7）")
        self.assertTrue(
            {PROMPT_LEN, PROMPT_LEN + 1, PROMPT_LEN + 2} <= positions,
            "response（含本步写入的 token）恒可见（I7）",
        )

    def test_focus_multi_segment_visibility(self) -> None:
        v = view(MODE_FOCUS, (1, 3))
        positions = set(v.positions())
        for seg_start, seg_end in (SEGMENTS[0], SEGMENTS[2]):
            self.assertTrue(set(range(seg_start, seg_end)) <= positions)
        self.assertEqual(positions & set(range(*SEGMENTS[1])), set())


if __name__ == "__main__":
    unittest.main()
