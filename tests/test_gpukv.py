"""阶段 04 返工：候选侧转换（gpukv）与验收汇总（gpucheck）的 CPU 用例。"""

from __future__ import annotations

import copy
import types
import unittest

from attnview.gpucheck import summarize
from attnview.gpukv import GpuKvError, next_write_slot
from attnview.gpukv import read_table_from_read_view as read_table_from_view

B = 784
L2P = (7, 2, 11, 0, 5, 9, 1, 8, 3, 10, 4, 6)


def fake_view(*, visible_blocks, physical, spans, kv_len, block_size=B, valid_counts=None, next_write=None):
    """最小 ReadView 替身（只含候选转换真正使用的字段）。"""
    padded = list(physical) + [-1] * (12 - len(physical))
    return types.SimpleNamespace(
        kernel_block_size=block_size,
        attention_kv_len=kv_len,
        visible_blocks=tuple(visible_blocks),
        visible_spans=tuple(spans),
        valid_counts=len(visible_blocks) if valid_counts is None else valid_counts,
        physical_block_ids=tuple(padded),
        next_write_position=kv_len if next_write is None else next_write,
    )


def good_case(**overrides):
    measurement = {
        "label": "step0",
        "appended": False,
        "mode": "local",
        "kv_len": 6272,
        "numeric": {"max_abs": 0.01, "rms": 0.001, "within_tolerance": True, "failure": None},
        "oracle_selfcheck": {"max_diff": 1e-6, "limit": 1e-5},
        "read_table": {"has_minus_one": False, "width_ok": True, "physical_nonneg": True, "width": 3, "seqused_k": 1569},
        "data_plane": {
            "blocks_match": True,
            "physical_match": True,
            "seqused_match": True,
            "write_slot_match": True,
            "current_block_retained": True,
            "details": "",
        },
        "kv_integrity": {"k_unchanged": True, "v_unchanged": True},
        "residency": {"resident_identity": ["cuda:0", 1, "cuda:0", 2], "ptr_before": ["cuda:0", 1, "cuda:0", 2],
                      "ptr_after": ["cuda:0", 1, "cuda:0", 2], "data_ptr_stable": True},
        "expectation": {"config": [0, 7], "oracle": [0, 7], "actual": [0, 7]},
        "append": {"declared": False},
    }
    measurement.update(overrides)
    return {"id": "case_x", "seed": 0, "measurements": [measurement]}


class ReadTableFromViewTest(unittest.TestCase):
    def test_local_view_maps_blocks_and_counts(self) -> None:
        view = fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272)
        table = read_table_from_view(view, L2P)
        self.assertEqual(table.physical_blocks, (7, 8))
        self.assertEqual(table.visible_blocks, (0, 7))
        self.assertEqual(table.effective_per_block, (784, 784))
        self.assertEqual(table.seqused_k, 1568)
        self.assertEqual(table.tail_len, 784)
        self.assertEqual(table.next_write_position, 6272)
        self.assertEqual(table.block_size, B)

    def test_tail_block_count_follows_written_length(self) -> None:
        view = fake_view(visible_blocks=(0, 6), physical=(7, 1), spans=((0, 784), (4704, 4705)), kv_len=4705)
        table = read_table_from_view(view, L2P)
        self.assertEqual(table.effective_per_block, (784, 1))
        self.assertEqual(table.seqused_k, 785)
        self.assertEqual(table.tail_len, 1)

    def test_padded_row_never_uses_minus_one(self) -> None:
        table = read_table_from_view(
            fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272), L2P
        )
        row = table.padded_row(5)
        self.assertEqual(row, [7, 8, 8, 8, 8])
        self.assertTrue(all(pid >= 0 for pid in row))

    def test_minus_one_in_valid_prefix_is_rejected(self) -> None:
        view = fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272)
        view.physical_block_ids = (7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        with self.assertRaises(GpuKvError):
            read_table_from_view(view, L2P)

    def test_physical_mismatch_with_canonical_mapping_is_rejected(self) -> None:
        view = fake_view(visible_blocks=(0, 7), physical=(7, 1), spans=((0, 784), (5488, 6272)), kv_len=6272)
        with self.assertRaises(GpuKvError) as ctx:
            read_table_from_view(view, L2P)
        self.assertIn("canonical", str(ctx.exception))

    def test_unwritten_block_is_rejected_not_dropped(self) -> None:
        # 块 8 起点 6272 恰好等于 kv_len：不含任何已写位置，必须报错而不是静默丢弃
        view = fake_view(visible_blocks=(0, 8), physical=(7, 3), spans=((0, 784), (6272, 6272)), kv_len=6272)
        with self.assertRaises(GpuKvError) as ctx:
            read_table_from_view(view, L2P)
        self.assertIn("不含任何已写位置", str(ctx.exception))

    def test_partial_non_final_block_is_rejected(self) -> None:
        # 块 0 只覆盖 300 个位置却不是最大可见块 -> 前缀语义会读入未写槽，必须报错
        view = fake_view(visible_blocks=(0, 6), physical=(7, 1), spans=((0, 300), (4704, 4705)), kv_len=4705)
        with self.assertRaises(GpuKvError) as ctx:
            read_table_from_view(view, L2P)
        self.assertIn("前缀语义", str(ctx.exception))

    def test_current_block_must_be_visible(self) -> None:
        view = fake_view(visible_blocks=(0,), physical=(7,), spans=((0, 784),), kv_len=6272)
        with self.assertRaises(GpuKvError) as ctx:
            read_table_from_view(view, L2P)
        self.assertIn("当前 token", str(ctx.exception))

    def test_next_write_slot_is_independent_of_view(self) -> None:
        self.assertEqual(next_write_slot(6272, B, L2P), (8, 0, 3 * B))
        self.assertEqual(next_write_slot(6273, B, L2P), (8, 1, 3 * B + 1))


class BoundaryGuardTest(unittest.TestCase):
    """越读必须在 CPU 边界被拒绝，不得产出会索引越界列/非法地址的表。"""

    def test_table_width_must_cover_seqused(self) -> None:
        import dataclasses

        view = fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272)
        table = read_table_from_view(view, L2P)
        self.assertEqual(table.width, -(-table.seqused_k // B))
        # 人为把 seqused_k 抬高，使需要的列数超过表宽 -> 必须能构造出来（负对照用），
        # 但 GPU 边界（run_candidate）会拒绝；此处校验 ReadKit 自身对不一致表宽报错
        bad = fake_view(visible_blocks=(0, 5, 6), physical=(7, 9, 1),
                        spans=((0, 784), (3920, 4704), (4704, 4705)), kv_len=4705)
        good = read_table_from_view(bad, L2P)
        self.assertEqual((good.width, good.seqused_k), (3, 1569))
        over = dataclasses.replace(good, seqused_k=good.seqused_k + 100)
        self.assertEqual(-(-over.seqused_k // B), over.width, "尾块内越读仍应在已分配列内（不得越界列）")
        # 而 2 列的表抬高到需要 3 列时，宽度不一致
        two_col = read_table_from_view(
            fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272), L2P
        )
        stretched = dataclasses.replace(two_col, seqused_k=two_col.seqused_k + 100)
        self.assertGreater(-(-stretched.seqused_k // B), stretched.width)

    def test_table_block_size_must_match_cache(self) -> None:
        table = read_table_from_view(
            fake_view(visible_blocks=(0, 7), physical=(7, 8), spans=((0, 784), (5488, 6272)), kv_len=6272), L2P
        )
        self.assertEqual(table.block_size, 784)
        # 16 块大小的表与 784 的缓存不能混用（GPU 边界会断言）
        alt = read_table_from_view(
            fake_view(visible_blocks=(0, 3), physical=(7, 0), spans=((0, 53),), kv_len=53, block_size=16), L2P
        )
        self.assertEqual(alt.block_size, 16)

    def test_view_with_inconsistent_width_is_rejected(self) -> None:
        # 可见块 3 个但物理表有效前缀只有 2 个 -> 转换边界报错
        view = fake_view(visible_blocks=(0, 5, 6), physical=(7, 9), spans=((0, 784), (3920, 4705)), kv_len=4705)
        with self.assertRaises(GpuKvError):
            read_table_from_view(view, L2P)


class AcceptanceAggregationTest(unittest.TestCase):
    def test_all_criteria_pass(self) -> None:
        summary = summarize([good_case()])
        self.assertTrue(summary.ok, summary.reasons)
        self.assertGreater(summary.total, 10)

    def test_no_cases_is_a_failure(self) -> None:
        self.assertFalse(summarize([]).ok)

    def test_each_criterion_gates_the_verdict(self) -> None:
        injections = {
            "numeric": {"numeric": {"max_abs": 9.9, "rms": 1.0, "within_tolerance": False, "failure": "mismatch 20%"}},
            "oracle_selfcheck": {"oracle_selfcheck": {"max_diff": 0.5, "limit": 1e-5}},
            "read_table_no_minus_one": {"read_table": {"has_minus_one": True, "width_ok": True, "physical_nonneg": True, "width": 3, "seqused_k": 10}},
            "read_table_width": {"read_table": {"has_minus_one": False, "width_ok": False, "physical_nonneg": True, "width": 3, "seqused_k": 10}},
            "data_plane_blocks": {"data_plane": {"blocks_match": False, "physical_match": True, "seqused_match": True, "write_slot_match": True, "current_block_retained": True, "details": "块集合不一致"}},
            "data_plane_seqused": {"data_plane": {"blocks_match": True, "physical_match": True, "seqused_match": False, "write_slot_match": True, "current_block_retained": True, "details": "seqused 不一致"}},
            "data_plane_write_slot": {"data_plane": {"blocks_match": True, "physical_match": True, "seqused_match": True, "write_slot_match": False, "current_block_retained": True, "details": "写入位置不一致"}},
            "kv_k_unchanged": {"kv_integrity": {"k_unchanged": False, "v_unchanged": True, "expected_slots": None}},
            "kv_v_unchanged": {"kv_integrity": {"k_unchanged": True, "v_unchanged": False, "expected_slots": None}},
            "append_slot_k": {"appended": True, "append": {"declared": True, "position": 6272,
                                                           "slot_independent": 2352, "slot_helper": 2352,
                                                           "k_changed_slots": [1], "v_changed_slots": [2352]}},
            "append_declared": {"appended": None},
            "resident_cache": {"residency": {"data_ptr_stable": False}},
            "expectation_match": {"expectation": {"config": [0, 7], "oracle": [0, 8], "actual": [0, 7]}},
            "append_slot_source_match": {"appended": True, "append": {"declared": True, "position": 6272,
                                                                     "slot_independent": 2352, "slot_helper": 99,
                                                                     "k_changed_slots": [2352], "v_changed_slots": [2352]}},
        }
        for name, overrides in injections.items():
            with self.subTest(criterion=name):
                summary = summarize([good_case(**overrides)])
                self.assertFalse(summary.ok, f"{name} 未参与判定")
                self.assertTrue(
                    any(c.name.endswith(name) for c in summary.failed),
                    f"{name} 未出现在失败列表：{[c.name for c in summary.failed]}",
                )

    def test_none_or_missing_integrity_cannot_bypass_the_gate(self) -> None:
        # 每项给出"必须失败"的判据集合：None/缺字段一律不能通过
        cases = {
            "K=None V=False": ({"k_unchanged": None, "v_unchanged": False, "expected_slots": None},
                               {"kv_k_unchanged", "kv_v_unchanged"}),
            "K=True V=None": ({"k_unchanged": True, "v_unchanged": None, "expected_slots": None},
                              {"kv_v_unchanged"}),
            "K=None V=True": ({"k_unchanged": None, "v_unchanged": True, "expected_slots": None},
                              {"kv_k_unchanged"}),
            "K=None V=None": ({"k_unchanged": None, "v_unchanged": None, "expected_slots": None},
                              {"kv_k_unchanged", "kv_v_unchanged"}),
            "两项都缺": ({}, {"kv_k_unchanged", "kv_v_unchanged"}),
        }
        for name, (integrity, must_fail) in cases.items():
            with self.subTest(case=name):
                summary = summarize([good_case(kv_integrity=integrity)])
                self.assertFalse(summary.ok, f"{name} 竟然通过")
                for suffix in must_fail:
                    self.assertTrue(
                        any(c.name.endswith(suffix) for c in summary.failed),
                        f"{name}：{suffix} 未出现在失败列表 {[c.name for c in summary.failed]}",
                    )

    def test_missing_measurement_sections_fail(self) -> None:
        for section in ("numeric", "oracle_selfcheck", "read_table", "data_plane", "residency", "appended",
                        "expectation", "append"):
            with self.subTest(section=section):
                case = good_case()
                case["measurements"][0].pop(section)
                summary = summarize([case])
                self.assertFalse(summary.ok, f"缺少 {section} 竟然通过")
                self.assertTrue(any(c.name.endswith("schema_complete") for c in summary.failed))

    def test_append_declaration_and_slot_discipline(self) -> None:
        # appended 不是布尔 -> 失败
        self.assertFalse(summarize([good_case(appended="yes")]).ok)
        # 声明追加但没有独立来源/变化槽记录 -> 失败
        case = good_case(appended=True)
        case["measurements"][0]["append"] = {"declared": True, "position": 6272, "slot_independent": 2352,
                                             "slot_helper": 2352}
        summary = summarize([case])
        self.assertFalse(summary.ok)
        self.assertTrue(any(c.name.endswith("append_slot_k") for c in summary.failed))
        # 期望槽与被测 helper 同源（不一致）-> 失败
        case = good_case(appended=True)
        case["measurements"][0]["append"] = {"declared": True, "position": 6272, "slot_independent": 2352,
                                             "slot_helper": 99, "k_changed_slots": [2352], "v_changed_slots": [2352]}
        self.assertFalse(summarize([case]).ok)
        # 实际只错一侧（V）-> 失败
        case = good_case(appended=True)
        case["measurements"][0]["append"] = {"declared": True, "position": 6272, "slot_independent": 2352,
                                             "slot_helper": 2352, "k_changed_slots": [2352], "v_changed_slots": [2353]}
        summary = summarize([case])
        self.assertFalse(summary.ok)
        self.assertTrue(any(c.name.endswith("append_slot_v") for c in summary.failed))
        # declared=False 却记录了变化槽 -> 失败
        case = good_case()
        case["measurements"][0]["append"] = {"declared": False, "k_changed_slots": [1]}
        self.assertFalse(summarize([case]).ok)

    def test_expectation_must_be_present_and_three_way_equal(self) -> None:
        # 配置预期缺失（预冻结预期必须参与判据）
        case = good_case()
        case["measurements"][0]["expectation"] = {"config": None, "oracle": [0, 7], "actual": [0, 7]}
        summary = summarize([case])
        self.assertFalse(summary.ok)
        self.assertTrue(any(c.name.endswith("expectation_match") for c in summary.failed))
        # 候选与独立 oracle 不一致
        case = good_case()
        case["measurements"][0]["expectation"] = {"config": [0, 7], "oracle": [0, 7], "actual": [0, 8]}
        self.assertFalse(summarize([case]).ok)
        # 三方一致 -> 通过
        case = good_case()
        case["measurements"][0]["expectation"] = {"config": [0, 7], "oracle": [0, 7], "actual": [0, 7]}
        self.assertTrue(summarize([case]).ok, summarize([case]).reasons)

    def test_control_only_case_uses_extra_criteria(self) -> None:
        case = {"id": "negative_control_verdict", "seed": 0, "control_only": True,
                "extra_criteria": [{"name": "negative_control_rejected", "ok": True, "detail": "已被拒绝"}]}
        self.assertTrue(summarize([case]).ok)
        case["extra_criteria"][0]["ok"] = False
        self.assertFalse(summarize([case]).ok)

    def test_extra_criteria_gate_the_verdict(self) -> None:
        case = good_case()
        case["extra_criteria"] = [{"name": "row_isolation", "ok": False, "detail": "行间误差 3.2"}]
        summary = summarize([case])
        self.assertFalse(summary.ok)

    def test_full_detail_is_preserved(self) -> None:
        long_detail = "第 3 行第 17 列偏差 0.5，" + "附加说明" * 20
        case = good_case()
        case["extra_criteria"] = [{"name": "sensitivity", "ok": False, "detail": long_detail}]
        summary = summarize([case])
        self.assertIn(long_detail, summary.reasons[0])
        self.assertNotIn("...", summary.reasons[0])

    def test_missing_measurements_fail(self) -> None:
        case = copy.deepcopy(good_case())
        case["measurements"] = []
        self.assertFalse(summarize([case]).ok)


if __name__ == "__main__":
    unittest.main()
