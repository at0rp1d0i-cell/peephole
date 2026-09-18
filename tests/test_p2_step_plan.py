"""阶段 05 检查点 1（纠偏后）CPU 验收：每步读取视图计划 `attnview.step_plan`。

覆盖 SUP-004-R1 §2 的风险 1–8（逐条对应见各测试类 docstring）：

1. `seqused_k` 是**压缩后的可见长度**（不等于 canonical 总长、也不等于可见块数×块大小）；
2. 当前 token 必须可见（不可见必须报错，不允许静默少读）；
3. 末尾单 token prefill 走原版，判据是 `is_prefilling` 而非 `query_len==1`；
4. 请求终结/abort：无计划不落位、注册表释放后取不到状态、finish 后不再喂 token；
5. 未支持的运行配置显式拒绝（`UnsupportedConfig`），不做静默降级；
6. 几何不由载荷提供：载荷出现几何字段/版本不符/长度与 span 非法一律报错；
7. `canonical_blocks` 含 `None` 直接报错（不过滤），`StepPlan.validate()` 拦下表宽/长度不一致；
8. `trace_from_plan` 字段与 plan 一致，且 `as_dict()` 只含原生 JSON 类型。

几何夹具为**真实运行期配置**：`kernel_block_size=784`、4 个 KV 组（FA 组 index=3）、
`blocks_per_kv_block=1`、`max_model_len=8192` → `max_width=11`；块大小不是 16。

边界：本文件只覆盖纯逻辑（CPU）。FA kernel 是否按 `seqused_k`/因果序数正确取数、落位到
FA 组 metadata 时是否污染共享 buffer 等，属于 GPU 侧接入验证，不在本文件范围内。
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview.gpukv import GpuKvError  # noqa: E402
from attnview.parser import MODE_GLOBAL, MODE_LOCAL  # noqa: E402
from attnview.readview import ReadViewError  # noqa: E402
from attnview.state import ProtocolRegistry, RequestProtocolState  # noqa: E402
from attnview.step_plan import (  # noqa: E402
    EXTRA_ARG_KEY,
    FORBIDDEN_PAYLOAD_KEYS,
    PROTOCOL_VERSION,
    AttnViewConfigError,
    DaRequestConfig,
    Geometry,
    StepPlan,
    UnsupportedConfig,
    build_step_plan,
    check_supported_config,
    resolve_canonical_mapping,
    should_apply_view,
    trace_from_plan,
)

KERNEL_BLOCK = 784
NUM_KV_GROUPS = 4
FA_GROUP_INDEX = 3
MAX_MODEL_LEN = 8192
PROMPT_LEN = 8 * KERNEL_BLOCK  # 6272：8 个 kernel 块
DECODE_KV_LEN = PROMPT_LEN + 1  # 6273：含本次 forward 写入的当前 token
LOCAL_WINDOW = (5 * KERNEL_BLOCK, PROMPT_LEN)  # 3920..6272（question 起点 → prompt 末尾）
SINK_SPAN = (0, KERNEL_BLOCK)
SEGMENTS = ((0, 100), (200, 300), (400, 500))
#: 真实夹具下 local 步的期望读取表：可见块（对齐后排除已写的中间块）× 有效位置数
EXPECTED_VISIBLE = (0, 5, 6, 7, 8)
EXPECTED_EFFECTIVE = (KERNEL_BLOCK, KERNEL_BLOCK, KERNEL_BLOCK, KERNEL_BLOCK, 1)
EXPECTED_SEQUSED_K = 4 * KERNEL_BLOCK + 1  # 3137
EXPECTED_MAX_WIDTH = 11


def runtime_geometry(**overrides) -> Geometry:
    kwargs = {
        "kernel_block_size": KERNEL_BLOCK,
        "num_kv_groups": NUM_KV_GROUPS,
        "fa_group_index": FA_GROUP_INDEX,
        "blocks_per_kv_block": 1,
        "max_model_len": MAX_MODEL_LEN,
    }
    kwargs.update(overrides)
    return Geometry(**kwargs)


def canonical_blocks(count: int = 12) -> tuple[int, ...]:
    """确定性、互不重复且非恒等的逻辑→物理映射（固定置换，便于发现索引错位）。"""
    ids = list(range(500, 500 + count))
    return tuple(ids[::2] + ids[1::2][::-1])


def base_payload(**overrides) -> dict:
    raw = {
        "protocol": PROTOCOL_VERSION,
        "prompt_len": PROMPT_LEN,
        "segment_spans": [list(s) for s in SEGMENTS],
        "local_window_span": [LOCAL_WINDOW[0], LOCAL_WINDOW[1]],
        "sink_span": [SINK_SPAN[0], SINK_SPAN[1]],
    }
    raw.update(overrides)
    return raw


def da_config(**overrides) -> DaRequestConfig:
    return DaRequestConfig.from_extra_args({EXTRA_ARG_KEY: base_payload(**overrides)})


def build_plan(
    mode: str = MODE_LOCAL,
    *,
    config: DaRequestConfig | None = None,
    attention_kv_len: int = DECODE_KV_LEN,
    effect_step: int = 1,
    geom: Geometry | None = None,
    blocks: object = None,
    refs: tuple[int, ...] = (),
    req_id: str = "req-0",
) -> StepPlan:
    return build_step_plan(
        req_id=req_id,
        mode=mode,
        refs=refs,
        config=config if config is not None else da_config(),
        geometry=geom if geom is not None else runtime_geometry(),
        canonical_blocks=canonical_blocks() if blocks is None else blocks,
        attention_kv_len=attention_kv_len,
        effect_step=effect_step,
    )


class SeqUsedKCompressionTest(unittest.TestCase):
    """风险 1：`seqused_k` 是压缩后的可见长度，由可见块的实际覆盖位置数决定。"""

    def test_local_decode_step_counts_only_visible_positions(self) -> None:
        plan = build_plan(MODE_LOCAL)
        self.assertEqual(plan.mode, MODE_LOCAL)
        self.assertEqual(plan.visible_logical_blocks, EXPECTED_VISIBLE)
        self.assertEqual(plan.effective_per_block, EXPECTED_EFFECTIVE)
        self.assertEqual(plan.tail_len, 1)  # 最大可见块取实际尾长
        self.assertEqual(plan.seqused_k, EXPECTED_SEQUSED_K)
        self.assertEqual(plan.seqused_k, sum(plan.effective_per_block))
        # attention_kv_len 是本次 attention 发生时的 canonical 有效长度（含当前 token）
        self.assertEqual(plan.attention_kv_len, DECODE_KV_LEN)
        self.assertEqual(plan.next_write_position, DECODE_KV_LEN)
        self.assertEqual(plan.written_before_step, PROMPT_LEN)
        # 表宽是运行期几何常量，needed_width 只覆盖可见前缀
        self.assertEqual(plan.block_size, KERNEL_BLOCK)
        self.assertEqual(plan.width, EXPECTED_MAX_WIDTH)
        self.assertEqual(plan.width, runtime_geometry().max_width)
        self.assertEqual(plan.needed_width, len(plan.visible_logical_blocks))
        # 压缩：既不等于 canonical 总长，也不等于可见块数 × 块大小
        self.assertNotEqual(plan.seqused_k, plan.attention_kv_len)
        self.assertNotEqual(plan.seqused_k, len(plan.visible_logical_blocks) * KERNEL_BLOCK)

    def test_excluded_blocks_are_written_but_deliberately_not_visible(self) -> None:
        """压缩来自"已写但被排除的中间块"，不是尾部未写槽。"""
        plan = build_plan(MODE_LOCAL)
        for block in (1, 2, 3, 4):
            self.assertLess(block * KERNEL_BLOCK, plan.attention_kv_len)  # 已写
            self.assertNotIn(block, plan.visible_logical_blocks)  # 但不读
        self.assertLess(plan.seqused_k, plan.attention_kv_len)

    def test_width_stays_constant_across_steps_while_tail_grows(self) -> None:
        first = build_plan(MODE_LOCAL, attention_kv_len=DECODE_KV_LEN, effect_step=1)
        second = build_plan(MODE_LOCAL, attention_kv_len=DECODE_KV_LEN + 1, effect_step=2)
        self.assertEqual(second.visible_logical_blocks, first.visible_logical_blocks)
        self.assertEqual(second.seqused_k, first.seqused_k + 1)
        self.assertEqual(second.tail_len, first.tail_len + 1)
        self.assertEqual(second.width, first.width)  # I5：步间常量
        self.assertEqual(second.needed_width, first.needed_width)

    def test_global_mode_reads_full_canonical_length(self) -> None:
        plan = build_plan(MODE_GLOBAL)
        self.assertEqual(plan.mode, MODE_GLOBAL)
        self.assertEqual(plan.seqused_k, plan.attention_kv_len)
        self.assertEqual(plan.seqused_k, DECODE_KV_LEN)
        self.assertEqual(plan.needed_width, len(plan.visible_logical_blocks))
        self.assertGreaterEqual(plan.width, plan.needed_width)


class CurrentTokenVisibilityTest(unittest.TestCase):
    """风险 2：当前 token 所在块必须可见；不可见必须报错而不是静默通过。"""

    def test_local_window_hiding_current_token_is_rejected(self) -> None:
        # kv_len 停在 prompt 末尾：当前 token 位置 6271 在块 7，而 sink/local 只覆盖块 0–1
        hidden = da_config(
            local_window_span=[KERNEL_BLOCK, 2 * KERNEL_BLOCK], sink_span=[0, 16]
        )
        with self.assertRaises(GpuKvError):
            build_plan(MODE_LOCAL, config=hidden, attention_kv_len=PROMPT_LEN)
        # 对照组：同一 kv_len，窗口覆盖末尾时当前 token 可见，必须能构造出计划
        control = build_plan(MODE_LOCAL, attention_kv_len=PROMPT_LEN)
        self.assertEqual(
            control.visible_logical_blocks[-1], (PROMPT_LEN - 1) // KERNEL_BLOCK
        )

    def test_unallocated_current_block_is_rejected(self) -> None:
        # 只分配到逻辑块 0..7：当前 token 落在未分配的逻辑块 8
        with self.assertRaises(ReadViewError):
            build_plan(MODE_LOCAL, blocks=canonical_blocks(8))


class PrefillGateTest(unittest.TestCase):
    """风险 3：末尾单 token prefill 走原版；判据是 `is_prefilling`，不是 `query_len==1`。"""

    CHUNKS = (PROMPT_LEN - 1, 1)  # 最后一个 prefill chunk 只调度 1 个 token
    TOKEN_TEXT = "<local>"  # 采样出的 g0：第 1 步 decode 起切到 local 模式

    def setUp(self) -> None:
        self.state = RequestProtocolState(
            "req-prefill", arm="da", num_segments=3, prompt_len=PROMPT_LEN
        )
        self.record = self.state.feed_generated_token(0, 101, self.TOKEN_TEXT)
        self.mode, self.refs = self.state.view_for_next_step()

    def test_single_token_final_prefill_chunk_stays_on_baseline(self) -> None:
        self.assertEqual(sum(self.CHUNKS), PROMPT_LEN)
        self.assertEqual(self.CHUNKS[-1], 1)  # 陷阱：这一步的 query_len 也是 1
        self.assertFalse(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=True,
                has_plan=True,
                request_uses_attnview=True,
            )
        )

    def test_decode_step_consuming_sampled_token_applies_view(self) -> None:
        self.assertEqual(self.record.written_kv_len, PROMPT_LEN)  # g0 尚未写入
        self.assertEqual(self.record.mode_after, MODE_LOCAL)
        self.assertEqual(self.mode, MODE_LOCAL)
        self.assertEqual(self.state.attention_kv_len_next, DECODE_KV_LEN)
        plan = build_plan(
            self.mode,
            attention_kv_len=self.state.attention_kv_len_next,
            effect_step=self.record.effect_step,
            refs=self.refs,
            req_id=self.state.request_id,
        )
        self.assertEqual(plan.req_id, "req-prefill")
        self.assertEqual(plan.effect_step, 1)  # t 采样、t+1 生效
        self.assertEqual(plan.attention_kv_len, DECODE_KV_LEN)
        self.assertEqual(plan.written_before_step, self.record.written_kv_len)
        self.assertEqual(plan.seqused_k, EXPECTED_SEQUSED_K)
        self.assertTrue(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=False,
                has_plan=True,
                request_uses_attnview=True,
            )
        )

    def test_same_query_len_different_prefill_flag_flips_the_gate(self) -> None:
        self.assertEqual(self.CHUNKS[-1], 1)
        prefill = should_apply_view(
            is_dummy_run=False,
            is_prefilling=True,
            has_plan=True,
            request_uses_attnview=True,
        )
        decode = should_apply_view(
            is_dummy_run=False,
            is_prefilling=False,
            has_plan=True,
            request_uses_attnview=True,
        )
        self.assertNotEqual(prefill, decode)


class LifecycleGateTest(unittest.TestCase):
    """风险 4：无计划/dummy/普通请求不落位；abort 释放后取不到状态；finish 后不再喂 token。"""

    def test_gate_refuses_without_plan_dummy_run_or_plain_request(self) -> None:
        self.assertFalse(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=False,
                has_plan=False,
                request_uses_attnview=True,
            )
        )
        self.assertFalse(
            should_apply_view(
                is_dummy_run=True,
                is_prefilling=False,
                has_plan=True,
                request_uses_attnview=True,
            )
        )
        self.assertFalse(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=False,
                has_plan=True,
                request_uses_attnview=False,
            )
        )
        self.assertTrue(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=False,
                has_plan=True,
                request_uses_attnview=True,
            )
        )

    def test_abort_releases_state_and_step_has_no_plan(self) -> None:
        registry = ProtocolRegistry()
        state = registry.create(
            "req-abort", arm="da", num_segments=3, prompt_len=PROMPT_LEN
        )
        self.assertEqual(registry.active_ids(), ("req-abort",))
        self.assertIs(registry.release("req-abort"), state)
        with self.assertRaises(KeyError):
            registry.get("req-abort")
        with self.assertRaises(KeyError):
            registry.release("req-abort")
        self.assertEqual(registry.active_ids(), ())
        # abort 之后没有计划可落位
        self.assertFalse(
            should_apply_view(
                is_dummy_run=False,
                is_prefilling=False,
                has_plan=False,
                request_uses_attnview=True,
            )
        )

    def test_finished_request_refuses_further_tokens(self) -> None:
        registry = ProtocolRegistry()
        state = registry.create(
            "req-finish", arm="da", num_segments=3, prompt_len=PROMPT_LEN
        )
        state.feed_generated_token(0, 101, "<local>")
        state.finish()
        self.assertTrue(state.closed)
        with self.assertRaises(RuntimeError):
            state.feed_generated_token(1, 102, "x")


class UnsupportedConfigTest(unittest.TestCase):
    """风险 5：未支持的运行配置显式拒绝（`UnsupportedConfig`），不静默降级。"""

    LEGAL = {
        "async_scheduling": False,
        "max_concurrent_batches": 1,
        "cudagraph_mode": "CUDAGraphMode.NONE",
        "enable_prefix_caching": False,
        "speculative_config": None,
    }

    def test_supported_configuration_is_accepted(self) -> None:
        self.assertIsNone(check_supported_config(**self.LEGAL))
        geom = runtime_geometry().validate()
        self.assertEqual(geom.max_width, EXPECTED_MAX_WIDTH)
        self.assertEqual(build_plan(MODE_LOCAL, geom=geom).width, geom.max_width)

    def test_unsupported_runtime_flags_are_rejected(self) -> None:
        cases = {
            "async_scheduling=True": dict(self.LEGAL, async_scheduling=True),
            "max_concurrent_batches=0": dict(self.LEGAL, max_concurrent_batches=0),
            "max_concurrent_batches=2": dict(self.LEGAL, max_concurrent_batches=2),
            "cudagraph_mode 非 NONE": dict(self.LEGAL, cudagraph_mode="CUDAGraphMode.FULL"),
            "enable_prefix_caching=True": dict(self.LEGAL, enable_prefix_caching=True),
            "speculative_config 非 None": dict(
                self.LEGAL, speculative_config={"method": "ngram"}
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(UnsupportedConfig):
                    check_supported_config(**kwargs)

    def test_blocks_per_kv_block_mismatch_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedConfig):
            runtime_geometry(blocks_per_kv_block=2).validate()
        with self.assertRaises(UnsupportedConfig):
            build_plan(MODE_LOCAL, geom=runtime_geometry(blocks_per_kv_block=2))

    def test_invalid_geometry_values_are_rejected(self) -> None:
        cases = {
            "kernel_block_size=0": dict(kernel_block_size=0),
            "num_kv_groups=0": dict(num_kv_groups=0),
            "fa_group_index 越界": dict(fa_group_index=NUM_KV_GROUPS),
            "max_model_len=0": dict(max_model_len=0),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(AttnViewConfigError):
                    runtime_geometry(**overrides).validate()
        # build_step_plan 必须自己校验运行期几何（而不是只信调用方）
        with self.assertRaises(AttnViewConfigError):
            build_plan(MODE_LOCAL, geom=runtime_geometry(fa_group_index=NUM_KV_GROUPS))


class PayloadGeometryInjectionTest(unittest.TestCase):
    """风险 6：几何/协议版本/prompt_len/span 一律由载荷校验拦截。"""

    def test_forbidden_geometry_keys_in_payload_are_rejected(self) -> None:
        keys = {"kernel_block_size", "max_width", "num_blocks"} | FORBIDDEN_PAYLOAD_KEYS
        for key in sorted(keys):
            with self.subTest(key=key):
                with self.assertRaises(AttnViewConfigError):
                    da_config(**{key: KERNEL_BLOCK})

    def test_protocol_version_must_match_exactly(self) -> None:
        self.assertEqual(da_config().protocol, PROTOCOL_VERSION)
        for bad in ("v0.9", "1.0", "V1.0", "", None):
            with self.subTest(protocol=bad):
                with self.assertRaises(AttnViewConfigError):
                    da_config(protocol=bad)
        raw = base_payload()
        raw.pop("protocol")
        with self.assertRaises(AttnViewConfigError):
            DaRequestConfig.from_extra_args({EXTRA_ARG_KEY: raw})

    def test_prompt_len_must_be_a_positive_int(self) -> None:
        self.assertEqual(da_config().prompt_len, PROMPT_LEN)
        for bad in (None, 0, -1, True, "6272", 6272.0):
            with self.subTest(prompt_len=bad):
                with self.assertRaises(AttnViewConfigError):
                    da_config(prompt_len=bad)
        raw = base_payload()
        raw.pop("prompt_len")
        with self.assertRaises(AttnViewConfigError):
            DaRequestConfig.from_extra_args({EXTRA_ARG_KEY: raw})

    def test_spans_must_stay_inside_the_prompt(self) -> None:
        beyond = PROMPT_LEN + 1
        cases = {
            "segment_spans": [[PROMPT_LEN, beyond]],
            "local_window_span": [0, beyond],
            "sink_span": [0, beyond],
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(AttnViewConfigError):
                    da_config(**{field: value})

    def test_accepted_payload_maps_fields_and_layout(self) -> None:
        cfg = da_config(enforce_global=True)
        self.assertEqual(cfg.prompt_len, PROMPT_LEN)
        self.assertEqual(cfg.segment_spans, SEGMENTS)
        self.assertEqual(cfg.local_window_span, LOCAL_WINDOW)
        self.assertEqual(cfg.sink_span, SINK_SPAN)
        self.assertTrue(cfg.enforce_global)
        layout = cfg.layout()
        self.assertEqual(layout.prompt_len, PROMPT_LEN)
        self.assertEqual(layout.segment_spans, SEGMENTS)
        self.assertEqual(layout.local_window_span, LOCAL_WINDOW)
        self.assertEqual(layout.sink_span, SINK_SPAN)

    def test_malformed_spans_and_extra_args_are_rejected(self) -> None:
        for field, value in (
            ("segment_spans", [[5, 3]]),
            ("segment_spans", [[-1, 10]]),
            ("segment_spans", [[5]]),
            ("segment_spans", "abc"),
            ("local_window_span", [10]),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(AttnViewConfigError):
                    da_config(**{field: value})
        raw = base_payload()
        raw.pop("segment_spans")
        for extra_args in (
            None,
            {},
            "attnview",
            {EXTRA_ARG_KEY: 5},
            {"other": {}},
            {EXTRA_ARG_KEY: raw},
        ):
            with self.subTest(extra_args=extra_args):
                with self.assertRaises(AttnViewConfigError):
                    DaRequestConfig.from_extra_args(extra_args)

    def test_missing_local_window_falls_back_for_global_but_not_local(self) -> None:
        raw = base_payload()
        raw.pop("local_window_span")
        raw.pop("sink_span")
        cfg = DaRequestConfig.from_extra_args({EXTRA_ARG_KEY: raw})
        self.assertEqual(cfg.layout().local_window_span, (0, 0))
        with self.assertRaises(AttnViewConfigError):
            build_plan(MODE_LOCAL, config=cfg)
        self.assertEqual(build_plan(MODE_GLOBAL, config=cfg).seqused_k, DECODE_KV_LEN)


class MappingAndTableWidthTest(unittest.TestCase):
    """风险 7：`None` 槽不得被过滤；`validate()` 拦下表宽/长度不一致。"""

    def test_unallocated_slot_is_rejected_not_filtered(self) -> None:
        # 过滤成 (10, 12) 会让逻辑块 2 变成"第 1 块"，静默错位
        with self.assertRaises(AttnViewConfigError):
            resolve_canonical_mapping([10, None, 12])
        blocks = list(canonical_blocks())
        blocks[3] = None
        with self.assertRaises(AttnViewConfigError):
            build_plan(MODE_LOCAL, blocks=blocks)

    def test_missing_empty_or_invalid_mapping_is_rejected(self) -> None:
        for bad in (None, (), [], [10, 10], [-3, 4]):
            with self.subTest(mapping=bad):
                with self.assertRaises(AttnViewConfigError):
                    resolve_canonical_mapping(bad)

    def test_step_plan_validate_rejects_inconsistent_tables(self) -> None:
        base = build_plan(MODE_GLOBAL)  # 9 个可见块、width=11、seqused_k==attention_kv_len
        self.assertIs(base.validate(), base)
        self.assertEqual(base.seqused_k, DECODE_KV_LEN)
        self.assertEqual(base.needed_width, len(base.visible_logical_blocks))
        cases = {
            # ceil(7000/784)=9 == 可见块数、width 足够 → 只有"超过 canonical 长度"这一条会失败
            "seqused_k > attention_kv_len": replace(base, seqused_k=7000),
            "needed_width != len(visible)": replace(
                base, visible_logical_blocks=base.visible_logical_blocks[:-1]
            ),
            "width < needed_width": replace(base, width=4),
            "tail_len 与最大可见块不一致": replace(base, tail_len=KERNEL_BLOCK),
        }
        for name, plan in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(AttnViewConfigError):
                    plan.validate()


class TraceTest(unittest.TestCase):
    """风险 8：trace 字段与 plan 一致，且 `as_dict()` 只含原生 JSON 类型。"""

    def test_trace_fields_follow_the_plan(self) -> None:
        plan = build_plan(MODE_LOCAL)
        trace = trace_from_plan(plan, parse_step=11, applied_step=12)
        self.assertEqual(trace.req_id, plan.req_id)
        self.assertEqual(trace.parse_step, 11)
        self.assertEqual(trace.effect_step, plan.effect_step)
        self.assertEqual(trace.applied_step, 12)
        self.assertEqual(trace.mode, plan.mode)
        self.assertEqual(trace.refs, plan.refs)
        self.assertEqual(trace.visible_blocks, plan.visible_logical_blocks)
        self.assertEqual(trace.effective_per_block, plan.effective_per_block)
        self.assertEqual(trace.seqused_k, plan.seqused_k)
        self.assertEqual(trace.width, plan.width)
        self.assertEqual(trace.tail_len, plan.tail_len)
        self.assertEqual(trace.attention_kv_len, plan.attention_kv_len)
        self.assertEqual(trace.next_write_position, plan.next_write_position)
        self.assertEqual(trace.notes, ())

    def test_trace_dict_is_json_native(self) -> None:
        plan = build_plan(MODE_LOCAL)
        record = trace_from_plan(plan, parse_step=11, applied_step=None).as_dict()
        self.assertIsNone(record["applied_step"])
        self.assertIs(type(record["req_id"]), str)
        for key in (
            "parse_step",
            "effect_step",
            "seqused_k",
            "width",
            "tail_len",
            "attention_kv_len",
            "next_write_position",
        ):
            self.assertIs(type(record[key]), int)
        for key in ("refs", "visible_blocks", "effective_per_block"):
            self.assertIs(type(record[key]), list)
            self.assertTrue(all(type(v) is int for v in record[key]))
        self.assertIs(type(record["notes"]), list)
        self.assertTrue(all(type(v) is str for v in record["notes"]))
        self.assertEqual(record["visible_blocks"], list(plan.visible_logical_blocks))
        self.assertEqual(record["effective_per_block"], list(plan.effective_per_block))
        self.assertEqual(record["seqused_k"], plan.seqused_k)
        self.assertEqual(record["tail_len"], plan.tail_len)
        self.assertEqual(record["next_write_position"], plan.next_write_position)
        self.assertEqual(json.loads(json.dumps(record)), record)


class StepPayloadTest(unittest.TestCase):
    """下推载荷（风险 6 的出站侧 + 风险 8 的类型要求）：不带几何、只用原生类型。"""

    def test_payload_carries_no_geometry_and_is_json_native(self) -> None:
        plan = build_plan(MODE_LOCAL)
        payload = plan.as_payload()
        self.assertFalse(set(payload) & FORBIDDEN_PAYLOAD_KEYS)
        self.assertEqual(payload["req_id"], plan.req_id)
        self.assertEqual(payload["mode"], plan.mode)
        self.assertEqual(payload["effect_step"], plan.effect_step)
        self.assertEqual(payload["visible_logical_blocks"], list(plan.visible_logical_blocks))
        self.assertEqual(payload["effective_per_block"], list(plan.effective_per_block))
        self.assertEqual(payload["seqused_k"], plan.seqused_k)
        self.assertEqual(payload["tail_len"], plan.tail_len)
        self.assertEqual(payload["attention_kv_len"], plan.attention_kv_len)
        self.assertEqual(payload["next_write_position"], plan.next_write_position)
        self.assertEqual(payload["written_before_step"], plan.written_before_step)
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_invalid_plan_cannot_be_pushed_down(self) -> None:
        with self.assertRaises(AttnViewConfigError):
            replace(build_plan(MODE_GLOBAL), width=4).as_payload()


if __name__ == "__main__":
    unittest.main()
