"""请求状态与生命周期验收（工作单 §4「状态隔离」面 + C3.7 输入伪标签不触发）。

注意：这是 **CPU 对象级**测试，只证明协议状态按 request_id 隔离；不代表 vLLM 的混批/取消/抢占已支持。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview.parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL  # noqa: E402
from attnview.state import ProtocolRegistry  # noqa: E402


def make_state(registry: ProtocolRegistry, request_id: str, num_segments: int = 3):
    return registry.create(
        request_id, arm="da", num_segments=num_segments, prompt_len=100
    )


def feed(state, text: str) -> None:
    for ch in text:
        state.feed_generated_token(len(state.steps), 0, ch)


class StateIsolationTest(unittest.TestCase):
    def test_two_requests_interleaved_do_not_share_state(self) -> None:
        reg = ProtocolRegistry()
        a = make_state(reg, "req-a")
        b = make_state(reg, "req-b")
        feed(a, '<focus magic_chunks="1">')
        feed(b, "<local>")
        feed(a, "value</focus>")
        feed(b, "planning")
        self.assertEqual(a.mode, MODE_GLOBAL)
        self.assertEqual(b.mode, MODE_LOCAL)
        self.assertEqual(a.refs, ())
        feed(b, "</local>")
        self.assertEqual(b.mode, MODE_GLOBAL)
        self.assertEqual(a.mode, MODE_GLOBAL)
        self.assertEqual(reg.active_ids(), ("req-a", "req-b"))

    def test_release_and_reuse_leaves_no_residue(self) -> None:
        reg = ProtocolRegistry()
        a = make_state(reg, "req-1")
        feed(a, "<local>leftover")
        reg.release("req-1")
        self.assertEqual(len(reg), 0)
        b = make_state(reg, "req-1")
        self.assertEqual(b.mode, MODE_GLOBAL)
        self.assertEqual(b.refs, ())
        self.assertEqual(b.generated_tokens, 0)
        self.assertEqual(b.anomalies, ())
        with self.assertRaises(KeyError):
            reg.create("req-1", arm="da", num_segments=3, prompt_len=10)

    def test_generation_stream_without_tags_keeps_default_state(self) -> None:
        """对象级局部保证：控制状态只由生成流驱动；正文字符串不经此接口（真实接线属接入测试）。"""
        reg = ProtocolRegistry()
        state = make_state(reg, "req-x")
        feed(state, "The document mentions ")
        self.assertEqual(state.mode, MODE_GLOBAL)
        self.assertEqual(state.anomalies, ())
        self.assertEqual(state.focus_stats()["focus_attempts"], 0)

    def test_finish_flushes_incomplete_tag(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-y")
        feed(state, '<focus magic_chunks="2"')
        events = state.finish()
        self.assertEqual([e.reason for e in events], ["incomplete_tag"])
        self.assertEqual(state.mode, MODE_GLOBAL)

    def test_token_index_must_be_contiguous(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-z")
        with self.assertRaises(ValueError):
            state.feed_generated_token(3, 0, "x")

    def test_step_record_follows_the_timing_table(self) -> None:
        """采样出第 t 个 token 后：已写仍是 prompt_len+t；消费它的那步 forward 才有上界 prompt_len+t+1。"""
        reg = ProtocolRegistry()
        state = make_state(reg, "req-w")  # prompt_len = 100
        first = state.feed_generated_token(0, 0, "<")
        self.assertEqual(first.written_kv_len, 100, "prefill 采出 g0 时 KV 仍长 P")
        self.assertEqual(first.next_write_position, 100)
        self.assertEqual(first.attention_kv_len_next, 101)
        record = state.feed_generated_token(1, 0, "local>")
        self.assertEqual(record.effect_step, 2)
        self.assertEqual(record.sampled_tokens, 2)
        self.assertEqual(record.written_kv_len, 101)
        self.assertEqual(record.next_write_position, 101)
        self.assertEqual(record.attention_kv_len_next, 102)
        self.assertEqual(state.written_kv_len, 101)
        self.assertEqual(state.attention_kv_len_next, 102)
        self.assertEqual(state.mode, MODE_LOCAL)

    def test_incomplete_focus_buffer_counts_as_attempt(self) -> None:
        """结束时的未闭合 `<focus magic_chunks="` 必须进分母（此前被 flush 漏统计）。"""
        reg = ProtocolRegistry()
        state = make_state(reg, "req-i")
        feed(state, '<focus magic_chunks="')
        state.finish()
        stats = state.focus_stats()
        self.assertEqual(stats["focus_attempts"], 1)
        self.assertEqual(stats["focus_successes"], 0)
        self.assertEqual([e.reason for e in state.flush_events], ["incomplete_tag"])
        self.assertEqual([e.name for e in state.flush_events], ["focus"])

    def test_overflow_focus_attempt_reaches_state_stats(self) -> None:
        """R1#4 漏项：状态级统计里，超长缓冲的 focus 尝试必须计 1 次（跨 token 也只记一次）。"""
        reg = ProtocolRegistry()
        state = make_state(reg, "req-of")
        state.feed_generated_token(0, 0, '<focus magic_chunks="')
        state.feed_generated_token(1, 0, "1" * 300)
        state.finish()
        stats = state.focus_stats()
        self.assertEqual(stats["focus_attempts"], 1)
        self.assertEqual(stats["focus_successes"], 0)
        self.assertEqual(state.mode, MODE_GLOBAL)

    def test_mismatched_close_keeps_mode_and_is_not_an_attempt(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-m")
        feed(state, "<local>x</focus>")
        self.assertEqual(state.mode, MODE_LOCAL, "不匹配的闭标签不得改写模式")
        self.assertEqual([e.reason for e in state.anomalies], ["mismatched_close"])
        self.assertFalse(state.anomalies[0].is_focus_attempt, "闭标签不计入 focus 分母")
        self.assertEqual(state.focus_stats()["focus_attempts"], 0)

    def test_focus_stats_count_attempts_and_successes(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-s")
        feed(state, '<focus magic_chunks="1">ok</focus>')
        feed(state, '<focus magic_chunks="9">bad</focus>')
        feed(state, "<focus>missing</focus>")
        feed(state, '<focus magic_chunks="2,3">ok</focus>')
        stats = state.focus_stats()
        self.assertEqual(stats["focus_attempts"], 4, "两次合法 + 越界 + 缺属性")
        self.assertEqual(stats["focus_successes"], 2)
        reasons = sorted(e.reason for e in state.anomalies)
        self.assertEqual(
            reasons,
            ["invalid_reference", "mismatched_close", "mismatched_close", "missing_attribute"],
            "声明作废后模式停在 global，随后的 </focus> 属不匹配闭标签",
        )

    def test_stop_token_is_sampled_but_not_forwarded(self) -> None:
        """末 token 触发停止：它被采样，但没有 forward 消费它 → 轨迹里不出现那一行。"""
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
        from attnview.readview import TokenLayout
        from attnview.trace import build_step_trace, independence_summary

        reg = ProtocolRegistry()
        state = make_state(reg, "req-stop", num_segments=2)  # prompt_len = 100
        for token_index, ch in enumerate("<local>x</local>"):
            state.feed_generated_token(token_index, 0, ch)
        state.mark_stop_token(len(state.steps) - 1)
        layout = TokenLayout(prompt_len=100, segment_spans=((10, 40), (50, 80)), local_window_span=(90, 100))
        table = tuple(range(200, 220))
        rows_all = build_step_trace(
            state, layout, canonical_blocks=table, kernel_block_size=16, max_width=len(table)
        )
        rows_stopped = build_step_trace(
            state, layout, canonical_blocks=table, kernel_block_size=16, max_width=len(table),
            final_token_forwarded=False,
        )
        self.assertEqual(len(rows_all) - len(rows_stopped), 1, "少一行 = 消费 stop token 的那次 forward")
        self.assertEqual(rows_stopped[-1].input_token_index, state.stop_token_index - 1)
        summary = independence_summary(rows_stopped)
        self.assertEqual(summary["forwards"], len(rows_stopped) - 1)
        self.assertEqual(state.trace()["stop_token_index"], state.stop_token_index)

    def test_trace_contains_protocol_fields(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-t")
        feed(state, '<focus magic_chunks="2">v</focus><answer>A</answer>')
        state.finish()
        trace = state.trace()
        for key in (
            "request_id",
            "final_mode",
            "focus_attempts",
            "focus_successes",
            "anomalies",
            "steps",
        ):
            self.assertIn(key, trace)
        with_events = next(s for s in trace["steps"] if s["events"])
        self.assertEqual(
            with_events["events"][0]["effect_step"], with_events["token_index"] + 1
        )


if __name__ == "__main__":
    unittest.main()
