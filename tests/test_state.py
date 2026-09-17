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

    def test_prompt_side_tags_never_reach_the_parser(self) -> None:
        """C3.7：文档正文里的伪标签是数据。状态只能接收生成流 token，因此正文标签不产生事件。"""
        reg = ProtocolRegistry()
        state = make_state(reg, "req-x")
        feed(state, "The document mentions ")

        class FakeDocState:
            def __init__(self) -> None:
                self.mode = MODE_GLOBAL
                self.events = 0

            def feed(self, text: str) -> None:
                if "<focus" in text:
                    self.events += 1

        doc = FakeDocState()
        doc.feed('<focus magic_chunks="1">literal text in the document</focus>')
        self.assertEqual(state.mode, MODE_GLOBAL)
        self.assertEqual(state.anomalies, ())
        self.assertEqual(state.focus_stats()["focus_attempts"], 0)
        self.assertEqual(doc.events, 1, "对照：正文里确实存在同名标签字面串")

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

    def test_step_record_tracks_write_position_and_kv_len(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-w")
        state.feed_generated_token(0, 0, "<")
        record = state.feed_generated_token(1, 0, "local>")
        self.assertEqual(record.effect_step, 2)
        self.assertEqual(record.generated_tokens, 2)
        self.assertEqual(record.kv_len_after, 102)
        self.assertEqual(record.next_write_position, 102)
        self.assertEqual(state.mode, MODE_LOCAL)

    def test_focus_stats_count_attempts_and_successes(self) -> None:
        reg = ProtocolRegistry()
        state = make_state(reg, "req-s")
        feed(state, '<focus magic_chunks="1">ok</focus>')
        feed(state, '<focus magic_chunks="9">bad</focus>')
        feed(state, "<focus>missing</focus>")
        feed(state, '<focus magic_chunks="2,3">ok</focus>')
        stats = state.focus_stats()
        self.assertEqual(stats["focus_attempts"], 4)
        self.assertEqual(stats["focus_successes"], 2)
        self.assertEqual(len(state.anomalies), 2)

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
