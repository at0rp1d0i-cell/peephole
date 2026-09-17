"""C3 增量解析验收（工作单 §4「增量解析」面）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview.parser import (  # noqa: E402
    MODE_FOCUS,
    MODE_GLOBAL,
    MODE_LOCAL,
    TagParser,
)

FOCUS_TAG = '<focus magic_chunks="1,2">'


class ParserTest(unittest.TestCase):
    def test_tag_split_at_every_char_boundary(self) -> None:
        text = FOCUS_TAG + "v</focus>"
        close_at = text.index(">")
        for cut in range(1, len(text)):
            parser = TagParser(num_segments=3)
            events = parser.feed(text[:cut], 0) + parser.feed(text[cut:], 1)
            opens = [e for e in events if e.mode_after == MODE_FOCUS]
            self.assertEqual(len(opens), 1, f"cut={cut}")
            self.assertEqual(opens[0].refs, (1, 2))
            expected_step = 1 if close_at < cut else 2
            self.assertEqual(opens[0].effect_step, expected_step, f"cut={cut}")
            # 闭合标签同样只在 `>` 到达时生效
            closes = [e for e in events if e.name == "focus" and e.mode_after == MODE_GLOBAL]
            self.assertEqual(len(closes), 1, f"cut={cut}")

    def test_no_premature_transition_before_gt(self) -> None:
        text = FOCUS_TAG
        close_at = text.index(">")
        for cut in range(1, close_at + 1):
            parser = TagParser(num_segments=3)
            events = parser.feed(text[:cut], 0)
            self.assertEqual([e for e in events if e.kind == "transition"], [], f"cut={cut}")
            self.assertEqual(parser.mode, MODE_GLOBAL)

    def test_char_by_char_effect_step_is_the_gt_token(self) -> None:
        parser = TagParser(num_segments=3)
        text = "<local>x</local>"
        events = []
        for index, ch in enumerate(text):
            events.extend(parser.feed(ch, index))
        transitions = [e for e in events if e.kind == "transition"]
        self.assertEqual([(t.name, t.mode_after, t.effect_step) for t in transitions],
                         [("local", MODE_LOCAL, text.index(">") + 1),
                          ("local", MODE_GLOBAL, text.rindex(">") + 1)])

    def test_multiple_events_in_one_token(self) -> None:
        parser = TagParser(num_segments=3)
        events = parser.feed("<global><local>", 7)
        self.assertEqual([(e.kind, e.mode_after) for e in events],
                         [("noop", MODE_GLOBAL), ("transition", MODE_LOCAL)])
        self.assertTrue(all(e.effect_step == 8 for e in events))
        events2 = parser.feed("</local><global>", 8)
        self.assertEqual([(e.kind, e.mode_after) for e in events2],
                         [("transition", MODE_GLOBAL), ("noop", MODE_GLOBAL)])

    def test_refs_dedupe_and_stable_sort(self) -> None:
        parser = TagParser(num_segments=5)
        parser.feed('<focus magic_chunks="3,1,2,2">', 0)
        self.assertEqual(parser.refs, (1, 2, 3))
        self.assertEqual(parser.mode, MODE_FOCUS)
        parser.feed('<focus magic_chunks="4,3">', 1)
        self.assertEqual(parser.refs, (3, 4))

    def test_invalid_reference_keeps_mode(self) -> None:
        parser = TagParser(num_segments=3)
        parser.feed('<focus magic_chunks="1">', 0)
        self.assertEqual(parser.refs, (1,))
        events = parser.feed('<focus magic_chunks="1,9">', 1)
        self.assertEqual(parser.mode, MODE_FOCUS)
        self.assertEqual(parser.refs, (1,), "整条声明作废，不得部分生效")
        self.assertEqual([e.reason for e in events], ["invalid_reference"])
        self.assertEqual(events[0].name, "focus")
        self.assertEqual(events[0].kind, "anomaly")

    def test_missing_and_malformed_attribute(self) -> None:
        cases = {
            "<focus>": "missing_attribute",
            "<focus magic_chunks=1>": "malformed_attribute",
            '<focus magic_chunks="">': "malformed_attribute",
            '<focus magic_chunks="a,b">': "malformed_attribute",
            '<focus magic_chunks="1" foo="bar">': "unexpected_attribute",
            '<focus magic_chunks="1" magic_chunks="2">': "duplicate_attribute",
            '<local x="1">': "unexpected_attribute",
            '<focus magic_chunks="1">': None,
        }
        for tag, expected in cases.items():
            parser = TagParser(num_segments=3)
            events = parser.feed(tag, 0)
            reasons = [e.reason for e in events if e.kind == "anomaly"]
            self.assertEqual(reasons, ([expected] if expected else []), tag)
            if expected:
                self.assertEqual(parser.mode, MODE_GLOBAL, "异常必须保持当前模式（C6.2）")

    def test_unknown_and_incomplete_tags(self) -> None:
        parser = TagParser(num_segments=3)
        self.assertEqual([e.reason for e in parser.feed("<foo>", 0)], ["unknown_tag"])
        self.assertEqual(parser.mode, MODE_GLOBAL)
        parser.feed("<focus", 1)
        flushed = parser.flush(2)
        self.assertEqual([e.reason for e in flushed], ["incomplete_tag"])
        self.assertEqual(parser.mode, MODE_GLOBAL)

    def test_focus_attempt_flag_marks_openings_only(self) -> None:
        parser = TagParser(num_segments=3)
        events = parser.feed('<focus magic_chunks="1">', 0)
        self.assertTrue(events[-1].is_focus_attempt)
        events = parser.feed("</focus>", 1)
        self.assertFalse(events[-1].is_focus_attempt, "闭标签不是一次 focus 调用")
        events = parser.feed("<focus>", 2)
        self.assertTrue(events[-1].is_focus_attempt, "缺属性的开标签仍计为尝试")
        parser.feed('<focus magic_chunks="9">', 3)
        self.assertTrue(parser.anomalies[-1].is_focus_attempt, "越界引用仍计为尝试")

    def test_buffer_overflow_is_text_then_recovers(self) -> None:
        parser = TagParser(num_segments=3, max_tag_buffer=32)
        events = parser.feed("<" + "z" * 40, 0)
        self.assertEqual([e.reason for e in events], ["tag_buffer_overflow"])
        self.assertEqual(parser.mode, MODE_GLOBAL)
        self.assertIn("zzz", "".join(parser.plain_text))
        events2 = parser.feed("<local>", 1)
        self.assertEqual([(e.kind, e.mode_after) for e in events2], [("transition", MODE_LOCAL)])

    def test_lt_not_starting_a_tag_is_plain_text(self) -> None:
        parser = TagParser(num_segments=3)
        for text in ("a < b", "1 <2>", "<<focus>"):
            parser = TagParser(num_segments=3)
            events = parser.feed(text, 0)
            self.assertEqual([e.kind for e in events if e.kind == "transition"], [], text)
            self.assertEqual(parser.mode, MODE_GLOBAL, text)
        parser = TagParser(num_segments=3)
        events = parser.feed("<<local>", 0)
        self.assertEqual([(e.kind, e.mode_after) for e in events], [("transition", MODE_LOCAL)])

    def test_answer_marker_does_not_change_mode(self) -> None:
        parser = TagParser(num_segments=3)
        parser.feed("<local>x</local>", 0)
        events = parser.feed("<answer>", 1)
        self.assertEqual([(e.kind, e.name) for e in events], [("marker", "answer")])
        self.assertEqual(parser.mode, MODE_GLOBAL)
        events2 = parser.feed("</answer>", 2)
        self.assertEqual([(e.kind, e.name) for e in events2], [("marker", "answer")])

    def test_matching_closing_tags_return_to_global(self) -> None:
        parser = TagParser(num_segments=3)
        parser.feed("<local>", 0)
        parser.feed("</local>", 1)
        self.assertEqual(parser.mode, MODE_GLOBAL)
        parser.feed('<focus magic_chunks="2">', 2)
        self.assertEqual(parser.mode, MODE_FOCUS)
        parser.feed("</focus>", 3)
        self.assertEqual(parser.mode, MODE_GLOBAL)
        self.assertEqual(parser.refs, ())

    def test_mismatched_closing_tag_is_an_anomaly_and_keeps_mode(self) -> None:
        cases = [
            ('<focus magic_chunks="1">', "</local>"),
            ("<local>", "</focus>"),
            ('<focus magic_chunks="1">', "</global>"),
            ("<local>", "<local></focus>"),
        ]
        for opening, closing in cases:
            parser = TagParser(num_segments=3)
            parser.feed(opening, 0)
            before = parser.mode
            events = parser.feed(closing, 1)
            anomalies = [e for e in events if e.kind == "anomaly"]
            self.assertTrue(anomalies, f"{opening} {closing}")
            self.assertEqual(anomalies[-1].reason, "mismatched_close")
            self.assertEqual(parser.mode, before, f"{opening} {closing} 必须保持模式")
            self.assertFalse(
                any(e.is_focus_attempt for e in events), "闭标签不得计入 focus 分母"
            )

    def test_whitespace_inside_tags_tolerated(self) -> None:
        parser = TagParser(num_segments=3)
        events = parser.feed('<focus   magic_chunks="2" >', 0)
        self.assertEqual([e.kind for e in events], ["transition"])
        self.assertEqual(parser.refs, (2,))


if __name__ == "__main__":
    unittest.main()
