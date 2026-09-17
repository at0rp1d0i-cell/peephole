"""E4 公开输出提取验收（工作单 §4「输出隔离」面）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview.extract import extract_public_output, public_stream_accounting  # noqa: E402

PATCH = """<global>Look at the diff. Magic Chunk 1 holds the file.</global><focus magic_chunks="1">def add(a, b):
    # keep this comment and indentation
    if a is None:
        return b

    return a + b</focus><local>Return the early-exit branch.</local><answer>def add(a, b):
    if a is None:
        return b

    return a + b
</answer>"""


class ExtractTest(unittest.TestCase):
    def test_only_the_final_answer_is_returned(self) -> None:
        raw = "<global>analysis</global><local>reasoning</local><answer>D</answer>"
        result = extract_public_output(raw)
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "D")
        self.assertNotIn("analysis", result.answer or "")
        self.assertNotIn("reasoning", result.answer or "")

    def test_code_and_patch_body_preserved_exactly(self) -> None:
        result = extract_public_output(PATCH)
        self.assertTrue(result.ok)
        expected = (
            "def add(a, b):\n    if a is None:\n        return b\n\n    return a + b"
        )
        self.assertEqual(result.answer, expected)

    def test_plain_words_and_foreign_tags_preserved(self) -> None:
        raw = (
            "<answer>The answer is that focus groups and the answer field are plain words; "
            "keep <div class=\"x\">markup</div> and <p>html</p> intact.</answer>"
        )
        result = extract_public_output(raw)
        self.assertTrue(result.ok)
        self.assertIn("focus groups", result.answer or "")
        self.assertIn("<div class=\"x\">markup</div>", result.answer or "")
        self.assertIn("<p>html</p>", result.answer or "")

    def test_missing_answer_returns_failure_without_leaking_internals(self) -> None:
        secret = "INTERNAL-ANALYSIS-TEXT"
        result = extract_public_output(f"<global>{secret}</global>no answer here")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "answer_missing")
        self.assertIsNone(result.answer)
        self.assertNotIn(secret, result.error_message or "")
        self.assertNotIn(secret, str(result.as_dict()))

    def test_truncated_answer_is_a_failure_not_a_partial_leak(self) -> None:
        secret = "INTERNAL-ANALYSIS-TEXT"
        result = extract_public_output(f"<local>{secret}</local><answer>partial answer")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "answer_unterminated")
        self.assertIsNone(result.answer)
        self.assertNotIn(secret, str(result.as_dict()))

    def test_empty_answer_region(self) -> None:
        result = extract_public_output("<local>x</local><answer></answer>")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "answer_empty")

    def test_multiple_answer_regions_use_last_complete_one(self) -> None:
        raw = "<answer>first</answer><local>check</local><answer>final</answer>"
        result = extract_public_output(raw)
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "final")
        self.assertEqual(result.answer_count, 2)

    def test_no_residue_between_calls(self) -> None:
        raw = "<local>x</local><answer>A</answer>"
        first = extract_public_output(raw).as_dict()
        extract_public_output("<answer>B</answer>")
        second = extract_public_output(raw).as_dict()
        self.assertEqual(first, second)

    def test_same_name_literal_collision_is_a_documented_limitation(self) -> None:
        """已知限制（范围假设，非保真通过）：正文里与控制语法完全同名的字面串会被过滤。"""
        raw = '<answer>Use the literal <focus magic_chunks="1"> tag in prose.</answer>'
        result = extract_public_output(raw)
        self.assertTrue(result.ok)
        self.assertNotIn('<focus magic_chunks="1">', result.answer or "")
        self.assertEqual(result.stripped_tags, ('<focus magic_chunks="1">',))

    def test_accounting_separates_internal_and_public_volume(self) -> None:
        raw = "<global>long analysis text</global><local>more</local><answer>D</answer>"
        acc = public_stream_accounting(raw)
        self.assertEqual(acc["answer_chars"], 1)
        self.assertGreater(acc["analysis_chars"], 0)
        self.assertEqual(
            acc["protocol_tag_count"], 4,
            "只统计规定的注意力标签（<answer> 包装不计入控制标签）",
        )
        self.assertTrue(acc["ok"])


if __name__ == "__main__":
    unittest.main()
