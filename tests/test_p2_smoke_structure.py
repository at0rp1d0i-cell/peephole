"""SUP-004 smoke:masked 结构检查的 CPU 用例(真实 harness 驱动,不加载模型/不跑 GPU)。

复用既有 `LayerCaptureTest._armed`/`FakeModel`/`FakeRunner`(真实 `_begin_step`/`_end_step`),
为 fake runner 补上**固定 pin 形状**的字段(`block_tables.input_block_tables`/`input_buffers.slot_mappings`),
再用 `smoke_structure.check_capture_structure` 逐步核对独立预期;并覆盖 fail-closed 分支。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attnview.smoke_structure import (  # noqa: E402
    StructureError,
    check_capture_structure,
    expectations_from_config,
)

PROMPT = 8
DECODES = 28
TIMELINE = REPO / "configs/p2-masked-prep/crossblock.json"
DOC = REPO / "evidence/p1-cpu/demo-fixtures.json"
SNAPSHOT = REPO / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class FakeTables:
    def __init__(self, rows):
        self.input_block_tables = rows


class FakeBuffers:
    def __init__(self, slots):
        self.slot_mappings = slots


class StructureCheckTest(unittest.TestCase):
    PROMPT = PROMPT
    DECODES = DECODES

    def setUp(self):
        from tests.test_p2_calib_hooks import FakeFlashAttentionImpl, FakeModel, LayerCaptureTest
        self._h = LayerCaptureTest("test_steps_layers_positions_and_metadata_are_real")
        self._h.setUp()
        self.addCleanup(self._h.tearDown)
        self.impls_cls, self.model_cls = FakeFlashAttentionImpl, FakeModel

    def _lifecycle(self):
        impls = [self.impls_cls() for _ in range(2)]          # 两个"层"
        model = self.model_cls(impls, prompt_len=self.PROMPT)
        capture, runner = self._h._armed(model, impls, prompt_len=self.PROMPT)
        runner.block_tables = FakeTables([[3, 5, 7, 9] * 8])
        runner.input_buffers = FakeBuffers(torch.arange(4096, dtype=torch.int64))
        runner.next_step(list(range(self.PROMPT)))
        model.forward(positions=torch.arange(self.PROMPT, dtype=torch.int64))
        for i in range(1, self.DECODES + 1):
            runner.next_step([self.PROMPT + i - 1])
            model.forward(positions=torch.tensor([self.PROMPT + i - 1], dtype=torch.int64))
        capture.disarm()
        return capture, runner

    def _expectations(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
        return expectations_from_config(timeline_config=TIMELINE, doc_fixture=DOC, tokenizer=tok,
                                        prompt_len=self.PROMPT, sink_span=(0, 2), local_window_span=(0, self.PROMPT),
                                        segment_spans=[(0, 3), (3, 6), (6, 8)], block_size=4,
                                        total_steps=self.DECODES + 1)

    def test_structure_matches_expectations_over_all_steps(self):
        capture, runner = self._lifecycle()
        exp = self._expectations()
        checks, evidence = check_capture_structure(capture=capture, expectations=exp, runner=runner,
                                                  trace={"override_steps": []}, prompt_len=self.PROMPT,
                                                  block_size=4, representative_decodes=(6, 7, 20, 25))
        self.assertTrue(all(c["ok"] for c in checks), [c for c in checks if not c["ok"]])
        self.assertEqual(len(evidence["steps"]), self.DECODES + 1)
        self.assertIn("global 恢复步存在且被核对", [c["name"] for c in checks])

    def test_trace_step_mapping_is_explicit(self):
        capture, runner = self._lifecycle()
        exp = self._expectations()
        checks, evidence = check_capture_structure(capture=capture, expectations=exp, runner=runner,
                                                  trace={"override_steps": [{"step": 6, "applied_view": "global"}]},
                                                  prompt_len=self.PROMPT, block_size=4, representative_decodes=(6, 7))
        self.assertIn("override.step=6 ⇒ capture forward=7 存在", [c["name"] for c in checks])
        self.assertEqual(evidence["trace"]["mapping"], "override.step = decode_index; forward = decode_index + 1")

    def test_missing_trace_fails_closed(self):
        capture, runner = self._lifecycle()
        with self.assertRaises(StructureError):
            check_capture_structure(capture=capture, expectations=self._expectations(), runner=runner, trace=None,
                                    prompt_len=self.PROMPT, block_size=4, representative_decodes=(6, 7))

    def test_missing_runner_fields_fail_closed(self):
        capture, runner = self._lifecycle()
        del runner.input_buffers
        with self.assertRaises(StructureError):
            check_capture_structure(capture=capture, expectations=self._expectations(), runner=runner,
                                    trace={"override_steps": []}, prompt_len=self.PROMPT, block_size=4,
                                    representative_decodes=(6, 7))


if __name__ == "__main__":
    unittest.main()
