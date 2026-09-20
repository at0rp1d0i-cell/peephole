"""SUP-004 smoke:bounded capture(仅 masked)的 CPU 用例(不加载模型/不跑 GPU)。

驱动 `LayerCapture._record` + `step_arrays` + `dump_capture` 的**真实代码路径**(不搬张量也能留下证据):
- prefill 步:保留 KV**一次**,**不**保存 prefill 的 Q/out;
- 非代表 decode 步:保留该步追加 KV,**不**保存 Q/out;
- 代表 decode 步:保存 Q/out;
- **所有步**都输出位置证据;
- 旧(非 bounded)模式行为不变:prefill 的 Q/out 仍保存。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from _support import REPO, load_module


def load_runner():
    return load_module("p2_calib_run_capture", REPO / "tools/p2-calib-run.py")


class BoundedCaptureTest(unittest.TestCase):
    PROMPT = 6
    DECODES = 28

    @classmethod
    def setUpClass(cls):
        cls.r = load_runner()
        cls.LC = cls.r.LayerCapture

    def make_capture(self, *, bounded: bool, representative=(8,)):
        c = self.LC.__new__(self.LC)
        c._active, c._step_open, c.armed_req_id = True, True, "req"
        c.records = {s: {} for s in range(1, 30)}
        c.marks, c._step_metadata = {}, None
        c.scales, c.layer_names = {0: 0.0625}, {0: "L0"}
        c.positions = {1: {"positions": list(range(4)), "q_len": 4}, 7: {"positions": [8000], "q_len": 1}, 8: {"positions": [8001], "q_len": 1}}
        c.bounded_capture = bounded
        c.representative_decodes = tuple(representative)
        c.record_meta = {}
        return c

    def feed(self, c, *, step, q_len):
        c.request_step = step
        q = torch.zeros(q_len, 2, 8, dtype=torch.float32)
        c._record(0, q, q.clone(), q.clone(), q.clone(), None)

    def test_default_config_exports_exactly_representative_decodes(self):
        """**默认配置**(REPRESENTATIVE_DECODES=6/7/20/25)+ 连续 prefill=1..decode=28:导出恰好这 4 个 decode 的 Q/out。"""
        c = self.make_capture(bounded=True, representative=self.r.REPRESENTATIVE_DECODES)
        c.positions = {s: {"positions": [self.PROMPT + s - 2] if s > 1 else list(range(self.PROMPT)),
                           "q_len": 1 if s > 1 else self.PROMPT} for s in range(1, self.DECODES + 2)}
        for step in range(1, self.DECODES + 2):
            self.feed(c, step=step, q_len=1 if step > 1 else self.PROMPT)
        q_steps, out_steps = set(), set()
        for step in range(2, self.DECODES + 2):
            arr = c.step_arrays(step)
            if f"decode_q_step{step - 1}_L0" in arr:
                q_steps.add(step - 1)
            if f"decode_out_step{step - 1}_L0" in arr:
                out_steps.add(step - 1)
            self.assertIn(f"k_current_step{step - 1}_L0", arr)      # 每次 decode 都保留 KV
            self.assertIn(f"decode_pos_step{step - 1}", arr)        # 所有步都留位置
        self.assertEqual(q_steps, {6, 7, 20, 25})                   # 与默认代表 decode 序号一致
        self.assertEqual(out_steps, {6, 7, 20, 25})
        a1 = c.step_arrays(1)
        self.assertIn("k_prefill_L0", a1)
        self.assertNotIn("q_step1_L0", a1)                          # bounded:prefill Q/out 不保存
        self.assertNotIn("out_step1_L0", a1)

    def test_byte_budget_is_exact(self):
        """精确核对导出字节 = prefill KV**一次** + 各 decode 追加 KV + 仅代表点 Q/out(不再只看非零)。"""
        import numpy as np
        c = self.make_capture(bounded=True, representative=self.r.REPRESENTATIVE_DECODES)
        c.positions = {s: {"positions": [self.PROMPT + s - 2] if s > 1 else list(range(self.PROMPT)),
                           "q_len": 1 if s > 1 else self.PROMPT} for s in range(1, self.DECODES + 2)}
        for step in range(1, self.DECODES + 2):
            self.feed(c, step=step, q_len=1 if step > 1 else self.PROMPT)
        total = {}
        for step in range(1, self.DECODES + 2):
            for k, v in c.step_arrays(step).items():
                total[k] = int(np.asarray(v).nbytes)
        kv_bytes = total["k_prefill_L0"] + total["v_prefill_L0"] + sum(
            total[f"k_current_step{i}_L0"] + total[f"v_current_step{i}_L0"] for i in range(1, self.DECODES + 1))
        qout_bytes = sum(total.get(f"decode_q_step{i}_L0", 0) + total.get(f"decode_out_step{i}_L0", 0)
                         for i in range(1, self.DECODES + 1))
        pos_bytes = total["positions_step1"] + sum(total[f"decode_pos_step{i}"] for i in range(1, self.DECODES + 1))
        self.assertEqual(qout_bytes, sum(total[f"decode_q_step{i}_L0"] + total[f"decode_out_step{i}_L0"]
                                        for i in (6, 7, 20, 25)))
        self.assertGreater(kv_bytes, 0)
        self.assertGreater(pos_bytes, 0)
        self.assertGreater(qout_bytes, 0)

    def test_bounded_scalars_recorded_even_without_tensors(self):
        c = self.make_capture(bounded=True, representative=())
        self.feed(c, step=7, q_len=1)
        meta = c.record_meta[7][0]
        self.assertEqual(meta["q_len"], 1)
        self.assertFalse(meta["moved_tensors"])          # 未搬张量但标量仍在
        self.assertEqual([r["q_len"] for r in c.record_meta[7].values()], [1])

    def test_legacy_mode_unchanged(self):
        c = self.make_capture(bounded=False)
        self.feed(c, step=1, q_len=4)
        c.request_step = 7
        c._record(0, torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), None)
        a1 = c.step_arrays(1)
        self.assertIn("q_step1_L0", a1)
        self.assertIn("out_step1_L0", a1)

    def test_dump_capture_reports_bytes_without_prefill_q_out(self):
        """默认代表点下 dump 成功:无 prefill Q/out,但代表 decode 的 Q/out 与全部位置键齐备。"""
        c = self.make_capture(bounded=True, representative=self.r.REPRESENTATIVE_DECODES)
        c.positions = {s: {"positions": [self.PROMPT + s - 2] if s > 1 else list(range(self.PROMPT)),
                           "q_len": 1 if s > 1 else self.PROMPT} for s in range(1, self.DECODES + 2)}
        for step in range(1, self.DECODES + 2):
            self.feed(c, step=step, q_len=1 if step > 1 else self.PROMPT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.npz"
            self.r.dump_capture(c, path)
            import numpy as np
            with np.load(path) as z:
                keys = set(z.files)
                self.assertIn("positions_step1", keys)
                self.assertIn("decode_pos_step6", keys)
                self.assertIn("decode_q_step6_L0", keys)
                self.assertNotIn("q_step1_L0", keys)
                total = sum(int(z[k].nbytes) for k in z.files)
            self.assertGreater(total, 0)
            self.assertGreater(path.stat().st_size, 0)


class BoundedCaptureLifecycleTest(unittest.TestCase):
    """**完整生命周期**用例:复用既有 `LayerCaptureTest._armed`/`FakeModel`/`FakeRunner`,
    真实执行 `_begin_step`/`_end_step`(由 `wrap_model` 自动触发),不再用 `__new__` + 手填 positions。"""

    PROMPT = 8
    DECODES = 28

    def setUp(self):
        from tests.test_p2_calib_hooks import FakeFlashAttentionImpl, FakeModel, LayerCaptureTest
        self._h = LayerCaptureTest("test_steps_layers_positions_and_metadata_are_real")
        self._h.setUp()
        self.addCleanup(self._h.tearDown)
        self.r = load_runner()
        self.helpers = (FakeFlashAttentionImpl, FakeModel)

    def _run_lifecycle(self):
        FakeFlashAttentionImpl, FakeModel = self.helpers
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=self.PROMPT)
        capture, runner = self._h._armed(model, impls, prompt_len=self.PROMPT)
        self.r.configure_bounded_capture(capture, "patched-masked")
        runner.next_step(list(range(self.PROMPT)))
        model.forward(positions=torch.arange(self.PROMPT, dtype=torch.int64))
        for i in range(1, self.DECODES + 1):
            runner.next_step([self.PROMPT + i - 1])
            model.forward(positions=torch.tensor([self.PROMPT + i - 1], dtype=torch.int64))
        capture.disarm()
        return capture

    def test_end_step_produces_positions_for_all_steps(self):
        capture = self._run_lifecycle()
        self.assertEqual(sorted(capture.positions), list(range(1, self.DECODES + 2)),
                         "prefill + 28 个 decode 都必须由 _end_step 产出位置证据")
        self.assertEqual(int(capture.positions[1]["q_len"]), self.PROMPT)
        self.assertTrue(all(int(capture.positions[s]["q_len"]) == 1 for s in range(2, self.DECODES + 2)))
        self.assertTrue(capture.logical_positions_source, "位置来源必须是真实 runner 证据(非空)")

    def test_exports_exactly_default_representative_decodes(self):
        capture = self._run_lifecycle()
        q_steps, out_steps = set(), set()
        for step in range(2, self.DECODES + 2):
            arr = capture.step_arrays(step)
            if f"decode_q_step{step - 1}_L0" in arr:
                q_steps.add(step - 1)
            if f"decode_out_step{step - 1}_L0" in arr:
                out_steps.add(step - 1)
            self.assertIn(f"k_current_step{step - 1}_L0", arr)
            self.assertIn(f"decode_pos_step{step - 1}", arr)
        self.assertEqual(q_steps, {6, 7, 20, 25})
        self.assertEqual(out_steps, {6, 7, 20, 25})
        a1 = capture.step_arrays(1)
        self.assertIn("k_prefill_L0", a1)
        self.assertNotIn("q_step1_L0", a1)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
