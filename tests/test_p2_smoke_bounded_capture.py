"""SUP-004 smoke:bounded capture(仅 masked)的 CPU 用例(不加载模型/不跑 GPU)。

驱动 `LayerCapture._record` + `step_arrays` + `dump_capture` 的**真实代码路径**(不搬张量也能留下证据):
- prefill 步:保留 KV**一次**,**不**保存 prefill 的 Q/out;
- 非代表 decode 步:保留该步追加 KV,**不**保存 Q/out;
- 代表 decode 步:保存 Q/out;
- **所有步**都输出位置证据;
- 旧(非 bounded)模式行为不变:prefill 的 Q/out 仍保存。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_runner():
    spec = importlib.util.spec_from_file_location("p2_calib_run_capture", REPO / "tools/p2-calib-run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class BoundedCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_runner()
        cls.LC = cls.r.LayerCapture

    def make_capture(self, *, bounded: bool, representative=(8,)):
        c = self.LC.__new__(self.LC)
        c._active, c._step_open, c.armed_req_id = True, True, "req"
        c.records, c.marks, c._step_metadata = {1: {}, 7: {}, 8: {}}, {}, None
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

    def test_bounded_keeps_kv_and_positions_but_drops_q_out(self):
        c = self.make_capture(bounded=True, representative=(8,))
        for step, q_len in ((1, 4), (7, 1), (8, 1)):
            self.feed(c, step=step, q_len=q_len)
        a1, a7, a8 = c.step_arrays(1), c.step_arrays(7), c.step_arrays(8)
        self.assertIn("k_prefill_L0", a1)
        self.assertIn("v_prefill_L0", a1)
        self.assertNotIn("q_step1_L0", a1)              # bounded:不保存 prefill Q/out
        self.assertNotIn("out_step1_L0", a1)
        self.assertIn("k_current_step6_L0", a7)          # 每次 decode 追加 KV
        self.assertNotIn("decode_q_step6_L0", a7)        # 非代表点:不保存 Q/out
        self.assertIn("decode_q_step7_L0", a8)           # 代表点:保存 Q/out
        self.assertIn("decode_out_step7_L0", a8)
        for arr, key in ((a1, "positions_step1"), (a7, "decode_pos_step6"), (a8, "decode_pos_step7")):
            self.assertIn(key, arr)                      # 所有步都留位置证据

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

    def test_dump_capture_reports_bytes_without_q_out(self):
        c = self.make_capture(bounded=True, representative=(8,))
        for step, q_len in ((1, 4), (7, 1), (8, 1)):
            self.feed(c, step=step, q_len=q_len)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.npz"
            self.r.dump_capture(c, path)
            import numpy as np
            with np.load(path) as z:
                keys = set(z.files)
                self.assertIn("positions_step1", keys)
                self.assertIn("decode_pos_step6", keys)
                self.assertNotIn("q_step1_L0", keys)
                total = sum(int(z[k].nbytes) for k in z.files)
            self.assertGreater(total, 0)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
