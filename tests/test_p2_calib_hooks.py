"""校准钩子（强制轨迹 / 完整 logits / 覆写 trace）的 CPU 真实调用测试。

要点：
- 两个钩子都必须在**未设置环境变量**时完全不介入（普通请求零成本、零行为差异）；
- 设环境变量时按真实调用验证：**原地**替换（同一块内存，worker 历史与宿主看到同一个 token）、
  原始采样被记录、完整 logits 落盘且步号递增、轨迹用尽/形状不符必须拒绝；
- trace 记录必须包含"未改写入参"的指纹（`data_ptr`/`_version`），且不读设备内容。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ADAPTER = REPO / "vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("attnview_adapter_calib", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["attnview_adapter_calib"] = module
    spec.loader.exec_module(module)
    return module


class CalibHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_adapter()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("ATTNVIEW_CALIB_FORCE", "ATTNVIEW_CALIB_FORCE_LOG",
                      "ATTNVIEW_CALIB_LOGITS", "ATTNVIEW_CALIB_TRACE")
        }
        self.mod._CALIB_STATE.update({"tokens": None, "step": 0})

    def tearDown(self) -> None:
        self.tmp.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    # --- 强制轨迹 --------------------------------------------------------- #

    def test_force_tokens_is_noop_without_env(self) -> None:
        sampled = torch.tensor([[7], [8]], dtype=torch.int32)
        self.assertFalse(self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=sampled), ["a", "b"]))
        self.assertEqual(sampled.tolist(), [[7], [8]], "未设开关时不得改写采样")

    def test_force_tokens_replaces_in_place_and_logs_raw(self) -> None:
        traj = self.dir / "traj.json"
        traj.write_text(json.dumps({"tokens": [[[101], [102]], [[103], [104]]]}))
        log = self.dir / "force.jsonl"
        os.environ["ATTNVIEW_CALIB_FORCE"] = str(traj)
        os.environ["ATTNVIEW_CALIB_FORCE_LOG"] = str(log)
        sampled = torch.tensor([[7], [8]], dtype=torch.int32)
        ptr = sampled.data_ptr()
        self.assertTrue(self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=sampled), ["a", "b"]))
        self.assertEqual(sampled.tolist(), [[101], [102]])
        self.assertEqual(sampled.data_ptr(), ptr, "必须原地替换：worker 历史与宿主看到的要是同一块内存")
        sampled2 = torch.tensor([[9], [10]], dtype=torch.int32)
        self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=sampled2), ["a", "b"])
        self.assertEqual(sampled2.tolist(), [[103], [104]])
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([r["step"] for r in records], [1, 2])
        self.assertEqual(records[0]["raw_sampled"], [[7], [8]], "原始采样必须留存")
        self.assertEqual(records[1]["raw_sampled"], [[9], [10]])
        self.assertIn("D2H", records[0]["sync_note"], "必须标注该钩子含同步（校准专用）")

    def test_force_tokens_rejects_shape_mismatch_and_exhausted_trajectory(self) -> None:
        traj = self.dir / "traj.json"
        traj.write_text(json.dumps({"tokens": [[[101]]]}))
        os.environ["ATTNVIEW_CALIB_FORCE"] = str(traj)
        sampled = torch.tensor([[7], [8]], dtype=torch.int32)  # 2 行 vs 轨迹 1 行
        with self.assertRaises(RuntimeError):
            self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=sampled), ["a", "b"])
        self.assertEqual(sampled.tolist(), [[7], [8]], "形状不符时不得写一半")
        ok = torch.tensor([[7]], dtype=torch.int32)
        self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=ok), ["a"])
        self.assertEqual(ok.tolist(), [[101]])
        with self.assertRaises(RuntimeError):
            self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=ok), ["a"])

    def test_force_tokens_rejects_flat_trajectory_format(self) -> None:
        traj = self.dir / "traj.json"
        traj.write_text(json.dumps({"tokens": [101, 102]}))  # 少了一层（每步内应有请求行）
        os.environ["ATTNVIEW_CALIB_FORCE"] = str(traj)
        sampled = torch.tensor([[7]], dtype=torch.int32)
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.calibration_force_tokens(SimpleNamespace(sampled_token_ids=sampled), ["a"])
        self.assertIn("tokens[step][row]", str(ctx.exception))
        self.assertEqual(sampled.tolist(), [[7]])

    # --- 完整 logits ------------------------------------------------------ #

    def test_capture_logits_is_noop_without_env(self) -> None:
        self.assertFalse(self.mod.calibration_capture_logits(torch.zeros(1, 4), SimpleNamespace(req_ids=["a"])))
        self.assertEqual(list(self.dir.iterdir()), [], "未设开关时不得写文件")

    def test_capture_logits_saves_full_vocab_fp32(self) -> None:
        target = self.dir / "logits.pt"
        os.environ["ATTNVIEW_CALIB_LOGITS"] = str(target)
        logits = torch.randn(2, 151936, dtype=torch.bfloat16)
        self.assertTrue(self.mod.calibration_capture_logits(logits, SimpleNamespace(req_ids=["a", "b"])))
        self.mod._CALIB_STATE["step"] = 1
        self.mod.calibration_capture_logits(logits[:1], SimpleNamespace(req_ids=["a"]))
        records = torch.load(target, weights_only=False)
        self.assertEqual([r["step"] for r in records], [1, 2])
        self.assertEqual(tuple(records[0]["logits"].shape), (2, 151936), "必须是**完整**词表，不是 top-k")
        self.assertEqual(records[0]["logits"].dtype, torch.float32)
        self.assertEqual(records[1]["shape"], [1, 151936])
        self.assertEqual(records[0]["req_ids"], ["a", "b"])

    # --- 覆写 trace ------------------------------------------------------- #

    def test_override_trace_records_inputs_and_noop_without_env(self) -> None:
        self.mod.calibration_note_override(None, req_ids=["a"], scheduler_output=SimpleNamespace(),
                                           inputs={"t": torch.zeros(2)})
        self.assertEqual(list(self.dir.iterdir()), [])
        trace = self.dir / "steps.jsonl"
        os.environ["ATTNVIEW_CALIB_TRACE"] = str(trace)
        table = torch.zeros((2, 3), dtype=torch.int32)
        override = SimpleNamespace(
            group_index=3,
            rows_with_override=(0,),
            max_seq_len=3137,
            seq_lens=torch.tensor([3137], dtype=torch.int32),
            plans={0: SimpleNamespace(visible_logical_blocks=(0, 5, 6, 7, 8))},
            block_table=torch.zeros((1, 11), dtype=torch.int32),
        )
        self.mod.calibration_note_override(
            override, req_ids=["a"],
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"a": 1}),
            inputs={"fa_group_table": table},
        )
        record = json.loads(trace.read_text().splitlines()[0])
        self.assertEqual(record["override"]["seqused_k"], [3137])
        self.assertEqual(record["override"]["visible_logical_blocks"]["0"], [0, 5, 6, 7, 8])
        self.assertEqual(record["inputs"]["fa_group_table"]["data_ptr"], table.data_ptr())
        self.assertIn("version", record["inputs"]["fa_group_table"])
        self.assertIn("stream", record)
        self.assertEqual(table.tolist(), [[0, 0, 0], [0, 0, 0]], "trace 不得改写入参")


if __name__ == "__main__":
    unittest.main()
