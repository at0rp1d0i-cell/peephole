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


def load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


class LayerCaptureTest(unittest.TestCase):
    """层观测必须按 pin 的**真实契约**接入：impl 是 ABC（非 nn.Module）、
    `forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)`、
    步边界由 `model.forward` 包裹（每步一次）——本类用两步两层的 CPU 真实调用核实。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = load_module_by_path("attnview_calib_driver", REPO / "tools/p2-calib-run.py")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_fake(self, *, prompt_len: int = 3, num_kv_heads: int = 2, head_dim: int = 4):
        torch_ = torch

        class FakeFlashAttentionImpl:
            def __init__(self, index: int) -> None:
                self.index = index

            def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None):
                return query + key.sum(dim=0, keepdim=True) + value.sum(dim=0, keepdim=True)

        class FakeModel:
            def __init__(self, impls) -> None:
                self.layers = [SimpleNamespace(self_attn=SimpleNamespace(impl=impl)) for impl in impls]
                self.calls = 0

            def forward(self, *args, **kwargs):
                # 镜像真实：**每步每层只调一次**，q_len = 本步调度的 token 数
                # （prefill 一整步一次，decode 单 query 一次）。
                self.calls += 1
                q_len = prompt_len if self.calls == 1 else 1
                q = torch_.randn(q_len, num_kv_heads, head_dim)
                k = torch_.randn(q_len, num_kv_heads, head_dim)
                v = torch_.randn(q_len, num_kv_heads, head_dim)
                return [layer.self_attn.impl.forward(None, q, k, v, None, None, torch_.zeros_like(q))
                        for layer in self.layers]

        impls = [FakeFlashAttentionImpl(0), FakeFlashAttentionImpl(1)]
        return FakeModel(impls), impls

    def test_two_steps_two_layers_records_are_complete(self) -> None:
        model, impls = self._make_fake()
        capture = self.driver.LayerCapture()
        for index, impl in enumerate(impls):
            capture.wrap_impl(index, impl)
        capture.wrap_model(model)
        model.forward()  # 步 0：prefill（q_len=3）
        model.forward()  # 步 1：decode（q_len=1）
        self.assertEqual(sorted(capture.records), [0, 1], "步边界必须每步一次，不能被覆盖")
        for step in (0, 1):
            self.assertEqual(sorted(capture.records[step]), [0, 1], "每步每层都要有记录")
        self.assertEqual(capture.records[0][0]["q_len"], 3)
        self.assertEqual(capture.records[1][0]["q_len"], 1)
        rec = capture.records[0][1]
        self.assertEqual(rec["q"].shape[-1], 4, "q/k/v 必须是真实张量（最后一维 = head_dim）")
        self.assertGreater(rec["out"].numel(), 0, "out 不能为空")
        self.assertTrue(torch.isfinite(rec["out"]).all())

    def test_dump_capture_matches_oracle_contract(self) -> None:
        model, impls = self._make_fake()
        capture = self.driver.LayerCapture()
        for index, impl in enumerate(impls):
            capture.wrap_impl(index, impl)
        capture.wrap_model(model)
        model.forward()
        model.forward()
        target = self.dir / "layers.npz"
        self.driver.dump_capture(capture, target)
        import numpy as np

        data = np.load(target)
        self.assertIn("k_prefill_L0", data)
        self.assertIn("v_prefill_L0", data)
        self.assertIn("q_step1_L0", data)
        self.assertIn("out_step1_L0", data)
        for key in ("scale", "num_heads", "num_kv_heads", "head_dim", "prompt_len"):
            self.assertIn(key, data, f"oracle 合同要求元数据 {key}")
        self.assertEqual(tuple(data["k_prefill_L0"].shape), (2, 3, 4))  # [kv_heads, prompt_len, head_dim]

    def test_find_fa_layers_fails_loudly_without_flash_impl(self) -> None:
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(
                            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(impl=object()))])
                        )
                    )
                )
            )
        )
        with self.assertRaises(RuntimeError):
            self.driver.find_fa_layers(llm)


class WatchdogTest(unittest.TestCase):
    """期限必须**真的能中断**：到期先落盘证据再退出（用子进程真实验证，不在本进程里自杀）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_watchdog_saves_evidence_and_exits_nonzero(self) -> None:
        script = Path(self.dir, "watchdog_case.py")
        script.write_text(
            "import sys, time\n"
            f"sys.path.insert(0, {str(REPO / 'tools')!r})\n"
            "from pathlib import Path\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('drv', " + repr(str(REPO / "tools/p2-calib-run.py")) + ")\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['drv'] = m; spec.loader.exec_module(m)\n"
            "with m.Watchdog(1.0, 'request', Path(sys.argv[1])):\n"
            "    time.sleep(30)\n"
            "print('不该走到这里')\n"
        )
        started = __import__("time").time()
        proc = __import__("subprocess").run(
            [sys.executable, str(script), self.dir.as_posix()], capture_output=True, text=True, timeout=30
        )
        elapsed = __import__("time").time() - started
        self.assertEqual(proc.returncode, 3, f"到期应以 3 退出（实际 {proc.returncode}）")
        self.assertLess(elapsed, 15, "看门狗必须在预算后很快生效（不能等阻塞返回）")
        evidence = self.dir / "budget_exceeded.request.json"
        self.assertTrue(evidence.exists(), "到期必须先落盘证据")
        payload = json.loads(evidence.read_text())
        self.assertEqual(payload["label"], "request")
        self.assertIn("elapsed_s", payload)


if __name__ == "__main__":
    unittest.main()
