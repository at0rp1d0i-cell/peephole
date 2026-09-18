"""校准钩子（控制文件绑定的强制轨迹 / 完整 logits / 覆写 trace）与驱动生命周期验收的 CPU 真实调用测试。

要点（本地复核 R1 后收紧）：

- 钩子只在**控制文件 `ATTNVIEW_CALIB_ARM` 指向的 JSON 存在**时才 armed；文件在 engine 就绪后由驱动写入，
  因此 warmup 期间（文件不存在）**任何**钩子都不得介入，也不得"绑定第一个请求"；
- 只在控制文件显式给出的 `target_req_id` 上生效：warmup / cleanup / 第二个请求都不改写、不写日志、
  不追加 logits 记录；
- 强制/捕获的计数各自独立（`force_step` / `logits_step`），都从 1 起；
- 清理验收按**可观察效果**判定：对 add_request/step/abort 都无动作的引擎必须 `ok=False`。
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
    """每个测试一份**全新**适配层模块：状态计数不跨测试泄漏，也不依赖内部键名。"""
    return load_module_by_path("attnview_adapter_calib", ADAPTER)


def _override():
    return SimpleNamespace(
        group_index=3,
        rows_with_override=(0,),
        max_seq_len=3137,
        seq_lens=torch.tensor([3137], dtype=torch.int32),
        plans={0: SimpleNamespace(visible_logical_blocks=(0, 5, 6, 7, 8))},
        block_table=torch.zeros((1, 11), dtype=torch.int32),
    )


class CalibHookTest(unittest.TestCase):
    """三钩子都按控制文件（`arm.json`）与显式 `target_req_id` 工作。"""

    ARM_ENV = "ATTNVIEW_CALIB_ARM"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (self.ARM_ENV, "ATTNVIEW_CALIB_FORCE", "ATTNVIEW_CALIB_FORCE_LOG",
                      "ATTNVIEW_CALIB_LOGITS", "ATTNVIEW_CALIB_TRACE")
        }
        self.mod = load_adapter()

    def tearDown(self) -> None:
        self.tmp.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    # --- 测试工具 --------------------------------------------------------- #

    def _sampled(self, rows):
        return torch.tensor(rows, dtype=torch.int32)

    def write_arm(self, *, target: str = "r-main", tokens=None, force_log=None,
                  logits=None, trace=None, name: str = "arm.json") -> Path:
        content = {
            "target_req_id": target,
            "tokens": tokens if tokens is not None else [],
            "force_log": str(force_log) if force_log is not None else None,
            "logits_path": str(logits) if logits is not None else None,
            "trace_path": str(trace) if trace is not None else None,
        }
        path = self.dir / name
        path.write_text(json.dumps(content))
        os.environ[self.ARM_ENV] = str(path)
        return path

    def force(self, req_ids, num_sampled, rows):
        sampled = self._sampled(rows)
        changed = self.mod.calibration_force_tokens(
            SimpleNamespace(sampled_token_ids=sampled), req_ids, torch.tensor(num_sampled)
        )
        return changed, sampled

    # --- 控制文件绑定 ----------------------------------------------------- #

    def test_arm_file_absent_means_not_armed(self) -> None:
        """文件不存在 ⇒ 所有钩子不介入、不写任何文件（warmup 期间的真实状态）。"""
        os.environ[self.ARM_ENV] = str(self.dir / "arm.json")  # 路径已设，但文件不存在
        sampled = self._sampled([[7]])
        self.assertFalse(self.mod.calibration_force_tokens(
            SimpleNamespace(sampled_token_ids=sampled), ["r-main"], torch.tensor([1])))
        self.assertFalse(self.mod.calibration_capture_logits(
            torch.zeros(1, 8), SimpleNamespace(req_ids=["r-main"])))
        self.mod.calibration_note_override(
            _override(), req_ids=["r-main"], scheduler_output=SimpleNamespace(num_scheduled_tokens={"r-main": 1}),
            inputs={})
        self.assertEqual(sampled.tolist(), [[7]], "未 armed 时不得改写采样")
        self.assertEqual(list(self.dir.iterdir()), [], "未 armed 时不得写任何文件")

    def test_arm_file_created_after_warmup_arms_hooks(self) -> None:
        """真实流程：warmup 期间文件不存在 ⇒ 不命中；engine 就绪后写入文件 ⇒ 目标请求被强制。"""
        os.environ[self.ARM_ENV] = str(self.dir / "arm.json")
        warm = self._sampled([[7]])
        self.assertFalse(self.mod.calibration_force_tokens(
            SimpleNamespace(sampled_token_ids=warm), ["warmup-1"], torch.tensor([1])))
        self.assertEqual(warm.tolist(), [[7]])
        self.write_arm(tokens=[[[101]]], force_log=self.dir / "force.jsonl")
        changed, main = self.force(["r-main"], [1], [[9]])
        self.assertTrue(changed)
        self.assertEqual(main.tolist(), [[101]])
        records = [json.loads(line) for line in (self.dir / "force.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in records], [1])

    def test_warmup_with_sampling_first_does_not_steal_binding(self) -> None:
        """有采样的 warmup 先出现：不得改写、不得写日志；随后目标请求才被强制且 `step` 从 1 起。"""
        force_log = self.dir / "force.jsonl"
        self.write_arm(target="r-main", tokens=[[[101]], [[102]]], force_log=force_log)
        changed, warm = self.force(["warmup-1"], [1], [[42]])
        self.assertFalse(changed, "warmup 请求不是 target_req_id，不得被强制")
        self.assertEqual(warm.tolist(), [[42]])
        self.assertFalse(force_log.exists(), "warmup 步不得写强制日志")
        changed, main = self.force(["r-main"], [1], [[6]])
        self.assertTrue(changed)
        self.assertEqual(main.tolist(), [[101]])
        changed, main2 = self.force(["r-main"], [1], [[6]])
        self.assertTrue(changed)
        self.assertEqual(main2.tolist(), [[102]])
        records = [json.loads(line) for line in force_log.read_text().splitlines()]
        self.assertEqual([r["step"] for r in records], [1, 2], "步号独立计数、从 1 起（不含 warmup）")
        self.assertEqual({r["req_id"] for r in records}, {"r-main"})

    def test_second_request_does_not_pollute_main_artifacts(self) -> None:
        """主请求跑完后，另一个 req_id 的消费步：不改写、不写日志、不追加 trace/logits 记录。"""
        force_log = self.dir / "force.jsonl"
        logits = self.dir / "logits.pt"
        trace = self.dir / "steps.jsonl"
        self.write_arm(tokens=[[[101]]], force_log=force_log, logits=logits, trace=trace)
        self.force(["r-main"], [1], [[7]])
        self.mod.calibration_capture_logits(torch.zeros(1, 8), SimpleNamespace(req_ids=["r-main"]))
        self.mod.calibration_note_override(
            _override(), req_ids=["r-main"],
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r-main": 1}), inputs={})
        before_force = force_log.read_text()
        before_logits = torch.load(logits, weights_only=False)
        before_trace = trace.read_text()

        changed, other = self.force(["calib-cleanup-2"], [1], [[55]])
        self.assertFalse(changed, "清理/第二个请求不得被强制")
        self.assertEqual(other.tolist(), [[55]])
        self.assertFalse(self.mod.calibration_capture_logits(
            torch.zeros(1, 8), SimpleNamespace(req_ids=["calib-cleanup-2"])))
        self.mod.calibration_note_override(
            None, req_ids=["calib-cleanup-2"],
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"calib-cleanup-2": 1}), inputs={})
        after_logits = torch.load(logits, weights_only=False)
        self.assertEqual(force_log.read_text(), before_force, "不得追加强制日志")
        self.assertEqual(len(after_logits), len(before_logits), "不得追加 logits 记录")
        self.assertEqual([r["step"] for r in after_logits], [r["step"] for r in before_logits])
        self.assertEqual(trace.read_text(), before_trace, "不得追加 trace 记录")

    def test_logits_step_counter_is_independent_from_force(self) -> None:
        """`logits_step` 与 `force_step` 互不影响：每个都从 1 起、各自独立递增。"""
        logits = self.dir / "logits.pt"
        force_log = self.dir / "force.jsonl"
        self.write_arm(tokens=[[[101]]], force_log=force_log, logits=logits)
        for _ in range(2):
            self.assertTrue(self.mod.calibration_capture_logits(
                torch.zeros(1, 8), SimpleNamespace(req_ids=["r-main"])))
        records = torch.load(logits, weights_only=False)
        self.assertEqual([r["step"] for r in records], [1, 2], "只调 logits 捕获时计数从 1 起")
        changed, main = self.force(["r-main"], [1], [[9]])
        self.assertTrue(changed)
        self.assertEqual(main.tolist(), [[101]], "logits 捕获不得消耗强制轨迹（仍是第 1 步）")
        forced = [json.loads(line) for line in force_log.read_text().splitlines()]
        self.assertEqual([r["step"] for r in forced], [1], "强制步号不受 logits 计数影响")
        self.assertTrue(self.mod.calibration_capture_logits(
            torch.zeros(1, 8), SimpleNamespace(req_ids=["r-main"])))
        records = torch.load(logits, weights_only=False)
        self.assertEqual([r["step"] for r in records], [1, 2, 3], "强制不得推进 logits 计数")

    # --- 强制轨迹 --------------------------------------------------------- #

    def test_force_tokens_replaces_in_place_and_logs_raw(self) -> None:
        force_log = self.dir / "force.jsonl"
        self.write_arm(tokens=[[[101]], [[102]]], force_log=force_log)
        sampled = self._sampled([[7]])
        pointer = sampled.data_ptr()
        self.assertTrue(self.mod.calibration_force_tokens(
            SimpleNamespace(sampled_token_ids=sampled), ["r-main"], torch.tensor([1])))
        self.assertEqual(sampled.tolist(), [[101]])
        self.assertEqual(sampled.data_ptr(), pointer, "必须原地替换：worker 历史与宿主看到同一块内存")
        second = self._sampled([[9]])
        self.mod.calibration_force_tokens(
            SimpleNamespace(sampled_token_ids=second), ["r-main"], torch.tensor([1]))
        self.assertEqual(second.tolist(), [[102]])
        records = [json.loads(line) for line in force_log.read_text().splitlines()]
        self.assertEqual([r["step"] for r in records], [1, 2])
        self.assertEqual(records[0]["req_id"], "r-main")
        self.assertEqual(records[0]["raw_sampled"], [[7]], "原始采样必须与 forced 分列留存")
        self.assertEqual(records[0]["forced"], [101])
        self.assertEqual(records[1]["raw_sampled"], [[9]])
        self.assertIn("D2H", records[0]["sync_note"], "必须标注该钩子含同步（校准专用）")

    def test_prefill_discard_does_not_consume_trajectory(self) -> None:
        """`num_sampled == 0` 的未完成 prefill 步：不消耗轨迹、不改写、不写日志。"""
        force_log = self.dir / "force.jsonl"
        self.write_arm(tokens=[[[101]]], force_log=force_log)
        changed, discarded = self.force(["r-main"], [0], [[7]])
        self.assertFalse(changed)
        self.assertEqual(discarded.tolist(), [[7]])
        self.assertFalse(force_log.exists(), "丢弃步不得写强制日志")
        changed, main = self.force(["r-main"], [1], [[7]])
        self.assertTrue(changed, "随后真正的消费步才消耗轨迹第 1 步")
        self.assertEqual(main.tolist(), [[101]])
        records = [json.loads(line) for line in force_log.read_text().splitlines()]
        self.assertEqual([r["step"] for r in records], [1])

    def test_exhausted_trajectory_raises_only_for_bound_request(self) -> None:
        self.write_arm(tokens=[[[101]]])
        _changed, main = self.force(["r-main"], [1], [[7]])
        self.assertEqual(main.tolist(), [[101]])
        with self.assertRaises(RuntimeError):
            self.force(["r-main"], [1], [[7]])
        changed, other = self.force(["r9"], [1], [[5]])
        self.assertFalse(changed, "轨迹用尽只对被绑定请求报错")

    def test_force_tokens_rejects_shape_mismatch_and_flat_format(self) -> None:
        self.write_arm(tokens=[[[101, 202]]])
        sampled = self._sampled([[7]])
        with self.assertRaises(RuntimeError):
            self.force(["r-main"], [1], [[7]])  # 轨迹 2 个 token vs 采样 1 个
        self.assertEqual(sampled.tolist(), [[7]], "形状不符时不得写一半")
        # 换一份**全新**模块状态再验格式错误（不依赖控制文件的 mtime 粒度）
        self.mod = load_adapter()
        flat = self.write_arm(name="flat.json", tokens=[101])
        self.assertTrue(flat.exists())
        with self.assertRaises(RuntimeError) as ctx:
            self.force(["r-main"], [1], [[7]])
        self.assertIn("tokens[step][row]", str(ctx.exception))
        self.assertEqual(sampled.tolist(), [[7]])

    # --- 完整 logits ------------------------------------------------------ #

    def test_capture_logits_saves_full_vocab_fp32(self) -> None:
        logits_path = self.dir / "logits.pt"
        self.write_arm(logits=logits_path)
        logits = torch.randn(2, 151936, dtype=torch.bfloat16)
        self.assertTrue(self.mod.calibration_capture_logits(logits, SimpleNamespace(req_ids=["r-main"])))
        self.mod.calibration_capture_logits(logits[:1], SimpleNamespace(req_ids=["r-main"]))
        records = torch.load(logits_path, weights_only=False)
        self.assertEqual([r["step"] for r in records], [1, 2])
        self.assertEqual(tuple(records[0]["logits"].shape), (2, 151936), "必须是**完整**词表，不是 top-k")
        self.assertEqual(records[0]["logits"].dtype, torch.float32)
        self.assertEqual(records[1]["shape"], [1, 151936])
        self.assertEqual(records[0]["req_ids"], ["r-main"])

    # --- 覆写 trace ------------------------------------------------------- #

    def test_override_trace_records_inputs_and_noop_without_arm(self) -> None:
        self.mod.calibration_note_override(
            _override(), req_ids=["r-main"],
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r-main": 1}),
            inputs={"t": torch.zeros(2)})
        self.assertEqual(list(self.dir.iterdir()), [], "未 armed 时不得写 trace")

        trace = self.dir / "steps.jsonl"
        self.write_arm(trace=trace)
        table = torch.zeros((2, 3), dtype=torch.int32)
        self.mod.calibration_note_override(
            _override(), req_ids=["r-main"],
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r-main": 1}),
            inputs={"fa_group_table": table})
        record = json.loads(trace.read_text().splitlines()[0])
        self.assertEqual(record["override"]["seqused_k"], [3137])
        self.assertEqual(record["override"]["visible_logical_blocks"]["0"], [0, 5, 6, 7, 8])
        self.assertEqual(record["inputs"]["fa_group_table"]["data_ptr"], table.data_ptr())
        self.assertIn("version", record["inputs"]["fa_group_table"])
        self.assertIn("stream", record)
        self.assertEqual(table.tolist(), [[0, 0, 0], [0, 0, 0]], "trace 不得改写入参")


class FakeFlashAttentionImpl:
    """最小真实契约替身：ABC 普通方法（非 nn.Module）、持有真实 `impl.scale`。"""

    def __init__(self, *, scale: float = 0.125) -> None:
        self.scale = scale

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        return query + key.sum(dim=0, keepdim=True) + value.sum(dim=0, keepdim=True)


class FakeModel:
    """镜像真实调用形态：每步一次 `forward(positions=...)`，每层一次 `impl.forward`。

    prefill 步 q_len=prompt_len，之后每步 1 个 token；`positions` 是**真实绝对位置**，
    与 `input_batch.positions` 同源（pin `worker/gpu/model_runner.py:1701-1702`）。
    """

    def __init__(self, impls, *, prompt_len: int = 3, num_heads: int = 4, head_dim: int = 8,
                 marker: object | None = None) -> None:
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(impl=impl)) for impl in impls]
        self.calls = 0
        self.prompt_len = prompt_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = torch.bfloat16
        self.marker = marker

    def forward(self, *args, **kwargs):
        positions = kwargs["positions"]
        q_len = int(positions.shape[0])
        self.calls += 1
        q = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        k = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        v = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        return [layer.self_attn.impl.forward(None, q, k, v, None, self.marker, torch.zeros_like(q))
                for layer in self.layers]


class LayerCaptureTest(unittest.TestCase):
    """层观测必须按 pin 的**真实契约**接入，并且只覆盖 armed 的目标请求。

    真实位置来自 model inputs 的 `positions`（不是 `range(q_len)`）、dtype 来自 `query.dtype`、
    scale 来自 `impl.scale`；元数据按层分键；每步逐步落盘。
    """

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _armed(self, model, impls, *, prompt_len: int = 3, stream: bool = True):
        capture = self.driver.LayerCapture(
            expected_layers={index: f"language_model.model.layers.{index}.self_attn.attn"
                             for index in range(len(impls))}
        )
        for index, impl in enumerate(impls):
            capture.wrap_impl(index, impl)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture" if stream else self.dir / "unused", prompt_len=prompt_len)
        return capture

    def test_steps_layers_positions_and_metadata_are_real(self) -> None:
        impls = [FakeFlashAttentionImpl(), FakeFlashAttentionImpl(scale=0.25)]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        model.forward(positions=torch.tensor([3], dtype=torch.int64))
        capture.disarm()
        model.forward(positions=torch.tensor([99], dtype=torch.int64))  # 窗口外（disarmed）

        self.assertEqual(sorted(capture.records), [1, 2], "只记录目标请求窗口内的步")
        for step in (1, 2):
            self.assertEqual(sorted(capture.records[step]), [0, 1], "每步每层都要有记录")
        self.assertEqual(capture.records[1][0]["q"].shape[0], 3)
        self.assertEqual(capture.records[2][0]["q"].shape[0], 1)
        self.assertEqual(capture.positions[1]["positions"], [0, 1, 2], "prefill 用真实绝对位置")
        self.assertEqual(capture.positions[2]["positions"], [3], "decode 位置来自真实 model inputs")
        self.assertEqual(capture.records[2][0]["device_dtype"], "torch.bfloat16", "设备真实 dtype（非统一 fp32 声明）")
        self.assertEqual(capture.records[2][0]["scale"], 0.125, "scale 必须是 impl.scale")
        self.assertEqual(capture.records[2][1]["scale"], 0.25, "元数据按层分键，不被最后一层覆盖")
        self.assertEqual(capture.decode_steps(), [1], "选定 decode 步清单可查（i = 生成 token 序号）")

    def test_dump_capture_export_keys_for_decode_history(self) -> None:
        impls = [FakeFlashAttentionImpl(), FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        for position in (3, 4, 5):
            model.forward(positions=torch.tensor([position], dtype=torch.int64))
        capture.disarm()
        target = self.dir / "layers.npz"
        self.driver.dump_capture(capture, target)

        import numpy as np

        data = np.load(target)
        self.assertIn("k_prefill_L0", data)
        self.assertIn("v_prefill_L0", data)
        self.assertIn("q_step1_L0", data)
        self.assertIn("out_step1_L0", data)
        self.assertEqual(tuple(data["k_prefill_L0"].shape), (4, 3, 8))  # [KVH, prompt_len, D]
        self.assertEqual(tuple(data["positions_step1"].shape), (3,))
        self.assertEqual(data["decode_steps"].tolist(), [1, 2, 3], "选定 decode 步清单写进捕获（i = 生成 token 序号）")
        self.assertEqual(data["decode_forwards"].tolist(), [2, 3, 4], "同一批步对应的内部 forward 步号")
        for index, position in ((1, 3), (2, 4), (3, 5)):
            self.assertIn(f"decode_q_step{index}_L0", data)
            self.assertIn(f"decode_out_step{index}_L0", data)
            self.assertIn(f"k_current_step{index}_L0", data, "每个 decode 步都要有当前写入 token 的 K")
            self.assertIn(f"v_current_step{index}_L0", data)
            self.assertEqual(data[f"decode_pos_step{index}"].dtype, np.int64)
            self.assertEqual(data[f"decode_pos_step{index}"].tolist(), [position],
                             "decode 位置必须是真实绝对位置（用于拼接 canonical 历史）")
            self.assertEqual(tuple(data[f"k_current_step{index}_L0"].shape), (4, 1, 8))
        self.assertNotIn("q_step2_L0", data, "decode 步不得冒充 prefill 步")
        for key in ("scale", "scale_source", "stored_dtype", "num_heads", "num_kv_heads",
                    "head_dim", "prompt_len", "layer_index", "capture_dtype_L0", "scale_L0",
                    "layer_name_L0", "capture_dtype_L1", "layer_name_L1", "decode_note"):
            self.assertIn(key, data, f"oracle 合同要求元数据 {key}")
        self.assertEqual(float(data["scale_L1"]), 0.125)
        self.assertEqual(str(data["scale_source"]), "impl.scale")
        self.assertEqual(str(data["capture_dtype_L0"]), "torch.bfloat16")
        self.assertEqual(str(data["stored_dtype"]), "float32")

    def test_each_step_is_flushed_to_disk_immediately(self) -> None:
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        first = self.dir / "capture" / "forward1.npz"
        self.assertTrue(first.exists(), "每步结束必须立刻落盘（异常也留有可审证据）")
        model.forward(positions=torch.tensor([3], dtype=torch.int64))
        self.assertTrue((self.dir / "capture" / "forward2.npz").exists())
        import numpy as np

        step2 = np.load(self.dir / "capture" / "forward2.npz")
        self.assertEqual(step2["decode_pos_step1"].tolist(), [3], "decode 键 i = 生成 token 序号")
        self.assertIn("k_current_step1_L0", step2)

    def test_capture_requires_real_scale_and_positions(self) -> None:
        """缺 `impl.scale` 或缺真实 `positions` ⇒ 明确报错，不得用推导值/`range(q_len)` 顶替。"""
        class NoScaleImpl(FakeFlashAttentionImpl):
            def __init__(self) -> None:
                super().__init__()
                del self.scale

        capture = self.driver.LayerCapture()
        with self.assertRaises(RuntimeError) as ctx:
            capture.wrap_impl(0, NoScaleImpl())
        self.assertIn("scale", str(ctx.exception))

        model = FakeModel([FakeFlashAttentionImpl()], prompt_len=3)
        capture = self.driver.LayerCapture()
        capture.wrap_impl(0, model.layers[0].self_attn.impl)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture", prompt_len=3)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward()  # 真实路径里 model inputs 一定带 positions
        self.assertIn("positions", str(ctx.exception))

    def test_positions_must_be_contiguous_absolute(self) -> None:
        """位置不是本步 token 的真实连续绝对位置时拒绝记录（不静默接受错误来源）。"""
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        with self.assertRaises(RuntimeError):
            model.forward(positions=torch.tensor([5, 6, 7], dtype=torch.int64))  # prefill 起点 != 0

    def test_other_request_window_is_not_captured(self) -> None:
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self.driver.LayerCapture(expected_layers={0: "layer0"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_model(model)
        model.forward(positions=torch.tensor([9], dtype=torch.int64))  # 未 armed：warmup 步
        self.assertEqual(capture.records, {}, "未 armed 的步不得进捕获")
        capture.arm("r-main", self.dir / "capture", prompt_len=3)
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        capture.disarm()
        model.forward(positions=torch.tensor([3], dtype=torch.int64))  # cleanup 步
        self.assertEqual(sorted(capture.records), [1], "只保留目标请求的步")

    def test_find_fa_layers_uses_runner_attn_groups_layer_names(self) -> None:
        """真实层级是嵌套的（`language_model.model.layers.*.self_attn.attn`），
        层集合事实来自 runner.attn_groups 的 `layer_names` —— 不能靠硬编码 `.layers`。"""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        class FakeFlashAttnImplStub:
            @staticmethod
            def forward(layer, query, key, value, kv_cache, attn_metadata, output, **kw):
                return query

        class FakeAttn(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.impl = FakeFlashAttnImplStub()

        root = torch.nn.Module()
        layers = torch.nn.ModuleDict()
        for i in range(2):
            attn_holder = torch.nn.Module()
            attn_holder.add_module("attn", FakeAttn())
            layer = torch.nn.Module()
            layer.add_module("self_attn", attn_holder)
            layers[str(i)] = layer
        inner = torch.nn.Module()
        inner.add_module("layers", layers)
        lm = torch.nn.Module()
        lm.add_module("model", inner)
        root.add_module("language_model", lm)

        fa_group = SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(block_size=784, num_kv_heads=4, head_size=128, dtype=torch.bfloat16),
            layer_names=["language_model.model.layers.1.self_attn.attn"],
        )
        gdn_group = SimpleNamespace(
            kv_cache_spec=MambaSpec(block_size=784, shapes=(), dtypes=()),
            layer_names=["language_model.model.layers.0.self_attn.attn"],
        )
        runner = SimpleNamespace(attn_groups=[[gdn_group], [fa_group]], model=root)
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner))
            )
        )
        model, found = self.driver.find_fa_layers(llm)
        self.assertIs(model, root)
        self.assertEqual([name for name, _ in found], ["language_model.model.layers.1.self_attn.attn"],
                         "必须只挑出 FullAttentionSpec 组里的真实层，且用真实嵌套名解析")
        self.assertIsInstance(found[0][1], FakeFlashAttnImplStub)

    def test_find_fa_layers_fails_loudly_without_matching_module(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        root = torch.nn.Module()
        root.add_module("layers", torch.nn.Module())
        group = SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(block_size=784, num_kv_heads=4, head_size=128, dtype=torch.bfloat16),
            layer_names=["does.not.exist"],
        )
        runner = SimpleNamespace(attn_groups=[[group]], model=root)
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner))
            )
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.driver.find_fa_layers(llm)
        self.assertIn("named_modules", str(ctx.exception))

    def test_dump_capture_rejects_missing_prefill_step(self) -> None:
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        del capture.positions[1]
        with self.assertRaises(RuntimeError):
            self.driver.dump_capture(capture, self.dir / "layers.npz")

    def test_accounting_two_token_request_passes_for_both_arm_types(self) -> None:
        """回归防线（本地复核指出）：单次 prefill 已采出 g0 ⇒ 2 个 token = 1 次 decode 步。

        覆盖两种臂：
        1) `original` 型：无控制文件、无强制钩子 —— 宿主消费计数 2、decode 捕获 1 ⇒ 门禁**通过**；
        2) `patched/forced` 型：同样步数下钩子计数 == 宿主计数 ⇒ 通过；
        并显式断言「拿 `force_step` 取代宿主计数来判 original 型」**必然失败**（防回退）。
        """
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture = self._armed(model, impls)
        model.forward(positions=torch.arange(3, dtype=torch.int64))  # prefill 消费步 → g0
        model.forward(positions=torch.tensor([3], dtype=torch.int64))  # 唯一一次 decode → g1
        capture.disarm()
        tokens = [1001, 1002]

        host_only = self.driver.capture_accounting(capture, tokens, consumption_steps=2)
        self.assertEqual(host_only["problems"], [], "original 型（仅宿主计数）必须通过")
        self.assertEqual(host_only["consumption_steps_host"], 2)
        self.assertEqual(host_only["decode_steps"], [1])
        self.assertEqual(host_only["prefill_steps"], [1])
        self.assertNotEqual(len(host_only["decode_steps"]), len(tokens),
                            "旧断言「decode 步数 == len(tokens)」在本场景必错 ⇒ 已改为按宿主消费计数核对")

        forced = self.driver.capture_accounting(capture, tokens, consumption_steps=2, hook_steps=2,
                                                hook_source="force.jsonl")
        self.assertEqual(forced["problems"], [], "forced 型（钩子计数与宿主一致）必须通过")
        divergent = self.driver.capture_accounting(capture, tokens, consumption_steps=2, hook_steps=1,
                                                   hook_source="force.jsonl")
        self.assertTrue(any("分叉" in problem for problem in divergent["problems"]))

        # 反例 1：original 型下若用 force_step（不存在 ⇒ 0/不可用）代替宿主计数 ⇒ 必须失败
        blind = self.driver.capture_accounting(capture, tokens, consumption_steps=0,
                                               hook_steps=None, hook_source=None)
        self.assertTrue(any("宿主消费步数" in problem for problem in blind["problems"]))
        # 反例 2：宿主计数不可观察 ⇒ 不得静默当成功
        unknown = self.driver.capture_accounting(capture, tokens, consumption_steps=None)
        self.assertTrue(any("不可观察" in problem for problem in unknown["problems"]))
        # 反例 3：多要一次 decode（把宿主计数当 3）也必须被拒
        wrong = self.driver.capture_accounting(capture, tokens, consumption_steps=3)
        self.assertTrue(any("decode 捕获步数" in problem for problem in wrong["problems"]))

    def test_phase_marker_disagreement_is_rejected(self) -> None:
        """prefill/decode 由真实位置判定，并与 attn_metadata 真实标记互核：不一致即报错。"""
        marker = SimpleNamespace(num_prefill_reqs=0, num_decode_reqs=1, num_prefill_tokens=0,
                                 num_decode_tokens=1, max_query_len=1)
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3, marker=marker)
        capture = self._armed(model, impls)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))  # 位置说 prefill、标记说 decode
        self.assertIn("相位判定不一致", str(ctx.exception))

    def test_mixed_prefill_decode_marker_is_rejected(self) -> None:
        marker = SimpleNamespace(num_prefill_reqs=1, num_decode_reqs=1, num_prefill_tokens=3,
                                 num_decode_tokens=1, max_query_len=3)
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3, marker=marker)
        capture = self._armed(model, impls)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("混合批", str(ctx.exception))


class DriverGuardTest(unittest.TestCase):
    """驱动入口的守卫：只写新 run、轨迹必须属于同一 prompt、缺件判据不得静默通过。"""

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_guard", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_output_dir_refuses_to_overwrite_existing_run(self) -> None:
        out = self.dir / "run-1"
        out.mkdir()
        (out / "manifest.json").write_text("{}")
        with self.assertRaises(RuntimeError) as ctx:
            self.driver.prepare_output_dir(out)
        self.assertIn("拒绝覆盖", str(ctx.exception))
        fresh = self.dir / "run-2"
        self.driver.prepare_output_dir(fresh)  # 新目录正常创建
        self.assertTrue(fresh.is_dir())

    def test_trajectory_must_match_prompt(self) -> None:
        path = self.dir / "traj.json"
        prompt_ids = [1, 2, 3, 4]
        path.write_text(json.dumps({
            "tokens": [[[9]]], "prompt_len": 3,
            "prompt_token_ids_sha256": self.driver.sha256_token_ids([1, 2, 3]),
        }))
        with self.assertRaises(RuntimeError) as ctx:
            self.driver.load_trajectory(path, prompt_ids=prompt_ids)
        self.assertIn("另一条 prompt", str(ctx.exception))
        path.write_text(json.dumps({
            "tokens": [[[9]]], "prompt_len": len(prompt_ids),
            "prompt_token_ids_sha256": self.driver.sha256_token_ids(prompt_ids),
        }))
        loaded = self.driver.load_trajectory(path, prompt_ids=prompt_ids)
        self.assertEqual(self.driver.trajectory_token_sequence(loaded), [9])

    def test_force_log_flags_warmup_pollution(self) -> None:
        log = self.dir / "force.jsonl"
        log.write_text("\n".join(json.dumps(record) for record in (
            {"kind": "force_tokens", "step": 1, "req_id": "warmup-1", "req_ids": ["warmup-1"],
             "raw_sampled": [[7]], "forced": [9]},
            {"kind": "force_tokens", "step": 1, "req_id": "r-main", "req_ids": ["r-main", "x"],
             "raw_sampled": [[8]], "forced": [10]},
        )) + "\n")
        report = self.driver.verify_force_log(log, req_id="r-main", trajectory_tokens=[10])
        problems = " | ".join(report["problems"])
        self.assertIn("warmup-1", problems, "必须指出非主请求的记录")
        self.assertIn("混入其它请求", problems)
        self.assertIn("不是从 1 连续", problems)


class FakeRequestOutput:
    def __init__(self, request_id: str, tokens: list[int], finished: bool = False) -> None:
        self.request_id = request_id
        self.outputs = [SimpleNamespace(token_ids=list(tokens), text="")]
        self.finished = finished


class ScriptedEngine:
    """按真实可观察语义脚本化的假引擎：产出 token、abort 从账本移除、正常终结即移除。

    `attnview` 属性**故意缺失**（原版臂形态）：协议状态不可观察，驱动必须明确记录并改用
    scheduler/未完成计数/worker req_ids 替代证据，而不是静默当成功。
    """

    def __init__(self) -> None:
        self.requests: dict[str, dict] = {}
        self.add_calls: list[str] = []
        self.step_calls = 0
        self.abort_calls: list[list[str]] = []
        self.scheduler = SimpleNamespace(requests=self.requests)
        self.engine_core = SimpleNamespace(scheduler=self.scheduler)

    def add_request(self, request_id, prompt, params) -> None:
        self.add_calls.append(str(request_id))
        self.requests[str(request_id)] = {"tokens": 0, "max_tokens": int(params.max_tokens)}

    def step(self):
        outputs = []
        for req_id, state in list(self.requests.items()):
            state["tokens"] += 1
            finished = state["tokens"] >= state["max_tokens"]
            outputs.append(FakeRequestOutput(req_id, [1000 + state["tokens"]], finished=finished))
            if finished:
                del self.requests[req_id]
        self.step_calls += 1
        return outputs

    def abort_request(self, request_ids) -> None:
        self.abort_calls.append([str(r) for r in request_ids])
        for req_id in request_ids:
            self.requests.pop(str(req_id), None)

    def get_num_unfinished_requests(self) -> int:
        return len(self.requests)


class NoopEngine:
    """对 add_request/step/abort 都无动作的引擎：清理验收必须 `ok=False`（旧实现返回 ok=True）。"""

    def add_request(self, request_id, prompt, params) -> None:
        return None

    def step(self):
        return []

    def abort_request(self, request_ids) -> None:
        return None


class CleanupCheckTest(unittest.TestCase):
    """清理验收只看**可观察效果**；无动作引擎必须失败。"""

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_cleanup", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_noop_engine_cleanup_is_failure(self) -> None:
        llm = SimpleNamespace(llm_engine=NoopEngine())
        result = self.driver.run_cleanup_check(
            llm, [1, 2, 3], payload={"protocol": "v1.0", "prompt_len": 3, "enforce_global": True},
            patched=False, out_dir=self.dir)
        self.assertFalse(result["ok"], "对 add_request/step/abort 都无动作的引擎不得判成功")
        self.assertTrue(result["failed_checks"])
        self.assertIn("cancel_req_produced_token", result["failed_checks"])

    def test_scripted_engine_cleanup_passes_with_documented_substitutes(self) -> None:
        engine = ScriptedEngine()
        llm = SimpleNamespace(llm_engine=engine)
        result = self.driver.run_cleanup_check(
            llm, [1, 2, 3], payload={"protocol": "v1.0", "prompt_len": 3, "enforce_global": True},
            patched=False, out_dir=self.dir)
        self.assertEqual(result["failed_checks"], [], f"实际失败项：{result['failed_checks']}")
        self.assertTrue(result["ok"])
        self.assertTrue(engine.abort_calls, "必须真的调用过 abort_request")
        # 协议状态不可观察时必须**明确记录**（不得静默当成功）
        state = result["observations"]["cancel_req_state"]
        self.assertFalse(state["observable"])
        self.assertIn("attnview", str(state["reason"]))
        checks = {c["check"]: c for c in result["checks"]}
        self.assertIn("cancel_req_state_unobservable_documented", checks)
        self.assertTrue(checks["cancel_req_state_unobservable_documented"]["ok"])
        self.assertIn("worker_req_ids_release_evidence_recorded", checks)
        self.assertIn("worker_req_ids_after_new_request", checks)
        self.assertIn("new_req_released_after_finish", checks)
        self.assertIn("normal_req_released_after_finish", checks)


class ChromeTraceSummaryTest(unittest.TestCase):
    """用**已提交的真实 trace** 做纯 CPU 回归：拷贝汇总必须从 `cat=='gpu_memcpy'` 读数。"""

    TRACE = REPO / "evidence/p3-calib/metadata-probe-gate4.chrome.json"

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = load_module_by_path("attnview_probe_under_test", REPO / "tools/p2-calib-metadata-probe.py")

    def test_summary_reads_bytes_and_stream_from_gpu_memcpy(self) -> None:
        summary = self.probe.summarize_chrome_trace(self.TRACE)
        htod = summary["HtoD"]
        self.assertEqual(htod["count"], 2, "候选段应恰好两次 H2D（索引 40B + 标量 4B）")
        self.assertEqual(htod["bytes"], 44, "字节数必须来自 args.bytes，而不是 0")
        self.assertEqual(htod["streams"], [7], "stream 必须来自 args.stream")
        self.assertEqual(htod["pinned_to_device"], 2, "必须是 Pinned→Device（异步 staging 生效）")
        self.assertEqual(summary["DtoH"]["count"], 0, "候选不得把设备数据读回 host")
        self.assertEqual(summary["DtoD"]["count"], 3, "三处设备内拷贝（块表 gather 写回 + 标量行拷贝）")
        self.assertTrue(any("Synchronize" in e["name"] for e in summary["sync_events"]),
                        "profiler 自身收尾同步要单列，便于区分于候选行为")

    def test_summary_is_not_fooled_by_non_memcpy_events(self) -> None:
        """`cpu_op`/`kernel` 等事件没有 bytes/stream，不得被算进拷贝汇总。"""
        summary = self.probe.summarize_chrome_trace(self.TRACE)
        total = summary["HtoD"]["count"] + summary["DtoH"]["count"] + summary["DtoD"]["count"]
        self.assertEqual(total, 5, "只统计 gpu_memcpy（该 trace 共 5 条）")


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

    def test_watchdog_marks_manifest_ended(self) -> None:
        """超时也必须补上 manifest 的 `ended_cst`/`exit_code`（不能只留起时记录）。"""
        script = Path(self.dir, "watchdog_manifest.py")
        manifest = self.dir / "manifest.json"
        manifest.write_text(json.dumps({"schema": "x", "started_cst": "t0", "exit_code": None}))
        script.write_text(
            "import sys, time, json\n"
            "from pathlib import Path\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('drv', " + repr(str(REPO / "tools/p2-calib-run.py")) + ")\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['drv'] = m; spec.loader.exec_module(m)\n"
            "with m.Watchdog(1.0, 'startup', Path(sys.argv[1]), Path(sys.argv[2])):\n"
            "    time.sleep(30)\n"
        )
        proc = __import__("subprocess").run(
            [sys.executable, str(script), self.dir.as_posix(), manifest.as_posix()],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 3)
        data = json.loads(manifest.read_text())
        self.assertEqual(data["exit_code"], 3)
        self.assertTrue(data["ended_cst"], "超时也要写结束时间戳")


if __name__ == "__main__":
    unittest.main()
