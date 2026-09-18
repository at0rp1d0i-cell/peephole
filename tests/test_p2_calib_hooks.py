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

import numpy as np
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

    - `positions` 与真实入口一致可以是 **2-D 批张量**（本机 original 实测 `(3, 1505)`）或 1-D；
    - 同时构造与 pin 同形的 `FlashAttentionMetadata` 关键字段（prefill/decode 计数、`seq_lens`、
      `block_table`），用于真实标记互核与执行视图（canonical/global）证据。
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
        self.last_metadata = None
        #: 仅供反例：让本步声明的 q_len 与 positions 行宽不一致（真实路径不会出现）
        self.q_len_override: int | None = None
        #: 仅供反例：省略 attn_metadata 的 max_query_len（与"未包裹 InputBatch"组合 ⇒ 相位证据不可用）
        self.omit_max_query_len = False

    def metadata_for(self, positions_row, q_len: int):
        """构造与**本 pin 实测**同形的 metadata：计数器默认**全 0（未填充）**，`max_query_len` 可用。"""
        last = int(positions_row[min(q_len, int(positions_row.shape[0])) - 1])
        canonical_seq_len = last + 1
        metadata = SimpleNamespace(
            num_prefill_reqs=0,
            num_decode_reqs=0,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            max_query_len=q_len,
            seq_lens=torch.tensor([canonical_seq_len], dtype=torch.int32),
            max_seq_len=canonical_seq_len,
            num_actual_tokens=q_len,
            block_table=torch.zeros((1, 4), dtype=torch.int32),
        )
        if self.omit_max_query_len:
            del metadata.max_query_len
        if self.marker is not None:  # 反例用：只改相位字段，视图证据仍保持 canonical
            for field in ("num_prefill_reqs", "num_decode_reqs", "num_prefill_tokens",
                          "num_decode_tokens", "max_query_len"):
                if hasattr(self.marker, field):
                    setattr(metadata, field, getattr(self.marker, field))
        return metadata

    def forward(self, *args, **kwargs):
        positions = kwargs["positions"]
        row = positions[0] if positions.dim() == 2 else positions  # 单活跃请求取第 0 行
        q_len = int(self.q_len_override or row.shape[0])
        self.calls += 1
        metadata = self.metadata_for(row, q_len)
        self.last_metadata = metadata
        q = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        k = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        v = torch.randn(q_len, self.num_heads, self.head_dim, dtype=self.dtype)
        return [layer.self_attn.impl.forward(None, q, k, v, None, metadata, torch.zeros_like(q))
                for layer in self.layers]


class FakeRunner:
    """本步 `InputBatch` 的最小替身（相位**权威证据**来源）。

    `prepare_inputs` 每次调用给出本步 InputBatch；`is_prefilling_np[目标行]` 是相位判定的唯一权威证据
    （pin `model_runner.py:1138,1345`）。默认按"首步 prefill、其后 decode"自动翻转；可用
    `is_prefilling` 显式覆盖，或用 `req_ids` 模拟"批次里没有目标请求"（⇒ 证据不可用）。
    """

    def __init__(self, *, is_prefilling: bool | None = None, req_ids=("r-main",),
                 auto_first_prefill: bool = True, logical_positions=None,
                 with_buffers: bool = True) -> None:
        self.is_prefilling = is_prefilling
        self.req_ids = list(req_ids)
        self.auto_first_prefill = auto_first_prefill
        self.prepare_calls = 0
        self.execute_model_state = None
        #: 本次真实的一维逻辑位置缓冲（pin `input_batch.py:29` `InputBuffers.positions`）
        self.logical_positions = list(logical_positions) if logical_positions is not None else None
        self.num_tokens = None
        self.input_buffers = (
            SimpleNamespace(positions=None if logical_positions is None
                            else torch.tensor(list(logical_positions), dtype=torch.int64))
            if with_buffers else None
        )

    def next_step(self, logical_positions):
        """模拟 runner 的每步准备：写入本步一维逻辑位置缓冲（真实缓冲）后再 `prepare_inputs`。"""
        self.logical_positions = [int(v) for v in logical_positions]
        self.num_tokens = len(self.logical_positions)
        self.input_buffers = SimpleNamespace(
            positions=torch.tensor(self.logical_positions, dtype=torch.int64))
        return self.prepare_inputs()

    def prepare_inputs(self, *_args, **_kwargs):
        self.prepare_calls += 1
        if self.is_prefilling is not None:
            prefill = bool(self.is_prefilling)
        else:
            prefill = bool(self.auto_first_prefill and self.prepare_calls == 1)
        num_tokens = int(self.num_tokens or 0) or len(self.logical_positions or ())
        self.last_batch = SimpleNamespace(
            req_ids=list(self.req_ids),
            is_prefilling_np=np.asarray([prefill], dtype=bool),
            num_tokens=num_tokens,
        )
        return self.last_batch


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

    def _armed(self, model, impls, *, prompt_len: int = 3, stream: bool = True,
               runner: "FakeRunner | None" = None):
        """装上捕获与**相位权威证据**（runner.prepare_inputs），返回 (capture, runner)。

        调用方在每步 forward 前调一次 `runner.prepare_inputs()`（与真实 runner 的顺序一致）。
        """
        runner = runner or FakeRunner()
        capture = self.driver.LayerCapture(
            runner=runner,
            expected_layers={index: f"language_model.model.layers.{index}.self_attn.attn"
                             for index in range(len(impls))}
        )
        for index, impl in enumerate(impls):
            capture.wrap_impl(index, impl)
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture" if stream else self.dir / "unused", prompt_len=prompt_len)
        return capture, runner

    def test_steps_layers_positions_and_metadata_are_real(self) -> None:
        impls = [FakeFlashAttentionImpl(), FakeFlashAttentionImpl(scale=0.25)]
        model = FakeModel(impls, prompt_len=3)
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        runner.next_step([3])
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
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        for position in (3, 4, 5):
            runner.next_step([position])
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
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        first = self.dir / "capture" / "forward1.npz"
        self.assertTrue(first.exists(), "每步结束必须立刻落盘（异常也留有可审证据）")
        runner.next_step([3])
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
        capture, runner = self._armed(model, impls)
        runner.next_step([5, 6, 7])  # 一维逻辑位置缓冲起点 != 0（真实来源不合格）
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("不是 0", str(ctx.exception))

    def test_mrope_axes_are_recorded_but_logical_positions_come_from_real_buffer(self) -> None:
        """真实反例回归：`model inputs.positions` 是 **mRoPE 三轴**（本机 original 实测 (3, 1505)），

        而 KV/oracle 用的 **canonical 逻辑位置**必须取自 runner 真实的一维逻辑位置缓冲
        （pin `input_batch.py:29` `InputBuffers.positions`，由 `prepare_pos_seq_lens` 写入）。

        本用例故意让"三轴"与"一维逻辑位置"**不相等**（三轴首轴 = 逻辑位置 + 100）：
        导出必须是逻辑位置，三轴只作记录；用 range(q_len) 或任一轴冒充逻辑位置会被起点/连续性检查拒绝。
        """
        prompt_len = 1505
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=prompt_len)
        runner = FakeRunner()
        capture = self.driver.LayerCapture(
            runner=runner, expected_layers={0: "language_model.model.layers.3.self_attn.attn"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture", prompt_len=prompt_len)

        logical = list(range(prompt_len))
        axes = torch.zeros(3, prompt_len, dtype=torch.int64)
        axes[0] = torch.arange(prompt_len, dtype=torch.int64) + 100  # 三轴与逻辑位置不同
        axes[1] = torch.arange(prompt_len, dtype=torch.int64) + 7
        runner.next_step(logical)
        model.q_len_override = prompt_len
        model.forward(positions=axes)

        step = capture.positions[1]
        self.assertEqual(step["positions"], logical, "导出必须是 canonical 逻辑位置（真实一维缓冲）")
        self.assertEqual(step["positions_source"],
                         "runner.input_buffers.positions[:num_tokens]（真实一维逻辑位置缓冲）")
        self.assertEqual(step["mrope_axes_shape"], [3, prompt_len], "mRoPE 三轴形状只作记录")
        self.assertIsNotNone(step["mrope_axes_head"])
        self.assertNotEqual(step["positions"][0], int(axes[0][0]),
                            "三轴不得冒充逻辑位置（本例三轴 = 逻辑位置 + 100）")

        # 反例：一维缓冲换成"三轴首轴"的值 ⇒ 起点/连续性检查必须拒绝
        capture2 = self.driver.LayerCapture(
            runner=runner, expected_layers={0: "language_model.model.layers.3.self_attn.attn"})
        capture2.wrap_impl(0, impls[0])
        capture2.wrap_runner_inputs(runner)
        capture2.wrap_model(model)
        capture2.arm("r-main", self.dir / "capture2", prompt_len=prompt_len)
        runner.next_step([v + 100 for v in range(prompt_len)])  # 冒充来源：三轴首轴
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=axes)
        self.assertIn("不是 0", str(ctx.exception))

    def test_logical_positions_require_consistent_real_quantities(self) -> None:
        """R2 §B：三者必须齐备且一致 —— 本步 `InputBatch.num_tokens` ↔ 逻辑位置缓冲长度 ↔ 捕获 q_len。

        缺失/不一致一律报错（**不回退 q_len、不取 max、不用 range(q_len)／三轴**），错误信息里要写明三个实际值。
        """
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)

        def build(runner, name):
            capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "L0"})
            capture.wrap_impl(0, impls[0])
            capture.wrap_runner_inputs(runner)
            capture.wrap_model(model)
            capture.arm("r-main", self.dir / name, prompt_len=3)
            return capture

        # ① 缓冲存在但长度不足 ⇒ 报错（写明三个值）
        runner = FakeRunner()
        build(runner, "capture-short")
        runner.next_step([0, 1])
        runner.last_batch.num_tokens = 3  # 真实字段说 3（与 q_len 一致），但缓冲容量只有 2
        runner.input_buffers.positions = torch.zeros(2, dtype=torch.int64)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))
        message = str(ctx.exception)
        self.assertIn("num_tokens=3", message)
        self.assertIn("buffer_len=2", message)
        self.assertIn("q_len=3", message)
        self.assertIn("不足以覆盖本步 token", message)

        # ② num_tokens 缺失 ⇒ 报错（不得回退 q_len）
        impls2 = [FakeFlashAttentionImpl()]
        model2 = FakeModel(impls2, prompt_len=3)
        runner2 = FakeRunner()
        capture2 = self.driver.LayerCapture(runner=runner2, expected_layers={0: "L0"})
        capture2.wrap_impl(0, impls2[0])
        capture2.wrap_runner_inputs(runner2)
        capture2.wrap_model(model2)
        capture2.arm("r-main", self.dir / "capture-no-num-tokens", prompt_len=3)
        runner2.next_step([0, 1, 2])
        runner2.last_batch.num_tokens = None  # 真实字段缺失
        with self.assertRaises(RuntimeError) as ctx2:
            model2.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("num_tokens 缺失或非正", str(ctx2.exception))
        self.assertIn("q_len=3", str(ctx2.exception))

        # ③ num_tokens != q_len ⇒ 报错（防回退：取 max(num_tokens,q_len) 或回退 q_len 都不会报错）
        impls3 = [FakeFlashAttentionImpl()]
        model3 = FakeModel(impls3, prompt_len=3)
        runner3 = FakeRunner()
        capture3 = self.driver.LayerCapture(runner=runner3, expected_layers={0: "L0"})
        capture3.wrap_impl(0, impls3[0])
        capture3.wrap_runner_inputs(runner3)
        capture3.wrap_model(model3)
        capture3.arm("r-main", self.dir / "capture-mismatch-num-tokens", prompt_len=3)
        runner3.next_step([0, 1, 2])
        runner3.last_batch.num_tokens = 2  # 与 q_len=3 不一致（此时缓冲容量 4096 充足）
        with self.assertRaises(RuntimeError) as ctx3:
            model3.forward(positions=torch.arange(3, dtype=torch.int64))
        message3 = str(ctx3.exception)
        self.assertIn("num_tokens=2", message3)
        self.assertIn("q_len=3", message3)
        self.assertIn("拒绝取 max 或回退", message3)
        self.assertGreater(max(2, 3), 3 - 1, "取 max(2,3)=3 会掩盖不一致 ⇒ 该写法被禁止")

        # ④ 缓冲缺失 ⇒ 报错
        impls4 = [FakeFlashAttentionImpl()]
        model4 = FakeModel(impls4, prompt_len=3)
        runner4 = FakeRunner(with_buffers=False)
        capture4 = self.driver.LayerCapture(runner=runner4, expected_layers={0: "L0"})
        capture4.wrap_impl(0, impls4[0])
        capture4.wrap_runner_inputs(runner4)
        capture4.wrap_model(model4)
        capture4.arm("r-main", self.dir / "capture-no-buffer", prompt_len=3)
        runner4.prepare_inputs()
        with self.assertRaises(RuntimeError) as ctx4:
            model4.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("取不到 runner.input_buffers.positions", str(ctx4.exception))

    def test_logical_positions_take_valid_prefix_of_larger_buffer(self) -> None:
        """容量 > num_tokens 属正常：只取本步**有效前缀**（前 num_tokens 个），尾部残留不得进参考。"""
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        runner = FakeRunner()
        capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "L0"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture-prefix", prompt_len=3)
        runner.next_step([0, 1, 2])
        # 尾部（容量剩余部分）塞入明显异常值：若被误当有效位置会立刻破坏连续/起点检查
        runner.input_buffers.positions[len(runner.logical_positions):] = 999
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        step = capture.positions[1]
        self.assertEqual(step["positions"], [0, 1, 2], "只取有效前缀")
        self.assertNotIn(999, step["positions"])
        self.assertGreaterEqual(int(runner.input_buffers.positions.shape[0]), 3, "容量 ≥ num_tokens 即可")

    def test_accounting_requires_canonical_global_view(self) -> None:
        """执行视图必须是 canonical/global：FA metadata 的 seq_lens[0] 必须 == 本步最后位置 + 1。"""
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
        model.forward(positions=torch.arange(3, dtype=torch.int64))
        capture.disarm()
        ok = self.driver.capture_accounting(capture, [1001], consumption_steps=1)
        self.assertEqual(ok["problems"], [])
        self.assertEqual(ok["view_evidence"]["1"]["seq_lens_first"], 3, "canonical 读长度上证据")
        # 反例：受限读视图（seq_lens 被缩短）⇒ 必须判失败
        capture.positions[1]["view"]["seq_lens_first"] = 1
        limited = self.driver.capture_accounting(capture, [1001], consumption_steps=1)
        self.assertTrue(any("canonical/global" in problem for problem in limited["problems"]))
        # 反例：视图证据不可观察 ⇒ 不得静默当成功
        capture.positions[1]["view"] = {"observable": False, "reason": "无 attn_metadata"}
        blind = self.driver.capture_accounting(capture, [1001], consumption_steps=1)
        self.assertTrue(any("无法取得 FA metadata 读长度证据" in problem for problem in blind["problems"]))

    def test_other_request_window_is_not_captured(self) -> None:
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3)
        runner = FakeRunner()
        capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "layer0"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        model.forward(positions=torch.tensor([9], dtype=torch.int64))  # 未 armed：warmup 步
        self.assertEqual(capture.records, {}, "未 armed 的步不得进捕获")
        capture.arm("r-main", self.dir / "capture", prompt_len=3)
        runner.next_step([0, 1, 2])
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
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
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
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
        model.forward(positions=torch.arange(3, dtype=torch.int64))  # prefill 消费步 → g0
        runner.next_step([3])
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

    def test_single_token_prefill_via_authoritative_evidence(self) -> None:
        """相位**唯一权威证据** = 本步 `InputBatch.is_prefilling_np[目标行]`（pin `model_runner.py:1138,1345`）。

        反例回归：单 token prefill（`is_prefilling_row=True` 且本步 q_len==1 且四个计数器全 0）
        ⇒ 必须判 **prefill 并通过**；若按 `q_len==1` 分类会判成 decode（明令禁止）⇒ 显式断言防回退。
        """
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=1)  # prompt 只有 1 个 token（合法的单 token prefill）
        runner = FakeRunner(is_prefilling=True)
        capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "L0"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture", prompt_len=1)
        runner.next_step([0])
        model.forward(positions=torch.tensor([0], dtype=torch.int64))  # q_len == 1
        markers = capture.positions[1]["phase_markers"]
        self.assertEqual(capture.positions[1]["phase"], "prefill")
        self.assertEqual(markers["decided_by"], "is_prefilling_np")
        self.assertEqual(markers["is_prefilling_row"], True)
        self.assertFalse(markers["counters_populated"], "全 0 计数器不得作相位证据")
        self.assertIn("全 0", markers["counters_note"])
        self.assertNotEqual(capture.positions[1]["phase"], "decode",
                            "按 q_len==1 会判成 decode ⇒ 必须以 is_prefilling_np 为准（防回退）")

    def test_phase_authoritative_evidence_unavailable_is_error(self) -> None:
        """取不到 `is_prefilling_np[目标行]` ⇒ 直接报错，**不得**用 q_len/max_query_len 兜底。"""
        impls = [FakeFlashAttentionImpl()]

        # (i) 未包裹 `prepare_inputs` ⇒ 本步没有 InputBatch：先撞上"三者一致"的严格校验
        #     （num_tokens 缺失），同样是**明确失败**而不是相位兜底。
        runner = FakeRunner()
        runner.next_step([0, 1, 2])
        model = FakeModel(impls, prompt_len=3)
        capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "L0"})
        capture.wrap_impl(0, impls[0])
        capture.wrap_model(model)  # 故意不 wrap_runner_inputs
        capture.arm("r-main", self.dir / "capture-no-batch", prompt_len=3)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("num_tokens 缺失或非正", str(ctx.exception), "未包裹 prepare_inputs")

        # (ii) 批次里没有目标请求 ⇒ 同样判证据不可用
        impls2 = [FakeFlashAttentionImpl()]
        model2 = FakeModel(impls2, prompt_len=3)
        runner2 = FakeRunner(req_ids=("other-req",))
        capture2 = self.driver.LayerCapture(runner=runner2, expected_layers={0: "L0"})
        capture2.wrap_impl(0, impls2[0])
        capture2.wrap_runner_inputs(runner2)
        capture2.wrap_model(model2)
        capture2.arm("r-main", self.dir / "capture-no-row", prompt_len=3)
        runner2.next_step([0, 1, 2])
        with self.assertRaises(RuntimeError) as ctx2:
            model2.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("相位权威证据不可用", str(ctx2.exception), "批次里没有目标请求")

    def test_phase_evidence_disagreement_and_mixed_counters_are_rejected(self) -> None:
        """证据与真实位置相位不一致 ⇒ 报错；计数器有值且 prefill/decode 并存 ⇒ 报错。"""
        impls = [FakeFlashAttentionImpl()]
        # (1) 证据说 decode、位置说 prefill
        model = FakeModel(impls, prompt_len=3)
        runner = FakeRunner(is_prefilling=False)
        capture = self.driver.LayerCapture(runner=runner)
        capture.wrap_impl(0, impls[0])
        capture.wrap_runner_inputs(runner)
        capture.wrap_model(model)
        capture.arm("r-main", self.dir / "capture-mismatch", prompt_len=3)
        runner.next_step([0, 1, 2])
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("相位判定不一致", str(ctx.exception))

        # (2) 计数器有值且混合 ⇒ 报错（此时计数器是交叉核对证据）
        marker = SimpleNamespace(num_prefill_reqs=1, num_decode_reqs=1, num_prefill_tokens=3,
                                 num_decode_tokens=1)
        impls2 = [FakeFlashAttentionImpl()]
        model2 = FakeModel(impls2, prompt_len=3, marker=marker)
        runner2 = FakeRunner(is_prefilling=True)
        capture2 = self.driver.LayerCapture(runner=runner2)
        capture2.wrap_impl(0, impls2[0])
        capture2.wrap_runner_inputs(runner2)
        capture2.wrap_model(model2)
        capture2.arm("r-main", self.dir / "capture-mixed", prompt_len=3)
        runner2.next_step([0, 1, 2])
        with self.assertRaises(RuntimeError) as ctx2:
            model2.forward(positions=torch.arange(3, dtype=torch.int64))
        self.assertIn("混合批", str(ctx2.exception))

    def test_mixed_prefill_decode_marker_is_rejected(self) -> None:
        marker = SimpleNamespace(num_prefill_reqs=1, num_decode_reqs=1, num_prefill_tokens=3,
                                 num_decode_tokens=1, max_query_len=3)
        impls = [FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=3, marker=marker)
        capture, runner = self._armed(model, impls)
        runner.next_step([0, 1, 2])
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


class PinAssignRequestIdTest(unittest.TestCase):
    """**真实** `InputProcessor.assign_request_id` 调用（不关随机化开关）：

    pin `v1/engine/input_processor.py:262-278` 把外部 id 换成 `f"{external}-{uuid8}"` 的内部 id，
    并把原值写入 `request.external_req_id`。
    """

    def setUp(self) -> None:
        from vllm.v1.engine.input_processor import InputProcessor

        self.assign = InputProcessor.assign_request_id

    def test_assign_request_id_appends_random_suffix(self) -> None:
        class _Stub:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.external_req_id = None

        request = _Stub("calib-main-original-20260918")
        self.assign(request)
        self.assertEqual(request.external_req_id, "calib-main-original-20260918",
                         "原外部 id 存进 external_req_id")
        self.assertNotEqual(request.request_id, request.external_req_id)
        suffix = request.request_id[len("calib-main-original-20260918-"):]
        self.assertEqual(len(suffix), 8, f"pin 追加 8 位随机后缀（实际 {request.request_id!r}）")
        # 再次调用同一对象必须拒绝（pin 显式校验 external_req_id 不得预置）
        with self.assertRaises(ValueError):
            self.assign(request)

    def test_assign_request_id_suffix_is_per_call_random(self) -> None:
        class _Stub:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.external_req_id = None

        first, second = _Stub("ext"), _Stub("ext")
        self.assign(first)
        self.assign(second)
        self.assertNotEqual(first.request_id, second.request_id, "同名外部 id 也得到不同内部 id")


class CumulativeOutputRegressionTest(unittest.TestCase):
    """输出形态回归（R2 §A）：`_merge_step_tokens` **按已知 output_kind 明确分支**，不靠前缀猜。

    pin 默认 `SamplingParams.output_kind=CUMULATIVE`（`sampling_params.py:317`）；四臂显式请求 **DELTA**。
    """

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_cumulative", REPO / "tools/p2-calib-run.py")

    def test_sampling_params_request_delta_against_pin_default(self) -> None:
        from vllm.sampling_params import RequestOutputKind

        self.assertEqual(RequestOutputKind.CUMULATIVE.value, 0, "pin 默认值仍是 CUMULATIVE")
        params = self.driver.sampling_params(8)
        self.assertEqual(params.output_kind, RequestOutputKind.DELTA, "驱动必须显式请求 DELTA")
        self.assertEqual(self.driver.output_mode_of(params), "delta", "合并形态由实际请求派生")

    def test_delta_mode_never_loses_tokens(self) -> None:
        """反例 1：显式 DELTA 下 `seen=[11]` + 行 `[11,12]` ⇒ 必须新增 **2** 个（11、12），不得丢 11。"""
        seen = [11]
        fresh = self.driver._merge_step_tokens(seen, [11, 12], mode="delta")
        self.assertEqual(fresh, [11, 12], "DELTA 整行采纳")
        self.assertEqual(len(fresh), 2)
        # 防回退：若按前缀猜（累计）会只取 [12]，丢掉 11
        guessed_wrong = [11, 12][len(seen):]
        self.assertNotEqual(fresh, guessed_wrong, "按前缀猜会丢 token ⇒ 不得采用该实现")

    def test_cumulative_equal_length_snapshot_adds_nothing(self) -> None:
        """反例 2：显式 CUMULATIVE 的等长重复快照 ⇒ 零新增（不得重复追加）。"""
        seen = [11, 12]
        self.assertEqual(self.driver._merge_step_tokens(seen, [11, 12], mode="cumulative"), [])
        self.assertEqual(self.driver._merge_step_tokens(seen, [11, 12, 13], mode="cumulative"), [13])

    def test_cumulative_prefix_mismatch_and_shorter_row_are_errors(self) -> None:
        """反例 3：累计行前缀不符 ⇒ 报错；行变短 ⇒ 报错。"""
        with self.assertRaises(RuntimeError) as ctx:
            self.driver._merge_step_tokens([11, 12], [11, 99, 13], mode="cumulative")
        self.assertIn("前缀与已见序列不符", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx2:
            self.driver._merge_step_tokens([11, 12], [11], mode="cumulative")
        self.assertIn("小于已见", str(ctx2.exception))
        with self.assertRaises(RuntimeError) as ctx3:
            self.driver._merge_step_tokens([11], [11], mode="guess")
        self.assertIn("未知输出形态", str(ctx3.exception))

    def test_drives_with_delta_mode_are_exact(self) -> None:
        """DELTA 引擎：步内多 token 的行不得被截断为 1 个（反例 1 的端到端形态）。"""
        engine = CumulativeEngine([100, 101], delta=True, row_width=2)  # 一步内两个 token 的 DELTA 行
        drove = self.driver.drive_main_request(engine, "ext-1", max_tokens=2, mode="delta")
        self.assertEqual(drove["tokens"], [100, 101], "DELTA 行 [100,101] 必须整段采纳（不得截成 1 个）")
        self.assertEqual(drove["consuming_steps"], 1)

    def test_cumulative_engine_requires_explicit_mode(self) -> None:
        engine = CumulativeEngine([100, 101, 102])
        drove = self.driver.drive_main_request(engine, "ext-1", max_tokens=3, mode="cumulative")
        self.assertEqual(drove["tokens"], [100, 101, 102], "显式累计模式下只计一次")
        self.assertEqual(drove["consuming_steps"], 3)


class PreflightFailureLandingTest(unittest.TestCase):
    """起时 manifest 之后的**所有**异常都必须有落点（本地复核反例：曾经只剩 source 快照）。

    覆盖：部署态前置校验（`deployment_fingerprint`）失败 ⇒ ①`<out>/manifest.json` 含
    `failures`/`ended_cst`/`exit_code`；②`<out>/error.txt` 含完整原始 traceback；③返回非零。
    """

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_preflight", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_preflight_exception_lands_in_manifest_and_error_txt(self) -> None:
        original_fingerprint = self.driver.deployment_fingerprint
        original_prompt = self.driver.build_prompt
        self.driver.deployment_fingerprint = lambda arm, **kw: (_ for _ in ()).throw(
            RuntimeError("部署态前置校验失败：checkout != post（模拟）"))
        self.driver.build_prompt = lambda: (
            SimpleNamespace(token_ids=(1, 2, 3), rendered="x", segment_spans=[(0, 1)],
                            scaffold=SimpleNamespace(local_window_span=(1, 3), sink_span=(0, 1))),
            {"protocol": "v1.0", "prompt_len": 3, "segment_spans": [[0, 1]],
             "local_window_span": [1, 3], "sink_span": [0, 1]},
        )
        try:
            code = self.driver.main(["--arm", "original", "--out", str(self.dir / "run-x")])
        finally:
            self.driver.deployment_fingerprint = original_fingerprint
            self.driver.build_prompt = original_prompt

        out = self.dir / "run-x"
        self.assertNotEqual(code, 0, "前置校验失败必须非零退出")
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["exit_code"], code)
        self.assertTrue(manifest["ended_cst"], "结束时间戳必须落盘")
        self.assertTrue(any("deployment" in failure or "exception" in failure
                            for failure in manifest.get("failures", [])),
                        f"failures 必须记录前置失败：{manifest.get('failures')}")
        self.assertIsNone(manifest["deployment"], "校验失败时 deployment 保持未写入")
        error = (out / "error.txt").read_text()
        self.assertIn("Traceback", error, "必须保留完整原始 traceback")
        self.assertIn("部署态前置校验失败", error)
        self.assertTrue((out / "source" / Path(self.driver.__file__).name).exists(),
                        "源码快照仍要留下（起时冻结先于校验）")


class DeploymentFingerprintTest(unittest.TestCase):
    """按臂核对的部署身份（本地复核反例：apply 之后 **pin checkout 也是 post**，不得再拿它比 pre）。

    用假部署树覆盖：`installed_root` / `pin_root`（checkout）/ `patch_root`（补丁树 + manifest）。
    """

    EDITED = "v1/core/sched/output.py"
    ADDED = "v1/engine/attnview_engine.py"

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_deploy", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pre = "A" * 8 + "\n"
        self.post = "B" * 8 + "\n"
        self.added = "C" * 8 + "\n"
        self.installed = self.root / "installed"
        self.checkout = self.root / "checkout"
        self.patch = self.root / "patch"
        for base in (self.installed, self.checkout, self.patch / "patched", self.patch / "files/vllm"):
            (base / "v1/core/sched").mkdir(parents=True, exist_ok=True)
            (base / "v1/engine").mkdir(parents=True, exist_ok=True)
        (self.patch / "patched" / self.EDITED).write_text(self.post)
        (self.patch / "files/vllm" / self.ADDED).write_text(self.added)
        (self.patch / "manifest.json").write_text(json.dumps({
            "generated_cst": "2026-09-18 14:30:03 +0800",
            "pin_commit": "deadbeef",
            "edits": {self.EDITED: {"pre_sha256": self.driver.sha256_text(self.pre),
                                    "post_sha256": self.driver.sha256_text(self.post)}},
            "new_files": [{"dest": self.ADDED, "sha256": self.driver.sha256_text(self.added)}],
        }))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _lay(self, *, installed_edited: str | None, installed_added: str | None,
             checkout_edited: str | None, checkout_added: str | None) -> None:
        for root, edited, added in ((self.installed, installed_edited, installed_added),
                                    (self.checkout, checkout_edited, checkout_added)):
            for path, content in ((root / self.EDITED, edited), (root / self.ADDED, added)):
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(content)

    def _fingerprint(self, arm: str) -> dict:
        return self.driver.deployment_fingerprint(
            arm, installed_root=self.installed, pin_root=self.checkout, patch_root=self.patch)

    def test_patched_arm_requires_both_copies_at_post(self) -> None:
        """真实反例：apply 同时改写两副本 ⇒ checkout 与 installed **都**必须 == post 才通过。

        - ① 两副本 == post（真实部署态）⇒ 通过；
        - ② checkout 已改（== post）但 installed != post ⇒ 报错并列出双方实际值与期望值；
        - ③ 防回退：以 `checkout == pre` 判 patched 臂 ⇒ 必然失败（旧口径的误判根源）。
        """
        self._lay(installed_edited=self.post, installed_added=self.added,
                  checkout_edited=self.post, checkout_added=self.added)
        evidence = self._fingerprint("patched-global")
        record = evidence["edited"][self.EDITED]
        self.assertEqual(record["installed_sha256"], self.driver.sha256_text(self.post))
        self.assertEqual(record["checkout_sha256"], self.driver.sha256_text(self.post))
        self.assertEqual(record["expected_for_arm"], self.driver.sha256_text(self.post))
        self.assertTrue(record["checkout_checked"], "patched 臂同样核对 checkout")
        self.assertTrue(evidence["added"][self.ADDED]["checkout_checked"])

        # ② installed 还是 pre（未部署）而 checkout 已是 post ⇒ 拒绝，并列出三值
        self._lay(installed_edited=self.pre, installed_added=None,
                  checkout_edited=self.post, checkout_added=self.added)
        with self.assertRaises(RuntimeError) as ctx:
            self._fingerprint("patched-global")
        message = str(ctx.exception)
        self.assertIn("installed=", message)
        self.assertIn("checkout=", message)
        self.assertIn("manifest.post=", message)

        # ③ checkout 停在 pre（apply 没改到它）⇒ 也必须拒绝；以 pre 判定 patched 臂是错的
        self._lay(installed_edited=self.post, installed_added=self.added,
                  checkout_edited=self.pre, checkout_added=None)
        with self.assertRaises(RuntimeError) as ctx2:
            self._fingerprint("patched-disabled")
        self.assertIn("checkout=", str(ctx2.exception))
        self.assertNotEqual(self.driver.sha256_text(self.pre), self.driver.sha256_text(self.post),
                            "以 checkout == pre 判 patched 臂会放过不一致部署 ⇒ 该写法被禁止")

    def test_original_arm_requires_pin_and_installed_at_pre(self) -> None:
        self._lay(installed_edited=self.pre, installed_added=None,
                  checkout_edited=self.pre, checkout_added=None)
        self._fingerprint("original")
        self._lay(installed_edited=self.post, installed_added=self.added,
                  checkout_edited=self.post, checkout_added=self.added)
        with self.assertRaises(RuntimeError) as ctx:
            self._fingerprint("original")
        self.assertIn("期望双方都 == manifest.pre=", str(ctx.exception))

    def test_manifest_tree_mismatch_is_refused(self) -> None:
        self._lay(installed_edited=self.post, installed_added=self.added,
                  checkout_edited=self.post, checkout_added=self.added)
        (self.patch / "patched" / self.EDITED).write_text("D" * 8 + "\n")
        with self.assertRaises(RuntimeError) as ctx:
            self._fingerprint("patched-global")
        self.assertIn("补丁 manifest 与补丁树不一致", str(ctx.exception))


class RunSourceFreezeTest(unittest.TestCase):
    """运行源冻结（R2）：脚本自身 sha256 + 源码快照落进 run 目录，运行期被改即判失败。"""

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_freeze", REPO / "tools/p2-calib-run.py")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_freeze_copies_script_and_records_hashes(self) -> None:
        frozen = self.driver.freeze_run_source(self.dir)
        script = REPO / "tools/p2-calib-run.py"
        self.assertEqual(frozen["script_sha256"], self.driver.sha256_file(script))
        snapshot = Path(frozen["source_snapshot"])
        self.assertTrue(snapshot.exists(), "源码快照必须落在 run 目录里")
        self.assertEqual(self.driver.sha256_file(snapshot), frozen["script_sha256"])
        self.assertIsNone(self.driver.assert_run_source_unchanged(frozen), "未改动 ⇒ 无漂移")

    def test_modified_script_is_detected(self) -> None:
        copy = self.dir / "p2-calib-run-copy.py"
        copy.write_text((REPO / "tools/p2-calib-run.py").read_text())
        frozen = {"script_path": str(copy), "script_sha256": self.driver.sha256_file(copy)}
        copy.write_text(copy.read_text() + "\n# 运行期被改\n")
        drift = self.driver.assert_run_source_unchanged(frozen)
        self.assertIsNotNone(drift)
        self.assertIn("运行期脚本被修改", drift)


class PinIdBoundaryTest(unittest.TestCase):
    """pin 的 request id 三层边界（真实调用）：

    1. `add_request` 返回**内部 id**（≠ 外部 id，pin `input_processor.py:262-278` 追加 8 位随机后缀）
       ⇒ 绑定/查询/协议状态一律用内部 id；
    2. `step()` 返回的 `RequestOutput.request_id` 是**外部 id**（pin `output_processor.py:380-381`）
       ⇒ 宿主消费计数/轨迹匹配用外部 id；
    3. `abort_request` 默认按**外部 id** 查 external→internal 映射（pin `llm_engine.py:212`、
       `output_processor.py:494-524`）；传内部 id 而不加 `internal=True` **不生效**。
    """

    def setUp(self) -> None:
        self.driver = load_module_by_path("attnview_calib_driver_ids", REPO / "tools/p2-calib-run.py")

    def test_submit_returns_internal_id_and_output_uses_external(self) -> None:
        engine = PinLikeEngine()
        ledger = self.driver.submit_request(engine, "ext-1", [1, 2, 3], SimpleNamespace(max_tokens=1))
        internal = ledger["internal_id"]
        self.assertNotEqual(internal, ledger["external_id"], "pin 会追加随机后缀 ⇒ 内部 id ≠ 外部 id")
        self.assertTrue(ledger["randomized"])
        self.assertEqual(engine.scheduler_requests(), {internal},
                         "调度器账本以内部 id 为键（外部 id 不在其中）")
        outputs = engine.step()
        self.assertEqual([out.request_id for out in outputs], ["ext-1"],
                         "宿主输出用外部 id 归属 ⇒ 宿主消费计数按外部 id 匹配")
        tokens = [int(t) for out in outputs for row in out.outputs for t in row.token_ids]
        self.assertEqual(tokens, [1001], "宿主侧消费 token 序列按外部 id 归属即可读到")

    def test_abort_by_external_works_and_internal_without_flag_does_not(self) -> None:
        engine = PinLikeEngine()
        ledger = self.driver.submit_request(engine, "ext-2", [1, 2, 3], SimpleNamespace(max_tokens=1))
        internal = ledger["internal_id"]
        engine.abort_request([internal])  # 与 pin 一致：默认按外部 id 查表 ⇒ 内部 id 命中不了
        self.assertIn(internal, engine.scheduler_requests(),
                      "不加 internal=True 传内部 id 不应取消成功（证明驱动必须用外部 id）")
        engine.abort_request(["ext-2"])  # 驱动采用的形式：默认 external 路径
        self.assertNotIn(internal, engine.scheduler_requests(), "默认（external）路径应取消同一请求")

    def test_override_trace_semantics_by_arm(self) -> None:
        """global/disabled 臂：无 override 记录算通过；有记录或"必须有记录"的断言都算失败。"""
        path = Path(tempfile.mkdtemp()) / "steps.jsonl"
        report = self.driver.verify_override_trace(path, req_id="r", expect_override_records=False)
        self.assertEqual(report["problems"], [], "global/disabled 臂允许没有 override 记录")
        record = {"kind": "override_step", "step": 1, "req_ids": ["r"], "override": {"group_index": 3}}
        path.write_text(json.dumps(record) + "\n")
        present = self.driver.verify_override_trace(path, req_id="r", expect_override_records=False)
        self.assertTrue(any("不应有受限读视图覆写记录" in problem for problem in present["problems"]),
                        "global 臂出现覆写记录即失败（不得为造 trace 改执行路径）")
        # 反回退：若把语义写成"必须有 override 记录"，真实 global/disabled 运行会被误判失败
        wrong = self.driver.verify_override_trace(path, req_id="r", expect_override_records=True)
        self.assertEqual(wrong["problems"], [])
        path.unlink()
        missing = self.driver.verify_override_trace(path, req_id="r", expect_override_records=True)
        self.assertTrue(missing["problems"], "「必须有 override」在无记录时必然失败 ⇒ 不能用于本阶段三臂")


class CumulativeEngine:
    """每轮返回**累计** token 列表的假引擎（pin 默认 `output_kind=CUMULATIVE` 的返回形态）。

    `delta=True` 时改为返回增量（用于确认两种形态都被正确处理）。
    """

    def __init__(self, tokens: list[int], *, delta: bool = False, request_id: str = "ext-1",
                 row_width: int = 1) -> None:
        self.tokens = list(tokens)
        self.delta = delta
        self.request_id = request_id
        self.row_width = int(row_width)
        self.emitted = 0
        self.rows_for_guard: list[list[list[int]]] = []

    def step(self):
        if self.emitted >= len(self.tokens):
            return []
        if self.delta:
            row = self.tokens[self.emitted: self.emitted + self.row_width]
            self.emitted += len(row)
        else:
            self.emitted += 1
            row = self.tokens[: self.emitted]
        self.rows_for_guard.append([list(row)])
        return [FakeRequestOutput(self.request_id, row, finished=self.emitted >= len(self.tokens))]


class FakeRequestOutput:
    def __init__(self, request_id: str, tokens: list[int], finished: bool = False) -> None:
        self.request_id = request_id
        self.outputs = [SimpleNamespace(token_ids=list(tokens), text="")]
        self.finished = finished


class PinLikeEngine:
    """按 pin 真实语义脚本化的假引擎（三条边界都模拟）：

    - `add_request` 返回**内部 id**（`f"{external}-{8 位十六进制}"`，模拟 `assign_request_id`）；
    - `step()` 的输出用**外部 id**（模拟 `RequestOutput.request_id`）；
    - `abort_request` 默认按**外部 id** 查 external→internal 映射（`internal=True` 才按内部 id）；
    - `scheduler.requests` 以**内部 id** 为键；`attnview` 故意缺失（original 臂形态：协议状态不可观察）。
    """

    def __init__(self) -> None:
        self.requests: dict[str, dict] = {}
        self.external_to_internal: dict[str, str] = {}
        self.internal_to_external: dict[str, str] = {}
        self.add_calls: list[str] = []
        self.abort_calls: list[list[str]] = []
        self.scheduler = SimpleNamespace(requests=self.requests)
        self.engine_core = SimpleNamespace(scheduler=self.scheduler)
        self._suffix = 0

    def scheduler_requests(self) -> set[str]:
        return set(self.requests)

    def add_request(self, request_id, prompt, params) -> str:
        external = str(request_id)
        self._suffix += 1
        internal = f"{external}-{self._suffix:08x}"  # 模拟 pin 的 8 位随机后缀
        self.add_calls.append(external)
        self.external_to_internal[external] = internal
        self.internal_to_external[internal] = external
        self.requests[internal] = {
            "external": external,
            "max_tokens": int(params.max_tokens),
            "tokens": 0,
        }
        return internal

    def step(self):
        outputs = []
        for internal, state in list(self.requests.items()):
            state["tokens"] += 1
            finished = state["tokens"] >= state["max_tokens"]
            outputs.append(
                FakeRequestOutput(state["external"], [1000 + state["tokens"]], finished=finished)
            )
            if finished:
                del self.requests[internal]
                self.external_to_internal.pop(state["external"], None)
                self.internal_to_external.pop(internal, None)
        return outputs

    def abort_request(self, request_ids, internal: bool = False) -> None:
        self.abort_calls.append([str(r) for r in request_ids])
        for request_id in request_ids:
            request_id = str(request_id)
            if internal:
                internal_id = request_id if request_id in self.requests else None
            else:
                internal_id = self.external_to_internal.get(request_id)
            if internal_id is None:
                continue  # 与 pin 一致：id 体系不匹配 ⇒ 什么都不发生
            state = self.requests.pop(internal_id, None)
            if state is not None:
                self.external_to_internal.pop(state["external"], None)
                self.internal_to_external.pop(internal_id, None)

    def get_num_unfinished_requests(self) -> int:
        return len(self.requests)


class NoopEngine:
    """对 add_request/step/abort 都无动作的引擎：清理验收必须 `ok=False`（旧实现返回 ok=True）。

    `add_request` 仍返回一个"看起来可用"的内部 id（模拟 API 层不报错），但从不产出 token、
    也从不登记状态 —— 判据必须落在**可观察效果**上。
    """

    def add_request(self, request_id, prompt, params) -> str:
        return f"{request_id}-00000000"

    def step(self):
        return []

    def abort_request(self, request_ids, internal: bool = False) -> None:
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
        """pin 真实 id 语义下（internal≠external、输出用 external、abort 默认按 external）清理验收必须通过。"""
        engine = PinLikeEngine()
        llm = SimpleNamespace(llm_engine=engine)
        result = self.driver.run_cleanup_check(
            llm, [1, 2, 3], payload={"protocol": "v1.0", "prompt_len": 3, "enforce_global": True},
            patched=False, out_dir=self.dir)
        self.assertEqual(result["failed_checks"], [], f"实际失败项：{result['failed_checks']}")
        self.assertTrue(result["ok"])
        self.assertTrue(engine.abort_calls, "必须真的调用过 abort_request")
        # abort 用的是**外部 id**（pin 默认路径），而查询/绑定用**内部 id**
        cancel_label = result["request_ids"]["cancel"]
        self.assertIn([cancel_label], engine.abort_calls)
        ids = result["observations"]["ids"]
        self.assertNotEqual(ids["cancel"]["internal_id"], ids["cancel"]["external_id"])
        self.assertIn("internal_id", ids["new"])
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
