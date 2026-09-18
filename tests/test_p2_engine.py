"""阶段 05 EngineCore 侧适配层的编排测试（CPU、假对象驱动、不 import vLLM/torch）。

覆盖 SUP-004-R1 §2 的：解析位置与顺序、终结/取消清理、计划字段（含当前 token 的尾长语义）、
不支持配置拒绝、几何必须来自 worker。

驱动方式：按路径加载 `vllm-patch/files/vllm/v1/engine/attnview_engine.py`（该模块只依赖
`attnview` 纯逻辑包），用 `SimpleNamespace` 造假 `SchedulerOutput` / `ModelRunnerOutput` /
`EngineCoreOutputs` 与假 scheduler（带 `kv_cache_manager.get_block_ids`）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from attnview.readview import ReadViewError  # noqa: E402
from attnview.step_plan import AttnViewConfigError, UnsupportedConfig  # noqa: E402

ENGINE_PATH = REPO / "vllm-patch/files/vllm/v1/engine/attnview_engine.py"

B = 784
PROMPT_LEN = 6272  # 8 个满块
FA_BLOCKS = [41, 7, 90, 12, 63, 28, 55, 88, 102]  # 9 块（含当前 token 所在块 8）
GEOMETRY = {
    "kernel_block_size": B,
    "num_kv_groups": 4,
    "fa_group_index": 3,
    "blocks_per_kv_block": 1,
    "max_model_len": 8192,
    "group_block_sizes": [B, B, B, B],
}
TOKEN_TEXT = {1: "<local>", 2: "x", 3: "a"}


def load_engine_module():
    spec = importlib.util.spec_from_file_location("attnview_engine_under_test", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(**over):
    raw = {
        "protocol": "v1.0",
        "prompt_len": PROMPT_LEN,
        "segment_spans": [[784, 2352], [2352, 3920], [3920, 5488]],
        "local_window_span": [4000, 6272],
        "sink_span": [0, 16],
    }
    raw.update(over)
    return {"attnview": raw}


def new_req(req_id: str, extra_args):
    return SimpleNamespace(req_id=req_id, sampling_params=SimpleNamespace(extra_args=extra_args))


def scheduler_output(new_reqs=(), finished=()):
    return SimpleNamespace(scheduled_new_reqs=list(new_reqs), finished_req_ids=set(finished))


def model_output(sampled=((1,),), req_ids=("r1",)):
    return SimpleNamespace(
        req_ids=list(req_ids),
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[list(x) for x in sampled],
    )


def engine_outputs(finish_reason=None, req_id="r1"):
    return {"c": SimpleNamespace(outputs=[SimpleNamespace(request_id=req_id, finish_reason=finish_reason)])}


def empty_outputs():
    return {"c": SimpleNamespace(outputs=[])}


class FakeScheduler:
    def __init__(self, blocks=None):
        self.blocks = blocks or [list(FA_BLOCKS)] * 4
        self.kv_cache_manager = SimpleNamespace(get_block_ids=lambda req_id: self.blocks)


class EngineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_engine_module()

    def make_engine(self, *, geometry_fetcher=None, scheduler=None):
        return self.mod.AttnViewEngine(
            None,
            scheduler or FakeScheduler(),
            geometry_fetcher=geometry_fetcher if geometry_fetcher is not None else (lambda: dict(GEOMETRY)),
            token_text_of=lambda token_id: TOKEN_TEXT.get(int(token_id), "a"),
        )


class RegistrationTest(EngineTestBase):
    def test_only_payload_requests_are_registered(self) -> None:
        engine = self.make_engine()
        registered = engine.register_new_requests(
            scheduler_output([new_req("r1", payload()), new_req("p1", None)])
        )
        self.assertEqual(registered, ["r1"])
        self.assertEqual(engine.registry.active_ids(), ("r1",))
        self.assertNotIn("p1", engine.registry.active_ids())

    def test_payload_with_geometry_is_rejected(self) -> None:
        engine = self.make_engine()
        with self.assertRaises(AttnViewConfigError):
            engine.register_new_requests(
                scheduler_output([new_req("r1", payload(kernel_block_size=B))])
            )
        with self.assertRaises(AttnViewConfigError):
            engine.register_new_requests(scheduler_output([new_req("r2", payload(protocol="v0.9"))]))

    def test_registration_is_idempotent(self) -> None:
        engine = self.make_engine()
        req = new_req("r1", payload())
        engine.register_new_requests(scheduler_output([req]))
        engine.register_new_requests(scheduler_output([req]))
        self.assertEqual(engine.registry.active_ids(), ("r1",))


class ParseAndPlanTest(EngineTestBase):
    def test_parse_advances_state_and_plan_includes_current_token(self) -> None:
        engine = self.make_engine()
        so = scheduler_output([new_req("r1", payload())])
        engine.register_new_requests(so)
        parsed = engine.parse_outputs(model_output(sampled=((1,),)))
        self.assertEqual(parsed, {"r1": 1})
        state = engine.registry.get("r1")
        self.assertEqual(state.mode, "local")
        # 采样出 g0 后：已写长度仍是 prompt_len（g0 尚未写入），下一次 forward 的上界含它
        self.assertEqual(state.written_kv_len, PROMPT_LEN)
        self.assertEqual(state.attention_kv_len_next, PROMPT_LEN + 1)

        # 第二次调用不喂新 token（本用例已在上面显式解析过 g0），避免重复推进生成流
        engine.on_step_outputs(so, model_output(sampled=((),)), empty_outputs())
        plans = engine.pending_plans()
        self.assertIsNotNone(plans)
        plan = plans["r1"]
        # 计划对应的是**下一次** forward：attention_kv_len 含本次将写入的当前 token
        self.assertEqual(plan["attention_kv_len"], PROMPT_LEN + 1)
        self.assertEqual(plan["written_before_step"], PROMPT_LEN)
        self.assertEqual(plan["next_write_position"], PROMPT_LEN + 1)
        self.assertEqual(plan["effect_step"], 1)
        self.assertEqual(plan["mode"], "local")
        self.assertEqual(
            plan["seqused_k"], sum(plan["effective_per_block"]), "seqused_k 必须等于各可见块有效位置数之和"
        )
        self.assertNotEqual(
            plan["seqused_k"], plan["attention_kv_len"], "压缩读取长度不得等于完整历史长度"
        )
        self.assertEqual(plan["effective_per_block"][-1], plan["tail_len"])
        # 载荷不含几何（块大小/表宽由 worker 侧的运行期配置补）
        for forbidden in ("kernel_block_size", "max_width", "num_blocks"):
            self.assertNotIn(forbidden, plan)

    def test_plan_attached_to_next_scheduler_output(self) -> None:
        engine = self.make_engine()
        so = scheduler_output([new_req("r1", payload())])
        engine.register_new_requests(so)
        engine.on_step_outputs(so, model_output(), empty_outputs())
        target = SimpleNamespace()
        engine.attach_plans(target)
        self.assertEqual(target.da_step_plans, engine.pending_plans())
        self.assertIn("r1", target.da_step_plans)

    def test_plain_request_is_never_planned(self) -> None:
        engine = self.make_engine()
        so = scheduler_output([new_req("p1", None)])
        engine.on_step_outputs(so, model_output(sampled=((3,),), req_ids=("p1",)), empty_outputs())
        self.assertIsNone(engine.pending_plans())


class LifecycleTest(EngineTestBase):
    def test_finished_request_is_dropped_and_not_planned(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.on_step_outputs(
            scheduler_output(finished={"r1"}), model_output(sampled=((),)), empty_outputs()
        )
        self.assertNotIn("r1", engine.registry.active_ids())
        self.assertIsNone(engine.pending_plans())

    def test_release_finished_in_step_records_trace_once(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.parse_outputs(model_output())
        released = engine.release_finished_in_step(engine_outputs(finish_reason="stop"))
        self.assertEqual(released, ["r1"])
        self.assertEqual(list(engine.registry.active_ids()), [])
        self.assertTrue(any("finished:stop" in t.get("note", "") for t in engine.traces))

    def test_double_release_is_safe(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.release("r1")
        engine.release("r1")  # 幂等
        self.assertEqual(engine.pending_plans(), None)


class UnsupportedConfigTest(EngineTestBase):
    def test_batch_queue_path_refuses_payload_requests(self) -> None:
        engine = self.make_engine()
        with self.assertRaises(UnsupportedConfig):
            engine.refuse_unsupported_step(scheduler_output([new_req("r1", payload())]))
        # 普通请求不在拒绝范围
        engine.refuse_unsupported_step(scheduler_output([new_req("p1", None)]))

    def test_geometry_must_come_from_worker(self) -> None:
        engine = self.mod.AttnViewEngine(
            None,
            FakeScheduler(),
            geometry_fetcher=None,
            token_text_of=lambda token_id: "a",
        )
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(self.mod.WorkerGeometryUnavailable):
            engine.on_step_outputs(
                scheduler_output(), model_output(), empty_outputs()
            )

    def test_blocks_per_kv_block_mismatch_is_refused(self) -> None:
        geometry = dict(GEOMETRY)
        geometry["blocks_per_kv_block"] = 2
        engine = self.make_engine(geometry_fetcher=lambda: geometry)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(UnsupportedConfig):
            engine.on_step_outputs(scheduler_output(), model_output(), empty_outputs())

    def test_visible_block_beyond_allocated_range_is_refused(self) -> None:
        # 只用 8 块（0..7），而 9 个可见块需要块 8（当前 token 所在块）
        engine = self.make_engine(scheduler=FakeScheduler(blocks=[list(FA_BLOCKS[:8])] * 4))
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        # 可见块越界由阶段 03 的视图不变量拦下（I3/I8），引擎不包装成配置错误
        with self.assertRaises(ReadViewError):
            engine.on_step_outputs(scheduler_output(), model_output(), empty_outputs())


class TraceTest(EngineTestBase):
    def test_dump_traces_writes_json_records(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        parsed = engine.parse_outputs(model_output())
        engine._record_traces_for(parsed)
        engine.note_application("r1", applied_step=1, plan_payload={
            "seqused_k": 3137, "tail_len": 1,
            "visible_logical_blocks": [0, 5, 6, 7, 8],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = engine.dump_traces(Path(tmp) / "traces.json")
            data = json.loads(path.read_text())
        self.assertTrue(data)
        kinds = {rec.get("note", "") for rec in data}
        self.assertTrue(any("worker 落位回填" in k for k in kinds))
        applied = [rec for rec in data if rec.get("applied_step") == 1]
        self.assertTrue(applied)
        self.assertEqual(applied[-1]["seqused_k"], 3137)


if __name__ == "__main__":
    unittest.main()
