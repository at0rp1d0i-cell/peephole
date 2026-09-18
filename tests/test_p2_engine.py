"""阶段 05 EngineCore 侧适配层测试（CPU、假对象驱动、不 import vLLM）。

覆盖 SUP-004-R1 §2 与本地复核指出的两类真实风险：
1. **计划必须按本次 `schedule()` 的结果构造**：块在本次调度才分配（prompt_len=6272 时，
   消费 g0 的那次 forward 要写块 8）；在 `schedule()` 之前索引 canonical 映射会越界。
   据此，`on_step_outputs` 只做解析/清理，计划在 `attach_plans` 里按 `num_computed_tokens +
   num_scheduled_tokens` 构造，并**按运行期进度**跳过 prefill（末尾 prefill chunk 也可能只调度 1 个 token）。
2. **真实 tokenizer 接线**：构造路径与 `patched/.../engine/core.py` 一致（`make_token_text_of`），
   不得依赖测试注入的假取词函数，否则首个真实采样 token 就会崩。
"""

from __future__ import annotations

import ast
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
PATCHED_CORE = REPO / "vllm-patch/patched/v1/engine/core.py"
SNAPSHOT = (
    REPO
    / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)

B = 784
PROMPT_LEN = 6272  # 8 个满块
FA_BLOCKS = [41, 7, 90, 12, 63, 28, 55, 88, 102]  # 9 块：块 8 给 g0 那一步
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


def request_state(num_prompt_tokens=PROMPT_LEN, num_computed_tokens=PROMPT_LEN):
    return SimpleNamespace(
        num_prompt_tokens=num_prompt_tokens, num_computed_tokens=num_computed_tokens
    )


class FakeScheduler:
    """最小假调度器：`requests`（进度账本）+ `kv_cache_manager.get_block_ids`。"""

    def __init__(self, blocks=None, requests=None):
        self.blocks = blocks if blocks is not None else [list(FA_BLOCKS)] * 4
        self.requests = requests if requests is not None else {"r1": request_state()}
        self.kv_cache_manager = SimpleNamespace(get_block_ids=lambda req_id: self.blocks)

    def allocate_block(self, block_id: int) -> None:
        for group in self.blocks:
            group.append(block_id)


def scheduler_output(new_reqs=(), finished=(), scheduled=None):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        finished_req_ids=set(finished),
        num_scheduled_tokens=dict(scheduled or {}),
    )


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


def stub_config(model_dir: Path | None = None):
    """与 core.py 同构的最小 vllm_config（真实 tokenizer 走 model_config 的字段）。"""
    if model_dir is None:
        return SimpleNamespace(model_config=None)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(model_dir),
            tokenizer=str(model_dir),
            tokenizer_mode="auto",
            trust_remote_code=False,
            revision=None,
            tokenizer_revision=None,
        )
    )


class EngineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_engine_module()

    def make_engine(self, *, scheduler=None, geometry_fetcher=None, token_text_of="default"):
        kwargs = {"geometry_fetcher": geometry_fetcher if geometry_fetcher is not None else (lambda: dict(GEOMETRY))}
        if token_text_of == "default":
            kwargs["token_text_of"] = lambda token_id: TOKEN_TEXT.get(int(token_id), "a")
        elif token_text_of is not None:
            kwargs["token_text_of"] = token_text_of
        return self.mod.AttnViewEngine(stub_config(), scheduler or FakeScheduler(), **kwargs)


class RegistrationTest(EngineTestBase):
    def test_only_payload_requests_are_registered(self) -> None:
        engine = self.make_engine()
        registered = engine.register_new_requests(
            scheduler_output([new_req("r1", payload()), new_req("p1", None)])
        )
        self.assertEqual(registered, ["r1"])
        self.assertEqual(engine.registry.active_ids(), ("r1",))

    def test_payload_with_geometry_or_wrong_protocol_is_rejected(self) -> None:
        engine = self.make_engine()
        with self.assertRaises(AttnViewConfigError):
            engine.register_new_requests(scheduler_output([new_req("r1", payload(kernel_block_size=B))]))
        with self.assertRaises(AttnViewConfigError):
            engine.register_new_requests(scheduler_output([new_req("r2", payload(protocol="v0.9"))]))

    def test_registration_is_idempotent(self) -> None:
        engine = self.make_engine()
        req = new_req("r1", payload())
        engine.register_new_requests(scheduler_output([req]))
        engine.register_new_requests(scheduler_output([req]))
        self.assertEqual(engine.registry.active_ids(), ("r1",))


class PrefillAndPlanTimingTest(EngineTestBase):
    """计划必须按本次 schedule 的结果构造；prefill 按运行期进度跳过。"""

    def _engine_with_state(self, *, blocks, num_computed, scheduled=1):
        scheduler = FakeScheduler(blocks=[list(blocks)] * 4,
                                  requests={"r1": request_state(num_computed_tokens=num_computed)})
        engine = self.make_engine(scheduler=scheduler)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.parse_outputs(model_output())  # 解析 g0：模式进入 local
        return engine, scheduler_output(scheduled={"r1": scheduled})

    def test_no_plan_during_prefill_even_single_token_chunk(self) -> None:
        # 末尾 prefill chunk：query_len == 1，但进度仍在 prompt 内 → 必须按原版走
        engine, so = self._engine_with_state(blocks=FA_BLOCKS[:8], num_computed=PROMPT_LEN - 1)
        self.assertIsNone(engine.attach_plans(so))
        self.assertIsNone(so.da_step_plans)

    def test_plan_after_schedule_uses_allocated_block_and_current_token(self) -> None:
        engine, so = self._engine_with_state(blocks=FA_BLOCKS[:9], num_computed=PROMPT_LEN)
        plans = engine.attach_plans(so)
        self.assertIsNotNone(plans)
        plan = so.da_step_plans["r1"]
        # 长度按本次调度进度算：num_computed(=prompt) + scheduled(1) = 6273（含本步写入的当前 token）
        self.assertEqual(plan["attention_kv_len"], PROMPT_LEN + 1)
        self.assertEqual(plan["next_write_position"], PROMPT_LEN + 1)
        self.assertEqual(plan["written_before_step"], PROMPT_LEN)
        self.assertEqual(plan["effect_step"], 1)
        self.assertEqual(plan["mode"], "local")
        self.assertIn(8, plan["visible_logical_blocks"], "当前 token 所在块必须在可见集内")
        self.assertEqual(plan["seqused_k"], sum(plan["effective_per_block"]))
        self.assertNotEqual(plan["seqused_k"], plan["attention_kv_len"])
        self.assertEqual(plan["effective_per_block"][-1], plan["tail_len"])
        for forbidden in ("kernel_block_size", "max_width", "num_blocks"):
            self.assertNotIn(forbidden, plan)

    def test_plan_before_block_allocation_would_overrun(self) -> None:
        # 回归守卫：若把计划构造挪回 schedule 之前（只有 8 块），必须越界报错而不是静默算错
        engine, so = self._engine_with_state(blocks=FA_BLOCKS[:8], num_computed=PROMPT_LEN)
        with self.assertRaises(ReadViewError):
            engine.attach_plans(so)

    def test_prompt_len_mismatch_is_refused(self) -> None:
        scheduler = FakeScheduler(
            blocks=[list(FA_BLOCKS[:9])] * 4,
            requests={"r1": request_state(num_prompt_tokens=PROMPT_LEN - 1, num_computed_tokens=PROMPT_LEN - 1)},
        )
        engine = self.make_engine(scheduler=scheduler)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(AttnViewConfigError):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))

    def test_no_plan_for_plain_request(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("p1", None)]))
        so = scheduler_output(scheduled={"p1": 1})
        self.assertIsNone(engine.attach_plans(so))


class LifecycleTest(EngineTestBase):
    def test_parse_uses_protocol_state_and_advances_lengths(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        parsed = engine.parse_outputs(model_output(sampled=((1,),)))
        self.assertEqual(parsed, {"r1": 1})
        state = engine.registry.get("r1")
        self.assertEqual(state.mode, "local")
        self.assertEqual(state.written_kv_len, PROMPT_LEN)  # g0 尚未写入
        self.assertEqual(state.attention_kv_len_next, PROMPT_LEN + 1)

    def test_finished_request_is_dropped_and_not_planned(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.on_step_outputs(scheduler_output(finished={"r1"}), model_output(sampled=((),)), empty_outputs())
        self.assertNotIn("r1", engine.registry.active_ids())
        self.assertIsNone(engine.attach_plans(scheduler_output(scheduled={"r1": 1})))

    def test_step_finish_releases_and_traces(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.parse_outputs(model_output())
        self.assertEqual(engine.release_finished_in_step(engine_outputs(finish_reason="stop")), ["r1"])
        self.assertTrue(any("finished:stop" in t.get("note", "") for t in engine.traces))

    def test_release_is_idempotent(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.release("r1")
        engine.release("r1")


class UnsupportedConfigTest(EngineTestBase):
    def test_batch_queue_path_refuses_payload_requests(self) -> None:
        engine = self.make_engine()
        with self.assertRaises(UnsupportedConfig):
            engine.refuse_unsupported_step(scheduler_output([new_req("r1", payload())]))
        engine.refuse_unsupported_step(scheduler_output([new_req("p1", None)]))

    def test_geometry_must_come_from_worker(self) -> None:
        engine = self.mod.AttnViewEngine(stub_config(), FakeScheduler(), geometry_fetcher=None,
                                         token_text_of=lambda token_id: "a")
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(self.mod.WorkerGeometryUnavailable):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))

    def test_blocks_per_kv_block_mismatch_is_refused(self) -> None:
        geometry = dict(GEOMETRY)
        geometry["blocks_per_kv_block"] = 2
        engine = self.make_engine(geometry_fetcher=lambda: geometry)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(UnsupportedConfig):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))


class TokenizerWiringTest(EngineTestBase):
    """真实 tokenizer 接线：构造路径与 core.py 相同，不得依赖测试注入。"""

    def test_default_construction_uses_real_tokenizer(self) -> None:
        if not SNAPSHOT.is_dir():
            self.skipTest("缺少本地模型快照（tokenizer）")
        engine = self.mod.AttnViewEngine(
            stub_config(SNAPSHOT),  # 与 core.py 一样只给 vllm_config
            FakeScheduler(blocks=[list(FA_BLOCKS[:9])] * 4,
                          requests={"r1": request_state(num_computed_tokens=PROMPT_LEN)}),
            geometry_fetcher=lambda: dict(GEOMETRY),
            # 刻意**不**注入 token_text_of：走 make_token_text_of 的真实路径
        )
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        from vllm.tokenizers import get_tokenizer  # noqa: PLC0415

        hf = get_tokenizer(str(SNAPSHOT), tokenizer_mode="auto", trust_remote_code=False)
        local_ids = hf.encode("<local>", add_special_tokens=False)
        self.assertTrue(local_ids)
        parsed = engine.parse_outputs(model_output(sampled=(tuple(local_ids),)))
        self.assertEqual(parsed, {"r1": len(local_ids)})
        self.assertEqual(engine.registry.get("r1").mode, "local", "真实 tokenizer 取词必须能驱动状态机")
        self.assertIsNotNone(engine.attach_plans(scheduler_output(scheduled={"r1": 1})))

    def test_missing_tokenizer_raises_clear_error(self) -> None:
        engine = self.mod.AttnViewEngine(stub_config(None), FakeScheduler(), geometry_fetcher=lambda: dict(GEOMETRY))
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(self.mod.TokenizerUnavailable):
            engine.parse_outputs(model_output())

    def test_core_py_passes_tokenizer_factory(self) -> None:
        self.assertTrue(PATCHED_CORE.is_file(), "缺少 patched core.py")
        tree = ast.parse(PATCHED_CORE.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AttnViewEngine"
        ]
        self.assertTrue(calls, "core.py 未构造 AttnViewEngine")
        kwargs = {kw.arg for kw in calls[0].keywords}
        self.assertIn(
            "token_text_of",
            kwargs,
            "core.py 必须显式注入 token_text_of=make_token_text_of(vllm_config)，"
            "否则首个真实采样 token 会因缺 tokenizer 而失败",
        )


class TraceTest(EngineTestBase):
    def test_dump_traces_writes_json(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine._record_traces_for(engine.parse_outputs(model_output()))
        engine.note_application("r1", applied_step=1, plan_payload={
            "seqused_k": 3137, "tail_len": 1, "visible_logical_blocks": [0, 5, 6, 7, 8]})
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(engine.dump_traces(Path(tmp) / "traces.json").read_text())
        self.assertTrue(data)
        applied = [rec for rec in data if rec.get("applied_step") == 1]
        self.assertEqual(applied[-1]["seqused_k"], 3137)


if __name__ == "__main__":
    unittest.main()
