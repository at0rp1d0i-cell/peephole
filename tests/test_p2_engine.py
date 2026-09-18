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
    # `@dataclass` 需要模块已在 sys.modules 里（否则 dataclasses 取 cls.__module__ 会失败）
    sys.modules["attnview_engine_under_test"] = module
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


def request_state(num_prompt_tokens=PROMPT_LEN, post_computed_tokens=PROMPT_LEN):
    """`num_computed_tokens` 是**调度之后**的值（见 FakeScheduler.schedule 的推进语义）。"""
    return SimpleNamespace(
        num_prompt_tokens=num_prompt_tokens, num_computed_tokens=post_computed_tokens
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

    def schedule(self, req_id: str, num_scheduled: int) -> None:
        """镜像真实 `Scheduler.schedule()` 的推进：`_update_after_schedule` 里
        `request.num_computed_tokens += num_scheduled_token`（`scheduler.py:1461`），
        且该调用发生在 `schedule()` **返回之前** → 之后读到的进度是 post 值。
        """
        self.requests[req_id].num_computed_tokens += num_scheduled


def scheduler_output(new_reqs=(), finished=(), scheduled=None, preempted=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        finished_req_ids=set(finished),
        num_scheduled_tokens=dict(scheduled or {}),
        preempted_req_ids=set(preempted),
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
        # 两个请求都在调度器账本里（存活性过滤发生在载荷校验之前，见 register_new_requests）
        scheduler = FakeScheduler(requests={"r1": request_state(), "r2": request_state()})
        engine = self.make_engine(scheduler=scheduler)
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
    """计划按**本次调度**构造；长度用调度器账本的 post 值，不加第二次；global 走原版。"""

    def _engine_with_state(self, *, blocks, post_computed, mode_token=1):
        scheduler = FakeScheduler(
            blocks=[list(blocks)] * 4,
            requests={"r1": request_state(post_computed_tokens=post_computed)},
        )
        engine = self.make_engine(scheduler=scheduler)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.parse_outputs(model_output(sampled=((mode_token,),)))  # 解析 g0：进入 local
        return engine, scheduler

    def test_decode_plan_uses_post_computed_length_exactly_once(self) -> None:
        # pre=6272（g0 已采样）→ 本步调度 1 个 token → post=6273
        engine, scheduler = self._engine_with_state(blocks=FA_BLOCKS[:9], post_computed=PROMPT_LEN)
        scheduler.schedule("r1", 1)
        so = scheduler_output(scheduled={"r1": 1})
        plan = engine.attach_plans(so)["r1"]
        self.assertEqual(plan["attention_kv_len"], PROMPT_LEN + 1, "post=6273，不得算成 6274")
        self.assertEqual(plan["next_write_position"], PROMPT_LEN + 1)
        self.assertEqual(plan["written_before_step"], PROMPT_LEN)
        self.assertEqual(plan["effect_step"], 1)
        self.assertEqual(plan["mode"], "local")
        self.assertIn(8, plan["visible_logical_blocks"])
        self.assertEqual(plan["seqused_k"], sum(plan["effective_per_block"]))
        self.assertLess(plan["seqused_k"], plan["attention_kv_len"])

    def test_last_single_token_prefill_chunk_has_no_plan(self) -> None:
        # 末尾 prefill chunk：pre=6271 → post=6272 == prompt_len；此刻模式仍是 global
        engine = self.make_engine(
            scheduler=FakeScheduler(
                blocks=[list(FA_BLOCKS[:8])] * 4,
                requests={"r1": request_state(post_computed_tokens=PROMPT_LEN - 1)},
            )
        )
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.attach_plans(scheduler_output(scheduled={"r1": 1}))  # 无 token 被解析 → 模式 global
        engine.attach_plans(scheduler_output(scheduled={"r1": 1}))
        self.assertIsNone(engine.pending_plans())
        self.assertEqual(engine.registry.get("r1").mode, "global")

    def test_global_mode_uses_vanilla_path(self) -> None:
        # 即使已进入 decode（post=6273），global 模式也不出计划 → 走原版读 metadata
        engine, scheduler = self._engine_with_state(blocks=FA_BLOCKS[:9], post_computed=PROMPT_LEN)
        scheduler.schedule("r1", 1)
        engine.registry.get("r1").parser.mode = "global"  # 模拟解析后仍为 global
        self.assertIsNone(engine.attach_plans(scheduler_output(scheduled={"r1": 1})))

    def test_non_global_mode_before_any_generated_token_is_refused(self) -> None:
        # 非 global 却尚未生成任何 token（pre < prompt_len）→ 协议/时序冲突，拒绝
        engine, scheduler = self._engine_with_state(blocks=FA_BLOCKS[:9], post_computed=PROMPT_LEN)
        scheduler.requests["r1"].num_computed_tokens = PROMPT_LEN  # post=6272 → pre=6271 < prompt
        with self.assertRaises(AttnViewConfigError):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))

    def test_plan_before_block_allocation_would_overrun(self) -> None:
        # 回归守卫：若在块 8 尚未分配时就出计划，必须越界报错而不是静默算错
        engine, scheduler = self._engine_with_state(blocks=FA_BLOCKS[:8], post_computed=PROMPT_LEN)
        scheduler.schedule("r1", 1)
        with self.assertRaises(ReadViewError):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))

    def test_prompt_len_mismatch_is_refused(self) -> None:
        scheduler = FakeScheduler(
            blocks=[list(FA_BLOCKS[:9])] * 4,
            requests={"r1": request_state(num_prompt_tokens=PROMPT_LEN - 1, post_computed_tokens=PROMPT_LEN - 1)},
        )
        engine = self.make_engine(scheduler=scheduler)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        with self.assertRaises(AttnViewConfigError):
            engine.attach_plans(scheduler_output(scheduled={"r1": 1}))

    def test_no_plan_for_plain_request(self) -> None:
        engine = self.make_engine()
        engine.register_new_requests(scheduler_output([new_req("p1", None)]))
        self.assertIsNone(engine.attach_plans(scheduler_output(scheduled={"p1": 1})))

    def test_progress_semantics_are_anchored_to_pin_source(self) -> None:
        """假 scheduler 的推进语义必须与 pin 一致：schedule() 返回前已 +num_scheduled。"""
        source = (REPO / "vllm/vllm/v1/core/sched/scheduler.py").read_text()
        self.assertIn("self._update_after_schedule(scheduler_output)", source)
        self.assertIn("request.num_computed_tokens += num_scheduled_token", source)


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


class EngineConfigGateTest(EngineTestBase):
    """首次登记 DA 请求时必须实际调用配置门禁（不是只定义/只测纯函数）。"""

    def test_registration_refuses_when_sync_but_prefix_caching_on(self) -> None:
        from vllm.config import CUDAGraphMode  # noqa: PLC0415

        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            max_concurrent_batches=1,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=8192),
        )
        engine = self.mod.AttnViewEngine(config, FakeScheduler(), geometry_fetcher=lambda: dict(GEOMETRY),
                                         token_text_of=lambda token_id: "a")
        with self.assertRaises(UnsupportedConfig):
            engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        # 普通请求不受影响（原版可复跑）
        engine.register_new_requests(scheduler_output([new_req("p1", None)]))

    def test_registration_passes_with_supported_config(self) -> None:
        from vllm.config import CUDAGraphMode  # noqa: PLC0415

        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            max_concurrent_batches=1,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=8192),
        )
        engine = self.mod.AttnViewEngine(config, FakeScheduler(blocks=[list(FA_BLOCKS[:9])] * 4,
                                                              requests={"r1": request_state()}),
                                         geometry_fetcher=lambda: dict(GEOMETRY),
                                         token_text_of=lambda token_id: "a")
        self.assertEqual(engine.register_new_requests(
            scheduler_output([new_req("r1", payload())])), ["r1"])


class AbortDuringStepTest(EngineTestBase):
    """执行中取消/抢占的 CPU 轨迹：schedule → 执行中 abort → 输出回收。

    pin 语义（本类另有源码锚定测试）：`Scheduler.schedule()` 返回前已把 `self.finished_req_ids`
    清空（`scheduler.py:1495`），所以当前 `SchedulerOutput.finished_req_ids` 是**上一步**的快照；
    执行中取消的请求由 `_process_aborts_queue` 处理，`update_from_output` 会跳过它（`:1880`）且
    **不产出 finish 项**，`_free_request` 会 `del self.requests[...]`（`:2512`）。
    """

    def _engine_with_parsed_g0(self):
        scheduler = FakeScheduler(blocks=[list(FA_BLOCKS[:9])] * 4, requests={"r1": request_state()})
        engine = self.make_engine(scheduler=scheduler)
        engine.register_new_requests(scheduler_output([new_req("r1", payload())]))
        engine.parse_outputs(model_output())  # g0 → local
        return engine, scheduler

    def test_aborted_during_step_is_not_parsed_and_released(self) -> None:
        engine, scheduler = self._engine_with_parsed_g0()
        steps_before = engine.registry.get("r1").generated_tokens
        scheduler.requests.pop("r1")  # 镜像 _free_request 的 `del self.requests[...]`
        so = scheduler_output(scheduled={"r1": 1})
        result = engine.on_step_outputs(
            so,
            model_output(sampled=((2,),)),  # 执行中确实采样出了一个 token
            empty_outputs(),                # 但被取消 → 没有 finish 项
        )
        self.assertEqual(result["cancelled"], ["r1"])
        self.assertEqual(result["parsed"], {}, "被取消请求的采样 token 不得进入解析")
        self.assertEqual(list(engine.registry.active_ids()), [], "状态必须释放")
        note = [t for t in engine.traces if t.get("note", "").endswith("_during_step")]
        self.assertEqual(len(note), 1)
        self.assertEqual(note[0]["note"], "cancelled_during_step")
        self.assertEqual(note[0]["generated_tokens_before_drop"], steps_before)
        self.assertIsNone(engine.attach_plans(so), "不得为已取消请求产出计划")

    def test_normal_stop_token_is_parsed_then_released(self) -> None:
        engine, scheduler = self._engine_with_parsed_g0()
        scheduler.requests.pop("r1")  # 正常终结同样会从账本移除
        so = scheduler_output(scheduled={"r1": 1})
        result = engine.on_step_outputs(
            so,
            model_output(sampled=((3,),)),
            engine_outputs(finish_reason="stop"),  # 但这一步**有** finish 项 → 正常终结
        )
        self.assertEqual(result["parsed"], {"r1": 1}, "正常终结的 stop token 仍须解析")
        self.assertEqual(result["cancelled"], [])
        self.assertEqual(list(engine.registry.active_ids()), [])
        self.assertTrue(any("finished:stop" in t.get("note", "") for t in engine.traces))

    def test_aborted_before_registration_is_not_registered(self) -> None:
        scheduler = FakeScheduler(requests={})  # 账本里没有它（例如调度前已取消）
        engine = self.make_engine(scheduler=scheduler)
        registered = engine.register_new_requests(
            scheduler_output([new_req("r1", payload())]), alive=set(scheduler.requests)
        )
        self.assertEqual(registered, [])
        self.assertEqual(list(engine.registry.active_ids()), [])
        self.assertEqual(engine.config_of(new_req("r1", payload())) is not None, True)

    def test_preempted_during_step_is_flagged_unsupported(self) -> None:
        engine, scheduler = self._engine_with_parsed_g0()
        scheduler.requests.pop("r1")
        so = scheduler_output(preempted={"r1"})
        result = engine.on_step_outputs(so, model_output(sampled=((2,),)), empty_outputs())
        self.assertEqual(result["preempted"], ["r1"])
        self.assertEqual(result["parsed"], {})
        note = [t for t in engine.traces if t.get("note") == "preempted_during_step"]
        self.assertEqual(len(note), 1)
        self.assertTrue(note[0]["unsupported"], "抢占属本阶段不支持路径 → 必须显式标记")

    def test_abort_signals_are_anchored_to_pin_source(self) -> None:
        source = (REPO / "vllm/vllm/v1/core/sched/scheduler.py").read_text()
        self.assertIn("self.finished_req_ids = set()", source, "schedule 返回前会清空上一步终结集合")
        self.assertIn("del self.requests[request.request_id]", source, "账本移除 = 请求已消失")
        self.assertIn("if request is None or request.is_finished():", source,
                      "被取消请求在 update_from_output 被跳过、不产 finish 项")


class TokenizerWiringTest(EngineTestBase):
    """真实 tokenizer 接线：构造路径与 core.py 相同，不得依赖测试注入。"""

    def test_default_construction_uses_real_tokenizer(self) -> None:
        if not SNAPSHOT.is_dir():
            self.skipTest("缺少本地模型快照（tokenizer）")
        scheduler = FakeScheduler(blocks=[list(FA_BLOCKS[:9])] * 4,
                                  requests={"r1": request_state(post_computed_tokens=PROMPT_LEN)})
        engine = self.mod.AttnViewEngine(
            stub_config(SNAPSHOT),  # 与 core.py 一样只给 vllm_config
            scheduler,
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
        scheduler.schedule("r1", 1)  # 镜像真实 schedule() 的进度推进（post=6273）
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
