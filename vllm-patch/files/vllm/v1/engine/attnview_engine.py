"""attnview 声明式读取：vLLM **EngineCore 侧** 适配层（阶段 05，版本锁定、可撤销）。

职责（且仅此）：
1. 启动时拒绝不支持的运行配置（同步调度/eager/关前缀缓存/关投机），fail-fast；
2. 从请求的 `SamplingParams.extra_args["attnview"]` 建请求级协议状态（复用阶段 03 的
   `ProtocolRegistry` / `RequestProtocolState` / 增量 UTF-8 解码器，不复制第二套）；
3. 每步输出解析后构造**下一步**的读取计划，并注入下一次 `SchedulerOutput`；
4. trace 与请求终结清理（含取消/抢占：不隐式回退 global，C6.2）。

跨进程边界：本模块只依赖纯 Python（不 import torch）；worker 侧落位见
`vllm/v1/worker/gpu/attnview_adapter.py`。视图算术见 `attnview.step_plan`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from attnview.decode import IncrementalDetokenizer
from attnview.state import ProtocolRegistry
from attnview.readview import MODE_GLOBAL
from attnview.step_plan import (
    EXTRA_ARG_KEY,
    AttnViewConfigError,
    DaRequestConfig,
    Geometry,
    UnsupportedConfig,
    build_step_plan,
    check_supported_config,
    trace_from_plan,
)

__all__ = [
    "AttnViewEngine",
    "WorkerGeometryUnavailable",
    "make_token_text_of",
    "TokenizerUnavailable",
]


class WorkerGeometryUnavailable(RuntimeError):
    """拿不到 worker 的实际几何时拒绝启用（不得用默认块大小兜底）。"""


class TokenizerUnavailable(RuntimeError):
    """拿不到 token 文本来源时拒绝启用（增量解析需要 token id → byte-level token 文本）。"""


def make_token_text_of(vllm_config: Any) -> Callable[[int], str]:
    """返回 `token_id -> byte-level token 文本` 的**懒加载**取值函数（真实 tokenizer）。

    - `vllm` 只在首次真正取词时导入（模块级保持零依赖，EngineCore 进程不因 import 而加载模型栈）；
    - 普通请求（无载荷）永不触发，因此原版可复跑不付 tokenizer 加载成本；
    - 取词用 `convert_ids_to_tokens`（byte-level BPE 域），再由 `decode.token_bytes` 还原原始字节，
      与阶段 03 的增量 UTF-8 解码口径一致。
    """
    cache: dict[str, Any] = {}

    def _load() -> Any:
        tokenizer = cache.get("tokenizer")
        if tokenizer is not None:
            return tokenizer
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            raise TokenizerUnavailable("vllm_config 上没有 model_config，无法确定 tokenizer")
        name = getattr(model_config, "tokenizer", None) or getattr(model_config, "model", None)
        if not name:
            raise TokenizerUnavailable("model_config 既无 tokenizer 也无 model，无法确定 tokenizer")
        try:
            # v0.29.0 的入口在 `vllm.tokenizers`（registry.get_tokenizer(name, ...)）
            from vllm.tokenizers import get_tokenizer
        except Exception as exc:  # pragma: no cover - 仅环境缺失时
            raise TokenizerUnavailable(f"无法导入 vLLM tokenizer 工具：{exc}") from exc
        revision = getattr(model_config, "tokenizer_revision", None) or getattr(
            model_config, "revision", None
        )
        tokenizer = get_tokenizer(
            name,
            tokenizer_mode=getattr(model_config, "tokenizer_mode", "auto"),
            trust_remote_code=bool(getattr(model_config, "trust_remote_code", False)),
            revision=revision,
        )
        cache["tokenizer"] = tokenizer
        return tokenizer

    def token_text_of(token_id: int) -> str:
        text = _load().convert_ids_to_tokens(int(token_id))
        return "" if text is None else str(text)

    return token_text_of


class AttnViewEngine:
    """EngineCore 侧的每步解析与计划构造（单实例，按 `req_id` 隔离状态）。"""

    def __init__(
        self,
        vllm_config: Any,
        scheduler: Any,
        *,
        geometry_fetcher: Callable[[], Mapping[str, Any]] | None = None,
        token_text_of: Callable[[int], str] | None = None,
        arm: str = "da",
    ) -> None:
        self._vllm_config = vllm_config
        self._scheduler = scheduler
        self.registry = ProtocolRegistry()
        self._geometry_fetcher = geometry_fetcher
        # 未显式注入时用**真实 tokenizer** 的懒加载取值函数（普通请求永不触发）。
        self._token_text_of = token_text_of if token_text_of is not None else make_token_text_of(vllm_config)
        self._arm = arm
        self._geometry: Geometry | None = None
        self._group_block_sizes: list[int] | None = None
        self._detokenizers: dict[str, IncrementalDetokenizer] = {}
        self._configs: dict[str, DaRequestConfig] = {}
        self._pending: dict[str, dict] | None = None
        self._config_checked = False
        self.traces: list[dict] = []
        self.notes: list[str] = []

    # --- 几何（一次性，取自 worker 实际配置） ------------------------------ #

    def set_geometry(self, geometry: Geometry, *, group_block_sizes: Sequence[int] | None = None) -> None:
        self._geometry = geometry.validate()
        if group_block_sizes is not None:
            self._group_block_sizes = [int(b) for b in group_block_sizes]

    def geometry(self) -> Geometry:
        if self._geometry is None:
            if self._geometry_fetcher is None:
                raise WorkerGeometryUnavailable(
                    "尚未取得 worker 的实际几何：必须由 worker 提供，不得默认块大小"
                )
            payload = dict(self._geometry_fetcher())
            self.set_geometry(
                Geometry(
                    kernel_block_size=int(payload["kernel_block_size"]),
                    num_kv_groups=int(payload["num_kv_groups"]),
                    fa_group_index=int(payload["fa_group_index"]),
                    blocks_per_kv_block=int(payload["blocks_per_kv_block"]),
                    max_model_len=int(payload["max_model_len"]),
                ),
                group_block_sizes=payload.get("group_block_sizes"),
            )
        assert self._geometry is not None
        return self._geometry

    def _token_text(self, token_id: int) -> str:
        if self._token_text_of is None:  # 构造时已兜底，正常不可达
            raise TokenizerUnavailable("token 文本来源缺失（应已由 make_token_text_of 兜底）")
        return self._token_text_of(token_id)

    # --- 请求登记 ----------------------------------------------------------- #

    @staticmethod
    def config_of(new_req: Any) -> DaRequestConfig | None:
        """从 `NewRequestData` 取协议配置；普通请求（无载荷）返回 `None`（直通）。"""
        sampling_params = getattr(new_req, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) if sampling_params else None
        if not extra_args or EXTRA_ARG_KEY not in extra_args:
            return None
        return DaRequestConfig.from_extra_args(extra_args)

    def _assert_supported_once(self) -> None:
        """用 EngineCore 侧的 `vllm_config` 做一次支持范围校验（结果缓存）。"""
        if self._config_checked:
            return
        cfg = self._vllm_config
        scheduler_config = getattr(cfg, "scheduler_config", None)
        compilation = getattr(cfg, "compilation_config", None)
        check_supported_config(
            async_scheduling=bool(getattr(scheduler_config, "async_scheduling", False)),
            max_concurrent_batches=int(getattr(cfg, "max_concurrent_batches", 1)),
            cudagraph_mode=getattr(compilation, "cudagraph_mode", None),
            enable_prefix_caching=bool(
                getattr(getattr(cfg, "cache_config", None), "enable_prefix_caching", False)
            ),
            speculative_config=getattr(cfg, "speculative_config", None),
        )
        self._config_checked = True

    def refuse_unsupported_step(self, scheduler_output: Any) -> None:
        """批次队列（异步调度）路径的守卫：出现带载荷的内部请求即**拒绝**。

        为什么必须显式拒绝：该路径会先发起第 t+1 步的 `execute_model` 再等待第 t 步结果，
        本适配层既不会解析也不会下推计划 —— 若不拒绝，DA 请求会**静默退化成原版读取**，
        那正是本阶段不允许的行为。
        """
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            try:
                config = self.config_of(new_req)
            except AttnViewConfigError:
                raise
            if config is not None:
                raise UnsupportedConfig(
                    f"attnview 内部请求 {getattr(new_req, 'req_id', '?')} 在异步调度（批次队列）"
                    "下不受支持：请使用 --no-async-scheduling（不静默退化为原版读取）"
                )

    def register_new_requests(self, scheduler_output: Any, *, alive: set[str] | None = None) -> list[str]:
        """为本步新调度且携带载荷的请求建协议状态（幂等：已存在则跳过）。

        `alive` 为调度器账本里的存活集合：**不在账本里的请求不登记**（例如本步开始前已取消），
        避免把已取消请求重新拉起来。
        """
        if alive is None:
            alive = set(self._requests())
        registered: list[str] = []
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = str(new_req.req_id)
            if req_id in self._configs or req_id not in alive:
                continue
            config = self.config_of(new_req)
            if config is None:
                continue
            # 门禁：首个 DA 请求启用前校验运行配置（不静默降级；普通请求不受影响）。
            self._assert_supported_once()
            self._configs[req_id] = config
            self.registry.create(
                req_id,
                arm=getattr(new_req, "attnview_arm", self._arm),
                num_segments=len(config.segment_spans),
                prompt_len=config.prompt_len,
            )
            self._detokenizers[req_id] = IncrementalDetokenizer(self._token_text)
            registered.append(req_id)
        return registered

    # --- 清理 --------------------------------------------------------------- #

    def release(self, req_id: str) -> None:
        """释放请求级状态（幂等）。"""
        self._configs.pop(req_id, None)
        self._detokenizers.pop(req_id, None)
        try:
            self.registry.release(req_id)
        except KeyError:
            pass
        if self._pending is not None:
            self._pending.pop(req_id, None)

    def drop_finished(self, finished_req_ids: Iterable[str]) -> list[str]:
        """丢弃上一步已终结/取消的请求状态：**不再解析、不再产出计划**。"""
        dropped: list[str] = []
        for req_id in finished_req_ids or ():
            req_id = str(req_id)
            if req_id in self._configs:
                dropped.append(req_id)
            self.release(req_id)
        return dropped

    # --- 解析本步输出 ------------------------------------------------------- #

    def parse_outputs(self, model_output: Any, *, allow: set[str] | None = None) -> dict[str, int]:
        """解析本步采样 token（只解析本请求自己的生成流，C3.7）。返回 req_id -> token 数。

        `allow`：允许被解析的请求集合。**执行中被取消/中止的请求必须在集合之外** ——
        它们本步的采样结果不进入解析与 trace（合同 C6.4 的清理要求）。
        """
        sampled = getattr(model_output, "sampled_token_ids", None)
        req_ids = list(getattr(model_output, "req_ids", ()) or ())
        index_of = getattr(model_output, "req_id_to_index", None) or {}
        parsed: dict[str, int] = {}
        if not sampled:
            return parsed
        for req_id in req_ids:
            if req_id not in self._configs:
                continue
            if allow is not None and req_id not in allow:
                continue
            row = index_of.get(req_id)
            if row is None or row >= len(sampled):
                continue
            state = self.registry.get(req_id)
            detok = self._detokenizers[req_id]
            count = 0
            for token_id in sampled[row] or ():
                text = detok.feed(int(token_id))
                state.feed_generated_token(state.generated_tokens, int(token_id), text)
                count += 1
            parsed[req_id] = count
        return parsed

    def release_finished_in_step(self, engine_core_outputs: Any) -> list[str]:
        """释放**本步**刚结束的请求（正常停止 / 长度上限），并冻结 trace。"""
        released: list[str] = []
        for outputs in (engine_core_outputs or {}).values():
            for out in getattr(outputs, "outputs", ()) or ():
                req_id = str(getattr(out, "request_id", ""))
                if req_id not in self._configs:
                    continue
                reason = getattr(out, "finish_reason", None)
                if reason is None:
                    continue
                state = self.registry.get(req_id)
                state.finish()
                self._record_trace(req_id, applied_step=None, note=f"finished:{reason}")
                self.release(req_id)
                released.append(req_id)
        return released

    # --- 计划构造 ----------------------------------------------------------- #

    def build_plans(self, scheduler_output: Any, *, blocks_of: Callable[[str], Sequence[Sequence[int]]]) -> dict[str, dict]:
        """**在 `schedule()` 之后**为本步构造读取计划。

        长度口径（与 pin 的调度器语义对齐）：`Scheduler.schedule()` 在返回**之前**已调用
        `_update_after_schedule`，其中执行 `request.num_computed_tokens += num_scheduled_token`
        （`vllm/v1/core/sched/scheduler.py:1461`）。因此调度器账本里的 `num_computed_tokens` 是
        **本步之后**的值：

        - `attention_kv_len = post_computed_tokens`（**不再加** `num_scheduled_tokens` —— 那会重复计一次；
          decode 步 `post = prompt + 已生成`，含本步写入的当前 token）；
        - `pre_computed = post - scheduled` 仅用于一致性校验。

        只对**非 global** 模式出计划：global（含全部 prefill 步）走原版读 metadata 路径；
        若在"尚未生成任何 token"（`pre < prompt_len`）时就出现非 global 模式，说明协议/时序有冲突，直接拒绝。
        """
        geometry = self.geometry()
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        plans: dict[str, dict] = {}
        for req_id in list(self._configs):
            config = self._configs[req_id]
            state = self.registry.get(req_id)
            request = self._requests().get(req_id)
            if request is None:
                # 本步已不在调度器账本里（终结/取消）：不产出计划
                continue
            prompt_len = int(getattr(request, "num_prompt_tokens", -1))
            if prompt_len != config.prompt_len:
                raise AttnViewConfigError(
                    f"{req_id}: 载荷 prompt_len={config.prompt_len} 与调度器 num_prompt_tokens="
                    f"{prompt_len} 不一致（布局事实冲突，拒绝出计划）"
                )
            scheduled = int(num_scheduled.get(req_id, 0))
            if scheduled <= 0:
                continue
            post_computed = int(getattr(request, "num_computed_tokens", 0))
            pre_computed = post_computed - scheduled
            if pre_computed < 0:
                raise AttnViewConfigError(
                    f"{req_id}: 调度后进度 {post_computed} 小于本步调度 token 数 {scheduled}"
                    "（进度账本异常）"
                )
            if state.mode == MODE_GLOBAL:
                # global 一律走原版读 metadata；prefill 步在此天然被跳过（此时模式必为 global）
                continue
            if pre_computed < prompt_len:
                raise AttnViewConfigError(
                    f"{req_id}: 尚未生成任何 token（pre_computed={pre_computed} < prompt_len="
                    f"{prompt_len}）却已是 {state.mode} 模式：协议/时序冲突，拒绝出计划"
                )
            groups = list(blocks_of(req_id))
            canonical = list(groups[geometry.fa_group_index])
            plan = build_step_plan(
                req_id=req_id,
                mode=state.mode,
                refs=state.refs,
                config=config,
                geometry=geometry,
                canonical_blocks=canonical,
                attention_kv_len=post_computed,
                effect_step=state.generated_tokens,
            )
            plans[req_id] = plan.as_payload()
        return plans

    def _requests(self) -> Mapping[str, Any]:
        return getattr(self._scheduler, "requests", None) or {}

    # --- 与 step 的接合 ----------------------------------------------------- #

    def attach_plans(self, scheduler_output: Any) -> dict[str, dict] | None:
        """**紧跟 `schedule()` 之后**调用：按本次调度的块与进度构造计划并注入。

        为什么必须在这里：块是按本次调度分配的 —— 例如 prompt_len=6272（8 块）时，
        消费 g0 的那次 forward 会写入块 8，而块 8 直到本次 `schedule()` 才分配；
        在 `schedule()` 之前索引 canonical 映射会越界。
        """
        if not self._configs:
            self._pending = None
            setattr(scheduler_output, "da_step_plans", None)
            return None
        manager = getattr(self._scheduler, "kv_cache_manager", None)
        if manager is None:
            raise RuntimeError("attnview: scheduler 上没有 kv_cache_manager，无法取 canonical 块")
        self._pending = self.build_plans(
            scheduler_output, blocks_of=lambda r: manager.get_block_ids(r)
        ) or None
        setattr(scheduler_output, "da_step_plans", self._pending)
        return self._pending

    @staticmethod
    def _finish_items(engine_core_outputs: Any) -> dict[str, Any]:
        """本步**正常终结**的请求 -> finish_reason（被取消的请求不会出现在这里）。"""
        items: dict[str, Any] = {}
        for outputs in (engine_core_outputs or {}).values():
            for item in getattr(outputs, "outputs", ()) or ():
                req_id = str(getattr(item, "request_id", ""))
                reason = getattr(item, "finish_reason", None)
                if req_id and reason is not None:
                    items[req_id] = reason
        return items

    def _drop_gone(self, *, alive: set[str], finished_now: Mapping[str, Any], preempted: set[str]) -> dict[str, list[str]]:
        """处理"账本里已消失、本步也没有正常终结项"的请求：**不解析**、直接释放并记 trace。

        这类请求是执行中被取消/中止（`_process_aborts_queue` 之后 `update_from_output` 会跳过、
        不产出 finish 项），也可能是被抢占（本阶段不支持 → 记 trace，驱动侧据此停该用例）。
        """
        dropped = {"cancelled": [], "preempted": []}
        for req_id in list(self._configs):
            if req_id in alive or req_id in finished_now:
                continue
            kind = "preempted" if req_id in preempted else "cancelled"
            state = None
            try:
                state = self.registry.get(req_id)
            except KeyError:
                pass
            self.traces.append(
                {
                    "req_id": req_id,
                    "note": f"{kind}_during_step",
                    "generated_tokens_before_drop": state.generated_tokens if state else None,
                    "mode_before_drop": state.mode if state else None,
                    "unsupported": kind == "preempted",
                }
            )
            self.release(req_id)
            dropped[kind].append(req_id)
        return dropped

    def on_step_outputs(self, scheduler_output: Any, model_output: Any, engine_core_outputs: Any) -> dict[str, Any]:
        """一步的解析与清理（**不**构造计划；计划在 `attach_plans` 里按调度结果构造）。

        顺序与判据（依 pin 的调度器语义）：

        1. 先丢弃 `SchedulerOutput.finished_req_ids`（这是**上一步**终结集合的快照 ——
           `Scheduler.schedule()` 在返回前已把 `self.finished_req_ids` 清空，`scheduler.py:1495`）；
        2. 核对调度器账本（`scheduler.requests`，`_free_request` 会 `del`，`:2512`）后**再登记**新请求 ——
           不在账本里的不登记；
        3. 账本里消失**且**本步无正常终结项的请求 = 执行中被取消/中止 → **不解析**、释放并记 trace；
        4. 只解析「仍存活」或「本步正常终结」的请求（正常终结的 stop token 仍应进入解析与 trace）；
        5. 释放本步正常终结者。
        """
        alive = set(self._requests())
        finished_now = self._finish_items(engine_core_outputs)
        self.drop_finished(getattr(scheduler_output, "finished_req_ids", ()) or ())
        self.register_new_requests(scheduler_output, alive=alive)
        preempted = {str(r) for r in (getattr(scheduler_output, "preempted_req_ids", None) or ())}
        dropped = self._drop_gone(alive=alive, finished_now=finished_now, preempted=preempted)
        parsed = self.parse_outputs(model_output, allow=alive | set(finished_now))
        self._record_traces_for(parsed)
        self.release_finished_in_step(engine_core_outputs)
        return {"parsed": parsed, "cancelled": dropped["cancelled"], "preempted": dropped["preempted"]}

    # --- trace ------------------------------------------------------------- #

    def _record_traces_for(self, parsed: Mapping[str, int]) -> None:
        for req_id, count in parsed.items():
            if count <= 0:
                continue
            state = self.registry.get(req_id)
            last = state.steps[-1]
            self.traces.append(
                {
                    "req_id": req_id,
                    "parse_step": last.token_index,
                    "effect_step": state.generated_tokens,
                    "applied_step": None,
                    "mode": last.mode_after,
                    "refs": list(last.refs),
                    "events": [e.kind for e in last.events],
                    "written_after": last.written_kv_len,
                    "attention_kv_len_next": last.attention_kv_len_next,
                }
            )

    def _record_trace(self, req_id: str, *, applied_step: int | None, note: str) -> None:
        state = self.registry.get(req_id)
        record = state.trace()
        record["note"] = note
        record["applied_step"] = applied_step
        self.traces.append(record)

    def note_application(self, req_id: str, *, applied_step: int, plan_payload: Mapping[str, Any]) -> None:
        """worker 实际落位后回填 applied_step（检查点 2 由 trace 汇聚处调用）。"""
        geometry = self.geometry()
        self.traces.append(
            {
                "req_id": req_id,
                "applied_step": int(applied_step),
                "seqused_k": int(plan_payload["seqused_k"]),
                "tail_len": int(plan_payload["tail_len"]),
                "visible_logical_blocks": list(plan_payload["visible_logical_blocks"]),
                "block_size": geometry.kernel_block_size,
                "width": geometry.max_width,
                "note": "worker 落位回填",
            }
        )

    def dump_traces(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.traces, ensure_ascii=False, indent=2))
        return target

    def pending_plans(self) -> dict[str, dict] | None:
        return self._pending
