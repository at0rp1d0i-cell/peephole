"""阶段 05 单请求接入：每步读取视图计划（纯逻辑，CPU 可跑）。

设计边界（依 SUP-004-R1）：
- 协议/视图语义只在本模块（阶段 03 已验收的 `readview`/`gpukv`）里，不复制第二套实现；
  vLLM 侧只做「取载荷 → 调用本模块 → 落位 metadata」。
- `seqused_k` **不是**完整历史长度、也不是可见块数 × 块大小：它等于各可见块被可见 span
  覆盖的位置数之和（最大可见块取实际尾长）。该算术完全交给
  `gpukv.read_table_from_read_view`，本模块不复写。
- `attention_kv_len` 是**本次 attention 发生时**的 canonical 有效长度，**含本次 forward
  正常写入的当前 token**；不得停在上一采样时刻。
- **几何（块大小/表宽/块数）不由调用者提供**：只接受运行期从实际 KV 缓存配置读到的
  `Geometry`；载荷里出现几何字段一律报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .gpukv import GpuKvError, read_table_from_read_view, validate_mapping
from .readview import (
    MODE_FOCUS,
    MODE_GLOBAL,
    MODE_LOCAL,
    ReadView,
    TokenLayout,
    ViewInputs,
    build_read_view,
)

EXTRA_ARG_KEY = "attnview"
PROTOCOL_VERSION = "v1.0"

#: 载荷里**禁止**出现的字段：几何必须来自运行期配置，避免调用者注入块大小/宽度。
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "kernel_block_size",
        "block_size",
        "max_width",
        "num_blocks",
        "blocks_per_kv_block",
        "fa_group_index",
        "num_kv_groups",
        "num_blocks_per_group",
    }
)


class AttnViewConfigError(ValueError):
    """载荷/配置不合法（调用方应修输入，不得放行）。"""


class UnsupportedConfig(RuntimeError):
    """当前运行配置不在本阶段支持范围内：拒绝服务，不做静默降级。"""


# --------------------------------------------------------------------------- #
# 载荷
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DaRequestConfig:
    """一次请求携带的协议配置（`SamplingParams.extra_args["attnview"]`）。

    `prompt_len` 与三个 span 是**协议/布局**事实（由驱动器在渲染 prompt 后算出），
    不是引擎几何；载荷不含任何块大小/宽度字段。
    """

    protocol: str
    prompt_len: int
    segment_spans: tuple[tuple[int, int], ...]
    local_window_span: tuple[int, int]
    sink_span: tuple[int, int]
    enforce_global: bool = False

    @staticmethod
    def from_extra_args(extra_args: Mapping[str, Any] | None) -> "DaRequestConfig":
        if not extra_args:
            raise AttnViewConfigError("extra_args 为空：内部请求必须携带协议配置")
        if not isinstance(extra_args, Mapping):
            raise AttnViewConfigError(f"extra_args 必须是映射，得到 {type(extra_args).__name__}")
        raw = extra_args.get(EXTRA_ARG_KEY)
        if raw is None:
            raise AttnViewConfigError(f"extra_args 缺少 {EXTRA_ARG_KEY!r} 键")
        if not isinstance(raw, Mapping):
            raise AttnViewConfigError(f"{EXTRA_ARG_KEY!r} 必须是映射")
        forbidden = sorted(set(raw) & FORBIDDEN_PAYLOAD_KEYS)
        if forbidden:
            raise AttnViewConfigError(
                f"载荷不得携带几何字段 {forbidden}：块大小/表宽必须取自运行期 KV 缓存配置"
            )
        protocol = str(raw.get("protocol", ""))
        if protocol != PROTOCOL_VERSION:
            raise AttnViewConfigError(f"协议版本不支持：{protocol!r}（期望 {PROTOCOL_VERSION!r}）")
        prompt_len = raw.get("prompt_len")
        if not isinstance(prompt_len, int) or isinstance(prompt_len, bool) or prompt_len <= 0:
            raise AttnViewConfigError(f"prompt_len 必须是正整数，得到 {prompt_len!r}")
        spans = _spans(raw.get("segment_spans"), "segment_spans", allow_empty=False)
        local_window = _span(raw.get("local_window_span"), "local_window_span", allow_empty=True)
        sink = _span(raw.get("sink_span", [0, 16]), "sink_span", allow_empty=False)
        for name, span in [
            ("sink_span", sink),
            ("local_window_span", local_window),
            *[(f"segment_spans[{i}]", s) for i, s in enumerate(spans)],
        ]:
            if span[1] > prompt_len:
                raise AttnViewConfigError(
                    f"{name}={span} 超出 prompt_len={prompt_len}（prompt 之外的区间不是输入侧 span）"
                )
        return DaRequestConfig(
            protocol=protocol,
            prompt_len=int(prompt_len),
            segment_spans=spans,
            local_window_span=local_window,
            sink_span=sink,
            enforce_global=bool(raw.get("enforce_global", False)),
        )

    def layout(self) -> TokenLayout:
        return TokenLayout(
            prompt_len=self.prompt_len,
            segment_spans=self.segment_spans,
            local_window_span=self.local_window_span,
            sink_span=self.sink_span,
        )


def _span(value: Any, name: str, *, allow_empty: bool) -> tuple[int, int]:
    if value is None and allow_empty:
        return (0, 0)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise AttnViewConfigError(f"{name} 必须是 [start, end] 二元组，得到 {value!r}")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise AttnViewConfigError(f"{name} 非法：{value!r}")
    return (start, end)


def _spans(value: Any, name: str, *, allow_empty: bool) -> tuple[tuple[int, int], ...]:
    if value is None:
        if allow_empty:
            return ()
        raise AttnViewConfigError(f"{name} 缺失")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AttnViewConfigError(f"{name} 必须是区间列表，得到 {type(value).__name__}")
    return tuple(_span(item, f"{name}[{i}]", allow_empty=False) for i, item in enumerate(value))


# --------------------------------------------------------------------------- #
# 运行期几何与支持范围
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Geometry:
    """实际运行期 KV 缓存几何（来自 worker 的 `KVCacheConfig`，不由调用者提供）。"""

    kernel_block_size: int
    num_kv_groups: int
    fa_group_index: int
    blocks_per_kv_block: int
    max_model_len: int

    def validate(self) -> "Geometry":
        if self.kernel_block_size <= 0:
            raise AttnViewConfigError(f"kernel_block_size 必须为正：{self.kernel_block_size}")
        if self.num_kv_groups <= 0:
            raise AttnViewConfigError(f"num_kv_groups 必须为正：{self.num_kv_groups}")
        if not 0 <= self.fa_group_index < self.num_kv_groups:
            raise AttnViewConfigError(
                f"fa_group_index={self.fa_group_index} 越界（共 {self.num_kv_groups} 组）"
            )
        if self.blocks_per_kv_block < 1:
            raise AttnViewConfigError(f"blocks_per_kv_block 必须 ≥1：{self.blocks_per_kv_block}")
        if self.blocks_per_kv_block != 1:
            # manager/kernel 块粒度不一致时，逻辑块↔kernel 列的换算会改变 seqused_k 的
            # 计量单位（kernel 块）；必须显式拒绝，不能算错。
            raise UnsupportedConfig(
                f"manager/kernel 块粒度不一致（blocks_per_kv_block={self.blocks_per_kv_block}）："
                "本阶段不支持，拒绝启用声明式读取"
            )
        if self.max_model_len <= 0:
            raise AttnViewConfigError(f"max_model_len 必须为正：{self.max_model_len}")
        return self

    @property
    def max_width(self) -> int:
        """一次生成内常量表宽（I5）：完整历史的最大块数，必 ≥ 任何一步的 `needed_width`。"""
        return -(-self.max_model_len // self.kernel_block_size)


def check_supported_config(
    *,
    async_scheduling: object,
    max_concurrent_batches: int,
    cudagraph_mode: object,
    enable_prefix_caching: object,
    speculative_config: object,
) -> None:
    """拒绝本阶段未支持的运行配置（不静默降级）。"""
    if async_scheduling:
        raise UnsupportedConfig(
            "async_scheduling=True：默认异步流水在未经额外依赖处理时无法保证"
            "「第 t 步解析、第 t+1 步生效」；本阶段拒绝该配置，请用 --no-async-scheduling"
        )
    if int(max_concurrent_batches) != 1:
        raise UnsupportedConfig(
            f"max_concurrent_batches={max_concurrent_batches}：批次队列会让下一步 metadata "
            "早于上一步 token 回到 host 构建；本阶段要求 ==1"
        )
    mode = str(cudagraph_mode)
    if mode not in ("CUDAGraphMode.NONE", "None", "0", "cudagraph_mode.NONE"):
        raise UnsupportedConfig(f"cudagraph_mode={cudagraph_mode}：本阶段只跑 eager（--enforce-eager）")
    if enable_prefix_caching:
        raise UnsupportedConfig("enable_prefix_caching=True：本阶段要求关闭前缀缓存")
    if speculative_config is not None:
        raise UnsupportedConfig("speculative_config 非空：本阶段要求关闭投机/MTP")


# --------------------------------------------------------------------------- #
# 每步计划
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepPlan:
    """某个 DA 请求在某一步 forward 的读取计划（下沉给 worker 的最小载荷）。"""

    req_id: str
    mode: str
    refs: tuple[int, ...]
    effect_step: int
    visible_logical_blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]
    seqused_k: int
    block_size: int
    width: int
    tail_len: int
    attention_kv_len: int
    next_write_position: int
    written_before_step: int

    @property
    def needed_width(self) -> int:
        """后端按前缀语义实际会索引的列数 = `ceil(seqused_k / kernel_block_size)`。"""
        return -(-self.seqused_k // self.block_size) if self.block_size > 0 else 0

    def validate(self) -> "StepPlan":
        if self.seqused_k <= 0:
            raise AttnViewConfigError(f"seqused_k 必须为正：{self.seqused_k}")
        if self.needed_width != len(self.visible_logical_blocks):
            raise AttnViewConfigError(
                f"needed_width={self.needed_width} 与可见块数 {len(self.visible_logical_blocks)} "
                "不一致：压缩表的前缀语义会索引不存在的列"
            )
        if self.width < self.needed_width:
            raise AttnViewConfigError(
                f"表宽 {self.width} < needed_width={self.needed_width}：会越界读"
            )
        if self.seqused_k > self.attention_kv_len:
            raise AttnViewConfigError(
                f"seqused_k={self.seqused_k} 超过 canonical 有效长度 {self.attention_kv_len}"
            )
        if self.tail_len != self.effective_per_block[-1]:
            raise AttnViewConfigError("tail_len 与最后一个可见块的有效位置数不一致")
        return self

    def as_payload(self) -> dict:
        """下推内容：只用原生类型（msgspec 可编码）；**不含几何**（worker 自取运行期配置）。"""
        self.validate()
        return {
            "req_id": self.req_id,
            "mode": self.mode,
            "refs": list(self.refs),
            "effect_step": self.effect_step,
            "visible_logical_blocks": list(self.visible_logical_blocks),
            "effective_per_block": list(self.effective_per_block),
            "seqused_k": self.seqused_k,
            "tail_len": self.tail_len,
            "attention_kv_len": self.attention_kv_len,
            "next_write_position": self.next_write_position,
            "written_before_step": self.written_before_step,
        }


def build_step_plan(
    *,
    req_id: str,
    mode: str,
    refs: Sequence[int],
    config: DaRequestConfig,
    geometry: Geometry,
    canonical_blocks: Sequence[int | None],
    attention_kv_len: int,
    effect_step: int,
) -> StepPlan:
    """构造一步的读取计划；算术全部复用已验收的 `readview` + `gpukv`。

    `attention_kv_len` 必须是**本次 forward 写入之后**的 canonical 有效长度（含当前 token）。
    `canonical_blocks` 是逻辑块 → 物理块 id 的完整映射（`None` = 未分配），索引即逻辑块号。
    """
    geometry.validate()
    if mode not in (MODE_GLOBAL, MODE_FOCUS, MODE_LOCAL):
        raise AttnViewConfigError(f"未知模式：{mode!r}")
    if effect_step < 0:
        raise AttnViewConfigError(f"effect_step 非法：{effect_step}")
    if int(attention_kv_len) <= 0:
        raise AttnViewConfigError(f"attention_kv_len 必须为正：{attention_kv_len}")

    index_to_block = resolve_canonical_mapping(canonical_blocks)
    if mode == MODE_LOCAL and config.local_window_span == (0, 0):
        raise AttnViewConfigError("local 模式需要 local_window_span（缺省 (0,0) 不可用）")

    inputs = ViewInputs(
        mode=mode,
        refs=tuple(int(r) for r in refs),
        layout=config.layout(),
        attention_kv_len=int(attention_kv_len),
        canonical_blocks=index_to_block,
        kernel_block_size=geometry.kernel_block_size,
        max_width=geometry.max_width,
        effect_step=int(effect_step),
    )
    view: ReadView = build_read_view(inputs)
    table = read_table_from_read_view(view, index_to_block)
    if len(table.physical_blocks) != len(table.visible_blocks):
        raise GpuKvError("读取表块数与可见块数不一致")
    plan = StepPlan(
        req_id=req_id,
        mode=view.mode,
        refs=tuple(view.declared_refs),
        effect_step=view.effect_step,
        visible_logical_blocks=tuple(int(b) for b in table.visible_blocks),
        effective_per_block=tuple(int(c) for c in table.effective_per_block),
        seqused_k=int(table.seqused_k),
        block_size=int(table.block_size),
        width=int(view.max_width),
        tail_len=int(table.tail_len),
        attention_kv_len=int(table.attention_kv_len),
        next_write_position=int(table.next_write_position),
        written_before_step=int(view.written_before_step),
    )
    return plan.validate()


def resolve_canonical_mapping(canonical_blocks: Sequence[int | None]) -> tuple[int, ...]:
    """`canonical_blocks` → `gpukv` 需要的完整物理映射。

    只接受**已分配前缀**：列表里出现 `None` 直接报错——过滤掉 `None` 会让后续索引整体前移，
    那样算出的块号是错的（这正是要避免的静默错误），因此不做任何过滤。
    """
    if canonical_blocks is None:
        raise AttnViewConfigError("canonical_blocks 缺失")
    raw = list(canonical_blocks)
    if not raw:
        raise AttnViewConfigError("canonical_blocks 为空")
    unallocated = [i for i, b in enumerate(raw) if b is None]
    if unallocated:
        raise AttnViewConfigError(
            f"canonical_blocks 含未分配槽 {unallocated}：只传已分配前缀，不要用 None 占位"
        )
    try:
        return validate_mapping(tuple(int(b) for b in raw))
    except GpuKvError as exc:
        raise AttnViewConfigError(f"canonical 块映射不合法：{exc}") from exc


# --------------------------------------------------------------------------- #
# 门禁与 trace
# --------------------------------------------------------------------------- #


def should_apply_view(
    *,
    is_dummy_run: bool,
    is_prefilling: bool,
    has_plan: bool,
    request_uses_attnview: bool,
) -> bool:
    """是否把计划落位到 FA 组 metadata。

    - `dummy_run`（profile/warmup 的 dummy 路径）一律不落位；
    - **prefill 一律走原版**：判据是运行期 `is_prefilling`（末尾 prefill chunk 可能只有 1 个
      token，不能用 `query_len==1` 代替）；
    - 普通请求（无载荷）直通，不创建协议状态。
    """
    if is_dummy_run or is_prefilling or not has_plan:
        return False
    return bool(request_uses_attnview)


@dataclass(frozen=True)
class TraceRecord:
    """逐步 trace（协议合同 C6.3 的字段子集 + 本阶段必需的时序字段）。"""

    req_id: str
    parse_step: int
    effect_step: int
    applied_step: int | None
    mode: str
    refs: tuple[int, ...]
    visible_blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]
    seqused_k: int
    width: int
    tail_len: int
    attention_kv_len: int
    next_write_position: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "parse_step": self.parse_step,
            "effect_step": self.effect_step,
            "applied_step": self.applied_step,
            "mode": self.mode,
            "refs": list(self.refs),
            "visible_blocks": list(self.visible_blocks),
            "effective_per_block": list(self.effective_per_block),
            "seqused_k": self.seqused_k,
            "width": self.width,
            "tail_len": self.tail_len,
            "attention_kv_len": self.attention_kv_len,
            "next_write_position": self.next_write_position,
            "notes": list(self.notes),
        }


def trace_from_plan(plan: StepPlan, *, parse_step: int, applied_step: int | None) -> TraceRecord:
    return TraceRecord(
        req_id=plan.req_id,
        parse_step=int(parse_step),
        effect_step=int(plan.effect_step),
        applied_step=applied_step,
        mode=plan.mode,
        refs=plan.refs,
        visible_blocks=plan.visible_logical_blocks,
        effective_per_block=plan.effective_per_block,
        seqused_k=plan.seqused_k,
        width=plan.width,
        tail_len=plan.tail_len,
        attention_kv_len=plan.attention_kv_len,
        next_write_position=plan.next_write_position,
    )
