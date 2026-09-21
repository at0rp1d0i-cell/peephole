"""测试专用独立 dense 参考（CPU）。

用途:为 P2 的 attention/logits 对照提供**独立**参考路径。本模块**不属于生产路径**:
默认关闭,只在测试工具显式启用且仅对目标 request id 生效;生产代码没有任何调用点。

独立性约束(见 `reports/p2-single/masked-reference-design.md`):
- 可见位置只由**协议原始 span + 独立时间线(mode/refs)**算出,不调用候选的筛选/块表转换 helper;
- K/V 取值用**物理块表 + 逐位置 gather**,不读压缩读表,不读未写/越界槽;
- 参考结果在 FP32 显式计算,再按**真实输出 dtype** cast。

接线约束(vLLM pin 98dff2a8):`unified_attention_with_output` 调用 `impl.forward` 后
**继续使用传入的 `output` 缓冲、忽略返回值** ⇒ 参考实现**必须原地写入 output**,只返回新张量无效。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "ReferenceError",
    "VisibleMask",
    "independent_visible_positions",
    "gather_positions",
    "dense_attention_fp32",
    "TestOnlyReferenceSwitch",
]


class ReferenceError(RuntimeError):
    """参考路径的输入或接线不满足前提(不静默近似)。"""


@dataclass(frozen=True)
class VisibleMask:
    """两类位置**分开记录**,用途不同:

    - `semantic_positions`:协议语义的可见位置(原始 span 并集,截到 kv_len);
    - `read_positions`:合同实际读取位置 = semantic **向块边界外扩后**再**截断到 kv_len**
      (与 `readview.build_read_view` 的 I1/I6 一致);**dense 参考必须用这一组**;
    - `blocks`/`effective_per_block`:读取位置对应的逻辑块集与每块有效位置数(块内尾块被截断)。
    """

    semantic_positions: tuple[int, ...]
    read_positions: tuple[int, ...]
    blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]
    kv_len: int
    current_token_position: int
    mode: str
    refs: tuple[int, ...]
    block_size: int

    def __post_init__(self) -> None:
        if self.kv_len <= 0:
            raise ReferenceError(f"kv_len 必须为正:{self.kv_len}")
        if self.block_size <= 0:
            raise ReferenceError("block_size 必须为正")
        if self.current_token_position != self.kv_len - 1:
            raise ReferenceError("当前 token 位置必须是 kv_len - 1")
        for name, seq in (("semantic", self.semantic_positions), ("read", self.read_positions)):
            if any(p < 0 or p >= self.kv_len for p in seq):
                raise ReferenceError(f"{name} 位置越界(不得包含未写或负数位置)")
            if sorted(set(seq)) != list(seq):
                raise ReferenceError(f"{name} 位置必须严格升序且去重")
        if not set(self.semantic_positions) <= set(self.read_positions):
            raise ReferenceError("语义位置必须被读取位置覆盖(外扩只增不减)")
        if sum(self.effective_per_block) != len(self.read_positions):
            raise ReferenceError("每块有效位置数之和必须等于读取位置数")
        if tuple(sorted(self.blocks)) != self.blocks or len(set(self.blocks)) != len(self.blocks):
            raise ReferenceError("块集必须严格升序且唯一")


def block_expanded_positions(
    semantic: Sequence[int], *, block_size: int, kv_len: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """语义位置 → (读取位置, 逻辑块集, 每块有效位置数)。

    合同顺序:按**连续段**向块边界外扩,再截断到 `kv_len`(不读未写位置)。
    """
    if block_size <= 0:
        raise ReferenceError("block_size 必须为正")
    if not semantic:
        raise ReferenceError("语义位置为空")
    spans: list[list[int]] = []
    for p in semantic:
        if spans and p == spans[-1][1] + 1:
            spans[-1][1] = p
        else:
            spans.append([p, p])
    out: set[int] = set()
    for s0, e0 in spans:
        b0, b1 = s0 // block_size, e0 // block_size
        out.update(range(b0 * block_size, min((b1 + 1) * block_size, kv_len)))
    read = tuple(sorted(out))
    blocks = tuple(sorted({p // block_size for p in read}))
    counts = tuple(max(0, min(kv_len, (b + 1) * block_size) - b * block_size) for b in blocks)
    return read, blocks, counts


def independent_visible_positions(
    *,
    mode: str,
    refs: Sequence[int],
    kv_len: int,
    prompt_len: int,
    sink_span: tuple[int, int],
    local_window_span: tuple[int, int],
    segment_spans: Sequence[tuple[int, int]],
    block_size: int,
) -> VisibleMask:
    """按协议语义(C4.1)由**原始 span** 独立算出 semantic 位置,再按合同规则外扩成读取位置。"""
    if mode not in ("global", "focus", "local"):
        raise ReferenceError(f"未知模式:{mode!r}")
    spans: list[tuple[int, int]] = [tuple(sink_span), tuple(local_window_span), (prompt_len, kv_len)]
    if mode == "global":
        spans.append((0, kv_len))
    elif mode == "focus":
        if not refs:
            raise ReferenceError("focus 模式必须给出引用")
        for r in refs:
            if not 1 <= int(r) <= len(segment_spans):
                raise ReferenceError(f"引用越界:{r}(共 {len(segment_spans)} 段)")
            spans.append(tuple(segment_spans[int(r) - 1]))
    semantic = tuple(sorted({p for s0, e0 in spans for p in range(max(0, int(s0)), min(int(e0), kv_len))}))
    read, blocks, counts = block_expanded_positions(semantic, block_size=block_size, kv_len=kv_len)
    return VisibleMask(semantic_positions=semantic, read_positions=read, blocks=blocks,
                       effective_per_block=counts, kv_len=kv_len,
                       current_token_position=kv_len - 1, mode=mode,
                       refs=tuple(int(r) for r in refs), block_size=block_size)


def gather_positions(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: Sequence[int],
    block_size: int,
    positions: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """按**物理块表**逐位置 gather K/V;不做填充、不读未分配/未写槽。

    `block_table[i]` = 逻辑块 i 的物理块号(允许非顺序,以区分逻辑/物理编号)。
    传入的 `positions` 必须是**外扩后的读取位置**(见 `VisibleMask.read_positions`)。
    """
    if block_size <= 0:
        raise ReferenceError("block_size 必须为正")
    if not positions:
        raise ReferenceError("可见位置为空")
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ReferenceError("K/V 缓存块数不一致")
    n_blocks = int(k_cache.shape[0])
    last = max(positions)
    need_blocks = last // block_size
    if need_blocks >= len(block_table):
        raise ReferenceError(f"位置 {last} 需要逻辑块 {need_blocks},但物理块表只有 {len(block_table)} 项")
    if min(positions) < 0 or k_cache.shape != v_cache.shape or k_cache.shape[1] != block_size:
        raise ReferenceError("Invalid positions or KV cache geometry")
    physical = [int(block_table[int(p) // block_size]) for p in positions]
    if any(not 0 <= p < n_blocks for p in physical):
        raise ReferenceError("Physical block for a visible position is out of bounds")
    # 批量 gather 只复制可见位置;不对整份原生 cache 做 contiguous。
    ids = torch.tensor(physical, device=k_cache.device, dtype=torch.long)
    offsets = torch.tensor([int(p) % block_size for p in positions], device=k_cache.device, dtype=torch.long)
    return k_cache[ids, offsets], v_cache[ids, offsets]


def dense_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal_mask: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 显式 QK/softmax/V。

    返回 `(未 cast 的 FP32 参考, 按 `dtype` cast 后的结果)`;调用方必须明确比较对象。
    `q` 形状 `[heads, d]`(单 token decode);`k`/`v` 形状 `[positions, kv_heads, d]`。
    """
    if q.dim() != 2 or k.dim() != 3 or v.shape != k.shape:
        raise ReferenceError(f"形状不匹配:q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    heads, d = q32.shape
    kv_heads = k32.shape[1]
    if heads % kv_heads != 0:
        raise ReferenceError(f"GQA 头数不整除:q_heads={heads} kv_heads={kv_heads}")
    group = heads // kv_heads
    qg = q32.reshape(kv_heads, group, d)                      # [kvh, group, d]
    # 逐位置 score: [kvh, group, positions]
    scores = torch.einsum("hgd,phd->hgp", qg, k32) * float(scale)
    if causal_mask:
        scores = scores.masked_fill(torch.arange(k32.shape[0]).view(1, 1, -1) > (k32.shape[0] - 1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hgp,phd->hgd", weights, v32).reshape(heads, d)
    if dtype is None:
        return out, out
    return out, out.to(dtype)


@dataclass
class TestOnlyReferenceSwitch:
    """测试专用开关:**默认关闭**;只对目标 request id + 指定层/步启用;异常后自动恢复(关闭)。

    与 vLLM 的接线约定:本类只提供"写入传入 output 缓冲"的入口,返回值恒为 None,
    因为 `unified_attention_with_output` 忽略 `impl.forward` 的返回值。
    """

    enabled: bool = False
    request_id: str | None = None
    layers: tuple[int, ...] = ()
    steps: tuple[int, ...] = ()
    scale: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    current_request_id: str | None = None
    """运行期正在驱动的请求 id(由 harness 在每步设置);未设置时 `should_override` 一律返回 False。"""
    current_step: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    restores: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def should_override(self, *, request_id: str, layer_idx: int, step: int) -> bool:
        if not self.enabled or self.request_id is None:
            return False
        if request_id != self.request_id:
            return False
        if self.layers and layer_idx not in self.layers:
            return False
        if self.steps and step not in self.steps:
            return False
        return True

    def forward_with_output(
        self,
        *,
        request_id: str,
        layer_idx: int,
        step: int,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: Sequence[int],
        block_size: int,
        mask: VisibleMask,
        output: torch.Tensor,
    ) -> None:
        """把参考结果**原地写入** `output`,返回 None(与生产接线一致)。

        任何异常都记录并关闭开关(恢复为默认关闭),不留下"半启用"状态。
        """
        if not self.should_override(request_id=request_id, layer_idx=layer_idx, step=step):
            raise ReferenceError("开关未对本次调用启用:调用方不应进入参考路径")
        try:
            k, v = gather_positions(k_cache, v_cache, block_table, block_size, mask.read_positions)
            fp32, casted = dense_attention_fp32(q, k, v, scale=self.scale, dtype=self.dtype)
            if tuple(output.shape) != tuple(casted.shape):
                raise ReferenceError(f"output 形状 {tuple(output.shape)} 与参考 {tuple(casted.shape)} 不一致")
            output.copy_(casted)                              # 原地写入:必须写传入缓冲
            self.calls.append({"request_id": request_id, "layer_idx": layer_idx, "step": step,
                               "semantic_positions": len(mask.semantic_positions),
                               "read_positions": len(mask.read_positions), "fp32_max_abs": float(fp32.abs().max()),
                               "wrote_in_place": True})
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.enabled = False
            self.restores.append("exception→disabled")
            raise


def return_only_variant(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """反例辅助:只**返回**新张量、不写传入缓冲(用于验证测试能检出这种错误接线)。"""
    _fp32, casted = dense_attention_fp32(q, k, v, scale=scale, dtype=dtype)
    return casted
