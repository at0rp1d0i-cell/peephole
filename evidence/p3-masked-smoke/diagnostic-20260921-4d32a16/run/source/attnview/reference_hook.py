"""测试专用参考挂接(CPU):按**固定 pin 的真实签名**把独立 dense 参考接进 impl 包装路径。

模式与 `tools/p2-calib-run.py` 的 `LayerCapture.wrap_impl` 一致:
保存 `original = impl.forward` → 安装同签名 wrapper → 调用方在 `finally` 里 `restore()`(恢复原方法并校验身份)。

wrapper 收到的就是真实调用形态:
`(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)`,
其中 `kv_cache` 为原生 4 维张量、`attn_metadata` 为真实 `FlashAttentionMetadata`;
模式/几何/独立时间线由 `TimelineBridge` 桥接(**不**塞进 metadata)。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .reference_bridge import TimelineBridge, perform_reference_attention_native
from .reference_dense import ReferenceError, TestOnlyReferenceSwitch

__all__ = ["TestOnlyReferenceAttachment"]

Entry = Callable[..., dict[str, Any]]


@dataclass
class TestOnlyReferenceAttachment:
    """把参考接到若干 `impl` 上;默认关闭、只对目标 request/层/步生效、带覆盖账本。"""

    switch: TestOnlyReferenceSwitch
    bridge: TimelineBridge | None = None
    request_idx: int = 0
    head_size: int = 0
    layers: tuple[int, ...] = ()
    steps: tuple[int, ...] = ()
    entry: Entry = perform_reference_attention_native
    ledger: list[dict[str, Any]] = field(default_factory=list)
    wrapped: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    restored: list[int] = field(default_factory=list)
    before_forward: Callable[..., None] | None = None

    def _call_entry(self, *, index: int, step: int, query, key, value, kv_cache, attn_metadata, output, impl):
        if self.bridge is None:
            raise ReferenceError("未提供 TimelineBridge:参考路径无法确定独立时间线/几何")
        impl_scale = getattr(impl, "scale", None)
        if impl_scale is None:
            raise ReferenceError(f"L{index} 的 impl 没有 .scale:拒绝用 head_dim 推导替代")
        if self.switch.scale and float(self.switch.scale) != float(impl_scale):
            raise ReferenceError(
                f"开关 scale={self.switch.scale} 与 impl.scale={impl_scale} 不一致:拒绝继续")
        return self.entry(self.switch, layer_idx=index, step_index_0based=step - 1, bridge=self.bridge,
                          request_idx=self.request_idx, query=query, key=key, value=value, kv_cache=kv_cache,
                          attn_metadata=attn_metadata, output=output, head_size=self.head_size,
                          impl_scale=float(impl_scale))

    def wrap_impl(self, index: int, impl: object) -> None:
        original = getattr(impl, "forward", None)
        if original is None or not callable(original):
            raise ReferenceError(f"impl 没有可包装的 forward:L{index}")
        if index in self.wrapped:
            raise ReferenceError(f"层 L{index} 已挂接,拒绝重复包装")
        self.wrapped[index] = (impl, original)

        def wrapper(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            rid = self.switch.current_request_id or ""
            step = self.switch.current_step
            # 先判未启用/非目标:按**原签名**直通,不做任何额外参数检查(默认关闭时行为不变)
            if not self.switch.enabled or rid != self.switch.request_id:
                self.ledger.append({"layer": index, "step": step, "request_id": rid, "action": "passthrough"})
                return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            try:
                if self.before_forward is not None:
                    self.before_forward(index=index, query=query, key=key, value=value,
                                        kv_cache=kv_cache, metadata=attn_metadata)
                if not self.switch.should_override(request_id=rid, layer_idx=index, step=step):
                    self.ledger.append({"layer": index, "step": step, "request_id": rid, "action": "passthrough"})
                    return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
                # 目标请求:pin 每次都传 output_scale/output_block_scale(可能为 None);
                # 显式 None = 未启用该特性,允许;非 None 或未知参数 = 量化/特殊特性,拒绝(不静默降级)。
                known = {"output_scale", "output_block_scale"}
                unsupported = {k: v for k, v in kwargs.items() if k not in known or v is not None}
                if args or unsupported:
                    raise ReferenceError(
                        f"Unsupported attention arguments: args={len(args)}, kwargs={sorted(unsupported)}")
                # 原调用者忽略返回值;用哨兵检出只返回新张量或遗漏写回。
                output.fill_(float("nan"))
                info = self._call_entry(index=index, step=step, query=query, key=key, value=value,
                                        kv_cache=kv_cache, attn_metadata=attn_metadata, output=output,
                                        impl=self.wrapped[index][0])
                if not torch.isfinite(output).all():
                    raise ReferenceError("Reference did not fully write the original output buffer, or produced nonfinite values")
                self.ledger.append({"layer": index, "step": step, "request_id": rid, "action": "overrode", **info})
                return None                     # 与生产接线一致:返回值被忽略,结果写在传入 output
            except Exception as exc:            # noqa: BLE001
                self.switch.enabled = False
                self.restore()
                self.ledger.append({"layer": index, "step": step, "request_id": rid,
                                    "action": "error_restored", "error": f"{type(exc).__name__}: {exc}"[:200]})
                raise

        impl.forward = wrapper

    def restore(self) -> None:
        """恢复**原方法本身**并断言身份一致(不是翻 bool)。"""
        for index, (impl, original) in list(self.wrapped.items()):
            impl.forward = original
            if impl.forward is not original:
                raise ReferenceError(f"层 L{index} 恢复失败:forward 身份不一致")
            if index not in self.restored:
                self.restored.append(index)

    def wrap_all(self, impls: Sequence[object]) -> None:
        for index, impl in enumerate(impls):
            self.wrap_impl(index, impl)

    def __enter__(self) -> TestOnlyReferenceAttachment:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.restore()
        return False

    def overrode(self) -> set[tuple[int, int]]:
        return {(e["layer"], e["step"]) for e in self.ledger if e["action"] == "overrode"}

    def passthrough(self, request_id: str | None = None) -> set[tuple[int, int]]:
        return {(e["layer"], e["step"]) for e in self.ledger
                if e["action"] == "passthrough" and (request_id is None or e["request_id"] == request_id)}

    def errors(self) -> list[dict[str, Any]]:
        return [e for e in self.ledger if e["action"].endswith("error_restored")]

    def missing(self, expected: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return expected - self.overrode()

    def unexpected(self, expected: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return self.overrode() - expected

    def metrics(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in e.items() if k not in ("action",)} for e in self.ledger if e["action"] == "overrode"]
