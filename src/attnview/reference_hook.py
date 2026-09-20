"""测试专用参考挂接(CPU):把独立 dense 参考接进既有 harness 的 impl 包装路径。

模式与 `tools/p2-calib-run.py` 的 `LayerCapture.wrap_impl` **一致**:
保存 `original = impl.forward` → 安装同签名 `wrapper` → 由调用方在 `finally` 里 `restore()`。

约束:
- **默认关闭**:未显式 `enabled=True` 且未命中目标 request/layer/step 时**逐字节走原实现**(passthrough);
- **恢复是恢复原方法本身**(不是翻 bool):`restore()` 把 `impl.forward` 还原为保存的原始 bound method 并断言身份;
- **异常安全**:参考路径抛错时立即 `restore()` 并关闭开关,后续调用回到原实现;
- **覆盖账本**:每次调用记 `(layer, step, action)`,用于检出"缺层/缺步"(期望被覆盖却走了 passthrough);
- 生产路径无调用点;不加载模型、不需要 GPU。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .reference_dense import (
    ReferenceError,
    TestOnlyReferenceSwitch,
    perform_reference_attention,
)

__all__ = ["TestOnlyReferenceAttachment"]


@dataclass
class TestOnlyReferenceAttachment:
    """把一个 `TestOnlyReferenceSwitch` 接到若干 `impl` 上,并记录覆盖账本。"""

    switch: TestOnlyReferenceSwitch
    layers: tuple[int, ...] = ()
    steps: tuple[int, ...] = ()
    ledger: list[dict[str, Any]] = field(default_factory=list)
    wrapped: dict[int, tuple[Any, Any]] = field(default_factory=dict)   # index -> (impl, original_forward)
    restored: list[int] = field(default_factory=list)

    # --- 挂接 / 恢复(与应用位置无关,只依赖 impl.forward 这一层) ------------- #
    def wrap_impl(self, index: int, impl: object) -> None:
        original = getattr(impl, "forward", None)
        if original is None or not callable(original):
            raise ReferenceError(f"impl 没有可包装的 forward:L{index}")
        if index in self.wrapped:
            raise ReferenceError(f"层 L{index} 已挂接,拒绝重复包装")
        self.wrapped[index] = (impl, original)

        def wrapper(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            rid = self.switch.current_request_id
            step = self.switch.current_step
            if not self.switch.should_override(request_id=rid or "", layer_idx=index, step=step):
                self.ledger.append({"layer": index, "step": step, "request_id": rid, "action": "passthrough"})
                return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            try:
                info = perform_reference_attention(self.switch, layer_idx=index, step=step, query=query,
                                                   kv_cache=kv_cache, attn_metadata=attn_metadata, output=output)
                self.ledger.append({"layer": index, "step": step, "request_id": rid, "action": "overrode", **info})
                return None                                # 与生产接线一致:返回值被忽略
            except Exception as exc:                       # noqa: BLE001
                self.switch.enabled = False
                self.restore()
                self.ledger.append({"layer": index, "step": step, "request_id": rid,
                                    "action": "error_restored", "error": f"{type(exc).__name__}: {exc}"[:200]})
                raise

        impl.forward = wrapper

    def restore(self) -> None:
        """恢复**原方法本身**并断言身份一致(不足即报错,不静默)。"""
        for index, (impl, original) in list(self.wrapped.items()):
            if index in self.restored:
                continue
            impl.forward = original
            if impl.forward is not original:
                raise ReferenceError(f"层 L{index} 恢复失败:forward 身份不一致")
            self.restored.append(index)

    def wrap_all(self, impls: Sequence[object]) -> None:
        for index, impl in enumerate(impls):
            self.wrap_impl(index, impl)

    # --- 上下文管理:保证 finally 恢复 ------------------------------------- #
    def __enter__(self) -> "TestOnlyReferenceAttachment":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.restore()
        return False

    # --- 账本查询 ---------------------------------------------------------- #
    def overrode(self) -> set[tuple[int, int]]:
        return {(e["layer"], e["step"]) for e in self.ledger if e["action"] == "overrode"}

    def passthrough(self) -> set[tuple[int, int]]:
        return {(e["layer"], e["step"]) for e in self.ledger if e["action"] == "passthrough"}

    def errors(self) -> list[dict[str, Any]]:
        return [e for e in self.ledger if e["action"].endswith("error_restored")]

    def missing(self, expected: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """期望被覆盖但实际未覆盖的 (layer, step) —— 检出"缺层/缺步"。"""
        return expected - self.overrode()

    def unexpected(self, expected: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return self.overrode() - expected
