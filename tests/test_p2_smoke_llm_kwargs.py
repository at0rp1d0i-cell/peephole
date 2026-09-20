"""SUP-004 smoke:LLM kwargs 与 prefill 生效值的 CPU 控制流测试(不加载模型)。

覆盖 advisor 指出的 `_run` 路径缺口:
- `arm_llm_overrides`:masked 固定 `max_num_batched_tokens=8192`;**旧三臂 kwargs 不变**(无该键);
- `read_max_num_batched_tokens`:必须在**构造成功之后**读取(传入未初始化的 llm ⇒ 明确报 unavailable,
  不静默、不抛 UnboundLocalError 掩盖);读出真实值;小于 prompt_len 时**拒绝**。
"""
from __future__ import annotations

import unittest

from _support import REPO, load_module


def load_runner():
    return load_module("p2_calib_run_kwargs", REPO / "tools/p2-calib-run.py")


class FakeScheduler:
    def __init__(self, value):
        self.max_num_batched_tokens = value


class FakeConfig:
    def __init__(self, value):
        self.scheduler_config = FakeScheduler(value)


class FakeEngine:
    def __init__(self, value):
        self.vllm_config = FakeConfig(value)


class FakeLLM:
    def __init__(self, value):
        self.llm_engine = FakeEngine(value)


class LlmKwargsAndEffectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_runner()

    def test_masked_fixes_prefill_bound(self):
        self.assertEqual(self.r.arm_llm_overrides("patched-masked"), {"max_num_batched_tokens": 8192})

    def test_legacy_arms_unchanged(self):
        for arm in ("original", "patched-disabled", "patched-global"):
            self.assertEqual(self.r.arm_llm_overrides(arm), {}, arm)

    def test_effective_value_read_after_construction(self):
        self.assertEqual(self.r.read_max_num_batched_tokens(FakeLLM(8192), prompt_len=7834), 8192)

    def test_effective_value_unavailable_is_recorded_not_raised(self):
        class NoEngine:
            llm_engine = None
        self.assertTrue(str(self.r.read_max_num_batched_tokens(NoEngine(), prompt_len=7834)).startswith("unavailable"))

    def test_effective_value_asserts_prompt_len(self):
        with self.assertRaises(RuntimeError):
            self.r.read_max_num_batched_tokens(FakeLLM(4096), prompt_len=7834)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
