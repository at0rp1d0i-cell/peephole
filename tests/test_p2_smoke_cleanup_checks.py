"""SUP-004 smoke:cleanup 阶段按 enforce_global 分支的控制流 CPU 测试(不加载模型/不跑 GPU)。

直接调用 runner 的纯函数 `cleanup_enforce_checks`,覆盖 True/False 两条分支:
- True(旧 global 臂):载荷必须 True;有非 global 步时必须有合规覆写 mark;全 global 时 marks=[] 正确。
- False(masked 臂):载荷必须 **False**;**不要求**任何覆写 mark,且必须**确认无强制全局标记**。
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_runner():
    spec = importlib.util.spec_from_file_location("p2_calib_run_cleanup", REPO / "tools/p2-calib-run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


GOOD_MARK = {"req_id": "new", "applied_view": "global", "protocol_mode": "local"}
BAD_MARK = {"req_id": "new", "applied_view": "local", "protocol_mode": "local"}


class CleanupEnforceChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = staticmethod(load_runner().cleanup_enforce_checks)

    def results(self, **kw):
        return {name: ok for name, ok, _msg in self.f(**kw)}

    def test_enforce_true_with_non_global_steps_requires_compliant_mark(self):
        r = self.results(enforce_global=True, config_enforce_global=True, marks=[GOOD_MARK],
                         non_global_steps=[3], observed_modes=["local"])
        self.assertTrue(r["new_req_payload_enforce_global"])
        self.assertTrue(r["new_req_enforce_global_step_recorded"])

    def test_enforce_true_with_bad_mark_fails(self):
        r = self.results(enforce_global=True, config_enforce_global=True, marks=[BAD_MARK],
                         non_global_steps=[3], observed_modes=["local"])
        self.assertFalse(r["new_req_enforce_global_step_recorded"])

    def test_enforce_true_all_global_accepts_empty_marks(self):
        r = self.results(enforce_global=True, config_enforce_global=True, marks=[],
                         non_global_steps=[], observed_modes=["global"])
        self.assertTrue(r["new_req_enforce_global_step_recorded"])

    def test_enforce_false_requires_false_config_and_no_marks(self):
        r = self.results(enforce_global=False, config_enforce_global=False, marks=[],
                         non_global_steps=[6, 7], observed_modes=["local"])
        self.assertTrue(r["new_req_payload_enforce_global"])
        self.assertTrue(r["masked_no_forced_global_marks"])

    def test_enforce_false_rejects_true_config(self):
        r = self.results(enforce_global=False, config_enforce_global=True, marks=[],
                         non_global_steps=[6], observed_modes=["local"])
        self.assertFalse(r["new_req_payload_enforce_global"])

    def test_enforce_false_rejects_present_marks(self):
        r = self.results(enforce_global=False, config_enforce_global=False, marks=[GOOD_MARK],
                         non_global_steps=[6], observed_modes=["local"])
        self.assertFalse(r["masked_no_forced_global_marks"])


if __name__ == "__main__":
    unittest.main()
