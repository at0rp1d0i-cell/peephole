"""pin 锚点：本项目的时序/清理语义依赖哪一份上游源码。

为什么用**文件哈希**而不是断言上游源码文本：`vllm/` 是 pin 的只读 checkout，它的字节变了
只可能是"pin 动了"或"checkout 被改过"，两种情况都必须人工重核下面这些语义依赖；而
`assertIn("request.num_computed_tokens += ...", 上游源码)` 会因为上游改个变量名就变红，
且并不证明我们的假件与它等价（后者由 `tests/test_p2_engine.py` 直接执行上游实现来验证）。

被锚定的语义依赖（全部在 `vllm/v1/core/sched/scheduler.py`）：

1. `schedule()` 返回**之前**调用 `_update_after_schedule`，按本次调度数推进
   `num_computed_tokens` —— `attnview_engine.build_plans` 的
   `attention_kv_len = post_computed_tokens` 依赖这一点；
2. `schedule()` 返回前清空 `finished_req_ids` —— `on_step_outputs` 读的是**上一步**的终结集合；
3. 取消/终结的请求会被摘出 `self.requests` 账本 —— `_drop_gone` 据此判定"执行中被取消"；
4. 被抢占的请求**不会**被摘出账本 —— `_reject_preempted` 据此把抢占与取消分开处理。

pin 升级时的更新方式：逐条核对上面 4 条是否仍成立（不成立则改对应实现），再更新本文件的
`PIN_COMMIT` 与 `PIN_ANCHORS`。
"""

from __future__ import annotations

import hashlib
import unittest

from _support import REPO

#: 与 `vllm-patch/manifest.json` 的 `pin_commit` 同源（该字段由 `tools/p2-gen-patch.py` 现算）。
PIN_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"

#: 相对 vLLM 包根的文件 → 该 pin 下的 sha256。
PIN_ANCHORS = {
    "v1/core/sched/scheduler.py": "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983",
}


class PinAnchorTest(unittest.TestCase):
    def test_anchored_upstream_bytes_match_pin(self) -> None:
        pkg = REPO / "vllm" / "vllm"
        if not pkg.is_dir():
            self.skipTest("缺少 pin 的 vLLM checkout（先跑 install-runtime.sh）")
        for rel, want in sorted(PIN_ANCHORS.items()):
            path = pkg / rel
            self.assertTrue(path.is_file(), f"锚定文件缺失：{rel}")
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                got,
                want,
                f"{rel} 的字节与 pin {PIN_COMMIT[:8]} 不一致：上游事实可能已变，"
                "先按本文件 docstring 的 4 条依赖逐条核对再更新哈希",
            )

    def test_manifest_pin_commit_matches_anchor(self) -> None:
        manifest = REPO / "vllm-patch" / "manifest.json"
        if not manifest.is_file():
            self.skipTest("缺少 vllm-patch/manifest.json（先跑 tools/p2-gen-patch.py）")
        import json

        recorded = json.loads(manifest.read_text())["pin_commit"]
        self.assertEqual(
            recorded,
            PIN_COMMIT,
            "清单里的 pin 与本文件的锚点不是同一个 revision：锚点必须跟着 pin 走",
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
