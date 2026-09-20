"""SUP-004 masked smoke:CPU 检查 runner 的 fixture builder 与轨迹绑定(不加载模型/不跑 GPU)。

- 实际**调用 runner 的** `build_prompt_from_fixture`(经 importlib 加载该脚本)并与已验收值比对;
- 篡改 expect 的 `prompt_len` 必须被**拒绝**;
- 生成器产出的轨迹必须能被 runner 的 `load_trajectory` **绑定**到同一 prompt;换一条 prompt 必须拒绝。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DOC = REPO / "evidence/p1-cpu/demo-fixtures.json"
EXPECT = REPO / "evidence/p3-calib/masked-prep/crossblock-note.json"
TIMELINE = REPO / "configs/p2-masked-prep/crossblock.json"
TRAJ = REPO / "evidence/p3-masked-smoke/traj-7834-generated.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("p2_calib_run", REPO / "tools/p2-calib-run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class SmokeFixtureBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DOC.exists() and EXPECT.exists() and TIMELINE.exists()):
            raise unittest.SkipTest("夹具不可用")
        cls.runner = load_runner()

    def test_builder_matches_accepted_fixture(self):
        prompt, payload, evidence = self.runner.build_prompt_from_fixture(DOC, EXPECT, TIMELINE)
        exp = json.loads(EXPECT.read_text())
        self.assertEqual(len(prompt.token_ids), exp["prompt_len"])
        self.assertEqual(payload["prompt_len"], exp["prompt_len"])
        self.assertEqual([list(x) for x in prompt.segment_spans], exp["segment_spans"])
        self.assertEqual(evidence["checked"]["prompt_len"], exp["prompt_len"])

    def test_builder_rejects_tampered_expectation(self):
        exp = json.loads(EXPECT.read_text())
        exp["prompt_len"] = int(exp["prompt_len"]) + 1
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "expect.json"
            bad.write_text(json.dumps(exp, ensure_ascii=False))
            with self.assertRaises(RuntimeError):
                self.runner.build_prompt_from_fixture(DOC, bad, TIMELINE)

    def test_trajectory_binds_to_same_prompt(self):
        prompt, _payload, _evidence = self.runner.build_prompt_from_fixture(DOC, EXPECT, TIMELINE)
        prompt_ids = [int(t) for t in prompt.token_ids]
        traj = self.runner.load_trajectory(TRAJ, prompt_ids=prompt_ids)
        self.assertEqual(len(traj["tokens"]), 29)
        self.assertEqual(json.loads(TRAJ.read_text())["consumed_steps"], 28)

    def test_trajectory_rejects_other_prompt(self):
        with self.assertRaises(RuntimeError):
            self.runner.load_trajectory(TRAJ, prompt_ids=[1, 2, 3])


if __name__ == "__main__":
    unittest.main()
