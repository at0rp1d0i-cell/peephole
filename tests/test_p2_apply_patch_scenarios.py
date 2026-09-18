"""部署脚本的事务/安全场景测试（CPU，全部在临时隔离树里跑，不碰真实运行环境）。

覆盖本地复核提出的三类风险：
1. **最后一个目标冲突** → 整体拒绝、不留半部署；
2. **apply 之后目标被他人改动** → revert 先全量核对、拒绝覆盖（`--force` 才继续）；
3. **包目录含无关文件** → 只删本次新增文件、空目录用 rmdir，无关文件保留。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools/p2-apply-patch.py"
MANIFEST = REPO / "vllm-patch/manifest.json"

SRC_REL = Path("vllm/vllm")
INSTALL_REL = Path("venvs/attnview/lib/python3.12/site-packages/vllm")
PACKAGE_REL = Path("venvs/attnview/lib/python3.12/site-packages/attnview")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeployScenarioTest(unittest.TestCase):
    def setUp(self) -> None:
        if not MANIFEST.is_file():
            self.skipTest("缺少 vllm-patch/manifest.json")
        self.manifest = json.loads(MANIFEST.read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(REPO / "vllm-patch", self.root / "vllm-patch")
        # 包文件（src/attnview/*.py）也是部署源，必须一并镜像到临时树
        shutil.copytree(REPO / "src", self.root / "src")
        (self.root / SRC_REL).mkdir(parents=True, exist_ok=True)
        (self.root / INSTALL_REL).mkdir(parents=True, exist_ok=True)
        for rel in self.manifest["edits"]:
            pristine = REPO / SRC_REL / rel
            for root in (self.root / SRC_REL, self.root / INSTALL_REL):
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pristine, dest)
        self.targets = [self.root / SRC_REL / r for r in self.manifest["edits"]] + [
            self.root / INSTALL_REL / r for r in self.manifest["edits"]
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_tool(self, action: str, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), action, "--repo-root", str(self.root), *extra],
            cwd=REPO,
            capture_output=True,
            text=True,
        )

    def snapshot(self, paths) -> dict[str, str]:
        return {str(p): sha256(p) for p in paths if p.is_file()}

    def last_modified_target(self) -> Path:
        return self.root / INSTALL_REL / sorted(self.manifest["edits"])[-1]

    # ------------------------------------------------------------------ #
    def test_last_target_conflict_blocks_whole_deploy(self) -> None:
        victim = self.last_modified_target()
        victim.write_text(victim.read_text() + "# tampered\n")
        before = self.snapshot(self.targets)

        result = self.run_tool("apply")

        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(self.targets), before, "拒绝部署时不得改动任何目标")
        self.assertFalse((self.root / "vllm-patch/deployed.json").exists(), "不得留下事务记录")

    def test_apply_verify_revert_round_trip(self) -> None:
        pre = self.snapshot(self.targets)
        self.assertEqual(self.run_tool("apply").returncode, 0)
        self.assertEqual(self.run_tool("verify").returncode, 0)
        self.assertEqual(
            sha256(self.root / INSTALL_REL / sorted(self.manifest["edits"])[-1]),
            self.manifest["edits"][sorted(self.manifest["edits"])[-1]]["post_sha256"],
        )
        self.assertTrue((self.root / PACKAGE_REL / "step_plan.py").is_file())
        self.assertEqual(self.run_tool("revert").returncode, 0)
        self.assertEqual(self.snapshot(self.targets), pre, "撤销后必须回到原始指纹")
        self.assertFalse((self.root / "vllm-patch/deployed.json").exists())
        self.assertFalse((self.root / PACKAGE_REL).exists(), "空包目录应被 rmdir 移除")

    def test_revert_refuses_when_target_changed_after_apply(self) -> None:
        self.assertEqual(self.run_tool("apply").returncode, 0)
        victim = self.last_modified_target()
        victim.write_text(victim.read_text() + "# changed after deploy\n")
        tampered = sha256(victim)
        others = self.snapshot([p for p in self.targets if p != victim])

        refused = self.run_tool("revert")

        self.assertEqual(refused.returncode, 7, refused.stdout + refused.stderr)
        self.assertEqual(sha256(victim), tampered, "拒绝时不得覆盖被改动的目标")
        self.assertEqual(self.snapshot([p for p in self.targets if p != victim]), others)
        self.assertTrue((self.root / "vllm-patch/deployed.json").exists())

        forced = self.run_tool("revert", "--force")
        self.assertEqual(forced.returncode, 0, forced.stdout + forced.stderr)
        self.assertEqual(
            sha256(victim), self.manifest["edits"][victim.relative_to(self.root / INSTALL_REL).as_posix()]["pre_sha256"]
        )

    def test_revert_keeps_unrelated_files_in_package_dir(self) -> None:
        self.assertEqual(self.run_tool("apply").returncode, 0)
        unrelated = self.root / PACKAGE_REL / "unrelated.txt"
        unrelated.write_text("not ours\n")

        result = self.run_tool("revert")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(unrelated.is_file(), "不得删除本次补丁之外的文件")
        self.assertTrue((self.root / PACKAGE_REL).is_dir(), "目录非空时不得删除")
        ours = {Path(item["file"]).name for item in self.manifest["package_files"]}
        leftovers = [p.name for p in (self.root / PACKAGE_REL).iterdir()]
        self.assertEqual(sorted(leftovers), ["unrelated.txt"], f"本次文件应已删除：{leftovers}")
        self.assertTrue(ours.isdisjoint(leftovers))
        self.assertIn("warn", (result.stdout + result.stderr).lower() + "warning")


if __name__ == "__main__":
    unittest.main()
