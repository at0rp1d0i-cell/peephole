"""交付可复现性：patch 来源必须真的在**提交**里，且内容与 manifest 声明一致。

背景（本地复核发现）：仓库 `.gitignore` 曾有未锚定的 `vllm/` 规则，会把
`vllm-patch/files/vllm/...` 一并忽略 → 两个新适配模块只存在于工作区、从未进入提交，
交付无法从 commit 复现。本测试用 `git archive HEAD` 导出**独立临时树**核对，杜绝该类问题。

说明：本测试断言的是**已提交状态**，因此需要在提交后再跑；若失败，说明"有产物没有入库"或
"入库的是旧版本"，这正是它要抓的。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "vllm-patch/manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


class CommittedDeliverableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPO / ".git").exists():
            raise unittest.SkipTest("不是 git 仓库")
        if not MANIFEST.is_file():
            raise unittest.SkipTest("缺少 vllm-patch/manifest.json")
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.export = tempfile.TemporaryDirectory()
        cls.root = Path(cls.export.name)
        head = git("rev-parse", "HEAD").stdout.strip()
        archive = subprocess.run(
            ["git", "archive", head], cwd=REPO, capture_output=True, check=True
        ).stdout
        tar = subprocess.run(
            ["tar", "-x", "-C", str(cls.root)], input=archive, capture_output=True
        )
        if tar.returncode != 0:
            raise unittest.SkipTest(f"git archive 解包失败：{tar.stderr.decode()[:200]}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.export.cleanup()

    def expected_paths(self) -> list[str]:
        paths = ["vllm-patch/manifest.json"]
        paths += [info["patched_path"] for info in self.manifest["edits"].values()]
        paths += [item["src"] for item in self.manifest["new_files"]]
        paths += [item["file"] for item in self.manifest["package_files"]]
        return paths

    def test_all_patch_sources_are_committed(self) -> None:
        missing = [p for p in self.expected_paths() if not (self.root / p).is_file()]
        self.assertEqual(
            missing,
            [],
            "以下 patch 来源不在 HEAD 提交中（可能被 .gitignore 忽略而未入库）："
            f"{missing}；先把它们 add+commit 再跑本测试",
        )

    def test_committed_contents_match_manifest_hashes(self) -> None:
        mismatches = []
        for info in self.manifest["edits"].values():
            path = self.root / info["patched_path"]
            if path.is_file() and sha256(path) != info["post_sha256"]:
                mismatches.append(info["patched_path"])
        for item in self.manifest["new_files"]:
            path = self.root / item["src"]
            if path.is_file() and sha256(path) != item["sha256"]:
                mismatches.append(item["src"])
        for item in self.manifest["package_files"]:
            path = self.root / item["file"]
            if path.is_file() and sha256(path) != item["sha256"]:
                mismatches.append(item["file"])
        self.assertEqual(
            mismatches, [], f"以下文件已提交但内容与 manifest 不一致（先提交最新版本）：{mismatches}"
        )

    def test_patch_sources_are_not_gitignored(self) -> None:
        ignored = [
            p for p in self.expected_paths() if git("check-ignore", "-q", p).returncode == 0
        ]
        self.assertEqual(ignored, [], f"以下交付文件仍被 .gitignore 忽略：{ignored}")

    def test_gitignore_rule_is_anchored(self) -> None:
        rule_hits = [
            line
            for line in (REPO / ".gitignore").read_text().splitlines()
            if line.strip() == "vllm/"
        ]
        self.assertEqual(
            rule_hits,
            [],
            "`.gitignore` 出现未锚定的 `vllm/`：会连 `vllm-patch/files/vllm/...` 一起忽略，"
            "应写成 `/vllm/`",
        )


if __name__ == "__main__":
    unittest.main()
