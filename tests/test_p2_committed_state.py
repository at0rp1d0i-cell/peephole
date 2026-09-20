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
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import REPO


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def git_bytes(*args: str) -> bytes:
    """取原始字节（`git show :path` 用于拿索引里的内容，不能按文本处理）。"""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True).stdout


class CommittedDeliverableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPO / ".git").exists():
            raise unittest.SkipTest("不是 git 仓库")
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
        archived_manifest = cls.root / "vllm-patch/manifest.json"
        if not archived_manifest.is_file():
            raise unittest.SkipTest("HEAD 里没有 vllm-patch/manifest.json")
        # 一律以**归档里的**清单为准：本用例断言的是"提交本身自洽"，
        # 与工作区是否有其它未提交改动无关（工作区与 HEAD 的一致性由
        # `test_patch_products_have_no_uncommitted_changes` 单独负责）。
        cls.manifest = json.loads(archived_manifest.read_text())

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

    def test_staged_products_match_staged_manifest(self) -> None:
        """**索引（即将提交的内容）**必须自洽：交付产物改了却没 `git add`、或只 add 了一半都算失败。

        用索引而不是工作区，因为本用例要在 `pre-commit` 门禁里也能通过——那时改动已 staged
        但尚未成为 HEAD，工作区"脏"是正常状态。
        """
        manifest_bytes = self.staged("vllm-patch/manifest.json")
        self.assertNotEqual(
            manifest_bytes, b"", "vllm-patch/manifest.json 不在索引里：交付产物必须先 git add"
        )
        manifest = json.loads(manifest_bytes)
        expected = ["vllm-patch/manifest.json"]
        expected += [info["patched_path"] for info in manifest["edits"].values()]
        expected += [item["src"] for item in manifest["new_files"]]
        expected += [item["file"] for item in manifest["package_files"]]
        expected += [f"vllm-patch/patched/{rel}" for rel in manifest["edits"]]

        mismatches = []
        for path in sorted(set(expected)):
            blob = self.staged(path)
            if not blob:
                mismatches.append(f"{path}（不在索引里）")
                continue
            want = self.expected_sha(manifest, path)
            if want is not None and hashlib.sha256(blob).hexdigest() != want:
                mismatches.append(f"{path}（索引内容与清单声明不一致）")
        self.assertEqual(mismatches, [], f"以下交付产物未与清单一起入库：{mismatches}")

    @staticmethod
    def expected_sha(manifest: dict, path: str) -> str | None:
        for _rel, info in manifest["edits"].items():
            if info["patched_path"] == path:
                return info["post_sha256"]
        for item in manifest["new_files"]:
            if item["src"] == path:
                return item["sha256"]
        for item in manifest["package_files"]:
            if item["file"] == path:
                return item["sha256"]
        return None

    def staged(self, path: str) -> bytes:
        """索引里该路径的内容；不在索引里则空。"""
        return git_bytes("show", f":{path}")

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
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
