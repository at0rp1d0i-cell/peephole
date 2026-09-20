"""部署脚本的事务/安全场景测试（CPU，全部在临时隔离树里跑，不碰真实运行环境）。

覆盖本地复核提出的风险：
1. **最后一个目标冲突** → 整体拒绝、不留半部署；
2. **apply 之后目标被他人改动** → revert 先全量核对、拒绝覆盖（`--force` 才继续）；
3. **包目录含无关文件** → 只删本次新增文件、空目录用 rmdir，无关文件保留；
4. （R2 第 6 项）**写入失败必须回滚** → 真实注入 `OSError` 于第二个部署 copy：已改写目标恢复原状、
   本事务新建文件删除、journal 清理、退出码非零；预检失败零写入；`--force` 不绕过备份完整性；
5. **崩溃留下的 `applying` 事务可恢复** → 真实崩溃（`os._exit`）mid-transaction 后，`revert` 按
   实际落位对象恢复并退出 0；
6. **字节码清理只针对本事务模块** → 相对路径按 `--repo-root` 解析，不删无关 `.pyc`。

失败注入一律走 `shutil.copy2` 运行时包装（进程内真正的调用路径），不做“源码里存在某语句”的文本断言。
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

from _lib import venv_vllm_root
from _support import REPO

TOOL = REPO / "tools/p2-apply-patch.py"
MANIFEST = REPO / "vllm-patch/manifest.json"

SRC_REL = Path("vllm/vllm")
#: 部署目标的**相对**路径：本用例在临时隔离树里部署，所以不能直接拿绝对 venv 路径。
INSTALL_REL = venv_vllm_root(REPO).relative_to(REPO)
PACKAGE_REL = INSTALL_REL.parent / "attnview"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: 在真实调用路径上注入故障：包装 `shutil.copy2`，仅对**部署目标**（备份落在 vllm-patch/orig/ 下，
#: 其余 dst 都是部署目标）计数，第 N 次调用时按 mode 抛 OSError 或直接硬崩溃退出。
#: 参数依次为：工具路径、隔离树根、mode（oserror|crash）、第几次部署 copy、异常消息。
INJECTED_RUNNER = """
import os, runpy, shutil, sys

tool, root, mode, index, message = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
# 忠实模拟 `python tools/p2-apply-patch.py`：真实脚本调用会把脚本所在目录放进 sys.path[0]，
# 工具因此能 `import _lib`（工具层共享模块）；-c 启动的包装脚本必须自己补上这一步。
sys.path.insert(0, os.path.dirname(os.path.abspath(tool)))
real_copy2 = shutil.copy2
state = {"deploy_copies": 0}


def shim(src, dst, *args, **kwargs):
    if "vllm-patch" not in os.fspath(dst).split(os.sep):
        state["deploy_copies"] += 1
        if state["deploy_copies"] == index:
            if mode == "crash":
                os._exit(9)  # 模拟硬崩溃/被杀：不执行任何回滚，journal 停在 applying
            raise OSError(message)
    return real_copy2(src, dst, *args, **kwargs)


shutil.copy2 = shim
sys.argv = [tool, "apply", "--repo-root", root]
runpy.run_path(tool, run_name="__main__")
"""


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
        # 全部部署目标（改写 + 新增 + 包文件），顺序与 tools/p2-apply-patch.py:plan_targets 相同，
        # 便于按“第 N 个部署 copy”定位注入点。
        self.dests = [
            self.root / root / rel
            for rel in sorted(self.manifest["edits"])
            for root in (SRC_REL, INSTALL_REL)
        ]
        self.dests += [
            self.root / root / item["dest"]
            for item in self.manifest["new_files"]
            for root in (SRC_REL, INSTALL_REL)
        ]
        self.dests += [
            self.root / PACKAGE_REL / Path(item["file"]).name
            for item in self.manifest["package_files"]
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_tool(self, action: str, *extra: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), action, "--repo-root", str(self.root), *extra],
            cwd=cwd or REPO,
            capture_output=True,
            text=True,
        )

    def run_tool_injected(
        self, mode: str, index: int, message: str = "injected deployment copy failure"
    ) -> subprocess.CompletedProcess:
        """跑真实 `apply`，在第 `index` 次部署 copy 注入故障（`oserror` 抛错 / `crash` 硬崩溃）。"""
        return subprocess.run(
            [sys.executable, "-c", INJECTED_RUNNER, str(TOOL), str(self.root), mode, str(index), message],
            cwd=REPO,
            capture_output=True,
            text=True,
        )

    def snapshot(self, paths) -> dict[str, str]:
        return {str(p): sha256(p) for p in paths if p.is_file()}

    def tree_snapshot(self) -> dict[str, str]:
        """整棵隔离树的路径→内容指纹（目录标记为 `<dir>`），用于断言“零写入 / 无残留”。"""
        out: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            key = path.relative_to(self.root).as_posix()
            out[key] = "<dir>" if path.is_dir() else sha256(path)
        return out

    def journal_path(self) -> Path:
        return self.root / "vllm-patch/deployed.json"

    def journal_records(self) -> dict[str, dict]:
        """journal 的 per-record 视图：`dest` → 记录（含 pending/writing/written 进度）。"""
        journal = json.loads(self.journal_path().read_text())
        return {record["dest"]: record for record in journal["files"]}

    def crash_apply(self, index: int = 2) -> dict[str, dict]:
        """在第 `index` 次部署 copy 处硬崩溃（`os._exit`），返回崩溃后的 journal 记录。"""
        crash = self.run_tool_injected("crash", index, "injected deployment copy failure")
        self.assertEqual(crash.returncode, 9, crash.stdout + crash.stderr)
        return self.journal_records()

    def tree_rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

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

    # ------------------------------------------------------------------ #
    # R2 第 6 项：写入失败的恢复语义
    # ------------------------------------------------------------------ #
    def test_injected_second_copy_failure_rolls_back_transaction(self) -> None:
        """复核反例：第二次部署 copy 抛 OSError → 已改写目标恢复、journal 清理、退出码非零。"""
        before = self.tree_snapshot()

        result = self.run_tool_injected("oserror", 2, "injected second deployment copy failure")

        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertIn("injected second deployment copy failure", result.stderr)
        self.assertEqual(self.tree_snapshot(), before, "回滚必须把隔离树还原到部署前（含备份目录/新增文件）")
        self.assertFalse(self.journal_path().exists(), "回滚后不得留下 applying 残留")
        # 注入点之前的目标曾被改写，回滚后必须回到原内容
        first = self.dests[0]
        self.assertEqual(sha256(first), before[first.relative_to(self.root).as_posix()])

        # 回滚是幂等的、且事务记录清理后仍可正常 apply/revert
        self.assertEqual(self.run_tool("apply").returncode, 0)
        self.assertEqual(self.run_tool("revert").returncode, 0)
        self.assertEqual(self.tree_snapshot(), before)

    def test_injected_failure_after_added_targets_deletes_them(self) -> None:
        """回滚必须覆盖“目标原本不存在”的一支：删除本事务新建的文件。"""
        before = self.tree_snapshot()
        first_added_copy = 2 * len(self.manifest["edits"]) + 1  # 第一个新增目标的 copy 序号

        result = self.run_tool_injected("oserror", first_added_copy + 2, "injected third deployment copy failure")

        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        # 此时前 16 个改写目标与第一个新增目标（两处根）都已落位，回滚要全部覆盖
        self.assertEqual(self.tree_snapshot(), before, "回滚必须还原改写目标并删除新建文件")
        added_dest = self.manifest["new_files"][0]["dest"]
        for root_rel in (SRC_REL, INSTALL_REL):
            self.assertFalse((self.root / root_rel / added_dest).exists(), "本事务新建的文件必须删除")
        self.assertFalse(self.journal_path().exists())
        self.assertFalse((self.root / "vllm-patch/orig").exists(), "备份目录应随回滚清理")

    def test_precheck_source_hash_mismatch_writes_nothing(self) -> None:
        """部署源哈希不符（新文件 / patched 替换文件）→ 拒绝且零写入。"""
        cases = {
            "new_file_source": self.manifest["new_files"][0]["src"],
            "patched_source": self.manifest["edits"][sorted(self.manifest["edits"])[0]]["patched_path"],
        }
        for name, src_rel in cases.items():
            with self.subTest(source=name):
                src = self.root / src_rel
                original = src.read_text()
                src.write_text(original + "# tampered source\n")
                before = self.tree_snapshot()

                result = self.run_tool("apply")

                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertIn("部署源哈希不符", result.stderr)
                self.assertEqual(self.tree_snapshot(), before, "预检失败时不得写入任何字节（含备份/日志）")
                self.assertFalse(self.journal_path().exists())
                src.write_text(original)

    def test_force_still_refuses_damaged_backup(self) -> None:
        """`--force` 只忽略现状核对问题，备份完整性（缺失/哈希不符）仍必须拒绝。"""
        self.assertEqual(self.run_tool("apply").returncode, 0)
        backup = self.root / "vllm-patch/orig" / SRC_REL / sorted(self.manifest["edits"])[0]
        self.assertTrue(backup.is_file(), f"缺少备份：{backup}")

        backup.write_text(backup.read_text() + "# corrupted backup\n")
        state = self.tree_snapshot()
        damaged = self.run_tool("revert", "--force")
        self.assertEqual(damaged.returncode, 7, damaged.stdout + damaged.stderr)
        self.assertIn("备份", damaged.stderr)
        self.assertEqual(self.tree_snapshot(), state, "--force 拒绝时不得改动任何文件")
        self.assertTrue(self.journal_path().is_file())

        backup.unlink()
        state = self.tree_snapshot()
        missing = self.run_tool("revert", "--force")
        self.assertEqual(missing.returncode, 7, missing.stdout + missing.stderr)
        self.assertEqual(self.tree_snapshot(), state, "备份缺失时 --force 也不得改动任何文件")

        help_text = self.run_tool("revert", "--help").stdout
        self.assertIn("备份完整性仍强制校验", help_text, "help 文字必须与实际行为一致")

    def test_revert_recovers_half_deployed_transaction(self) -> None:
        """崩溃留下的 `applying` 半部署：revert 按实际落位对象恢复并退出 0（不得拒绝）。"""
        pre = self.snapshot(self.dests)
        crash = self.run_tool_injected("crash", 2, "injected second deployment copy failure")
        self.assertEqual(crash.returncode, 9, crash.stdout + crash.stderr)
        journal = json.loads(self.journal_path().read_text())
        self.assertEqual(journal["state"], "applying", "崩溃应留下 applying 事务")
        changed = [p for p, h in self.snapshot(self.dests).items() if pre[p] != h]
        self.assertEqual(len(changed), 1, f"崩溃前应恰好改写一个目标：{changed}")
        self.assertTrue((self.root / "vllm-patch/orig").is_dir(), "备份应已落盘")

        result = self.run_tool("revert")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(self.dests), pre, "半部署必须被完整恢复")
        self.assertFalse(self.journal_path().exists(), "恢复成功后应清除事务记录")
        self.assertFalse((self.root / "vllm-patch/orig").exists())

    def test_revert_recovers_applying_journal_without_any_write(self) -> None:
        """journal 停在 `applying` 但目标一个都没改（记录与实际不一致）时同样必须恢复成功。"""
        pre = self.snapshot(self.dests)

        crash = self.run_tool_injected("crash", 1, "injected first deployment copy failure")

        self.assertEqual(crash.returncode, 9, crash.stdout + crash.stderr)
        self.assertEqual(self.snapshot(self.dests), pre, "首个 copy 前崩溃：目标应全部保持原样")
        self.assertEqual(json.loads(self.journal_path().read_text())["state"], "applying")

        result = self.run_tool("revert")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(self.dests), pre)
        self.assertFalse(self.journal_path().exists())

    def test_revert_preserves_external_changes_to_untouched_targets(self) -> None:
        """崩溃后：本事务未触及的目标被外部改动/外部新建 → revert 一律保留并报告。"""
        pre = self.snapshot(self.dests)
        records = self.crash_apply(2)

        # journal 必须能区分“已写”与“未触及”
        self.assertEqual(records[self.tree_rel(self.dests[0])]["state"], "written")
        self.assertEqual(records[self.tree_rel(self.dests[1])]["state"], "writing")
        self.assertEqual(records[self.tree_rel(self.dests[2])]["state"], "pending")
        planned_new = self.dests[2 * len(self.manifest["edits"])]
        self.assertEqual(records[self.tree_rel(planned_new)]["state"], "pending")

        # 外部改动一个本事务未触及的改写目标；外部新建一个本事务计划但未写的新增目标
        external_target = self.dests[3]
        external_target.write_text(external_target.read_text() + "# external edit\n")
        external_hash = sha256(external_target)
        planned_new.parent.mkdir(parents=True, exist_ok=True)
        planned_new.write_text("externally created file\n")
        external_new_hash = sha256(planned_new)

        result = self.run_tool("revert")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sha256(external_target), external_hash, "未触及目标的外部改动不得被覆盖")
        self.assertEqual(sha256(planned_new), external_new_hash, "外部新建文件不得被删除")
        self.assertIn("preserved:", result.stderr)
        for path in (self.tree_rel(external_target), self.tree_rel(planned_new)):
            self.assertIn(path, result.stderr, "被保留的对象必须在输出里点名")
        # 本事务写入过的对象恢复干净，其余未触及目标保持原样
        for path, digest in pre.items():
            if path in (str(external_target), str(planned_new)):
                continue
            self.assertEqual(sha256(Path(path)), digest, f"{path} 应回到部署前内容")
        self.assertFalse(self.journal_path().exists(), "事务记录应清除")

    def test_revert_preserves_written_target_changed_externally(self) -> None:
        """崩溃后：本事务写过但随后被外部改动的目标 → 保留并告警，不得用备份覆盖。"""
        records = self.crash_apply(2)
        written = self.dests[0]
        self.assertEqual(records[self.tree_rel(written)]["state"], "written")
        written.write_text(written.read_text() + "# external edit after our write\n")
        external_hash = sha256(written)

        result = self.run_tool("revert")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sha256(written), external_hash, "写入后被外部改动的内容不得被覆盖")
        self.assertIn("preserved:", result.stderr)
        self.assertIn(self.tree_rel(written), result.stderr)
        self.assertIn("写入后被外部改动", result.stderr)
        self.assertFalse(self.journal_path().exists())

    def test_force_revert_overrides_external_edits_but_not_pending(self) -> None:
        """`--force` 可强制覆盖已写目标的外部改动；`pending`（本事务未触及）仍然不动。"""
        pre = self.snapshot(self.dests)
        self.crash_apply(2)
        written = self.dests[0]
        written.write_text(written.read_text() + "# external edit\n")
        planned_new = self.dests[2 * len(self.manifest["edits"])]
        planned_new.parent.mkdir(parents=True, exist_ok=True)
        planned_new.write_text("externally created file\n")
        external_new_hash = sha256(planned_new)

        result = self.run_tool("revert", "--force")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sha256(written), pre[str(written)], "--force 应按备份恢复本事务写过的目标")
        self.assertEqual(sha256(planned_new), external_new_hash, "pending 目标即便 --force 也不得动")
        self.assertIn("preserved:", result.stderr)
        self.assertFalse(self.journal_path().exists())

    def test_verify_after_crash_ignores_pending_records(self) -> None:
        """崩溃后 `verify` 只判定本事务实际写过的对象：未写入的 pending 记录不算差异。"""
        self.crash_apply(2)

        result = self.run_tool("verify")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("本事务未写入", result.stdout)

    def test_bytecode_cleanup_only_touches_this_transaction(self) -> None:
        """字节码清理按 `--repo-root` 解析、只删本事务落位模块的 `.pyc`（含包目录条目）。"""
        self.assertEqual(self.run_tool("apply").returncode, 0)
        added_mod = self.manifest["new_files"][0]["dest"]
        edited_mod = sorted(self.manifest["edits"])[0]
        package_mod = Path(self.manifest["package_files"][0]["file"]).name

        def fake_pyc(cache_dir: Path, name: str) -> Path:
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / name
            path.write_bytes(b"fake pyc\n")
            return path

        ours, kept = [], []
        for root_rel in (SRC_REL, INSTALL_REL):
            ours.append(fake_pyc(self.root / root_rel / Path(added_mod).parent / "__pycache__",
                                 f"{Path(added_mod).stem}.cpython-312.pyc"))
            ours.append(fake_pyc(self.root / root_rel / Path(edited_mod).parent / "__pycache__",
                                 f"{Path(edited_mod).stem}.cpython-312.pyc"))
            kept.append(fake_pyc(self.root / root_rel / Path(added_mod).parent / "__pycache__",
                                 "unrelated_module.cpython-312.pyc"))
        pkg_cache = self.root / PACKAGE_REL / "__pycache__"
        ours.append(fake_pyc(pkg_cache, f"{Path(package_mod).stem}.cpython-312.pyc"))
        kept.append(fake_pyc(pkg_cache, "other_pkg_module.cpython-312.pyc"))

        # 诱饵：漏了 `layout.repo` 前缀的相对路径实现会把清理落到 CWD（这里放可观测的诱饵）
        decoy_cwd = self.root / "decoy-cwd"
        decoy = fake_pyc(decoy_cwd / Path(added_mod).parent / "__pycache__",
                         f"{Path(added_mod).stem}.cpython-312.pyc")

        result = self.run_tool("revert", cwd=decoy_cwd)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for path in ours:
            self.assertFalse(path.exists(), f"本事务模块的字节码应被清理：{path}")
        for path in kept:
            self.assertTrue(path.is_file(), f"无关字节码不得删除：{path}")
        self.assertTrue(decoy.is_file(), "字节码清理必须基于 --repo-root 解析，不得落到 CWD")
        self.assertIn("cleaned:", result.stderr)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
