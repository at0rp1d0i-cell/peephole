#!/usr/bin/env python3
"""阶段 05 窄 patch 的部署 / 校验 / 撤销（版本锁定、可撤销、不覆盖他人改动）。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py apply   [--repo-root DIR]
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py verify  [--repo-root DIR]
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py revert  [--repo-root DIR] [--force]

事务语义（依 SUP-004-R1 §3 与本地复核意见）：
- **先全目标预检、再动任何文件**：任一目标的前置哈希不符/新增目标已存在 → 整体拒绝（退出码 4），
  不留下半部署状态；
- 预检通过后先落**备份与事务日志**（`deployed.json` 标 `applying` + 完整目标清单），再逐个复制；
  任一步失败即按日志自动回滚（退出码 5）；
- `revert` **先全量核对**每个目标的现状哈希（须等于记录的后置哈希）与备份哈希（须等于记录的前置哈希），
  任一不符即整体拒绝（退出码 7；`--force` 可强制执行）；随后才恢复/删除；
- 只删除**本次新增**的文件；包目录用 `rmdir` 逐级清理，**绝不** `rmtree`，
  目录内若仍有无关文件则保留目录并给出告警。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent.parent
PKG_SRC_REL = Path("vllm/vllm")  # 源码 checkout 的包根
PKG_INSTALL_REL = Path("venvs/attnview/lib/python3.12/site-packages/vllm")  # 运行时副本
PKG_INSTALL_PACKAGE_REL = Path("venvs/attnview/lib/python3.12/site-packages/attnview")


class Layout:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.patch_root = repo / "vllm-patch"
        self.orig_root = self.patch_root / "orig"
        self.journal = self.patch_root / "deployed.json"
        self.manifest_path = self.patch_root / "manifest.json"
        self.pkg_src = repo / PKG_SRC_REL
        self.pkg_install = repo / PKG_INSTALL_REL
        self.pkg_installed_package = repo / PKG_INSTALL_PACKAGE_REL

    def load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise SystemExit(f"缺少 {self.manifest_path}：先运行 tools/p2-gen-patch.py")
        return json.loads(self.manifest_path.read_text())

    def load_journal(self) -> dict | None:
        return json.loads(self.journal.read_text()) if self.journal.is_file() else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800")


def plan_targets(layout: Layout, manifest: dict) -> list[dict]:
    """目标清单：`{src, dest, pre, post, kind, package}`（相对 repo 记录 dest/备份路径）。"""
    targets: list[dict] = []
    for rel, info in sorted(manifest["edits"].items()):
        src = layout.repo / info["patched_path"]
        for root in (layout.pkg_src, layout.pkg_install):
            targets.append(
                {
                    "src": src,
                    "dest": root / rel,
                    "pre": info["pre_sha256"],
                    "post": info["post_sha256"],
                    "kind": "modified",
                    "package": False,
                }
            )
    for item in manifest["new_files"]:
        src = layout.repo / item["src"]
        for root in (layout.pkg_src, layout.pkg_install):
            targets.append(
                {
                    "src": src,
                    "dest": root / item["dest"],
                    "pre": None,
                    "post": item["sha256"],
                    "kind": "added",
                    "package": False,
                }
            )
    for item in manifest["package_files"]:
        src = layout.repo / item["file"]
        targets.append(
            {
                "src": src,
                "dest": layout.pkg_installed_package / Path(item["file"]).name,
                "pre": None,
                "post": item["sha256"],
                "kind": "added",
                "package": True,
            }
        )
    return targets


def backup_path(layout: Layout, dest: Path) -> Path:
    return layout.orig_root / dest.relative_to(layout.repo)


def rel(layout: Layout, path: Path) -> str:
    try:
        return str(path.relative_to(layout.repo))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def do_apply(layout: Layout, manifest: dict) -> int:
    if layout.load_journal() is not None:
        print(f"已存在事务记录 {layout.journal}：先 revert（或人工清理）再 apply", file=sys.stderr)
        return 4
    targets = plan_targets(layout, manifest)

    # --- 阶段 1：全目标预检（不修改任何文件） ---
    problems: list[str] = []
    for t in targets:
        if not Path(t["src"]).is_file():
            problems.append(f"缺少部署源：{rel(layout, Path(t['src']))}")
            continue
        dest = Path(t["dest"])
        if not dest.exists():
            if t["kind"] == "modified":
                problems.append(f"目标缺失（应为已存在文件）：{rel(layout, dest)}")
            continue
        current = sha256_file(dest)
        if t["kind"] == "modified":
            if current != t["pre"]:
                problems.append(
                    f"前置哈希不符 {rel(layout, dest)}：当前 {current[:12]} != 期望 {t['pre'][:12]}"
                    "（文件已被他人改动或版本不符）"
                )
        else:  # added
            if current == t["post"]:
                problems.append(f"新增目标已存在且内容一致（疑似已部署）：{rel(layout, dest)}")
            else:
                problems.append(f"新增目标已存在且内容不同：{rel(layout, dest)}（{current[:12]}）")
    if problems:
        print(f"预检失败：{len(problems)} 项冲突，未修改任何文件", file=sys.stderr)
        for line in problems:
            print("  " + line, file=sys.stderr)
        return 4

    # --- 阶段 2：备份 + 事务日志（先落盘，再动文件） ---
    records: list[dict] = []
    for t in targets:
        dest = Path(t["dest"])
        record = {
            "dest": rel(layout, dest),
            "src": rel(layout, Path(t["src"])),
            "pre": t["pre"],
            "post": t["post"],
            "kind": t["kind"],
            "package": t["package"],
            "backup": None,
        }
        if t["kind"] == "modified":
            backup = backup_path(layout, dest)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, backup)
            if sha256_file(backup) != t["pre"]:
                print(f"备份校验失败：{rel(layout, backup)}", file=sys.stderr)
                shutil.rmtree(layout.orig_root, ignore_errors=True)
                return 4
            record["backup"] = rel(layout, backup)
        records.append(record)
    layout.journal.write_text(
        json.dumps(
            {
                "state": "applying",
                "started_cst": now_cst(),
                "pin_commit": manifest.get("pin_commit"),
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # --- 阶段 3：复制并逐项校验；失败即自动回滚 ---
    for record in records:
        dest = layout.repo / record["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(layout.repo / record["src"], dest)
        got = sha256_file(dest)
        if got != record["post"]:
            print(
                f"部署后哈希不符：{record['dest']} {got[:12]} != {record['post'][:12]}，自动回滚",
                file=sys.stderr,
            )
            _rollback(layout, records)
            return 5

    journal = json.loads(layout.journal.read_text())
    journal["state"] = "deployed"
    journal["deployed_cst"] = now_cst()
    layout.journal.write_text(json.dumps(journal, ensure_ascii=False, indent=2))
    print(f"部署完成：{len(records)} 个目标（源码 checkout + site-packages 两处），事务记录 {rel(layout, layout.journal)}")
    return 0


def _rollback(layout: Layout, records: list[dict]) -> None:
    for record in records:
        dest = layout.repo / record["dest"]
        if record["kind"] == "modified" and record["backup"]:
            shutil.copy2(layout.repo / record["backup"], dest)
        elif dest.exists():
            dest.unlink()
    shutil.rmtree(layout.orig_root, ignore_errors=True)
    if layout.journal.is_file():
        layout.journal.unlink()
    print("已自动回滚", file=sys.stderr)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def do_verify(layout: Layout) -> int:
    journal = layout.load_journal()
    if journal is None:
        print("未部署（无事务记录）")
        return 0
    bad = []
    for record in journal["files"]:
        dest = layout.repo / record["dest"]
        if not dest.is_file():
            bad.append(f"缺失 {record['dest']}")
            continue
        got = sha256_file(dest)
        if got != record["post"]:
            bad.append(f"哈希不符 {record['dest']} {got[:12]} != {record['post'][:12]}")
    print(f"校验（state={journal.get('state')}）：{len(journal['files'])} 个目标，"
          + ("全部一致" if not bad else "存在差异"))
    for line in bad:
        print("  " + line)
    return 0 if not bad else 6


# --------------------------------------------------------------------------- #
# revert
# --------------------------------------------------------------------------- #


def do_revert(layout: Layout, *, force: bool) -> int:
    journal = layout.load_journal()
    if journal is None:
        print("未部署：无需撤销")
        return 0
    records = journal["files"]

    # --- 阶段 1：全量核对现状与备份（不修改任何文件） ---
    problems: list[str] = []
    for record in records:
        dest = layout.repo / record["dest"]
        if not dest.exists():
            problems.append(f"目标缺失：{record['dest']}")
            continue
        current = sha256_file(dest)
        if current != record["post"]:
            problems.append(
                f"现状与部署时不一致 {record['dest']}：当前 {current[:12]} != 部署 {record['post'][:12]}"
                "（部署后被他人改动）"
            )
        if record["kind"] == "modified":
            backup = layout.repo / str(record["backup"])
            if not backup.is_file():
                problems.append(f"缺少备份：{record['backup']}")
            elif sha256_file(backup) != record["pre"]:
                problems.append(f"备份哈希不符：{record['backup']}")
    if problems and not force:
        print(f"撤销前核对失败：{len(problems)} 项，未修改任何文件（可用 --force 强制执行）", file=sys.stderr)
        for line in problems:
            print("  " + line, file=sys.stderr)
        return 7
    if problems:
        print(f"警告：--force 忽略 {len(problems)} 项核对问题", file=sys.stderr)

    # --- 阶段 2：只恢复/删除本次的文件 ---
    leftovers: list[str] = []
    for record in records:
        dest = layout.repo / record["dest"]
        if record["kind"] == "modified":
            backup = layout.repo / str(record["backup"])
            if backup.is_file():
                shutil.copy2(backup, dest)
            else:
                leftovers.append(f"无法恢复（缺备份）：{record['dest']}")
        elif dest.exists():
            dest.unlink()

    # --- 阶段 3a：清掉本次部署产生的字节码缓存（属本次副产物） ---
    # 包目录整个是我们创建的，其 __pycache__ 自然是我们文件编译出来的；vLLM 侧只删
    # 与本 patch 新增模块同名的 .pyc（该 __pycache__ 目录可能还含其它模块的缓存）。
    cleaned: list[str] = []
    package_cache = layout.pkg_installed_package / "__pycache__"
    if package_cache.is_dir():
        shutil.rmtree(package_cache, ignore_errors=True)
        cleaned.append(rel(layout, package_cache))
    for record in records:
        if record["kind"] != "added":
            continue
        dest = Path(record["dest"])
        cache_dir = dest.parent / "__pycache__"
        if not cache_dir.is_dir():
            continue
        for stale in cache_dir.glob(f"{dest.stem}.*.pyc"):
            stale.unlink()
            cleaned.append(rel(layout, stale))

    # --- 阶段 3：包目录只清本次文件，空目录用 rmdir ---
    # 说明：目录里若还有**非本次**文件，属于他人内容 —— 保留文件与目录并**告警**，
    # 不算撤销失败（本节只负责清掉自己那份）。
    warnings: list[str] = []
    package_dir = layout.pkg_installed_package
    if package_dir.is_dir():
        ours = {Path(r["dest"]).name for r in records if r.get("package")}
        extra = sorted(p.name for p in package_dir.iterdir() if p.name not in ours)
        if extra:
            warnings.append(f"包目录仍有非本次文件，已保留：{extra}")
        # 自底向上 rmdir：不删除任何非空目录
        for path in sorted(package_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            package_dir.rmdir()
        except OSError:
            if not extra:
                warnings.append(f"包目录非空，未删除：{rel(layout, package_dir)}")

    # --- 阶段 4：收尾 ---
    restore_errors = [
        r["dest"] for r in records
        if r["kind"] == "modified" and sha256_file(layout.repo / r["dest"]) != r["pre"]
    ]
    for dest in restore_errors:
        leftovers.append(f"恢复后哈希不符：{dest}")
    if leftovers:
        print("撤销未完全成功（保留事务记录以便处理）：", file=sys.stderr)
        for line in leftovers:
            print("  " + line, file=sys.stderr)
        return 7
    shutil.rmtree(layout.orig_root, ignore_errors=True)
    layout.journal.unlink()
    for line in cleaned:
        print("cleaned: " + line, file=sys.stderr)
    for line in warnings:
        print("warning: " + line, file=sys.stderr)
    print(f"撤销完成：{len(records)} 个目标已恢复/删除，指纹已校验" + (f"（{len(warnings)} 条告警）" if warnings else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("apply", "verify", "revert"))
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO))
    ap.add_argument("--force", action="store_true", help="revert 时忽略现状核对问题（仍校验备份）")
    args = ap.parse_args()
    layout = Layout(Path(args.repo_root).resolve())
    manifest = layout.load_manifest()
    if args.action == "apply":
        return do_apply(layout, manifest)
    if args.action == "verify":
        return do_verify(layout)
    return do_revert(layout, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
