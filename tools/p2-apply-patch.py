#!/usr/bin/env python3
"""阶段 05 窄 patch 的部署 / 校验 / 撤销（版本锁定、可撤销、不覆盖他人改动）。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py apply   [--repo-root DIR]
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py verify  [--repo-root DIR]
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py revert  [--repo-root DIR] [--force]

事务语义（依 SUP-004-R1 §3、SUP-004-R2 第 6 项与本地复核意见）：
- **先全量预检、再动任何文件**：逐个核对部署源哈希（新文件 + patched 替换文件，须等于记录的 post）
  与目标状态（改写目标须存在且等于 pre；新增目标不得存在）；任一不符 → 整体拒绝（退出码 4），
  不留下半部署状态；
- 预检通过后先落**备份与事务日志**（`deployed.json` 标 `applying` + 完整目标清单），并立即校验
  备份完整性（须等于记录的前置哈希），全部落盘成功后才开始改写目标；
- 写入路径（`mkdir` / `copy2` / 部署后哈希 / 日志定稿）任一步失败 → 只回滚**本事务实际写过的对象**
  （改写目标回填已验证的备份、新建目标删除、本事务新建的空目录逐个 `rmdir`），清除事务记录后
  以退出码 5 退出；回滚自身失败则**保留**事务记录，交给 `revert` 按未完成事务继续恢复，仍返回非零；
- `revert` 有两个入口：`state == "deployed"` 时**先全量核对**每个目标的现状哈希（须等于记录的后置
  哈希）与备份哈希（须等于记录的前置哈希），现状不符整体拒绝（退出码 7，`--force` 可强制执行），
  **备份完整性任何情况下都不可绕过**；`state != "deployed"`（崩溃/中断留下的 `applying`）时不再
  要求现状匹配，直接按已落位对象恢复并正常退出（退出码 0，幂等）；
- 只删除**本次新增**的文件；包目录用 `rmdir` 逐级清理，**绝不** `rmtree`，
  目录内若仍有无关文件则保留目录并给出告警；
- 字节码清理只针对**本事务落位的模块**：按 `layout.repo / record['dest']` 定位其 `__pycache__`
  下的同名 `.pyc` 逐个删除，不整目录删除（同一目录里可能有他人/先前部署的缓存）。
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


class BackupIntegrityError(RuntimeError):
    """备份内容与记录的前置哈希不符：属预检类拒绝，绝不放行。"""


class DeploymentError(RuntimeError):
    """部署过程中的一致性失败（如部署后哈希不符）。"""


def sha256_or_none(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def backup_problems_of(layout: Layout, record: dict) -> list[str]:
    """`record` 的备份完整性检查（缺失/哈希不符）；非改写目标返回空。"""
    if record["kind"] != "modified":
        return []
    backup_rel = record.get("backup")
    backup = layout.repo / str(backup_rel)
    if not backup_rel or not backup.is_file():
        return [f"缺少备份：{backup_rel}"]
    if sha256_file(backup) != record["pre"]:
        return [f"备份哈希不符：{backup_rel}"]
    return []


def missing_dirs(dest_dir: Path, stop: Path) -> list[Path]:
    """`dest_dir` 到 `stop`（不含）之间当前不存在的目录链（即 `mkdir` 会新建的），由深到浅。"""
    missing: list[Path] = []
    cur = dest_dir
    while cur != stop and stop in cur.parents and not cur.exists():
        missing.append(cur)
        cur = cur.parent
    return missing


def discard_journal(layout: Layout) -> None:
    """尽力清除事务记录（只在事务对象已恢复、或目标文件从未被改动时调用）。"""
    try:
        layout.journal.unlink(missing_ok=True)
    except OSError as exc:
        print(f"warning: 事务记录未能删除：{rel(layout, layout.journal)}（{exc!r}）", file=sys.stderr)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def do_apply(layout: Layout, manifest: dict) -> int:
    if layout.load_journal() is not None:
        print(f"已存在事务记录 {layout.journal}：先 revert（或人工清理）再 apply", file=sys.stderr)
        return 4
    targets = plan_targets(layout, manifest)

    # --- 阶段 1：全量预检（不修改任何文件） ---
    # 部署源（patched 替换文件 + 新文件 + 包文件）哈希须等于记录的 post；改写目标须存在且等于 pre；
    # 新增目标不得存在。任一不符 → 整体拒绝。
    problems: list[str] = []
    for t in targets:
        src = Path(t["src"])
        dest = Path(t["dest"])
        if not src.is_file():
            problems.append(f"缺少部署源：{rel(layout, src)}")
        else:
            src_hash = sha256_or_none(src)
            if src_hash != t["post"]:
                problems.append(
                    f"部署源哈希不符 {rel(layout, src)}：当前 {str(src_hash)[:12]} != 期望 {t['post'][:12]}"
                    "（patched 产物被改动或清单过期）"
                )
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

    # --- 阶段 2：备份 + 事务日志（先落盘并校验备份完整性，再动目标文件） ---
    records: list[dict] = []
    try:
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
                    raise BackupIntegrityError(f"备份哈希不符：{rel(layout, backup)}")
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
    except BackupIntegrityError as exc:
        print(f"{exc}；备份阶段拒绝，未修改任何目标文件", file=sys.stderr)
        shutil.rmtree(layout.orig_root, ignore_errors=True)
        discard_journal(layout)
        return 4
    except OSError as exc:
        print(f"备份/事务记录落盘失败：{exc!r}，未修改任何目标文件", file=sys.stderr)
        shutil.rmtree(layout.orig_root, ignore_errors=True)
        discard_journal(layout)
        return 5

    # --- 阶段 3：复制并逐项校验；任一步失败即只回滚本事务已落位的对象 ---
    attempted: list[dict] = []
    created_dirs: list[Path] = []
    try:
        for record in records:
            dest = layout.repo / record["dest"]
            created_dirs.extend(missing_dirs(dest.parent, layout.repo))
            dest.parent.mkdir(parents=True, exist_ok=True)
            attempted.append(record)  # 先登记：copy2 可能写到一半才失败
            shutil.copy2(layout.repo / record["src"], dest)
            got = sha256_file(dest)
            if got != record["post"]:
                raise DeploymentError(
                    f"部署后哈希不符：{record['dest']} {got[:12]} != {record['post'][:12]}"
                )
    except (OSError, DeploymentError) as exc:
        print(f"部署失败：{exc}；自动回滚已落位的 {len(attempted)} 个对象", file=sys.stderr)
        _rollback(layout, attempted, created_dirs=created_dirs)
        return 5

    try:
        journal = json.loads(layout.journal.read_text())
        journal["state"] = "deployed"
        journal["deployed_cst"] = now_cst()
        layout.journal.write_text(json.dumps(journal, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"事务记录定稿失败：{exc!r}；自动回滚全部 {len(records)} 个已落位对象", file=sys.stderr)
        _rollback(layout, records, created_dirs=created_dirs)
        return 5

    print(f"部署完成：{len(records)} 个目标（源码 checkout + site-packages 两处），事务记录 {rel(layout, layout.journal)}")
    return 0


def _rollback(layout: Layout, records: list[dict], *, created_dirs: list[Path] | None = None) -> list[str]:
    """按记录回滚**本事务实际写过的对象**（改写目标回填备份，新建目标删除）。

    只处理调用方传入的记录 —— 未触及的目标不会被写回。幂等：目标已处于前置状态时跳过写回，
    重复调用无副作用。返回未能完全恢复的清单（空表示回滚成功，事务记录随之清除）。
    """
    problems: list[str] = []
    for record in records:
        dest = layout.repo / record["dest"]
        if record["kind"] == "modified":
            backup_rel = record.get("backup")
            backup = layout.repo / str(backup_rel)
            if not backup_rel or not backup.is_file() or sha256_file(backup) != record["pre"]:
                problems.append(f"回滚缺少可信备份：{record['dest']}（backup={backup_rel}）")
                continue
            if dest.is_file() and sha256_file(dest) == record["pre"]:
                continue  # 目标未被改动（或已回滚过）：幂等跳过
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, dest)
            except OSError as exc:
                problems.append(f"回滚写入失败：{record['dest']}（{exc!r}）")
                continue
            if sha256_file(dest) != record["pre"]:
                problems.append(f"回滚后哈希不符：{record['dest']}")
        elif dest.exists():
            try:
                dest.unlink()
            except OSError as exc:
                problems.append(f"回滚删除失败：{record['dest']}（{exc!r}）")
    # 本事务新建的空目录逐个 rmdir（只删空目录，绝不 rmtree 无关内容）
    for path in sorted(set(created_dirs or []), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    if problems:
        print("自动回滚未完成（保留事务记录，可用 revert 继续恢复）：", file=sys.stderr)
        for line in problems:
            print("  " + line, file=sys.stderr)
        return problems
    shutil.rmtree(layout.orig_root, ignore_errors=True)
    discard_journal(layout)
    print("已自动回滚", file=sys.stderr)
    return problems


def clean_bytecode(layout: Layout, records: list[dict]) -> list[str]:
    """删除**本事务落位模块**对应的字节码条目（`.pyc`），返回清理清单。

    路径一律以 `layout.repo` 为基准解析（`record['dest']` 是相对 repo 记录的），
    只删与其 `__pycache__` 下同名词条匹配的 `.pyc` 文件，绝不整目录删除：
    同一个 `__pycache__` 里可能还有其它模块（先前部署或他人）的缓存。
    """
    cleaned: list[str] = []
    for record in records:
        dest = layout.repo / record["dest"]
        cache_dir = dest.parent / "__pycache__"
        if not cache_dir.is_dir():
            continue
        for stale in sorted(cache_dir.glob(f"{dest.stem}.*.pyc")):
            try:
                stale.unlink()
            except OSError as exc:
                print(f"warning: 字节码清理失败 {rel(layout, stale)}（{exc!r}）", file=sys.stderr)
                continue
            cleaned.append(rel(layout, stale))
    return cleaned


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
    state = journal.get("state")

    # --- 阶段 1：核对（不修改任何文件） ---
    # `deployed`：先全量核对现状哈希与备份哈希；`applying`（崩溃/中断的半部署）：不要求现状匹配，
    # 直接按已落位对象恢复。备份完整性任何情况下都不可绕过（--force 也不例外）。
    if state == "deployed":
        state_problems: list[str] = []
        backup_problems: list[str] = []
        for record in records:
            dest = layout.repo / record["dest"]
            if not dest.exists():
                state_problems.append(f"目标缺失：{record['dest']}")
            else:
                current = sha256_file(dest)
                if current != record["post"]:
                    state_problems.append(
                        f"现状与部署时不一致 {record['dest']}：当前 {current[:12]} != 部署 {record['post'][:12]}"
                        "（部署后被他人改动）"
                    )
            backup_problems.extend(backup_problems_of(layout, record))
        if backup_problems:
            print(f"撤销前核对失败：{len(backup_problems)} 项备份问题，未修改任何文件"
                  "（备份完整性不可用 --force 绕过）", file=sys.stderr)
            for line in backup_problems:
                print("  " + line, file=sys.stderr)
            return 7
        if state_problems and not force:
            print(f"撤销前核对失败：{len(state_problems)} 项，未修改任何文件（可用 --force 强制执行）", file=sys.stderr)
            for line in state_problems:
                print("  " + line, file=sys.stderr)
            return 7
        if state_problems:
            print(f"警告：--force 忽略 {len(state_problems)} 项现状核对问题", file=sys.stderr)
    else:
        backup_problems = [p for record in records for p in backup_problems_of(layout, record)]
        if backup_problems:
            print(f"事务未完成（state={state}）且备份不可信：{len(backup_problems)} 项，未修改任何文件", file=sys.stderr)
            for line in backup_problems:
                print("  " + line, file=sys.stderr)
            return 7
        print(f"事务未完成（state={state}）：按实际落位情况恢复，不要求现状与部署后哈希一致", file=sys.stderr)

    # --- 阶段 2：只恢复/删除本次的文件（幂等：已回到前置状态的目标不动） ---
    leftovers: list[str] = []
    for record in records:
        dest = layout.repo / record["dest"]
        if record["kind"] == "modified":
            backup = layout.repo / str(record["backup"])
            if not backup.is_file():
                leftovers.append(f"无法恢复（缺备份）：{record['dest']}")
                continue
            if dest.is_file() and sha256_file(dest) == record["pre"]:
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, dest)
            except OSError as exc:
                leftovers.append(f"恢复失败：{record['dest']}（{exc!r}）")
        elif dest.exists():
            # 新增目标：部署预检已保证它此前不存在，故此刻存在即本事务写入（可能是半截文件）——
            # 按“实际已写入的对象”删除，正是崩溃恢复要做的。
            try:
                dest.unlink()
            except OSError as exc:
                leftovers.append(f"删除失败：{record['dest']}（{exc!r}）")

    # --- 阶段 3a：清掉本次落位模块的字节码缓存（属本次副产物） ---
    cleaned = clean_bytecode(layout, records)

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
    for record in records:
        dest = layout.repo / record["dest"]
        if record["kind"] == "modified":
            if not dest.is_file() or sha256_file(dest) != record["pre"]:
                leftovers.append(f"恢复后哈希不符：{record['dest']}")
        elif dest.exists():
            leftovers.append(f"新增目标未被删除：{record['dest']}")
    if leftovers:
        print("撤销未完全成功（保留事务记录以便处理）：", file=sys.stderr)
        for line in leftovers:
            print("  " + line, file=sys.stderr)
        return 7
    shutil.rmtree(layout.orig_root, ignore_errors=True)
    discard_journal(layout)
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
    ap.add_argument("--force", action="store_true", help="revert 时忽略现状核对问题（备份完整性仍强制校验，不可绕过）")
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
