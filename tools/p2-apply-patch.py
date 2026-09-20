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
- 预检通过后先落**备份与事务日志**（`deployed.json` 标 `applying` + 完整目标清单，逐条 `state:
  pending`），并立即校验备份完整性（须等于记录的前置哈希），全部落盘成功后才开始改写目标；
- **写入进度逐步落盘**：每个目标先标 `writing` 落盘、用「同目录临时文件 + `os.replace`」原子落位，
  校验后标 `written` 落盘。因此磁盘上的 journal 始终能回答“本事务实际写过哪些对象”，且目标只会是
  `pre` 或 `post`，不会出现半截文件；
- 写入路径（`mkdir` / 落位 / 部署后哈希 / 日志落盘）任一步失败 → 只回滚**本事务实际写过的对象**
  （改写目标回填已验证的备份、新建目标删除、本事务新建的空目录逐个 `rmdir`），清除事务记录后
  以退出码 5 退出；回滚自身失败则**保留**事务记录，交给 `revert` 按未完成事务继续恢复，仍返回非零；
- `revert` 有两个入口：
  * `state == "deployed"`：**先全量核对**每个目标的现状哈希（须等于记录的后置哈希）与备份哈希
    （须等于记录的前置哈希），现状不符整体拒绝（退出码 7，`--force` 可强制执行），
    **备份完整性任何情况下都不可绕过**；
  * `state != "deployed"`（崩溃/中断留下的 `applying`）：**只处理本事务实际写过的对象** ——
    `written`/`writing`（旧格式无 `state` 字段的记录视同 `writing`）按内容判归属：
    当前内容 == 记录 `post` → 是本事务写的（改写目标回填备份、新增目标删除）；内容 == `pre`
    或新增目标不存在 → 本事务的写入没有落位，跳过；其余（写入后被外部改动、被删除、外部新建）
    → **保留并告警，绝不覆盖**（`--force` 才强制按备份恢复/删除）。`pending` 记录是明确的
    “本事务未触及”证据，**任何情况下都不动它**（`--force` 也不动）；
- 退出码：0 = 撤销完成（本事务的对象已恢复/删除；若保留了外部状态，stderr 有 `preserved:` 提示
  与计数）；4 = 完整性/预检拒绝；5 = 写入失败并已回滚；6 = `verify` 发现差异；
  7 = 核对失败或存在无法恢复的对象（事务记录保留）。
  保留外部状态仍算撤销完成（本事务的契约是“只动自己写过的”，此时契约已满足；外部改动不是本事务的
  责任，也不该被无声覆盖），由 `preserved:` 行与 exit 0 区分于“无保留的干净撤销”。
- 只删除**本次新增**的文件；包目录用 `rmdir` 逐级清理，**绝不** `rmtree`，
  目录内若仍有无关文件则保留目录并给出告警；
- 字节码清理只针对**本事务落位的模块**：按 `layout.repo / record['dest']` 定位其 `__pycache__`
  下的同名 `.pyc` 逐个删除，不整目录删除（同一目录里可能有他人/先前部署的缓存）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from _lib import now_cst, sha256_file, venv_site_packages

DEFAULT_REPO = Path(__file__).resolve().parent.parent
PKG_SRC_REL = Path("vllm/vllm")  # 源码 checkout 的包根


class Layout:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.patch_root = repo / "vllm-patch"
        self.orig_root = self.patch_root / "orig"
        self.journal = self.patch_root / "deployed.json"
        self.manifest_path = self.patch_root / "manifest.json"
        self.pkg_src = repo / PKG_SRC_REL
        # 运行时副本：venv 的 site-packages（`venvs/attnview/lib/python3.*/`，次版本不写进代码）
        site_packages = venv_site_packages(repo)
        self.pkg_install = site_packages / "vllm"
        self.pkg_installed_package = site_packages / "attnview"

    def load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise SystemExit(f"缺少 {self.manifest_path}：先运行 tools/p2-gen-patch.py")
        return json.loads(self.manifest_path.read_text())

    def load_journal(self) -> dict | None:
        return json.loads(self.journal.read_text()) if self.journal.is_file() else None


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
    for path in (layout.journal, layout.journal.with_name(layout.journal.name + ".tmp")):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"warning: 事务记录未能删除：{rel(layout, path)}（{exc!r}）", file=sys.stderr)


def save_journal(layout: Layout, journal: dict) -> None:
    """原子落盘事务记录（同目录临时文件 + `os.replace`），崩溃不会留下半截 JSON。"""
    tmp = layout.journal.with_name(layout.journal.name + ".tmp")
    tmp.write_text(json.dumps(journal, ensure_ascii=False, indent=2))
    os.replace(tmp, layout.journal)


def tmp_path_of(dest: Path) -> Path:
    """部署落位用的同目录临时文件（`os.replace` 前的中间态）。"""
    return dest.parent / f".{dest.name}.p2-deploy-tmp"


def replace_copy(src: Path, dest: Path) -> None:
    """原子落位单个文件：先写同目录临时文件再 `os.replace`。

    目的是让目标只有 `pre` / `post` 两种内容，崩溃不会留下半截文件 —— 这是
    “journal 里的 `writing` 记录只能靠内容判归属”能成立的前提。
    """
    tmp = tmp_path_of(dest)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def target_state(layout: Layout, record: dict) -> str:
    """目标当前内容相对本事务记录的状态：`post` / `pre` / `missing` / `other`（外部内容）。"""
    dest = layout.repo / record["dest"]
    if not dest.exists():
        return "missing"
    if not dest.is_file():
        return "other"
    got = sha256_file(dest)
    if got == record["post"]:
        return "post"
    if record["kind"] == "modified" and got == record["pre"]:
        return "pre"
    return "other"


def perturbed(record: dict, cur: str) -> bool:
    """`pending` 目标是否已被外部改动（用于报告，不用于归属推断）。"""
    return cur != ("pre" if record["kind"] == "modified" else "missing")


def recovery_action(record: dict, cur: str, *, force: bool) -> str:
    """崩溃恢复时对单条记录的动作：`skip` / `undo` / `preserve`。

    `pending`（journal 明示本事务未触及）永不 undo，只报告；`writing`/`written`（旧格式无
    `state` 字段视同 `writing`）以内容判归属：`post` 才是本事务写的，`pre`/不存在表示写入未落位，
    其余一律 `preserve`（`--force` 才改成 `undo`）。
    """
    if (record.get("state") or "unknown") == "pending":
        return "preserve" if perturbed(record, cur) else "skip"
    if cur == "post":
        return "undo"
    if cur == "pre" or (record["kind"] == "added" and cur == "missing"):
        return "skip"
    return "undo" if force else "preserve"


def undo_record(layout: Layout, record: dict) -> list[str]:
    """撤销本事务写入的单个对象（改写目标回填备份、新增目标删除），返回问题清单。幂等。"""
    dest = layout.repo / record["dest"]
    tmp = tmp_path_of(dest)
    problems: list[str] = []
    if record["kind"] == "modified":
        backup_rel = record.get("backup")
        backup = layout.repo / str(backup_rel)
        if not backup_rel or not backup.is_file():
            problems.append(f"无法恢复（缺备份）：{record['dest']}")
        elif sha256_file(backup) != record["pre"]:
            problems.append(f"备份哈希不符，拒绝用于恢复：{backup_rel}")
        else:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, dest)
            except OSError as exc:
                problems.append(f"恢复失败：{record['dest']}（{exc!r}）")
            else:
                if sha256_file(dest) != record["pre"]:
                    problems.append(f"恢复后哈希不符：{record['dest']}")
    elif dest.exists():
        try:
            dest.unlink()
        except OSError as exc:
            problems.append(f"删除失败：{record['dest']}（{exc!r}）")
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return problems


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
                "state": "pending",  # pending → writing → written，随写入逐步落盘
            }
            if t["kind"] == "modified":
                backup = backup_path(layout, dest)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, backup)
                if sha256_file(backup) != t["pre"]:
                    raise BackupIntegrityError(f"备份哈希不符：{rel(layout, backup)}")
                record["backup"] = rel(layout, backup)
            records.append(record)
        journal = {
            "state": "applying",
            "started_cst": now_cst(),
            "pin_commit": manifest.get("pin_commit"),
            "files": records,
        }
        save_journal(layout, journal)
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

    # --- 阶段 3：逐个原子落位并逐项校验；任一步失败即只回滚本事务已落位的对象 ---
    attempted: list[dict] = []
    created_dirs: list[Path] = []
    try:
        for record in records:
            dest = layout.repo / record["dest"]
            record["state"] = "writing"  # 意图先落盘：崩溃后这一条也视为“本事务动过”
            save_journal(layout, journal)
            attempted.append(record)
            created_dirs.extend(missing_dirs(dest.parent, layout.repo))
            dest.parent.mkdir(parents=True, exist_ok=True)
            replace_copy(layout.repo / record["src"], dest)
            got = sha256_file(dest)
            if got != record["post"]:
                raise DeploymentError(
                    f"部署后哈希不符：{record['dest']} {got[:12]} != {record['post'][:12]}"
                )
            record["state"] = "written"
            save_journal(layout, journal)
    except (OSError, DeploymentError) as exc:
        print(f"部署失败：{exc}；自动回滚已落位的 {len(attempted)} 个对象", file=sys.stderr)
        _rollback(layout, attempted, created_dirs=created_dirs)
        return 5

    try:
        journal["state"] = "deployed"
        journal["deployed_cst"] = now_cst()
        save_journal(layout, journal)
    except OSError as exc:
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
        if record["kind"] == "modified" and target_state(layout, record) == "pre":
            continue  # 未落位（或已回滚过）：幂等跳过
        problems.extend(undo_record(layout, record))
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
    deployed = journal.get("state") == "deployed"
    bad: list[str] = []
    unwritten: list[str] = []
    for record in journal["files"]:
        # 未完成事务里 `pending`（本事务从未触及）与“写入未落位”的记录不能按“应为 post”判失败
        rstate = record.get("state") or "unknown"
        if not deployed and rstate == "pending":
            unwritten.append(record["dest"])
            continue
        cur = target_state(layout, record)
        if cur == "post":
            continue
        if not deployed and (cur == "pre" or (record["kind"] == "added" and cur == "missing")):
            unwritten.append(record["dest"])
            continue
        bad.append(f"{record['dest']}：当前 {cur}，期望 {record['post'][:12]}")
    print(f"校验（state={journal.get('state')}）：{len(journal['files'])} 个目标"
          + (f"（{len(unwritten)} 个本事务未写入/未落位）" if unwritten else "")
          + "，" + ("全部一致" if not bad else "存在差异"))
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
    preserved: list[str] = []

    # --- 阶段 1：核对 + 制定计划（不修改任何文件） ---
    # `deployed`：全量核对现状，任一不符整体拒绝（退出码 7，`--force` 才继续）；
    # 未完成事务（`applying` 等）：只处理 journal 标记为本事务实际写过的对象，外部状态一律保留。
    # 备份完整性（缺失/哈希不符）任何情况下都不可绕过（`--force` 也不例外）。
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
        plan = [("undo", record) for record in records]
    else:
        print(f"事务未完成（state={state}）：只处理本事务实际写过的对象", file=sys.stderr)
        plan = []
        plan_problems: list[str] = []
        for record in records:
            action = recovery_action(record, target_state(layout, record), force=force)
            if action == "undo":
                plan_problems.extend(backup_problems_of(layout, record))
            elif action == "preserve":
                rstate = record.get("state") or "unknown"
                reason = "本事务未触及" if rstate == "pending" else "写入后被外部改动"
                preserved.append(f"{record['dest']}（{reason}）")
            plan.append((action, record))
        if plan_problems:
            print(f"撤销前核对失败：{len(plan_problems)} 项备份问题，未修改任何文件"
                  "（备份完整性不可用 --force 绕过）", file=sys.stderr)
            for line in plan_problems:
                print("  " + line, file=sys.stderr)
            return 7

    # --- 阶段 2：只恢复/删除计划中 `undo` 的对象（恢复内容来自已校验的备份，重复调用无副作用） ---
    leftovers: list[str] = []
    acted = [record for action, record in plan if action == "undo"]
    for record in acted:
        leftovers.extend(undo_record(layout, record))

    # --- 阶段 3a：清掉本次落位模块的字节码缓存（属本次副产物；未触及的模块不动） ---
    cleaned = clean_bytecode(layout, acted)

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

    # --- 阶段 4：收尾（只校验本事务真正恢复/删除过的对象） ---
    for record in acted:
        cur = target_state(layout, record)
        if record["kind"] == "modified":
            if cur != "pre":
                leftovers.append(f"恢复后与前置内容不一致：{record['dest']}（{cur}）")
        elif cur != "missing":
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
    for line in preserved:
        print("preserved: " + line, file=sys.stderr)
    summary = f"撤销完成：{len(acted)} 个目标已恢复/删除，指纹已校验"
    notes = []
    if warnings:
        notes.append(f"{len(warnings)} 条告警")
    if preserved:
        notes.append(f"{len(preserved)} 个外部对象被保留")
    print(summary + (f"（{'、'.join(notes)}）" if notes else ""))
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
