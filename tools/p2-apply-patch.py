#!/usr/bin/env python3
"""阶段 05 窄 patch 的部署 / 校验 / 撤销（版本锁定、可撤销、不覆盖他人改动）。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py apply    # 部署（拒绝未知前置哈希）
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py verify   # 校验当前部署与 manifest 一致
    "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py revert   # 撤销：恢复原文件 + 删除新增文件 + 校验指纹

设计要点（依 SUP-004-R1 §3）：
- 运行时 import 的是 site-packages 副本，源码 checkout 不生效 → **两处都部署**；
- `apply` 前要求每个目标文件的当前 sha256 等于 manifest 记录的前置哈希（未知即拒绝），
  并把原文件字节保存到 `vllm-patch/orig/`（撤销的唯一来源）；
- `revert` 后**必须**校验：被改文件回到前置哈希、新增文件已删除、包目录已移除；
  不以「两副本彼此相同」作为恢复成功的判据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCH_ROOT = REPO / "vllm-patch"
ORIG_ROOT = PATCH_ROOT / "orig"
DEPLOYED = PATCH_ROOT / "deployed.json"

SRC_PKG_ROOT = REPO / "vllm" / "vllm"  # 源码包的包根（vllm/vllm/...）
INSTALL_ROOT = REPO / "venvs/attnview/lib/python3.12/site-packages"  # 运行时副本
INSTALL_PKG_ROOT = INSTALL_ROOT / "vllm"
PACKAGE_INSTALL_ROOT = INSTALL_ROOT / "attnview"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800")


def load_manifest() -> dict:
    path = PATCH_ROOT / "manifest.json"
    if not path.is_file():
        raise SystemExit("缺少 vllm-patch/manifest.json：先运行 tools/p2-gen-patch.py")
    return json.loads(path.read_text())


def load_deployed() -> dict | None:
    return json.loads(DEPLOYED.read_text()) if DEPLOYED.is_file() else None


def targets(manifest: dict) -> list[tuple[Path, Path, str, str]]:
    """返回 `(部署源, 部署目标, 期望前置哈希, 期望后置哈希)`；前置哈希为空表示新增文件。"""
    out: list[tuple[Path, Path, str, str]] = []
    for rel, info in sorted(manifest["edits"].items()):
        src = REPO / info["patched_path"]
        for root in (SRC_PKG_ROOT, INSTALL_PKG_ROOT):
            out.append((src, root / rel, info["pre_sha256"], info["post_sha256"]))
    for item in manifest["new_files"]:
        src = REPO / item["src"]
        for root in (SRC_PKG_ROOT, INSTALL_PKG_ROOT):
            out.append((src, root / item["dest"], "", item["sha256"]))
    for item in manifest["package_files"]:
        src = REPO / item["file"]
        rel = Path(item["file"]).name
        out.append((src, PACKAGE_INSTALL_ROOT / rel, "", item["sha256"]))
    return out


def do_apply(manifest: dict) -> int:
    if load_deployed() is not None:
        print("已处于部署状态：先 revert 再 apply", file=sys.stderr)
        return 2
    records = []
    for src, dest, pre, post in targets(manifest):
        if not src.is_file():
            print(f"缺少部署源：{src}", file=sys.stderr)
            return 3
        if dest.exists():
            current = sha256_file(dest)
            if pre and current != pre:
                print(
                    f"拒绝部署：{dest} 的当前哈希 {current[:12]} != 期望前置 {pre[:12]}"
                    "（文件已被他人改动或版本不符）",
                    file=sys.stderr,
                )
                return 4
            if not pre and current != post:
                print(
                    f"拒绝部署：新增目标 {dest} 已存在且与 patch 不一致（{current[:12]}）",
                    file=sys.stderr,
                )
                return 4
            orig = ORIG_ROOT / Path(str(dest).replace(str(REPO) + "/", ""))
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, orig)
            records.append(
                {"dest": str(dest.relative_to(REPO)), "orig": str(orig.relative_to(REPO)),
                 "pre": current, "post": post, "kind": "modified"}
            )
        else:
            records.append({"dest": str(dest.relative_to(REPO)), "orig": None,
                            "pre": None, "post": post, "kind": "added"})
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        got = sha256_file(dest)
        if got != post:
            print(f"部署后哈希不符：{dest} {got[:12]} != {post[:12]}", file=sys.stderr)
            return 5
    DEPLOYED.write_text(
        json.dumps(
            {"deployed_cst": now_cst(), "pin_commit": manifest.get("pin_commit"), "files": records},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"部署完成：{len(records)} 个目标（源码 checkout + site-packages 两处）")
    return 0


def do_verify(manifest: dict) -> int:
    deployed = load_deployed()
    if deployed is None:
        print("未部署（无 vllm-patch/deployed.json）")
        return 0
    bad = []
    for rec in deployed["files"]:
        dest = REPO / rec["dest"]
        if not dest.is_file():
            bad.append(f"缺失 {rec['dest']}")
            continue
        got = sha256_file(dest)
        if got != rec["post"]:
            bad.append(f"哈希不符 {rec['dest']} {got[:12]} != {rec['post'][:12]}")
    print(f"校验：{len(deployed['files'])} 个目标，" + ("全部一致" if not bad else "存在差异"))
    for line in bad:
        print("  " + line)
    return 0 if not bad else 6


def do_revert(manifest: dict) -> int:
    deployed = load_deployed()
    if deployed is None:
        print("未部署：无需撤销")
        return 0
    problems: list[str] = []
    for rec in deployed["files"]:
        dest = REPO / rec["dest"]
        if rec["kind"] == "added":
            if dest.is_file():
                dest.unlink()
            if dest.is_file():
                problems.append(f"新增文件未删除：{rec['dest']}")
            continue
        orig = REPO / rec["orig"]
        if not orig.is_file():
            problems.append(f"缺少原始副本：{rec['orig']}")
            continue
        shutil.copy2(orig, dest)
        got = sha256_file(dest)
        if got != rec["pre"]:
            problems.append(f"恢复后哈希不符 {rec['dest']} {got[:12]} != {rec['pre'][:12]}")
    # 包目录：仅当已无新增文件时移除
    if PACKAGE_INSTALL_ROOT.is_dir():
        shutil.rmtree(PACKAGE_INSTALL_ROOT)
        if PACKAGE_INSTALL_ROOT.is_dir():
            problems.append(f"包目录未删除：{PACKAGE_INSTALL_ROOT}")
    for rec in deployed["files"]:
        dest = REPO / rec["dest"]
        if dest.exists() and rec["kind"] == "added":
            problems.append(f"残留新增文件：{rec['dest']}")
    # 收尾：清空 orig 暂存并删除 deployed 记录
    if not problems:
        if ORIG_ROOT.is_dir():
            shutil.rmtree(ORIG_ROOT)
        DEPLOYED.unlink()
        print(f"撤销完成：{len(deployed['files'])} 个目标已恢复/删除，指纹已校验")
        return 0
    print("撤销存在问题（未清理 orig/deployed.json）：", file=sys.stderr)
    for line in problems:
        print("  " + line, file=sys.stderr)
    return 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("apply", "verify", "revert"))
    args = ap.parse_args()
    manifest = load_manifest()
    if args.action == "apply":
        return do_apply(manifest)
    if args.action == "verify":
        return do_verify(manifest)
    return do_revert(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
