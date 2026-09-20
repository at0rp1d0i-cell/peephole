#!/usr/bin/env python3
"""证据索引生成与自检：`reports/evidence-index.md` 是唯一的机械事实来源。

## 为什么存在

公开仓的证据索引以前是手写的，同一批文件在 8 份文档里各有一张 size/sha256 表，于是必然漂移
（实测：清单写 360 个文件时树里已有 371 个；写"工作区无未跟踪 evidence"时有 82 个；脱敏打散 15 行
已登记哈希）。本工具把机械事实收敛到一个生成物：事实从 `git ls-files` 与工作区算出，
**规则与登记**写在 `reports/evidence-registry.json`，改规则就改数据文件。

## 登记的两类

| 类 | 位置 | 语义 |
| --- | --- | --- |
| `registered_removed` | 已移出树、原件归档在 `archive_root` | 字节不随仓发行；hash 是剥离当时的实测值，仓内无法再复算，因此必须锁死 |
| `registered_only` | 仍在盘上、被 `.gitignore` 排除 | 大件 dump / 未验收件；`git add -A` 不得带走（A2 检查） |

## 自检项（`--check` 的非零退出条件）

- `A1` 同一路径既发布又登记（规则自相矛盾）。
- `A2` `registered_only` 项没有被 git 忽略：下一次 `git add -A` 会把它带走。
- `A3` `registered_only` glob 在磁盘上匹配不到任何文件（规则成了空话）。
- `A4` 悬空引用：文档/配置里写了 `evidence/...` 路径，但既不在库中、也不在任一登记表里。
- `A5` 入库文件间存在 ≥ 16 KiB 的逐字节重复对（`allow_duplicate` 允许的除外）。
- `A6` 生成的索引与磁盘上的 `reports/evidence-index.md` 不一致（索引过期）。
- `A7` （提示，不判失败）文档引用了**树外登记件**：路径在库中已不存在，表述必须与
  `registered_removed` 的状态一致（不得再写"保留在 evidence/.../"）。
- `A8` 入库文件没被任何规则覆盖（status 停在默认值 `unclassified`）——状态表把"未分类"定义为
  门禁失败，因此必须补规则，不能只让它安静地出现在生成物里。
- `A9` `registered_sets` 的 `count`/`bytes` 与磁盘不一致。口径：`glob` 展开 **减去**已逐个登记的
  成员（树内上表 / 树外登记表）后必须等于登记值——组级数字不再是无人复算的字面量。
  该组声明为 `remote-only`（字节留在数据盘）时，盘上只剩"已逐个登记"的成员（或一个都没有）
  则降级为提示：新 clone 没有数据盘，属预期差异，不是漂移。
- 提示（不判失败）：工作区未跟踪、未忽略的产物（进行中的 run），入库前必须先在规则表里分类。

## 用法

    python3 tools/pub-evidence-registry.py --write     # 重新生成 reports/evidence-index.md
    python3 tools/pub-evidence-registry.py --check     # 只读复核（默认动作）

退出码：`0` 一致且无违规 / `1` 有违规或索引过期 / `2` 用法或读取错误。
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as globmod
import json
import os
import re
import subprocess
import sys

from _lib import load_json, sha256_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE = "reports/evidence-registry.json"
INDEX_FILE = "reports/evidence-index.md"
SCOPE = ("evidence/", "logs/")
DUP_MIN_BYTES = 16 * 1024
DUP_WARN_BYTES = 1024
# 引用扫描面：人读的文档 + 机器读的配置。tools/tests 排除在外——那里的 `evidence/...`
# 多是用法示例与输出目录默认值，不是对已发布产物的声明。
SCAN_DIRS = ("reports", "configs")
SCAN_FILES = ("README.md", "CONTRIBUTING.md", ".gitmessage")
REF_RE = re.compile(r"evidence/[A-Za-z0-9_./+*-]+")


def run_git(*args: str, stdin: str = None) -> str:
    return subprocess.run(
        ["git", "-C", REPO, *args], input=stdin, capture_output=True, text=True
    ).stdout


def abs_path(rel: str) -> str:
    return os.path.join(REPO, rel)


def in_scope(path: str) -> bool:
    return path.startswith(SCOPE)


def expand_glob(pattern: str) -> list:
    hits = [
        p
        for p in globmod.glob(abs_path(pattern), recursive=True)
        if os.path.isfile(p)
    ]
    return sorted(os.path.relpath(p, REPO) for p in hits)


def matches_any(path: str, patterns: list) -> str:
    for g in patterns:
        base = g[:-3] if g.endswith("/**") else g.rstrip("/")
        if path == base or fnmatch.fnmatch(path, g) or path.startswith(base + "/"):
            return g
    return None


def classify(path: str, rules: list, default: str) -> tuple:
    for rule in rules:
        if matches_any(path, [rule["glob"]]):
            return rule["status"], rule.get("note", "")
    return default, ""


def build(doc: dict) -> dict:
    tracked_all = [p for p in run_git("ls-files").splitlines() if p]
    tracked = [p for p in tracked_all if in_scope(p)]
    untracked = [
        p
        for p in run_git("ls-files", "--others", "--exclude-standard").splitlines()
        if p and in_scope(p)
    ]
    rules, default = doc["rules"], doc.get("default_status", "unclassified")

    published = [
        {
            "path": p,
            "bytes": os.path.getsize(abs_path(p)),
            "sha256": sha256_file(abs_path(p)),
            "status": classify(p, rules, default)[0],
            "note": classify(p, rules, default)[1],
        }
        for p in sorted(tracked)
    ]
    removed = sorted(doc.get("registered_removed", []), key=lambda r: r["path"])
    only = []
    for entry in doc.get("registered_only", []):
        for path in expand_glob(entry["glob"]):
            only.append(
                {
                    "path": path,
                    "bytes": os.path.getsize(abs_path(path)),
                    "sha256": sha256_file(abs_path(path)),
                    "glob": entry["glob"],
                    "status": entry["status"],
                    "reason": entry["reason"],
                }
            )
    only_paths = {r["path"] for r in only}
    pending = [
        {
            "path": p,
            "bytes": os.path.getsize(abs_path(p)),
            "sha256": sha256_file(abs_path(p)),
            "status": "pending",
            "note": "工作区未跟踪、未忽略；入库前先在规则表里分类",
        }
        for p in sorted(untracked)
        if p not in only_paths
    ]
    return {
        "doc": doc,
        "published": published,
        "removed": removed,
        "only": only,
        "pending": pending,
        "tracked": set(tracked),
        "tracked_all": set(tracked_all),
    }


def find_violations(model: dict) -> tuple:
    doc, published, only = (
        model["doc"],
        model["published"],
        model["only"],
    )
    errors, notes, env_notes = [], [], []
    only_globs = [e["glob"] for e in doc.get("registered_only", [])]

    both = [r["path"] for r in published if matches_any(r["path"], only_globs)]
    if both:
        errors.append("A1 既发布又登记：" + ", ".join(both))

    if only:
        ignored = set(
            run_git(
                "check-ignore", "--stdin", "-z", stdin="\0".join(r["path"] for r in only)
            ).split("\0")
        )
        missing = sorted({r["path"] for r in only} - ignored)
        if missing:
            errors.append(
                f"A2 登记项未被 git 忽略（{len(missing)} 个）——git add -A 会带走它们："
                + ", ".join(missing[:8])
            )

    for entry in doc.get("registered_only", []):
        if not expand_glob(entry["glob"]):
            errors.append(f"A3 登记 glob 在磁盘上无匹配：{entry['glob']}")

    known = model["tracked_all"] | {r["path"] for r in only} | {
        g["path"] for g in doc.get("gone", [])
    } | {r["path"] for r in model["removed"]}
    scan_files = [p for p in model["tracked_all"] if p.startswith(SCAN_DIRS)]
    scan_files += [f for f in SCAN_FILES if f in model["tracked_all"]]
    refs = {}
    for f in sorted(set(scan_files)):
        if f not in SCAN_FILES and os.path.splitext(f)[1] not in (".md", ".json", ".py", ".sh"):
            continue
        try:
            with open(abs_path(f), encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        for token in REF_RE.findall(text):
            token = token.rstrip(".,;:)]}`\"'")
            if "*" in token:
                continue
            refs.setdefault(token, set()).add(f)
    removed_by_dir, removed_dirs = {}, {}
    for row in model["removed"]:
        removed_by_dir[row["path"]] = row["status"]
        parts = row["path"].split("/")
        for i in range(1, len(parts)):
            removed_dirs["/".join(parts[:i]).rstrip("/")] = row["status"]
    declared_dirs = set(removed_dirs) | {p.rstrip("/") for p in only_globs}

    ignored_cache: dict = {}

    def is_ignored(token: str) -> bool:
        if token not in ignored_cache:
            ignored_cache[token] = (
                subprocess.run(
                    ["git", "-C", REPO, "check-ignore", "-q", token], capture_output=True
                ).returncode
                == 0
            )
        return ignored_cache[token]

    for token, where in sorted(refs.items()):
        bare = token.rstrip("/")
        if (
            token in known
            or bare in known
            or matches_any(token, only_globs)
            or bare in declared_dirs
            or os.path.exists(abs_path(token))
            or is_ignored(token)          # 被 .gitignore 声明的张量 dump 等：按规则不发行
            or token.endswith("-")        # 散文里的前缀写法（如 `e5-` 系列），非路径声明
        ):
            continue
        errors.append(f"A4 悬空引用 {token}（出现在 {', '.join(sorted(where)[:3])}）")

    external = {}
    for token, where in sorted(refs.items()):
        status = removed_by_dir.get(token) or removed_dirs.get(token.rstrip("/"))
        if status is None:
            continue
        for f in where:
            if f == INDEX_FILE:      # 生成物自身登记全部树外件，不算表述漂移
                continue
            external.setdefault(f, set()).add(f"{token.rstrip('/')}（{status}）")
    for f, tokens in sorted(external.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        notes.append(
            f"A7 {f} 引用 {len(tokens)} 个树外登记件，表述需与登记状态一致：" + "、".join(sorted(tokens))
        )

    default_status = doc.get("default_status", "unclassified")
    unclassified = [r["path"] for r in published if r["status"] == default_status]
    if unclassified:
        errors.append(
            f"A8 入库文件未分类（规则表没有覆盖，status 停在默认值 {default_status!r}，"
            f"{len(unclassified)} 个）——状态表把未分类定义为门禁失败，必须补规则："
            + ", ".join(unclassified[:8])
        )

    # A9：集合登记的 count/bytes 必须能从磁盘复算。
    # 口径：glob 展开 − 已逐个登记（树内 published / 树外 removed）；登记表里的字面量数字
    # 一旦与磁盘漂移（多一个 run、把某个成员降级为逐条登记）就必须失败。
    published_paths = {r["path"] for r in published}
    removed_paths = {r["path"] for r in model["removed"]}
    for entry in doc.get("registered_sets", []):
        expanded = set(expand_glob(entry["glob"]))
        if not expanded:
            # `remote-only` 声明"字节不在仓内、留在数据盘"：在没有挂数据盘的检出（例如新 clone）上
            # 这属正常差异，无法复算 → 只提示；其它状态的组里应当有件在盘上，缺失仍是错误。
            if entry.get("status") == "remote-only":
                # 环境相关提示：只打印、**不进生成物**——否则索引会随机器状态漂移（新 clone 上 A6 永远红）。
                env_notes.append(
                    f"A9 集合登记无法复算（原始件不在本机盘上，属预期）：`{entry['glob']}`"
                )
            else:
                errors.append(f"A9 集合登记在磁盘上无匹配：{entry['glob']}")
            continue
        derived = expanded - published_paths - removed_paths
        if not derived and entry.get("status") == "remote-only" and entry["count"] > 0:
            # 组里只剩"已逐个登记"的成员、原始件（.npz/.pt 等）不在本机盘上（例如新 clone）。
            # 环境相关提示：只打印、**不进生成物**——否则索引会随机器状态漂移（新 clone 上 A6 永远红）。
            env_notes.append(f"A9 集合登记无法复算（原始件不在本机盘上，属预期）：`{entry['glob']}`")
            continue
        got_count = len(derived)
        got_bytes = sum(os.path.getsize(abs_path(p)) for p in derived)
        if (got_count, got_bytes) != (entry["count"], entry["bytes"]):
            errors.append(
                f"A9 集合登记与磁盘不一致 `{entry['glob']}`：登记 {entry['count']:,} 个 / "
                f"{entry['bytes']:,} B，实测 {got_count:,} 个 / {got_bytes:,} B"
            )

    allow = [e["glob"] for e in doc.get("allow_duplicate", [])]
    by_hash = {}
    for row in published:
        by_hash.setdefault(row["sha256"], []).append(row)
    for _, rows in sorted(by_hash.items(), key=lambda kv: sorted(r["path"] for r in kv[1])):
        if len(rows) < 2 or rows[0]["bytes"] < DUP_WARN_BYTES:
            continue
        paths = sorted(r["path"] for r in rows)
        if all(matches_any(p, allow) for p in paths):
            notes.append(f"同哈希（allow_duplicate 声明）：{', '.join(paths)}")
        elif rows[0]["bytes"] < DUP_MIN_BYTES:
            notes.append(f"同哈希（{rows[0]['bytes']:,} B，小于阈值、保留）：{', '.join(paths)}")
        else:
            errors.append(f"A5 入库文件逐字节重复（{rows[0]['bytes']:,} B）：{', '.join(paths)}")
    return errors, notes, env_notes, refs


def render(model: dict, notes: list) -> str:
    doc, published = model["doc"], model["published"]
    statuses = doc.get("status_label", {})
    order = doc.get("status_order", [])
    groups = {s: [r for r in published if r["status"] == s] for s in order}
    total = sum(r["bytes"] for r in published)
    out = [
        "# 证据索引（生成物，勿手改）",
        "",
        f"生成器：`python3 tools/pub-evidence-registry.py --write`；规则与登记：`{STATUS_FILE}`。",
        "本表覆盖 `evidence/` 与 `logs/`。机械事实（路径/字节/sha256/状态）一律现算，"
        "复核：`python3 tools/pub-evidence-registry.py --check`。",
        "",
        "## 发布规则",
        "",
    ]
    out += [f"- **{r['id']}** {r['text']}" for r in doc["rule"]]
    out += [
        "",
        "## 状态表",
        "",
        "| status | 含义 |",
        "| --- | --- |",
    ]
    out += [f"| `{s}` | {statuses.get(s, s)} |" for s in order]
    out += [
        "",
        f"树内产物 **{len(published)} 个 / {total:,} B**；"
        f"树外登记 **{len(model['removed'])} 个**；"
        f"在盘未跟踪 **{len(model['only'])} 个**。",
    ]
    archive_root = doc.get("archive_root")
    if archive_root:
        out.append(
            f"树外登记件的原件归档在 `{archive_root}`（远端数据盘），"
            "字节与 sha256 以本表为准；仓内不发行其字节。"
        )
    out.append("")

    for status in order:
        rows = groups.get(status, [])
        if not rows:
            continue
        out += [
            f"## 树内：{statuses.get(status, status)}（{len(rows)} 个 / {sum(r['bytes'] for r in rows):,} B）",
            "",
            "| 文件 | 字节 | sha256 | 说明 |",
            "| --- | ---: | --- | --- |",
        ]
        out += [
            f"| `{r['path']}` | {r['bytes']:,} | `{r['sha256']}` | {r['note']} |"
            for r in rows
        ]
        out.append("")

    if model["removed"]:
        out += [
            f"## 树外登记（{len(model['removed'])} 个 / "
            f"{sum(r['bytes'] for r in model['removed']):,} B）",
            "",
            "字节不随仓发行；哈希为剥离当时的实测值。",
            "",
            "| 文件 | 字节 | sha256 | 状态 | 原因 |",
            "| --- | ---: | --- | --- | --- |",
        ]
        out += [
            f"| `{r['path']}` | {r['bytes']:,} | `{r['sha256']}` | {r['status']} | {r.get('reason','')} |"
            for r in model["removed"]
        ]
        out.append("")

    if model["only"]:
        out += [
            f"## 在盘未入库（{len(model['only'])} 个 / "
            f"{sum(r['bytes'] for r in model['only']):,} B）",
            "",
            "留在数据盘、被 `.gitignore` 排除；A2 保证 `git add -A` 不会带走。",
            "",
            "| 文件 | 字节 | sha256 | 状态 | 原因 |",
            "| --- | ---: | --- | --- | --- |",
        ]
        out += [
            f"| `{r['path']}` | {r['bytes']:,} | `{r['sha256']}` | {r['status']} | {r['reason']} |"
            for r in model["only"]
        ]
        out.append("")

    sets = doc.get("registered_sets", [])
    if sets:
        out += [
            f"## 在盘未入库集合（{len(sets)} 组 / {sum(r['bytes'] for r in sets):,} B）",
            "",
            "按 R9 的类型规则排除（张量/序列化大件）；不与任何文档逐条绑定的组只登记组级事实，",
            "被文档引用的成员在上表逐个登记。计数口径：`glob` 展开 **减去**已逐个登记的成员"
            "（树内上表 / 树外登记表），由 A9 逐组复算。",
            "",
            "| glob | 文件数 | 合计字节 | 状态 | 说明 |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        out += [
            f"| `{r['glob']}` | {r['count']:,} | {r['bytes']:,} | {r['status']} | {r['reason']} |"
            for r in sets
        ]
        out.append("")

    gone = doc.get("gone", [])
    if gone:
        out += [
            f"## 已声明缺失 / 未执行（{len(gone)} 条）",
            "",
            "被文档或 manifest 声明、但任何位置都没有的路径；留档以免每次审计重新判一遍。",
            "",
            "| 路径 | 状态 | 说明 |",
            "| --- | --- | --- |",
        ]
        out += [f"| `{g['path']}` | {g['status']} | {g['reason']} |" for g in gone]
        out.append("")

    out += ["## 自检", ""]
    out += [f"- {n}" for n in notes] or ["- 无同哈希对"]
    out += [
        f"- 入库 {len(published)} 个 / 树外登记 {len(model['removed'])} 个 / 在盘未入库 {len(model['only'])} 个",
        f"- 引用扫描面：{', '.join(SCAN_DIRS)} + {', '.join(SCAN_FILES)}",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="重新生成索引")
    group.add_argument("--check", action="store_true", help="只读复核（默认动作）")
    args = ap.parse_args(argv)

    try:
        doc = load_json(abs_path(STATUS_FILE))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取 {STATUS_FILE} 失败：{exc}", file=sys.stderr)
        return 2
    doc.setdefault("rules", [])

    model = build(doc)
    errors, notes, env_notes, _ = find_violations(model)
    rendered = render(model, notes)
    # 先把违规打印出来，再决定是否落盘：`--write` 也会写出一个已知违规的索引，
    # 顺序反了会让操作者先看到"written"再看到 FAIL。
    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
    if args.write:
        with open(abs_path(INDEX_FILE), "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(
            f"written: {INDEX_FILE}（树内 {len(model['published'])} / 树外 {len(model['removed'])} / "
            f"在盘未入库 {len(model['only'])}）"
        )
    elif os.path.exists(abs_path(INDEX_FILE)):
        with open(abs_path(INDEX_FILE), encoding="utf-8") as fh:
            committed = fh.read()
        if committed != rendered:
            errors.append(f"A6 {INDEX_FILE} 与工作区不一致（重新 --write）")
    else:
        errors.append(f"A6 {INDEX_FILE} 不存在")

    for w in env_notes:
        print(f"WARN {w}")

    if model["pending"]:
        print(
            f"WARN 工作区未分类产物 {len(model['pending'])} 个（{sum(r['bytes'] for r in model['pending']) / 1048576:.2f} MiB）："
            f"{', '.join(r['path'] for r in model['pending'][:3])}{' …' if len(model['pending']) > 3 else ''}"
        )
    if errors:
        # 上面已经逐条打印过；A6 之类在落盘决定之后才发现，补打一次即可（幂等、无重复风险）。
        late = [e for e in errors if e.startswith("A6")]
        for e in late:
            print(f"FAIL {e}", file=sys.stderr)
        return 1
    print("OK 索引与工作区一致，无悬空引用与未声明重复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
