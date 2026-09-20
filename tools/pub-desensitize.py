#!/usr/bin/env python3
"""发布前脱敏：把仓库里的内部标识换成公开占位符。

## 边界（读之前先读这段）

**规则可公开，原值键必须留在仓外。**
本文件是发布仓的一部分，所以它只允许出现「公开的替换规则」——正则形状、
占位符字面量、规则顺序、退出码语义。真实值（容器主机名、GPU UUID、内部协调目录的
用户名/实例段等）是被替换**对象**，它们不得出现在本文件、本仓的注释、报告或提交信息里。
没有「真实值 → 占位符」的对照表，因为占位符是**固定 token**：不需要映射就不能被反推，
也就没有键需要保管。判断标准：把本文件和 `reports/desensitization.md` 一起公开，
读者的收获只有「这些形状的串会被换成这些字面量」，而不是任何一条真实标识。

## 与历史脱敏的关系

三条默认规则与 `/root/autodl-tmp/sanitize-tree.sh`（`git filter-branch --tree-filter` 用的
执行者，作用于**全部历史**）逐条一致，包括主机名只吃 `[a-z0-9]+-[a-z0-9]+` 两段这一点——
所以 `autodl-container-<id>-<id>-storage` 这类带后缀的写法会变成 `container-host-storage`
（后缀是有意的存储卷命名，不是标识的一部分）。本工具是同一策略的仓内可复跑版本：
历史重写之后，它用来在最终工作树上复核（`--check` 必须 0 命中）并覆盖后续新增的文件。
两条实现若不一致，`--check` 就会与历史里的字节对不上，所以**改规则必须同时改两处**。

## 规则表

`RULES` 是**有序**表。默认三条（主机名 / GPU UUID / 协调目录）与 tree-filter 一致；
绝对路径规则默认关闭，见下。

`/root/autodl-tmp/attnview` 等绝对路径是 `env.sh` / `install-runtime.sh` / `tools/*.py` 里的
**功能性默认值**（活值）：改写会打断本地可复现链条与 `vllm-patch` 部署指纹，故默认保留，
放在 `--include-paths` 开关后。这些规则内部仍按「先长后短」排列：`/root/autodl-tmp/attnview`
先于 `/root/autodl-tmp`，`/root/autodl-tmp` 先于 `/root`；前一条吃掉的区间后一条看不到。
`--include-paths` 下**跳过本工具自身**：规则表与注释是策略文本，路径规则会把第 3 条规则的
字面量也换成占位符，工具从此认不出真实路径（自毁）；默认三条规则仍会扫描自身。

## 从不动的东西

1. 二进制（含 NUL）与 > `MAX_TEXT_BYTES` 的文件：跳过并计入跳过清单（证据大件在索引里只登记
   size+sha256；tree-filter 侧同样用 `grep -I` 跳过二进制）。
2. 被本仓其他文本文件记过 sha256 的文件：**照改**，但改完会列出「哪个文件的哪一行记的哈希
   过期了」以及可直接替换的新表行——这些登记处要同步更新，否则审计链对不上（见报告）。
3. `--include-paths` 下的本工具自身（原因见上一节），跳过原因记为 `self(policy)`。

## 行为

- 幂等：`--apply` 之后再 `--check`，命中数必为 0（exit 0）。
- 无副作用默认：默认动作是 `--check`，只读；唯一的写盘动作是显式 `--apply`。

## 退出码

- `0` 无命中
- `1` 有命中（`--check` 的失败信号）
- `2` 用法/路径错误
- `3` `--apply` 写后自检失败

## 用法

    python3 tools/pub-desensitize.py --check                     # 只报告（默认动作）
    python3 tools/pub-desensitize.py --apply                     # 就地改写，打印每文件改动次数
    python3 tools/pub-desensitize.py --check --include-paths     # 连绝对路径规则一起查
    python3 tools/pub-desensitize.py --check --paths 'evidence/**/*.md'
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量（规则表：可公开）
# --------------------------------------------------------------------------- #

MAX_TEXT_BYTES = 10 * 1024 * 1024  # 超过则跳过（证据大件只登记 size+sha256，不入库）
CONTAINER_HOST_PLACEHOLDER = "container-host"
GPU_UUID_PLACEHOLDER = "GPU-<redacted>"
SUPERVISION_PLACEHOLDER = "/path/to/supervision"
ATTNVIEW_ROOT_PLACEHOLDER = "/path/to/attnview"
DATA_DISK_PLACEHOLDER = "/data"
GENERIC_HOME_PLACEHOLDER = "/home/user"

# 64 位十六进制串：可能是「别的文件记住了这个文件的 sha256」的证据
SHA256_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    replacement: str
    needs_include_paths: bool
    note: str


# 顺序 = 应用顺序。前 3 条与 /root/autodl-tmp/sanitize-tree.sh 逐条一致（作用于全部历史）。
RULES: tuple[Rule, ...] = (
    Rule(
        name="container-host",
        pattern=r"autodl-container-[a-z0-9]+-[a-z0-9]+",
        replacement=CONTAINER_HOST_PLACEHOLDER,
        needs_include_paths=False,
        note="容器主机名；只吃两段 id，故 -storage 之类后缀保留为 container-host-storage",
    ),
    Rule(
        name="gpu-uuid",
        pattern=r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        replacement=GPU_UUID_PLACEHOLDER,
        needs_include_paths=False,
        note="GPU UUID → 固定 token（无编号、无映射表）",
    ),
    Rule(
        name="supervision-root",
        # `[-]` 而不是裸 `-`：让**本文件里这个模式自身的字面文本**不匹配它自己的规则。
        # 否则 --check 永远非 0，且 --apply 会改写本工具自己的规则表（自毁）。
        # 语义与裸 `-` 完全等价（单字符类），tree-filter 侧用字面 `-`，两侧替换结果一致。
        pattern=r"/root/autodl-tmp/attnview[-]supervision",
        replacement=SUPERVISION_PLACEHOLDER,
        needs_include_paths=False,
        note="内部协调目录（字面前缀替换，其后路径自然保留）",
    ),
    Rule(
        name="abs-attnview-root",
        pattern=r"/root/(?:autodl-tmp/)?attnview(?![A-Za-z0-9_-])([^\s\"'`]*)",
        replacement=ATTNVIEW_ROOT_PLACEHOLDER + r"\1",
        needs_include_paths=True,
        note="仓库自身的绝对根路径（保留其后的相对尾巴）",
    ),
    Rule(
        name="abs-data-disk",
        pattern=r"/root/autodl-tmp(?![A-Za-z0-9_-])([^\s\"'`]*)",
        replacement=DATA_DISK_PLACEHOLDER + r"\1",
        needs_include_paths=True,
        note="数据盘绝对路径（须在 abs-attnview-root 之后、abs-root-home 之前）",
    ),
    Rule(
        name="abs-root-home",
        pattern=r"/root(?![A-Za-z0-9_.-])([^\s\"'`]*)",
        replacement=GENERIC_HOME_PLACEHOLDER + r"\1",
        needs_include_paths=True,
        note="其余 /root 下路径与裸 /root（HOME=/root 之类）",
    ),
)

_COMPILED = {r.name: re.compile(r.pattern) for r in RULES}


# --------------------------------------------------------------------------- #
# 路径集合
# --------------------------------------------------------------------------- #


def repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True, check=True).stdout.strip()).resolve()


def git_paths(root: Path) -> list[str]:
    """已跟踪 ∪ 未跟踪非忽略（gitignore 命中的大件不会出现在这里）。"""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p)


def expand_globs(root: Path, patterns: list[str]) -> tuple[list[str], list[str]]:
    picked: set[str] = set()
    empty: list[str] = []
    for pat in patterns:
        base = pat if os.path.isabs(pat) else str(root / pat)
        hits = [p for p in glob.glob(base, recursive=True) if Path(p).is_file()]
        if not hits:
            empty.append(pat)
            continue
        for p in hits:
            rp = str(Path(p).resolve())
            try:
                picked.add(str(Path(rp).relative_to(root)))
            except ValueError:
                picked.add(rp)  # 仓外文件：保持绝对路径
    return sorted(picked), empty


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #


@dataclass
class Hit:
    rel: str
    rule: str
    line: int


def read_text_bytes(path: Path) -> bytes | None:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    return raw


class Scanner:
    def __init__(self, root: Path, *, include_paths: bool):
        self.root = root
        self.include_paths = include_paths
        self.skipped: dict[str, str] = {}          # rel → 原因
        self.texts: dict[str, bytes] = {}          # 仅文本文件
        self.distinct: dict[str, set[str]] = {r.name: set() for r in RULES}

    def active_rules(self) -> list[Rule]:
        return [r for r in RULES if self.include_paths or not r.needs_include_paths]

    def rewrite(self, rel: str, text: str) -> tuple[str, dict[str, int], list[Hit]]:
        """按 RULES 顺序串行改写；返回 (新文本, 每规则次数, 命中明细)。

        命中明细来自**同一次串行过程**，所以前一条规则吃掉的区间不会在后一条里重复出现
        （否则 `/root/autodl-tmp/attnview/x` 会被三条路径规则各报一遍）。
        规则的模式都不含换行、占位符也不含换行，所以行号在整轮改写里保持有效。
        """
        counts: dict[str, int] = {}
        hits: list[Hit] = []
        for rule in self.active_rules():
            rx = _COMPILED[rule.name]
            n = 0

            def repl(m: re.Match, *, _rule: Rule = rule, _text: str = text) -> str:
                nonlocal n
                n += 1
                self.distinct[_rule.name].add(m.group(0))
                hits.append(Hit(rel, _rule.name, _text.count("\n", 0, m.start()) + 1))
                return m.expand(_rule.replacement)

            text = rx.sub(repl, text)
            if n:
                counts[rule.name] = n
        return text, counts, hits

    # -- 输入 ------------------------------------------------------------- #

    def load(self, rels: list[str]) -> None:
        self_rel = self._self_rel()
        for rel in rels:
            # 路径规则会改写本工具的规则表与注释（策略文本，不是数据）：`--apply --include-paths`
            # 一旦命中自身，第 3 条规则的字面量会被换成占位符，工具从此再也认不出真实路径。
            # 因此含路径规则时跳过自身；默认三条规则仍然扫描自身，真实标识不会被漏掉。
            if self.include_paths and self_rel and rel == self_rel:
                self.skipped[rel] = "self(policy)"
                continue
            path = Path(rel) if os.path.isabs(rel) else self.root / rel
            raw = read_text_bytes(path)
            if raw is None:
                self.skipped[rel] = self._skip_reason(path)
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                self.skipped[rel] = "not-utf8"
                continue
            self.texts[rel] = raw

    def _self_rel(self) -> str:
        """本工具在仓内的相对路径；不在仓内（或路径对不上）时返回空串。"""
        try:
            return str(Path(__file__).resolve().relative_to(self.root))
        except ValueError:
            return ""

    @staticmethod
    def _skip_reason(path: Path) -> str:
        if not path.is_file():
            return "missing"
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                return f"over-{MAX_TEXT_BYTES}B"
            if b"\0" in path.read_bytes():
                return "binary(NUL)"
        except OSError:
            return "unreadable"
        return "not-text"

    def pinned_map(self) -> dict[str, list[str]]:
        """rel → 记录过该文件 sha256 的其他文件（本仓内的审计链）。"""
        by_sha: dict[str, list[str]] = {}
        for rel, raw in self.texts.items():
            by_sha.setdefault(hashlib.sha256(raw).hexdigest(), []).append(rel)
        pinned: dict[str, list[str]] = {}
        for rel, raw in self.texts.items():
            for token in SHA256_TOKEN.findall(raw.decode("utf-8", "replace")):
                for target in by_sha.get(token, ()):
                    if target != rel:
                        pinned.setdefault(target, []).append(rel)
        return {k: sorted(set(v)) for k, v in pinned.items()}


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def print_table(rows: list[tuple[str, str, list[Hit]]], totals: dict[str, int],
                distinct: dict[str, set[str]], rules: list[Rule]) -> None:
    print("规则命中汇总（文件 / 规则 / 次数）")
    print(f"{'FILE':<64} {'RULE':<20} {'N':>4}  LINES")
    for rel, rule, rh in rows:
        lines = ",".join(str(h.line) for h in rh[:12]) + ("…" if len(rh) > 12 else "")
        print(f"{rel:<64} {rule:<20} {len(rh):>4}  {lines}")
    print()
    print("规则小计（只打印计数与去重个数，不打印原值）")
    for rule in rules:
        if totals.get(rule.name):
            print(f"  {rule.name:<20} 命中 {totals[rule.name]:>5}  文件 "
                  f"{len({r for r, ru, _ in rows if ru == rule.name}):>3}  去重值 "
                  f"{len(distinct[rule.name]):>3}")
    n_files = len({r for r, _, _ in rows})
    print()
    print(f"合计 {sum(totals.values())} 处 / {n_files} 个文件")


def print_pin_warnings(pinned: dict[str, list[str]], touched: list[str]) -> None:
    flagged = [rel for rel in touched if rel in pinned]
    if not flagged:
        return
    print()
    print("[pin] 以下文件被本仓其他文件登记过 sha256——本次改写会让那些登记行过期：")
    for rel in flagged:
        for holder in pinned[rel]:
            print(f"  {holder} 记的 {rel}")
    print("  处理方式见 reports/desensitization.md：同步更新登记行（新 sha256 见下），"
          "或在登记处注明该文件已按公开规则改写。")


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pub-desensitize-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, stat.S_IMODE(mode))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pub-desensitize.py",
        description="把仓库里的容器主机名 / GPU UUID / 协调目录（可选：绝对路径）换成公开占位符。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__.split("## 用法")[-1]).strip(),
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="只报告（默认动作）；有命中则 exit 1")
    mode.add_argument("--apply", action="store_true", help="就地改写命中并打印每文件改动次数")
    ap.add_argument("--paths", nargs="+", metavar="GLOB",
                    help="自定义文件集合（默认 = 已跟踪 ∪ 未跟踪非忽略）")
    ap.add_argument("--include-paths", action="store_true",
                    help="同时应用绝对路径规则（默认关闭：路径是功能性默认值；此模式跳过本工具自身）")
    ap.add_argument("--repo", metavar="DIR", help="仓库根（默认取 git rev-parse --show-toplevel）")
    args = ap.parse_args(argv)

    root = repo_root(args.repo)
    if args.paths:
        rels, empty = expand_globs(root, args.paths)
        for pat in empty:
            print(f"[warn] --paths 模式无匹配：{pat}", file=sys.stderr)
        if not rels:
            print("--paths 未匹配到任何文件", file=sys.stderr)
            return 2
    else:
        rels = git_paths(root)

    scanner = Scanner(root, include_paths=args.include_paths)
    scanner.load(rels)
    pinned = scanner.pinned_map()

    print(f"仓库 {root}")
    print(f"扫描 {len(scanner.texts)} 个文本文件"
          + (f"，跳过 {len(scanner.skipped)} 个" if scanner.skipped else "")
          + f"；规则集 {len(scanner.active_rules())}/{len(RULES)} 条（"
          + ("含绝对路径" if args.include_paths else "默认三条：主机名 / GPU UUID / 协调目录")
          + f"）；模式 {'apply' if args.apply else 'check'}")
    for rel, why in sorted(scanner.skipped.items()):
        print(f"  [skip] {rel}：{why}")
    print()

    totals: dict[str, int] = {}
    rows: list[tuple[str, str, list[Hit]]] = []
    per_file: dict[str, list[Hit]] = {}
    for rel, raw in scanner.texts.items():
        _, counts, hits = scanner.rewrite(rel, raw.decode("utf-8", "replace"))
        if not hits:
            continue
        per_file[rel] = hits
        by_rule: dict[str, list[Hit]] = {}
        for h in hits:
            by_rule.setdefault(h.rule, []).append(h)
        for rule_name, rh in sorted(by_rule.items()):
            rows.append((rel, rule_name, rh))
            totals[rule_name] = totals.get(rule_name, 0) + len(rh)

    print_table(sorted(rows), totals, scanner.distinct, scanner.active_rules())
    print_pin_warnings(pinned, sorted(per_file))

    touched = sorted(per_file)
    if not args.apply:
        if touched:
            print("\n[check] 存在命中；用 --apply 就地改写")
            return 1
        print("\n[check] 0 命中")
        return 0

    rewritten: list[tuple[str, str, str, int]] = []   # rel, before, after, n
    verify_fail: list[str] = []
    for rel in touched:
        raw = scanner.texts[rel]
        path = Path(rel) if os.path.isabs(rel) else root / rel
        new_text, counts, _ = scanner.rewrite(rel, raw.decode("utf-8"))
        atomic_write(path, new_text.encode("utf-8"))
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        rewritten.append((rel, hashlib.sha256(raw).hexdigest(), after, sum(counts.values())))
        print(f"[write] {rel}：改写 {sum(counts.values())} 处（"
              + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())) + "）")
        back = read_text_bytes(path)
        if back is None:
            verify_fail.append(f"{rel}：写后不可读")
            continue
        leftover = scanner.rewrite(rel, back.decode("utf-8", "replace"))[2]
        if leftover:
            verify_fail.append(f"{rel}：写后仍有 {len(leftover)} 处命中（"
                               + ", ".join(sorted({h.rule for h in leftover})) + "）")

    print()
    print("改动文件 sha256（前 / 后）")
    print(f"{'FILE':<64} {'N':>4}  {'BEFORE':<16} {'AFTER':<16}")
    for rel, before, after, n in rewritten:
        print(f"{rel:<64} {n:>4}  {before[:16]} {after[:16]}")
    if not rewritten:
        print("  （无）")

    stale = [(rel, after) for rel, _, after, _ in rewritten if rel in pinned]
    if stale:
        print()
        print("[pin] 需要同步更新的登记行：")
        for rel, after in stale:
            for holder in pinned[rel]:
                print(f"  {holder}: | `{rel}` | {(root / rel).stat().st_size:,} | `{after}` |")

    if verify_fail:
        print("\n[apply] 写后自检失败：", file=sys.stderr)
        for line in verify_fail:
            print("  " + line, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
