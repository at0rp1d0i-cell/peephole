#!/usr/bin/env python3
"""oracle 全量报告 → 聚合段（可复核派生）。

背景：`p2-calib-oracle.py` 产出的是全量报告，其 `comparisons` 逐比较明细占了
绝大部分体积（24,192 行 / 约 6.4 MB，全文件 8.27 MB）。入库版本只保留**结论所需
的全部数值段**：除 `comparisons` 外的原样字段、`decode` 逐比较行（112 行，是
`out_norm` 中位/最大等被引用数字的来源），以及全量件的字节数与 sha256。

被剥离的 `prefill` 逐比较行不以统计口径外的形式保留，但派生段给出逐 scope 的
计数与量级摘要，使「剥离后结论仍可核对、且剥离范围可界定」。

用法：

    python3 tools/p2-oracle-aggregate.py --from evidence/p3-calib/oracle-original-3a.json
    python3 tools/p2-oracle-aggregate.py --check        # 用全量件复核已入库的聚合段

`--check` 需要全量件在本机可达（数据盘）。全量件入库被移除后，其**字节数与 sha256
登记在 `reports/evidence-index.md` 的树外登记表**（状态源 `reports/evidence-registry.json`）；`--check` 同时核对这两项。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

AGGREGATE_SUFFIX = ".aggregates.json"
# 仓内全量件的规范相对路径：聚合段与它的相对位置绑定，因此 `--check` 的比对对象
# 恒为仓内聚合段，而 `--from` 只决定「全量件现在在哪」（数据盘归档也适用）。
DEFAULT_SOURCE = "evidence/p3-calib/oracle-original-3a.json"
OMITTED_KEY = "comparisons"
# 逐比较行按 scope 选择性保留：被报告直接引用的 scope 必须留在聚合段里。
PUBLISHED_SCOPES = ("decode",)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values: list) -> dict:
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def derive(full: dict, source: str) -> dict:
    """由全量报告构建聚合段。纯函数：同输入必得同输出。"""
    rows = full.get(OMITTED_KEY, [])
    if not isinstance(rows, list):
        raise SystemExit(f"{source}: `{OMITTED_KEY}` 不是列表，无法派生")

    aggregate = {k: v for k, v in full.items() if k != OMITTED_KEY}
    aggregate["comparisons_published"] = [
        r for r in rows if r.get("scope") in PUBLISHED_SCOPES
    ]
    aggregate["comparisons_published_note"] = (
        "仅保留 " + "/".join(PUBLISHED_SCOPES) + " 逐比较行：报告引用的 out_norm "
        "中位与最大值由这些行直接算出（见 reports/p2-single/masked-contract-proposal.md）。"
    )
    aggregate["comparisons_omitted"] = {
        "count": len(rows) - len(aggregate["comparisons_published"]),
        "scopes": sorted({r.get("scope") for r in rows} - set(PUBLISHED_SCOPES)),
        "note": "逐比较明细留远端数据盘；本段只给逐 scope 量级摘要。",
    }
    aggregate["derived"] = {
        scope: {
            "out_norm": quantiles(
                [r.get("out_norm") for r in rows if r.get("scope") == scope]
            ),
            "max_abs_err_max": max(
                (
                    r["max_abs_err"]
                    for r in rows
                    if r.get("scope") == scope and "max_abs_err" in r
                ),
                default=None,
            ),
        }
        for scope in sorted({r.get("scope") for r in rows if r.get("scope")})
    }
    aggregate["derived"]["all"] = {
        "out_norm": quantiles([r.get("out_norm") for r in rows])
    }
    aggregate["full_file"] = {
        "path": os.path.relpath(source, os.path.dirname(os.path.dirname(source)) or "."),
        "bytes": os.path.getsize(source),
        "sha256": sha256_file(source),
        "note": "全量件不随仓发行；字节数与 sha256 为剥离当时的实测值。",
    }
    return aggregate


def aggregate_path_for(source: str) -> str:
    return source[: -len(".json")] + AGGREGATE_SUFFIX


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--from",
        dest="source",
        default=DEFAULT_SOURCE,
        help="全量报告路径（默认 evidence/p3-calib/oracle-original-3a.json）",
    )
    ap.add_argument("--out", default=None, help="聚合段输出路径（默认 <全量件>.aggregates.json）")
    ap.add_argument(
        "--check",
        action="store_true",
        help="用全量件复核已入库的聚合段（含字节数与 sha256）",
    )
    args = ap.parse_args(argv)

    target = args.out or aggregate_path_for(DEFAULT_SOURCE)
    if not os.path.exists(args.source):
        print(f"全量件不存在：{args.source}", file=sys.stderr)
        return 2

    derived = derive(json.load(open(args.source, encoding="utf-8")), args.source)

    if args.check:
        if not os.path.exists(target):
            print(f"聚合段不存在：{target}", file=sys.stderr)
            return 2
        committed = json.load(open(target, encoding="utf-8"))
        committed_full = committed.get("full_file", {})
        problems = []
        if committed_full.get("sha256") != derived["full_file"]["sha256"]:
            problems.append(
                f"full_file.sha256 不一致：登记 {committed_full.get('sha256')} / 实测 {derived['full_file']['sha256']}"
            )
        if committed_full.get("bytes") != derived["full_file"]["bytes"]:
            problems.append(
                f"full_file.bytes 不一致：登记 {committed_full.get('bytes')} / 实测 {derived['full_file']['bytes']}"
            )
        for key in sorted(set(committed) | set(derived)):
            if committed.get(key) != derived.get(key):
                problems.append(f"字段 `{key}` 与全量件派生结果不一致")
        if problems:
            for p in problems:
                print(f"MISMATCH {p}", file=sys.stderr)
            return 1
        print(
            f"OK {target} 与 {args.source} 派生结果一致"
            f"（{derived['full_file']['bytes']} B / {derived['full_file']['sha256'][:16]}…）"
        )
        return 0

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(derived, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(
        f"written: {target}（{os.path.getsize(target)} B；"
        f"保留 {len(derived['comparisons_published'])} 行逐比较，"
        f"剥离 {derived['comparisons_omitted']['count']} 行）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
