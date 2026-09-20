#!/usr/bin/env python
"""E2 校验：以 Hub 元数据（E1 的 JSON）为准，逐文件核对本地快照的大小与 sha256。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e2-verify.py --snapshot DIR --identity JSON --out DIR

说明：Hub 侧只提供 LFS 文件的 sha256（lfs.sha256）；非 LFS 文件用 blob_id 记录但不能由它复算
本地 sha256（blob_id 是服务端 object id），因此对非 LFS 文件只校大小并另行记录本地 sha256。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lib import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()

    snap = Path(args.snapshot)
    ident = json.loads(Path(args.identity).read_text())
    entry = ident["repos"][args.repo]
    expected = {f["path"]: f for f in entry["files"]}

    rows = []
    total_bytes = 0
    n_size_ok = n_size_bad = 0
    n_lfs_ok = n_lfs_bad = 0
    n_missing = 0
    for path, meta in sorted(expected.items()):
        local = snap / path
        row = {"path": path, "expected_size": meta["size"], "expected_lfs_sha256": meta["lfs_sha256"]}
        if not local.exists():
            row["status"] = "MISSING"
            n_missing += 1
            rows.append(row)
            continue
        size = local.stat().st_size
        total_bytes += size
        row["local_size"] = size
        row["size_ok"] = size == meta["size"]
        n_size_ok += row["size_ok"]
        n_size_bad += not row["size_ok"]
        if not args.skip_hash:
            digest = sha256_file(local)
            row["local_sha256"] = digest
            if meta["lfs_sha256"]:
                row["lfs_sha256_ok"] = digest == meta["lfs_sha256"]
                n_lfs_ok += row["lfs_sha256_ok"]
                n_lfs_bad += not row["lfs_sha256_ok"]
        row["status"] = "ok" if row["size_ok"] else "SIZE_MISMATCH"
        rows.append(row)

    extra = sorted(p.name for p in snap.iterdir() if p.name not in expected)
    report = {
        "snapshot": str(snap),
        "revision": entry["revision_resolved"],
        "hub_file_count": len(expected),
        "local_file_count": len(rows) - n_missing,
        "missing": n_missing,
        "extra_local_files": extra,
        "size_matches": n_size_ok,
        "size_mismatches": n_size_bad,
        "lfs_sha256_matches": n_lfs_ok,
        "lfs_sha256_mismatches": n_lfs_bad,
        "hashed": not args.skip_hash,
        "total_local_bytes": total_bytes,
        "hub_total_bytes": entry["total_bytes"],
        "total_bytes_match": total_bytes == entry["total_bytes"],
        "files": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2-verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
    bad = [r["path"] for r in rows if r["status"] != "ok"]
    print("bad:", bad if bad else "none")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
