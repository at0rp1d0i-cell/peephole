#!/usr/bin/env python
"""生成证据文件清单（路径 / 字节 / sha256），供 evidence-index.md 内嵌。

用法：
  "$ATTNVIEW_PYTHON" /root/attnview/tools/gen-file-inventory.py --root /root/attnview --out FILE \
      evidence/p0-model logs/serve-e4.log logs/serve-e6.log ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _lib import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("targets", nargs="+")
    args = ap.parse_args()
    root = Path(args.root)

    rows: list[tuple[str, int, str]] = []
    for t in args.targets:
        p = root / t
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    rows.append((str(f.relative_to(root)), f.stat().st_size, sha256_file(f)))
        elif p.is_file():
            rows.append((str(p.relative_to(root)), p.stat().st_size, sha256_file(p)))
        else:
            rows.append((f"{t} (MISSING)", 0, "-"))

    lines = ["| 文件 | 字节 | sha256 |", "| --- | ---: | --- |"]
    for path, size, digest in rows:
        lines.append(f"| `{path}` | {size:,} | `{digest}` |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {args.out} ({len(rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
