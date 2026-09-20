#!/usr/bin/env python
"""E1: 通过 Hub API 复核两个候选模型的 revision / 文件清单 / 总字节。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e1-model-identity.py --out DIR

只读操作：不发 HTTP 下载请求（除 API 元数据），不落权重。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from _lib import MODEL_REVISION

CANDIDATES = [
    ("Qwen/Qwen3.8-27B", MODEL_REVISION, True),
    ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", False),
]
# 工作单 §3 E1 记录的对照值
RECORDED = {
    "Qwen/Qwen3.8-27B": {"files": 32, "bytes": 55_586_114_863},
    "Qwen/Qwen3.6-27B": {"files": 29, "bytes": 55_586_107_940},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    report: dict[str, object] = {
        "endpoint": os.environ.get("HF_ENDPOINT"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "huggingface_hub": __import__("huggingface_hub").__version__,
        "repos": {},
    }
    rc = 0
    for repo_id, revision, is_main in CANDIDATES:
        entry: dict[str, object] = {"revision_requested": revision, "is_main_download_target": is_main}
        try:
            info = api.model_info(repo_id, revision=revision, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - 失败要如实记录类型
            entry["error"] = f"{type(exc).__name__}: {exc}"
            rc = 1
            report["repos"][repo_id] = entry
            print(f"[FAIL] {repo_id}@{revision}: {entry['error']}", flush=True)
            continue

        siblings = list(info.siblings or [])
        files = []
        total = 0
        for s in siblings:
            size = s.size if s.size is not None else 0
            total += size
            files.append(
                {
                    "path": s.rfilename,
                    "size": s.size,
                    "lfs_sha256": (s.lfs or {}).get("sha256") if s.lfs else None,
                    "blob_id": s.blob_id,
                }
            )
        files.sort(key=lambda f: f["path"])
        entry.update(
            {
                "revision_resolved": info.sha,
                "revision_matches_request": info.sha == revision,
                "private": info.private,
                "gated": info.gated,
                "license": (info.card_data or {}).get("license") if info.card_data else None,
                "tags": list(info.tags or []),
                "file_count": len(files),
                "total_bytes": total,
                "total_gib": round(total / 1024**3, 3),
                "recorded_baseline": RECORDED.get(repo_id),
                "delta_vs_recorded": {
                    "files": len(files) - RECORDED[repo_id]["files"],
                    "bytes": total - RECORDED[repo_id]["bytes"],
                },
                "files": files,
            }
        )
        report["repos"][repo_id] = entry
        print(
            f"[OK] {repo_id}\n  sha={info.sha}\n  match={info.sha == revision} gated={info.gated} private={info.private}\n"
            f"  files={len(files)} (recorded {RECORDED[repo_id]['files']})\n"
            f"  bytes={total} (recorded {RECORDED[repo_id]['bytes']}, delta {total - RECORDED[repo_id]['bytes']})\n"
            f"  GiB={total / 1024**3:.3f}",
            flush=True,
        )

    (out / "e1-model-identity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {out / 'e1-model-identity.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
