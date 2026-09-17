#!/usr/bin/env python
"""E5: 对已启动的原版服务发一次固定短请求，保存完整请求体/响应体与耗时。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e5-request.py --out DIR [--port 8000]

固定 prompt、temperature=0、固定 seed；不做任何质量结论。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Reply with exactly one sentence: what is the capital of France?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tag", default="e5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 64,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    t1 = time.monotonic()
    elapsed = t1 - t0

    (out / f"{args.tag}-request.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / f"{args.tag}-response-raw.json").write_bytes(raw)

    record = {"url": url, "http_status": status, "elapsed_seconds": round(elapsed, 3), "request": body}
    try:
        parsed = json.loads(raw)
        record["response"] = parsed
        choice = (parsed.get("choices") or [{}])[0]
        record["summary"] = {
            "finish_reason": choice.get("finish_reason"),
            "message_keys": sorted((choice.get("message") or {}).keys()),
            "content": (choice.get("message") or {}).get("content"),
            "reasoning_content_present": "reasoning_content" in (choice.get("message") or {}),
            "usage": parsed.get("usage"),
        }
    except json.JSONDecodeError as exc:
        record["response_parse_error"] = str(exc)
        record["response_text"] = raw.decode("utf-8", "replace")

    (out / f"{args.tag}-record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("written:", out / f"{args.tag}-record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
