#!/usr/bin/env python
"""E5 附加：证明服务端 chat template 的 enable_thinking 开关真实生效，并与本地渲染交叉验证。

方法（全部走服务端口，不额外加载模型）：
  1. 同一 messages、同一 port，分别以 enable_thinking=false / true 调 /tokenize，比较 prompt token 数；
     两者必须不同（false 应显著少，因为不注入 reasoning 指令、且以空 think 块收尾）。
  2. 用本地 transformers + 同一模板渲染同一 messages，比较 token 数与 token id 序列，
     确认服务端渲染与本地渲染一致（token-span 映射后续要用）。
  3. 用 /detokenize 打印服务端渲染后的完整 prompt 文本（可读证据）。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" tools/e5-prompt-check.py --out DIR --port 8000 --snapshot DIR
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

MESSAGES = [
    {"role": "system", "content": "You are a precise assistant. Answer with a single short sentence."},
    {"role": "user", "content": "Name the capital of France."},
]


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    report: dict[str, object] = {"messages": MESSAGES, "server": {}, "local": {}}
    for label, flag in [("thinking_disabled", False), ("thinking_enabled", True)]:
        body = {
            "model": args.model,
            "messages": MESSAGES,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": flag},
        }
        tok = post(f"{base}/tokenize", body)
        ids = tok.get("tokens") or tok.get("input_ids") or []
        text = post(f"{base}/detokenize", {"model": args.model, "tokens": ids}).get("prompt", "")
        report["server"][label] = {"enable_thinking": flag, "token_count": len(ids), "token_ids": ids, "prompt_text": text}
        (out / f"e5-prompt-server-{label}.txt").write_text(text, encoding="utf-8")

    tok_local = AutoTokenizer.from_pretrained(args.snapshot)
    for label, flag in [("thinking_disabled", False), ("thinking_enabled", True)]:
        text = tok_local.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=flag)
        ids = list(tok_local(text, add_special_tokens=False)["input_ids"])
        report["local"][label] = {"enable_thinking": flag, "token_count": len(ids), "token_ids": ids, "prompt_text": text}

    s_off, s_on = report["server"]["thinking_disabled"], report["server"]["thinking_enabled"]
    l_off = report["local"]["thinking_disabled"]
    report["checks"] = {
        "server_flag_changes_prompt": s_off["token_count"] != s_on["token_count"],
        "server_off_tokens": s_off["token_count"],
        "server_on_tokens": s_on["token_count"],
        "server_matches_local_off_count": s_off["token_count"] == l_off["token_count"],
        "server_matches_local_off_ids": s_off["token_ids"] == l_off["token_ids"],
        "server_prompt_equals_local_off_text": s_off["prompt_text"] == l_off["prompt_text"],
    }
    (out / "e5-prompt-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    print("\nserver(off) prompt:", repr(s_off["prompt_text"]))
    print("server(on)  prompt:", repr(s_on["prompt_text"])[:200], "...")
    print("\nwritten:", out / "e5-prompt-check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
