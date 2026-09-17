#!/usr/bin/env python
"""E3 附加：记录协议标签在目标 tokenizer 下的切分行为（纯 CPU，只读）。

用途：下一阶段的增量解析与 token-span 映射需要知道控制标签是否落在词表里、
以及标签文本会切成哪些 token（能否被"跨 token 声明"这类构造命中）。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e3-probe-protocol-tokens.py --snapshot DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PROBES = [
    "<global>",
    "</global>",
    "<focus>",
    "<focus magic_chunks=\"8\">",
    "</focus>",
    "<local>",
    "</local>",
    "<answer>",
    "</answer>",
    "<answer>42</answer>",
]
MESSAGES = [
    {"role": "system", "content": "You are precise."},
    {
        "role": "user",
        "content": (
            "Read the doc. Declare <global> first. Then "
            '<focus magic_chunks="8">PASSAGE-17</focus> and <local> the rest. '
            "Finally output <answer>42</answer>."
        ),
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.snapshot)
    vocab = tok.get_vocab()
    report: dict[str, object] = {
        "vocab_size": len(vocab),
        "probes": {},
        "protocol_tags_in_vocab": {
            t: (t in vocab) for t in ["<global>", "</global>", "<focus>", "</focus>", "<local>", "</local>", "<answer>", "</answer>"]
        },
    }
    for probe in PROBES:
        ids = tok(probe, add_special_tokens=False)["input_ids"]
        report["probes"][probe] = {
            "ids": ids,
            "pieces": [tok.convert_ids_to_tokens(i) for i in ids],
            "decode_roundtrip": tok.decode(ids),
        }

    text = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    report["mixed_prompt"] = {
        "kwargs": {"enable_thinking": False},
        "prompt_text": text,
        "token_count": len(ids),
        "token_ids": ids,
        "tokens": [tok.convert_ids_to_tokens(i) for i in ids],
    }
    (out / "e3-protocol-tokens.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "e3-prompt-protocol-mixed.txt").write_text(text, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
