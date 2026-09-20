#!/usr/bin/env python3
"""由已验收 7834 夹具生成 **29 采样 / 28 消费** 的强制轨迹文件(纯 CPU, 不加载模型)。

- prompt 用与 runner 相同的真实 renderer 路径重渲染,并逐项校验(与夹具一致否则拒绝);
- 生成段 token 取夹具 `steps[].token_id`(28 个被后继 forward 消费的步)+ 1 个终止采样 token;
- 轨迹文件字段与 `p2-calib-run.py:load_trajectory` 一致:
  `tokens[step][row] = [token_ids...]`,`prompt_len`,`prompt_token_ids_sha256`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from _lib import MODEL_REVISION

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SNAPSHOT = REPO / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots" / MODEL_REVISION


def sha256_token_ids(ids) -> str:
    """与 runner 的 `sha256_token_ids` 完全一致:`sha256_text(json.dumps([int(t) for t in ids]))`。"""
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-fixture", type=Path, required=True,
                    help="含 document/question 的夹具(evidence/p1-cpu/demo-fixtures.json)")
    ap.add_argument("--expect", type=Path, required=True,
                    help="已验收产物(提供 filler_units/fine_units/context_sha256/prompt_len/spans/steps)")
    ap.add_argument("--timeline-config", type=Path, required=True,
                    help="提供 filler_unit/fine_char 的配置(configs/p2-masked-prep/crossblock.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="patched-masked")
    args = ap.parse_args()

    from attnview.fixture_rebuild import rebuild_prompt

    fx = json.loads(args.expect.read_text())

    prompt, _payload, evidence = rebuild_prompt(doc_fixture=args.doc_fixture, expect_fixture=args.expect,
                                                timeline_config=args.timeline_config, snapshot=SNAPSHOT)
    ids = [int(t) for t in prompt.token_ids]
    ctx_hash = evidence["context_sha256"]
    consumed = [int(s["token_id"]) for s in fx["steps"]]              # 28 个被后继 forward 消费
    terminal = int(fx["steps"][-1]["token_id"])                       # 终止采样 token(不被消费)
    tokens = [[[t]] for t in consumed + [terminal]]
    out = {"arm": args.arm, "source": "tools/p2-masked-smoke-trajectory.py",
           "doc_fixture": str(args.doc_fixture), "expect_fixture": str(args.expect),
           "context_sha256": ctx_hash, "prompt_len": len(ids),
           "prompt_token_ids_sha256": sha256_token_ids(ids),
           "sampled_steps": len(tokens), "consumed_steps": len(consumed),
           "note": "tokens[step][0]=[token_id];前 28 步被后继 forward 消费,第 29 个仅采样(终止)",
           "tokens": tokens}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({"prompt_len": len(ids), "sampled": len(tokens), "consumed": len(consumed),
                      "prompt_token_ids_sha256": out["prompt_token_ids_sha256"][:16]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
