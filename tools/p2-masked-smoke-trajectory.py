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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SNAPSHOT = REPO / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


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

    from transformers import AutoTokenizer

    from attnview.prompt import _tokenize_with_offsets, render_arm
    from attnview.segmenter import build_offsets_index, segment_context

    doc = json.loads(args.doc_fixture.read_text())
    fx = json.loads(args.expect.read_text())
    tcfg = json.loads(args.timeline_config.read_text())
    ctx = doc["document"] + ("\n\n" + tcfg["filler_unit"]) * int(fx["filler_units"]) + tcfg["fine_char"] * int(fx["fine_units"])
    ctx_hash = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
    if ctx_hash != fx["context_sha256"]:
        raise SystemExit(f"重建 context 哈希不一致,拒绝:got={ctx_hash[:16]} want={fx['context_sha256'][:16]}")
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    _ids, offsets = _tokenize_with_offsets(tok, ctx)
    segs = segment_context(ctx, build_offsets_index(offsets))
    prompt = render_arm("da", segs, doc["question"], ctx, tok, enable_thinking=False)
    ids = [int(t) for t in prompt.token_ids]
    got = {"prompt_len": len(ids), "segment_spans": [list(x) for x in prompt.segment_spans],
           "local_window_span": list(prompt.scaffold.local_window_span), "sink_span": list(prompt.scaffold.sink_span),
           "token_ids_sha256": hashlib.sha256(bytes(str(list(ids)), "utf-8")).hexdigest()}
    want = {k: fx[k] for k in got}
    if got != want:
        raise SystemExit(f"重渲染不一致,拒绝:got={got} want={want}")

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
