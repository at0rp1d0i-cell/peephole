"""按已验收夹具重建 prompt(共享逻辑):runner 与轨迹生成器都用这一份。

输入三类事实:
- **doc fixture**(`evidence/p1-cpu/demo-fixtures.json`):`document` / `question`;
- **timeline config**(`configs/p2-masked-prep/crossblock.json`):`filler_unit` / `fine_char`;
- **expect fixture**(`evidence/p3-calib/masked-prep/crossblock-note.json`):
  `filler_units` / `fine_units` / `context_sha256` / `prompt_len` / `segment_spans` /
  `local_window_span` / `sink_span` / `token_ids_sha256`。

重建后**逐项校验**,任一不一致即抛错拒绝运行(避免"命令像 7834、实际渲染别的 prompt")。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["FixtureMismatch", "rebuild_prompt"]


class FixtureMismatch(RuntimeError):
    """重建结果与已验收值不一致。"""


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rebuild_prompt(*, doc_fixture: Path, expect_fixture: Path, timeline_config: Path, snapshot: Path,
                   tokenizer_hash: str = "", template_hash: str = "") -> tuple[Any, dict, dict]:
    from transformers import AutoTokenizer

    from .prompt import _tokenize_with_offsets, render_arm
    from .segmenter import build_offsets_index, segment_context

    doc = json.loads(Path(doc_fixture).read_text())
    exp = json.loads(Path(expect_fixture).read_text())
    cfg = json.loads(Path(timeline_config).read_text())
    context = (doc["document"] + ("\n\n" + cfg["filler_unit"]) * int(exp["filler_units"])
               + cfg["fine_char"] * int(exp["fine_units"]))
    ctx_hash = _sha_text(context)
    if ctx_hash != exp["context_sha256"]:
        raise FixtureMismatch(f"重建 context 哈希不一致:got={ctx_hash[:16]} want={exp['context_sha256'][:16]}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=False)
    _ids, offsets = _tokenize_with_offsets(tokenizer, context)
    segments = segment_context(context, build_offsets_index(offsets))
    prompt = render_arm("da", segments, doc["question"], context, tokenizer,
                        tokenizer_hash=tokenizer_hash, template_hash=template_hash, enable_thinking=False)
    ids = [int(t) for t in prompt.token_ids]
    got = {
        "prompt_len": len(ids),
        "segment_spans": [list(x) for x in prompt.segment_spans],
        "local_window_span": list(prompt.scaffold.local_window_span),
        "sink_span": list(prompt.scaffold.sink_span),
        "token_ids_sha256": _sha_text(str(list(ids))),
    }
    want = {k: exp[k] for k in got}
    if got != want:
        raise FixtureMismatch(f"重渲染与已验收值不一致,拒绝运行:got={got} want={want}")
    payload = {
        "protocol": "v1.0", "prompt_len": len(ids),
        "segment_spans": [list(x) for x in prompt.segment_spans],
        "local_window_span": list(prompt.scaffold.local_window_span),
        "sink_span": list(prompt.scaffold.sink_span),
    }
    evidence = {"doc_fixture": str(doc_fixture), "expect_fixture": str(expect_fixture),
                "timeline_config": str(timeline_config), "context_sha256": ctx_hash, "checked": got}
    return prompt, payload, evidence
