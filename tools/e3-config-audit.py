#!/usr/bin/env python
"""E3: 核对本地模型快照的 config / tokenizer / chat template，并渲染样例 prompt。

用法：
  source /root/attnview/env.sh
  "$ATTNVIEW_PYTHON" /root/attnview/tools/e3-config-audit.py --snapshot DIR --out DIR

只读：不下载、不改文件。输出 JSON + 渲染后的 prompt 文本（供后续 token-span 使用）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a precise assistant. Answer with a single short sentence."},
    {"role": "user", "content": "Name the capital of France."},
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    snap = Path(args.snapshot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((snap / "config.json").read_text())
    tcfg = json.loads((snap / "tokenizer_config.json").read_text())
    gcfg = json.loads((snap / "generation_config.json").read_text())
    tcfg_text = cfg["text_config"]
    layer_types = tcfg_text["layer_types"]

    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    lin_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    other = sorted(set(layer_types) - {"full_attention", "linear_attention"})

    kv_heads = tcfg_text["num_key_value_heads"]
    head_dim = tcfg_text["head_dim"]
    n_full = len(full_layers)
    kv_bytes_per_token = 2 * n_full * kv_heads * head_dim * 2  # K+V, fp16/bf16 元素

    # index.json：checkpoint 张量总量（方案 §3.2 引用的 55,562,855,904 B）
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensor_bytes = index["metadata"]["total_size"]

    report = {
        "snapshot": str(snap),
        "config": {
            "architectures": cfg["architectures"],
            "model_type": cfg["model_type"],
            "language_model_only": cfg.get("language_model_only"),
            "has_vision_config": "vision_config" in cfg,
            "image_token_id": cfg.get("image_token_id"),
            "video_token_id": cfg.get("video_token_id"),
            "text_dtype": tcfg_text.get("dtype"),
            "quantization_config": cfg.get("quantization_config"),
            "num_hidden_layers": tcfg_text["num_hidden_layers"],
            "layer_type_counts": {
                "full_attention": n_full,
                "linear_attention": len(lin_layers),
                "other": other,
            },
            "full_attention_layer_indices": full_layers,
            "full_attention_interval": tcfg_text.get("full_attention_interval"),
            "num_attention_heads": tcfg_text["num_attention_heads"],
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "gqa_ratio": tcfg_text["num_attention_heads"] / kv_heads,
            "attn_output_gate": tcfg_text.get("attn_output_gate"),
            "output_gate_type": tcfg_text.get("output_gate_type"),
            "max_position_embeddings": tcfg_text["max_position_embeddings"],
            "rope_theta": tcfg_text["rope_parameters"].get("rope_theta"),
            "rope_type": tcfg_text["rope_parameters"].get("rope_type"),
            "partial_rotary_factor": tcfg_text.get("partial_rotary_factor"),
            "mrope_interleaved": tcfg_text["rope_parameters"].get("mrope_interleaved"),
            "mrope_section": tcfg_text["rope_parameters"].get("mrope_section"),
            "vocab_size": tcfg_text["vocab_size"],
            "tie_word_embeddings": tcfg_text["tie_word_embeddings"],
            "hidden_size": tcfg_text["hidden_size"],
            "linear_attention": {
                "linear_num_key_heads": tcfg_text.get("linear_num_key_heads"),
                "linear_num_value_heads": tcfg_text.get("linear_num_value_heads"),
                "linear_key_head_dim": tcfg_text.get("linear_key_head_dim"),
                "linear_value_head_dim": tcfg_text.get("linear_value_head_dim"),
                "linear_conv_kernel_dim": tcfg_text.get("linear_conv_kernel_dim"),
                "mamba_ssm_dtype": tcfg_text.get("mamba_ssm_dtype"),
            },
            "mtp_num_hidden_layers": tcfg_text.get("mtp_num_hidden_layers"),
            "eos_token_id": tcfg_text["eos_token_id"],
            "bos_token_id": tcfg_text["bos_token_id"],
            "pad_token_id": tcfg_text["pad_token_id"],
        },
        "kv_arithmetic": {
            "formula": "2 (K+V) * n_full_layers * num_key_value_heads * head_dim * 2 bytes",
            "n_full_layers": n_full,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "bytes_per_token": kv_bytes_per_token,
            "kib_per_token": kv_bytes_per_token / 1024,
            "plan_baseline_bytes_per_token": 65536,
            "matches_plan": kv_bytes_per_token == 65536,
            "logical_kv_by_seqlen": {
                f"{n // 1024}K": {
                    "bytes": kv_bytes_per_token * n,
                    "gib": round(kv_bytes_per_token * n / 1024**3, 3),
                }
                for n in (32768, 65536, 131072, 262144, 1000000)
            },
        },
        "generation_config": gcfg,
        "checkpoint_index": {
            "metadata_total_size": tensor_bytes,
            "plan_quoted_tensor_bytes": 55_562_855_904,
            "matches_plan": tensor_bytes == 55_562_855_904,
            "num_shards": len(sorted({v for v in weight_map.values()})),
        },
        "tokenizer_config": {
            "tokenizer_class": tcfg["tokenizer_class"],
            "model_max_length": tcfg["model_max_length"],
            "bos_token": tcfg["bos_token"],
            "eos_token": tcfg["eos_token"],
            "pad_token": tcfg["pad_token"],
            "unk_token": tcfg["unk_token"],
            "add_bos_token": tcfg.get("add_bos_token"),
            "additional_special_tokens": tcfg["additional_special_tokens"],
            "extra_special_tokens": tcfg.get("extra_special_tokens"),
            "added_tokens": {
                str(k): {"content": v["content"], "special": v["special"]}
                for k, v in sorted(tcfg["added_tokens_decoder"].items(), key=lambda x: int(x[0]))
            },
            "chat_template_sha256": sha256(snap / "chat_template.jinja"),
            "chat_template_in_tokenizer_config_identical": tcfg["chat_template"]
            == (snap / "chat_template.jinja").read_text(),
            "chat_template_flags": {
                "mentions_enable_thinking": "enable_thinking" in tcfg["chat_template"],
                "mentions_preserve_thinking": "preserve_thinking" in tcfg["chat_template"],
                "mentions_reasoning_effort": "reasoning_effort" in tcfg["chat_template"],
                "default_reasoning_effort": re.search(
                    r"reasoning_effort\|default\('([^']+)'\)", tcfg["chat_template"]
                ).group(1)
                if re.search(r"reasoning_effort\|default\('([^']+)'\)", tcfg["chat_template"])
                else None,
            },
            "file_sha256": {
                f.name: sha256(f)
                for f in sorted(snap.iterdir())
                if f.is_file() and not f.name.startswith("model-")
            },
        },
        "renders": {},
    }

    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)
    for label, kwargs in [
        ("thinking_disabled", {"enable_thinking": False}),
        ("thinking_default", {}),
    ]:
        text = tok.apply_chat_template(
            SAMPLE_MESSAGES, tokenize=False, add_generation_prompt=True, **kwargs
        )
        # 与 vLLM 的实际路径一致：先渲染文本，再对文本做一次 tokenize。
        enc = tok(text, add_special_tokens=False)
        ids = list(enc["input_ids"])
        report["renders"][label] = {
            "kwargs": kwargs,
            "prompt_text": text,
            "prompt_text_repr": repr(text),
            "token_count": len(ids),
            "token_ids": ids,
            "first_40_piece_tokens": [
                tok.convert_ids_to_tokens(i) for i in ids[:40]
            ],
        }

    (out / "e3-config-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for label in report["renders"]:
        (out / f"e3-prompt-{label}.txt").write_text(
            report["renders"][label]["prompt_text"], encoding="utf-8"
        )

    print(json.dumps({k: v for k, v in report.items() if k != "tokenizer_config"}, ensure_ascii=False, indent=2)[:6000])
    print("\n--- renders ---")
    for label, r in report["renders"].items():
        print(f"[{label}] kwargs={r['kwargs']} tokens={r['token_count']}")
        print(repr(r["prompt_text"]))
    print("\nwritten:", out / "e3-config-audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
