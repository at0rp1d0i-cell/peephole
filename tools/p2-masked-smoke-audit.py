#!/usr/bin/env python3
"""CPU-only post-run observations at four captured decode points; no pass threshold."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from _lib import sha256_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from attnview.reference_dense import dense_attention_fp32  # noqa: E402
from attnview.smoke_structure import expectations_from_config  # noqa: E402


def error_metrics(actual, reference):
    diff = actual.float() - reference.float()
    return {"max_abs": float(diff.abs().max()), "rms": float(diff.square().mean().sqrt()),
            "relative_l2": float(diff.norm() / reference.float().norm().clamp_min(1e-30)),
            "nonfinite": int((~torch.isfinite(actual)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("refusing to overwrite an audit")
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("masked_structure", {}).get("ok"):
        raise RuntimeError("structure gate must pass before interpreting captured numeric errors")
    for path, digest in manifest["sources"]["input_hashes"].items():
        if sha256_file(REPO / path) != digest:
            raise RuntimeError(f"input changed: {path}")
    fixture = manifest["fixture_evidence"]
    timeline = REPO / fixture["timeline_config"]
    config = json.loads(timeline.read_text())
    from transformers import AutoTokenizer
    payload = manifest["payload"]
    expected = expectations_from_config(
        timeline_config=timeline, doc_fixture=fixture["doc_fixture"],
        tokenizer=AutoTokenizer.from_pretrained(manifest["model"]["snapshot"], trust_remote_code=False),
        block_size=config["block_size"], total_steps=29,
        **{key: payload[key] for key in ("prompt_len", "sink_span", "local_window_span", "segment_spans")})
    capture_path = args.run / "capture/layers.npz"
    digest = sha256_file(capture_path)
    if digest != manifest["capture"]["npz_sha256"]:
        raise RuntimeError("capture hash mismatch")
    rows, tensor_bytes = [], 0
    with np.load(capture_path) as arrays:
        tensor_bytes = sum(arrays[name].nbytes for name in arrays.files)
        layers = arrays["layer_index"].tolist()
        for layer in layers:
            k_parts = [torch.from_numpy(arrays[f"k_prefill_L{layer}"]).transpose(0, 1)]
            v_parts = [torch.from_numpy(arrays[f"v_prefill_L{layer}"]).transpose(0, 1)]
            for d in range(1, 29):
                k_parts.append(torch.from_numpy(arrays[f"k_current_step{d}_L{layer}"]).transpose(0, 1))
                v_parts.append(torch.from_numpy(arrays[f"v_current_step{d}_L{layer}"]).transpose(0, 1))
                if d not in (6, 7, 20, 25):
                    continue
                exp = expected[d + 1]
                positions = [p for block, count in zip(exp.blocks, exp.effective_per_block, strict=True)
                             for p in range(block * config["block_size"], block * config["block_size"] + count)]
                k, v = torch.cat(k_parts), torch.cat(v_parts)
                assert len(k) == exp.kv_len == len(v)
                q = torch.from_numpy(arrays[f"decode_q_step{d}_L{layer}"])[:, 0]
                out = torch.from_numpy(arrays[f"decode_out_step{d}_L{layer}"])[:, 0]
                ref, cast = dense_attention_fp32(q, k[positions], v[positions],
                                                scale=float(arrays[f"scale_L{layer}"]), dtype=torch.bfloat16)
                rows.append({"layer": layer, "decode": d, "mode": exp.mode, "read_length": len(positions),
                             "candidate_vs_fp32": error_metrics(out, ref),
                             "candidate_vs_bf16_reference": error_metrics(out, cast)})
    logits = torch.load(args.run / "logits.pt", weights_only=False)
    result = {
        "audit_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "audit_script_sha256": sha256_file(Path(__file__)), "run_head": manifest["head"],
        "manifest_sha256": sha256_file(manifest_path), "capture_sha256": digest,
        "capture_file_bytes": capture_path.stat().st_size, "capture_array_bytes": tensor_bytes,
        "boundary": "Local attention arithmetic on candidate-trajectory Q/K/V; not independent model logits, cache-write-content verification, quality or performance acceptance. No numeric pass threshold.",
        "observations": rows,
        "logits": {"records": len(logits), "nonfinite": sum(int((~torch.isfinite(r["logits"])).sum()) for r in logits),
                   "max_abs": max(float(r["logits"].abs().max()) for r in logits)},
        "summary": {key: {metric: max(row[key][metric] for row in rows)
                           for metric in ("max_abs", "rms", "relative_l2", "nonfinite")}
                    for key in ("candidate_vs_fp32", "candidate_vs_bf16_reference")},
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
