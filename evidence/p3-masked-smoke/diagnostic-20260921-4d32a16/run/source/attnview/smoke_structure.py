"""Masked diagnostic: independent expectations and immutable in-forward evidence.

No candidate selection helpers are used. Runtime tables supply physical allocation,
while the declared script and original renderer spans supply the expected reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .reference_dense import independent_visible_positions


class StructureError(RuntimeError):
    """Missing or inconsistent diagnostic evidence."""


@dataclass(frozen=True)
class StepExpectation:
    forward: int
    decode_index: int | None
    mode: str
    refs: tuple[int, ...]
    kv_len: int
    blocks: tuple[int, ...]
    effective_per_block: tuple[int, ...]

    @property
    def total_read(self):
        return sum(self.effective_per_block)


def expectations_from_config(*, timeline_config, doc_fixture, tokenizer, prompt_len,
                             sink_span, local_window_span, segment_spans, block_size,
                             total_steps):
    cfg = json.loads(Path(timeline_config).read_text())
    declarations, token_count = [], 0
    for piece in cfg["generation_script"]:
        token_count += len(tokenizer.encode(piece["text"], add_special_tokens=False))
        declarations.append((token_count - 1, piece["expected_mode_after"], tuple(piece["refs"])))
    result = {}
    for forward in range(1, total_steps + 1):
        mode, refs = "global", ()
        # Prefill consumes no generated token. Decode d consumes generated token d-1.
        if forward > 1:
            for parse_index, declared_mode, declared_refs in declarations:
                if parse_index <= forward - 2:
                    mode, refs = declared_mode, declared_refs
        kv_len = prompt_len + forward - 1
        mask = independent_visible_positions(
            mode=mode, refs=refs, kv_len=kv_len, prompt_len=prompt_len,
            sink_span=sink_span, local_window_span=local_window_span,
            segment_spans=segment_spans, block_size=block_size)
        result[forward] = StepExpectation(
            forward, forward - 1 if forward > 1 else None, mode, refs, kv_len,
            tuple(mask.blocks), tuple(mask.effective_per_block))
    return result


def fa_bindings(runner):
    """Resolve cache group indices from actual FullAttentionSpec, never group zero."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    bindings = {}
    for group_index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, FullAttentionSpec):
            continue
        block_size = int(group.kv_cache_spec.block_size)
        if (runner.block_tables.block_sizes[group_index] != block_size
                or runner.block_tables.kernel_block_sizes[group_index] != block_size):
            raise StructureError("diagnostic requires manager/kernel block sizes to match")
        for name in group.layer_names:
            if name in bindings:
                raise StructureError(f"duplicate FA layer {name}")
            bindings[name] = {"group_index": group_index, "block_size": block_size}
    actual = [name for groups in runner.attn_groups for group in groups
              if isinstance(group.kv_cache_spec, FullAttentionSpec) for name in group.layer_names]
    if not bindings or sorted(bindings) != sorted(actual):
        raise StructureError("FA cache groups and attention groups disagree")
    return bindings


def snapshot_layer(*, runner, batch, context, layer_name, binding, metadata, req_id, q_len):
    """Called inside the actual FA forward; all tensor values become owned Python lists."""
    req_ids = list(batch.req_ids)
    if req_ids != [req_id]:
        raise StructureError(f"unexpected current batch {req_ids}, target={req_id}")
    row = req_ids.index(req_id)
    if int(batch.num_tokens) != q_len:
        raise StructureError("batch token count differs from actual FA query")
    positions = runner.input_buffers.positions[:q_len].detach().cpu().tolist()
    b = binding["block_size"]
    nblocks = (max(positions) + b) // b
    tables = runner.block_tables.input_block_tables
    table = tables[binding["group_index"]]
    if table.ndim != 2 or table.shape[1] < nblocks:
        raise StructureError("canonical table must be [requests, blocks] with sufficient columns")
    canonical = table[row, :nblocks].detach().cpu().tolist()
    seq_len = int(metadata.seq_lens[row].item())
    read_blocks = (seq_len + b - 1) // b
    return {
        "req_id": req_id, "req_ids": req_ids, "row": row, "layer_name": layer_name,
        **binding, "positions": positions, "q_len": q_len,
        "canonical": canonical, "seq_len": seq_len,
        "physical": metadata.block_table[row, :read_blocks].detach().cpu().tolist(),
        "slots": context.slot_mapping[layer_name][:q_len].detach().cpu().tolist(),
        "is_prefilling": bool(batch.is_prefilling_np[row]),
    }


def check_capture_structure(*, capture, expectations, trace, prompt_len, block_size,
                            representative_decodes, req_id):
    """Fail closed on exact coverage, each layer's values, and both trace categories."""
    checks, evidence = [], {"steps": {}}

    def equal(name, actual, expected):
        if actual != expected:
            raise StructureError(f"{name}: actual={str(actual)[:400]} expected={str(expected)[:400]}")
        checks.append({"name": name, "ok": True})

    try:
        bindings = capture.structure_bindings
        equal("FA layer names", sorted(capture.layer_names.values()), sorted(bindings))
        for field in (capture.structure, capture.positions, capture.records):
            equal("forward coverage", sorted(field), sorted(expectations))
        if not bindings or not expectations:
            raise StructureError("empty layer/step expectations")
        previous = {}
        for forward, exp in sorted(expectations.items()):
            equal("expectation forward", exp.forward, forward)
            equal(f"f{forward} layers", sorted(capture.structure[forward]), sorted(bindings))
            equal(f"f{forward} captured layers", sorted(capture.records[forward]), sorted(capture.layer_names))
            positions = list(range(prompt_len)) if forward == 1 else [prompt_len + forward - 2]
            equal(f"f{forward} logical positions", capture.positions[forward]["positions"], positions)
            equal(f"f{forward} phase", capture.positions[forward]["phase"], "prefill" if forward == 1 else "decode")
            for name, binding in bindings.items():
                rec = capture.structure[forward][name]
                prefix = f"f{forward}/{name}"
                for key, expected in {"req_id": req_id, "req_ids": [req_id], "row": 0,
                                      "layer_name": name, "positions": positions,
                                      "q_len": len(positions), "is_prefilling": forward == 1,
                                      "block_size": block_size, "group_index": binding["group_index"]}.items():
                    equal(f"{prefix} {key}", rec[key], expected)
                canonical = rec["canonical"]
                equal(f"{prefix} canonical length", len(canonical), (exp.kv_len + block_size - 1) // block_size)
                if name in previous:
                    equal(f"{prefix} canonical prefix stable", canonical[:len(previous[name])], previous[name])
                previous[name] = canonical
                equal(f"{prefix} read length", rec["seq_len"], exp.total_read)
                equal(f"{prefix} physical blocks", rec["physical"], [canonical[b] for b in exp.blocks])
                equal(f"{prefix} canonical write slots", rec["slots"],
                      [canonical[p // block_size] * block_size + p % block_size for p in positions])
            for index, rec in capture.records[forward].items():
                expected_keys = {"k", "v"}
                if forward - 1 in representative_decodes:
                    expected_keys |= {"q", "out"}
                equal(f"f{forward}/L{index} bounded tensors", set(rec) & {"k", "v", "q", "out"}, expected_keys)
            evidence["steps"][str(forward)] = {
                "decode": exp.decode_index, "mode": exp.mode, "refs": exp.refs,
                "blocks": exp.blocks, "effective_per_block": exp.effective_per_block,
                "read_length": exp.total_read,
            }
        equal("representative coverage", sorted(set(representative_decodes) & {f - 1 for f in expectations}),
              sorted(representative_decodes))
        equal("enforce_global marks", trace["enforce_global_steps"], [])
        overrides = trace["override_steps"]
        restricted = {f - 1: e for f, e in expectations.items() if e.mode != "global"}
        equal("restricted trace coverage", [r["step"] for r in overrides], sorted(restricted))
        for record in overrides:
            exp = restricted[record["step"]]
            equal("override request", record["req_ids"], [req_id])
            equal("override kind", record["kind"], "override_step")
            equal("override scheduled tokens", record["num_scheduled_tokens"], {req_id: 1})
            view = record["override"]
            groups = {b["group_index"] for b in bindings.values()}
            equal("single FA group", len(groups), 1)
            equal("override group", view["group_index"], next(iter(groups)))
            equal("override rows", view["rows_with_override"], [0])
            equal("override length", view["seqused_k"], [exp.total_read])
            equal("override max length", view["max_seq_len"], exp.total_read)
            equal("override blocks", view["visible_logical_blocks"], {"0": list(exp.blocks)})
        parsed = trace["parsed_steps"]
        equal("parser step coverage", [r["parse_step"] for r in parsed], list(range(len(expectations))))
        for forward, exp in expectations.items():
            if forward == 1:
                continue
            record = parsed[forward - 2]
            equal("parser request", record["req_id"], req_id)
            equal("parser effect step", record["effect_step"], forward - 1)
            equal("parser mode", record["mode"], exp.mode)
            equal("parser refs", record["refs"], list(exp.refs))
        if not any(e.mode != "global" for e in expectations.values()):
            raise StructureError("diagnostic contains no restricted step")
        equal("final global recovery", expectations[max(expectations)].mode, "global")
    except (KeyError, AttributeError, TypeError, IndexError) as exc:
        raise StructureError(f"missing/malformed evidence: {exc}") from exc
    evidence["trace_mapping"] = "override.step = decode; forward = decode + 1"
    return checks, evidence
