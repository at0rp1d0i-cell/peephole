"""Test-only reference lifecycle, driven by the calibration harness's real forwards.

The candidate adapter is disabled for this request. Its canonical allocation is
checked in each layer before the independent mask reads the reference's own KV.
"""
from __future__ import annotations

import math

import torch

from .reference_bridge import TimelineBridge, unpack_native_kv
from .reference_dense import ReferenceError, TestOnlyReferenceSwitch
from .reference_hook import TestOnlyReferenceAttachment
from .smoke_structure import snapshot_layer


class ReferenceRun:
    def __init__(self, bridge: TimelineBridge, *, total_forwards: int):
        self.bridge = bridge
        self.total_forwards = total_forwards
        self.capture = None
        self.attachment = None
        self.request_id = None
        self.snapshots = {}
        self.completed = []
        self.input_generation = 0
        self.previous_canonical = {}

    def install(self, capture, fa_layers):
        self.capture = capture
        head_sizes = set()
        for name, impl in fa_layers:
            # These fields all exist in the pinned FlashAttentionImpl. Missing
            # fields are unsupported evidence, not assumed defaults.
            expected = {
                "alibi_slopes": None, "sliding_window": (-1, -1),
                "logits_soft_cap": 0, "kv_sharing_target_layer_name": None,
                "sinks": None, "dcp_world_size": 1, "vllm_flash_attn_version": 2,
                "attn_type": "decoder",
            }
            for field, value in expected.items():
                if getattr(impl, field, object()) != value:
                    raise ReferenceError(f"{name}: unsupported impl.{field}")
            if getattr(impl, "kv_cache_dtype", None) not in ("auto", "bfloat16"):
                raise ReferenceError(f"{name}: quantized/unknown KV dtype")
            heads, kv_heads = impl.num_heads, impl.num_kv_heads
            if kv_heads <= 0 or heads % kv_heads or not math.isfinite(impl.scale) or impl.scale <= 0:
                raise ReferenceError(f"{name}: invalid GQA geometry/scale")
            head_sizes.add(impl.head_size)
        if len(head_sizes) != 1:
            raise ReferenceError("reference requires a single FA head size")
        if any(b["block_size"] != self.bridge.kernel_block_size for b in capture.structure_bindings.values()):
            raise ReferenceError("runtime FA block size differs from fixed fixture")
        switch = TestOnlyReferenceSwitch(layers=tuple(capture.layer_names),
                                        steps=tuple(range(1, self.total_forwards)))
        self.attachment = TestOnlyReferenceAttachment(
            switch, bridge=self.bridge, head_size=head_sizes.pop(), before_forward=self.before_layer)
        try:
            for index, (_name, impl) in enumerate(fa_layers):
                self.attachment.wrap_impl(index, impl)
        except BaseException:
            self.restore()
            raise

    def arm(self, request_id):
        if self.request_id is not None:
            raise ReferenceError("reference run cannot reuse a previous request's state")
        self.request_id = str(request_id)
        switch = self.attachment.switch
        switch.request_id = self.request_id
        switch.enabled = True
        self.bridge.current_request_id = self.request_id

    def begin_step(self):
        capture = self.capture
        forward = capture.request_step
        phase = capture.batch_phase[forward]
        if (capture.input_generation <= self.input_generation
                or not phase.get("available") or phase.get("req_ids") != [self.request_id]
                or phase.get("row") != 0):
            raise ReferenceError("missing fresh prepare_inputs or wrong target request")
        self.input_generation = capture.input_generation
        if forward != len(self.completed) + 1 or forward > self.total_forwards:
            raise ReferenceError("unexpected reference forward coverage")
        prefill = forward == 1
        if phase["is_prefilling_row"] != prefill:
            raise ReferenceError("reference phase differs from actual InputBatch")
        expected = list(range(self.bridge.prompt_len)) if prefill else [self.bridge.prompt_len + forward - 2]
        actual, _ = capture._logical_positions_from_runner(len(expected))
        if actual != expected:
            raise ReferenceError("reference logical positions differ from fixed trajectory")
        self.attachment.switch.current_request_id = phase["req_ids"][0]
        self.attachment.switch.current_step = forward - 1
        if not prefill:
            self.bridge.is_prefilling_by_step[forward - 2] = False
        self.snapshots[forward] = {}

    def before_layer(self, *, index, query, key, value, kv_cache, metadata):
        from vllm.forward_context import get_forward_context

        capture = self.capture
        forward = capture.request_step
        name = capture.layer_names[index]
        if not capture._step_open or name in self.snapshots[forward]:
            raise ReferenceError("reference layer outside forward or duplicate layer")
        rec = snapshot_layer(
            runner=capture.runner, batch=capture.current_input_batch, context=get_forward_context(),
            layer_name=name, binding=capture.structure_bindings[name], metadata=metadata,
            req_id=self.request_id, q_len=int(query.shape[0]))
        b = self.bridge.kernel_block_size
        kv_len = self.bridge.prompt_len + forward - 1
        expected_positions = list(range(self.bridge.prompt_len)) if forward == 1 else [kv_len - 1]
        row = rec["canonical"]
        if (rec["positions"] != expected_positions or rec["seq_len"] != kv_len
                or rec["physical"] != row or len(row) != (kv_len + b - 1) // b
                or len(set(row)) != len(row) or any(p < 0 or p >= kv_cache.shape[0] for p in row)
                or rec["slots"] != [row[p // b] * b + p % b for p in expected_positions]):
            raise ReferenceError("reference requires canonical reads, positions and write slots")
        previous = self.previous_canonical.get(name, [])
        if row[:len(previous)] != previous:
            raise ReferenceError("reference canonical prefix changed during request")
        self.previous_canonical[name] = list(row)
        impl = self.attachment.wrapped[index][0]
        if (query.shape[1:] != (impl.num_heads, impl.head_size)
                or kv_cache.shape[1:] != (impl.num_kv_heads, b, 2 * impl.head_size)
                or kv_cache.dtype != torch.bfloat16):
            raise ReferenceError("reference tensors differ from actual impl geometry")
        if forward > 1:
            # KV update must precede this hook, including the current token.
            kc, vc = unpack_native_kv(kv_cache, impl.head_size)
            phys, offset = row[(kv_len - 1) // b], (kv_len - 1) % b
            if not torch.equal(kc[phys, offset], key[0]) or not torch.equal(vc[phys, offset], value[0]):
                raise ReferenceError("current K/V not written to canonical slot before reference")
        self.snapshots[forward][name] = rec

    def end_step(self):
        forward = self.capture.request_step
        self._check_step(forward)
        self.completed.append(forward)
        self.attachment.switch.current_request_id = None

    def _check_step(self, forward):
        if set(self.snapshots.get(forward, {})) != set(self.capture.layer_names.values()):
            raise ReferenceError(f"reference missing FA layer at forward {forward}")
        rows = [r for r in self.attachment.ledger if r["step"] == forward - 1]
        expected_action = "passthrough" if forward == 1 else "overrode"
        actual = [(r["layer"], r["request_id"], r["action"]) for r in rows]
        expected = [(i, self.request_id, expected_action) for i in self.capture.layer_names]
        if sorted(actual) != sorted(expected):
            raise ReferenceError(f"reference ledger mismatch at forward {forward}: {actual}")

    def validate(self):
        if self.completed != list(range(1, self.total_forwards + 1)):
            raise ReferenceError("reference missing forward coverage")
        if len(self.attachment.ledger) != self.total_forwards * len(self.capture.layer_names):
            raise ReferenceError("reference unexpected ledger records")
        for forward in self.completed:
            self._check_step(forward)

    def disarm(self):
        if self.attachment is not None:
            self.attachment.switch.enabled = False
            self.attachment.switch.current_request_id = None

    def restore(self):
        self.disarm()
        if self.attachment is not None:
            self.attachment.restore()

    def report(self):
        return {
            "request_id": self.request_id, "completed_forwards": list(self.completed),
            "prefill": "original global, independently initialized request",
            "decode": "own Q/canonical KV, independent mask, FP32 dense cast to output dtype",
            "ledger": self.attachment.ledger if self.attachment else [],
            "snapshots": self.snapshots,
            "restored_layers": self.attachment.restored if self.attachment else [],
            "enabled": self.attachment.switch.enabled if self.attachment else False,
        }
