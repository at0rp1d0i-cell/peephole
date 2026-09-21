"""CPU evidence for actual driver control flow, not model/GDN numerical acceptance."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from _support import REPO
from attnview.reference_bridge import (
    TimelineBridge,
    perform_reference_attention_native,
    resolve_geometry,
    unpack_native_kv,
)
from attnview.reference_dense import ReferenceError, gather_positions
from attnview.reference_dense import TestOnlyReferenceSwitch as Switch
from attnview.reference_hook import TestOnlyReferenceAttachment as Attachment
from attnview.reference_run import ReferenceRun
from tests.test_p2_calib_hooks import FakeRunner, load_module_by_path


class FullAttentionSpec:
    block_size = 4


class CpuFlashImpl:
    """Pinned feature surface; prefill output is deliberately distinguishable."""
    num_heads, num_kv_heads, head_size, scale = 4, 2, 8, 0.25
    alibi_slopes = sinks = kv_sharing_target_layer_name = None
    sliding_window = (-1, -1)
    logits_soft_cap = 0
    dcp_world_size = 1
    vllm_flash_attn_version = 2
    supports_quant_query_input = True  # Pin reports capability even for BF16 inputs.
    attn_type, kv_cache_dtype = "decoder", "auto"

    def __init__(self):
        self.calls = 0

    def forward(self, layer, query, key, value, kv_cache, metadata, output, **kwargs):
        self.calls += 1
        output.fill_(0.5)
        return output


class CpuModel:
    """Write native KV before impl.forward; propagate original output, ignoring return."""
    def __init__(self, runner, context, *, corrupt=None):
        self.runner, self.context, self.corrupt = runner, context, corrupt
        self.impls = [CpuFlashImpl(), CpuFlashImpl()]
        self.names = [f"model.layers.{i}.attn" for i in (3, 7)]
        self.caches = [torch.full((24, 2, 4, 16), float("nan"), dtype=torch.bfloat16) for _ in self.impls]
        self.calls = 0
        self.outputs = []
        self.state = torch.zeros(4, 8, dtype=torch.bfloat16)

    def named_modules(self):
        return [(name, NS(impl=impl)) for name, impl in zip(self.names, self.impls, strict=True)]

    def forward(self, *, positions):
        self.calls += 1
        qlen = len(positions)
        kvlen = int(positions[-1]) + 1
        row = self.runner.block_tables.input_block_tables[1]
        metadata = NS(seq_lens=torch.tensor([kvlen], dtype=torch.int32), block_table=row.clone(),
                      query_start_loc=torch.tensor([0, qlen], dtype=torch.int32), causal=True,
                      num_actual_tokens=qlen, max_query_len=qlen, max_seq_len=kvlen,
                      num_decode_reqs=0, num_decode_tokens=0, num_prefill_reqs=0, num_prefill_tokens=0,
                      use_cascade=False, mm_prefix_query_range_tensor=None)
        physical = row[0, positions // 4]
        offsets = positions % 4
        self.context.slot_mapping = dict.fromkeys(self.names, physical * 4 + offsets)
        for index, impl in enumerate(self.impls):
            q = torch.ones(qlen, 4, 8, dtype=torch.bfloat16) * (index + 1) / 8 + self.state
            k = torch.arange(qlen * 2 * 8, dtype=torch.float32).reshape(qlen, 2, 8).to(torch.bfloat16) / 128
            v = (k + (positions[:, None, None] % 7) / 16 + self.state.mean()).to(torch.bfloat16)
            kc, vc = unpack_native_kv(self.caches[index], 8)
            kc[physical, offsets], vc[physical, offsets] = k, v
            if self.corrupt is not None:
                self.corrupt(self, index, metadata, kc, vc, physical, offsets)
            if getattr(self, "skip_layer", None) == index:
                continue
            output = torch.empty_like(q)
            impl.forward(None, q, k, v, self.caches[index], metadata, output,
                         output_scale=None, output_block_scale=None)
            self.state = output[-1].clone()
        self.outputs.append(self.state.clone())
        return self.state


class ReferenceLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_module_by_path("reference_driver", REPO / "tools/p2-calib-run.py")
        # Import before patch.dict(sys.modules): lazy torch registrations cannot
        # be undone by removing their modules at the end of a test.
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(str(cls.driver.SNAPSHOT), trust_remote_code=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.context = NS(slot_mapping={})
        self.stack.enter_context(patch.dict(sys.modules, {
            "vllm.v1.kv_cache_interface": NS(FullAttentionSpec=FullAttentionSpec),
            "vllm.forward_context": NS(get_forward_context=lambda: self.context),
        }))

    def make(self, *, corrupt=None, total=4):
        runner = FakeRunner()
        model = CpuModel(runner, self.context, corrupt=corrupt)
        fa = NS(kv_cache_spec=FullAttentionSpec(), layer_names=model.names)
        gdn = NS(kv_cache_spec=object(), layer_names=["gdn"])
        runner.kv_cache_config = NS(kv_cache_groups=[gdn, fa])
        runner.attn_groups = [[gdn], [fa]]
        # Negative padding is deliberately not a valid physical allocation.
        row = torch.tensor([[5, 1, 7, 2, 9, 3, 11, 4, 13, 6, 15, 8, 17, 10, 19, 12, -1]])
        runner.block_tables = NS(input_block_tables=[torch.full_like(row, 999), row],
                                 block_sizes=[4, 4], kernel_block_sizes=[4, 4])
        bridge = TimelineBridge(prompt_len=6, kernel_block_size=4, sink_span=(0, 1),
                                local_window_span=(4, 6), segment_spans=((1, 4),),
                                declarations=((0, "local", ()), (1, "focus", (1,)), (2, "global", ())))
        reference = ReferenceRun(bridge, total_forwards=total)
        return model, runner, reference

    def install(self, model, runner, reference):
        with patch.object(self.driver, "_model_and_runner", return_value=(model, runner)):
            capture = self.driver.install_capture(None, reference=reference)
        self.addCleanup(capture.restore)
        self.driver.configure_bounded_capture(capture, "patched-reference", representative_decodes=(1, 2, 3))
        capture.arm("r-main", self.root / "capture", prompt_len=6)
        return capture

    def drive(self, model, runner, *, steps=4):
        for forward in range(1, steps + 1):
            positions = list(range(6)) if forward == 1 else [forward + 4]
            runner.next_step(positions)
            model.forward(positions=torch.tensor(positions))

    def test_gdn_state_probe_requires_separate_storage_from_fa_cache(self):
        class Module:
            def __init__(self, fa_cache, gdn_cache):
                self.fa = NS(kv_cache=fa_cache)
                self.gdn = NS(kv_cache=gdn_cache)

            def named_modules(self):
                return [("", self), ("fa", self.fa), ("gdn", self.gdn)]

        fa_cache = torch.zeros(2, 4)
        gdn_cache = [torch.zeros(2, 3), torch.zeros(2, 3)]
        model = Module(fa_cache, gdn_cache)
        runner = NS(kv_caches=[fa_cache, *gdn_cache])
        capture = self.driver.LayerCapture(runner=runner, expected_layers={0: "fa"}, model=model)
        capture.configure_state_capture(model=model, required=True)
        self.assertTrue(capture.state_capture["enabled"])
        self.assertEqual(capture.state_capture["alias_pair_count"], 0)
        self.assertEqual(capture.state_capture["gdn_layers"], ["gdn"])

        aliased = Module(fa_cache, [fa_cache, torch.zeros(2, 3)])
        bad = self.driver.LayerCapture(runner=runner, expected_layers={0: "fa"}, model=aliased)
        bad.configure_state_capture(model=aliased, required=True)
        self.assertTrue(bad.state_capture["shared_backing_expected"])
        self.assertGreater(bad.state_capture["alias_pair_count"], 0)

    def test_real_begin_end_global_prefill_decode_write_and_restore(self):
        model, runner, ref = self.make()
        original = [i.forward for i in model.impls] + [model.forward, runner.prepare_inputs]
        capture = self.install(model, runner, ref)
        self.drive(model, runner)
        ref.validate()
        self.assertEqual([i.calls for i in model.impls], [1, 1])
        self.assertTrue(torch.equal(model.outputs[0], torch.full((4, 8), 0.5, dtype=torch.bfloat16)))
        self.assertFalse(torch.equal(model.outputs[1], model.outputs[0]))
        self.assertEqual([r["mode"] for r in ref.attachment.metrics()[::2]], ["local", "focus", "global"])
        self.assertEqual(ref.snapshots[4][model.names[0]]["canonical"], [5, 1, 7])
        self.assertEqual(ref.snapshots[4][model.names[0]]["slots"], [28])
        self.assertEqual(ref.completed, [1, 2, 3, 4])
        self.assertNotIn("q", capture.records[1][0])
        capture.restore()
        restored = [i.forward for i in model.impls] + [model.forward, runner.prepare_inputs]
        self.assertEqual(restored, original)
        self.assertFalse(ref.attachment.switch.enabled)

    def test_corrupt_actual_layer_inputs_fail(self):
        def wrong_length(m, i, md, *a): md.seq_lens.add_(-1)
        def wrong_table(m, i, md, *a): md.block_table[0, 0] = 0
        def wrong_slot(m, i, md, *a): self.context.slot_mapping[m.names[i]].add_(1)
        def unwritten(m, i, md, kc, vc, phys, offsets): kc[phys, offsets] = 99
        def missing(m, i, md, *a): m.skip_layer = 1
        for kind, change in {"length": wrong_length, "table": wrong_table, "slot": wrong_slot,
                             "unwritten": unwritten, "missing": missing}.items():
            with self.subTest(kind=kind):
                model, runner, ref = self.make(corrupt=lambda *a, change=change: change(*a) if a[0].calls == 2 else None)
                capture = self.install(model, runner, ref)
                with self.assertRaises((ReferenceError, RuntimeError)):
                    self.drive(model, runner)
                capture.restore()
                self.assertFalse(ref.attachment.switch.enabled)
                self.assertEqual(ref.completed, [1])

    def test_request_phase_position_and_stale_prepare_fail_before_attention(self):
        for kind in ("request", "phase", "position", "stale"):
            with self.subTest(kind=kind):
                model, runner, ref = self.make()
                capture = self.install(model, runner, ref)
                self.drive(model, runner, steps=1)
                runner.req_ids = ["other"] if kind == "request" else ["r-main"]
                runner.is_prefilling = kind == "phase"
                if kind != "stale":
                    runner.next_step([0] if kind == "position" else [6])
                with self.assertRaises(ReferenceError):
                    model.forward(positions=torch.tensor([6]))
                self.assertEqual([i.calls for i in model.impls], [1, 1])
                capture.restore()

    def test_missing_steps_or_ledger_identity_are_rejected(self):
        model, runner, ref = self.make()
        self.install(model, runner, ref)
        self.drive(model, runner)
        for field, value in (("request_id", "other"), ("layer", 99), ("step", 99), ("action", "passthrough")):
            with self.subTest(field=field):
                old = ref.attachment.ledger[2][field]
                ref.attachment.ledger[2][field] = value
                with self.assertRaises(ReferenceError):
                    ref.validate()
                ref.attachment.ledger[2][field] = old
        ref.completed.pop()
        with self.assertRaises(ReferenceError):
            ref.validate()

    def test_unknown_none_and_return_only_restore_methods(self):
        for kind in ("unknown_none", "return_only"):
            with self.subTest(kind=kind):
                model, runner, ref = self.make()
                original = model.impls[0].forward
                capture = self.install(model, runner, ref)
                self.drive(model, runner, steps=1)
                if kind == "return_only":
                    ref.attachment.entry = lambda *a, **k: {}
                else:
                    inner = model.impls[0].forward
                    model.impls[0].forward = lambda *a, inner=inner, **k: inner(*a, **k, unknown=None)
                runner.next_step([6])
                with self.assertRaises(ReferenceError):
                    model.forward(positions=torch.tensor([6]))
                capture.restore()
                self.assertEqual(model.impls[0].forward, original)

    def test_disabled_and_non_target_preserve_known_none_kwargs(self):
        for enabled, current in ((False, "target"), (True, "other")):
            impl = CpuFlashImpl()
            at = Attachment(Switch(enabled=enabled, request_id="target", current_request_id=current))
            original = impl.forward
            with at:
                at.wrap_impl(0, impl)
                out = torch.zeros(1, 4, 8)
                impl.forward(None, out, None, None, None, None, out,
                             output_scale=None, output_block_scale=None)
                self.assertEqual(impl.calls, 1)
            self.assertEqual(impl.forward, original)

    def test_feature_gate_rejects_special_attention_and_geometry(self):
        for field, value in (("alibi_slopes", [1.0]), ("sliding_window", (8, 0)),
                             ("logits_soft_cap", 1.0), ("kv_cache_dtype", "fp8"),
                             ("vllm_flash_attn_version", 3), ("head_size", 16)):
            with self.subTest(field=field):
                model, runner, ref = self.make()
                setattr(model.impls[0], field, value)
                with self.assertRaises(ReferenceError):
                    self.install(model, runner, ref)

    def test_canonical_padding_and_native_stride_gather(self):
        _, _, ref = self.make()
        ref.bridge.is_prefilling_by_step[2] = False
        md = NS(block_table=torch.tensor([[5, 1, 7, -1, 999]]), seq_lens=torch.tensor([9]),
                query_start_loc=torch.tensor([0, 1]))
        self.assertEqual(resolve_geometry(bridge=ref.bridge, attn_metadata=md, request_idx=0,
                                         step_index_0based=2)["block_table_row"], [5, 1, 7])
        cache = torch.arange(24 * 2 * 4 * 16).reshape(24, 2, 4, 16).to(torch.bfloat16)
        k, v = unpack_native_kv(cache, 8)
        self.assertEqual(k.untyped_storage().data_ptr(), cache.untyped_storage().data_ptr())
        positions = [0, 1, 4, 8]
        actual_k, actual_v = gather_positions(k, v, [5, 1, 7], 4, positions)
        for j, p in enumerate(positions):
            self.assertTrue(torch.equal(actual_k[j], cache[[5, 1, 7][p // 4], :, p % 4, :8]))
            self.assertTrue(torch.equal(actual_v[j], cache[[5, 1, 7][p // 4], :, p % 4, 8:]))
        for field, value in (("seq_lens", None), ("seq_lens", torch.tensor([8])),
                             ("query_start_loc", torch.tensor([1, 2])),
                             ("block_table", torch.tensor([[5, 1]]))):
            bad = copy.copy(md)
            setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ReferenceError):
                resolve_geometry(bridge=ref.bridge, attn_metadata=bad, request_idx=0, step_index_0based=2)

    def test_dense_gqa_reads_only_expanded_positions_including_current_token(self):
        bridge = TimelineBridge(prompt_len=10, kernel_block_size=2, sink_span=(0, 1),
                                local_window_span=(8, 10), segment_spans=((2, 4),),
                                declarations=((0, "local", ()),), is_prefilling_by_step={0: False})
        row = [5, 1, 7, 2, 9, 3]
        cache = torch.full((12, 2, 2, 16), float("nan"), dtype=torch.bfloat16)
        keys, values = unpack_native_kv(cache, 8)
        # Deliberately leave excluded blocks and the unwritten tail as NaN.
        for pos in (0, 1, 8, 9, 10):
            keys[row[pos // 2], pos % 2] = 0
            for head in range(2):
                values[row[pos // 2], pos % 2, head] = pos + 100 * head
        md = NS(seq_lens=torch.tensor([11]), block_table=torch.tensor([row + [-1]]),
                query_start_loc=torch.tensor([0, 1]), num_actual_tokens=1, causal=True)
        output = torch.empty(1, 4, 8, dtype=torch.bfloat16)
        perform_reference_attention_native(
            Switch(), layer_idx=0, step_index_0based=0, bridge=bridge, request_idx=0,
            query=torch.zeros_like(output), kv_cache=cache, attn_metadata=md,
            output=output, head_size=8, impl_scale=0.25)
        means = torch.tensor([28 / 5, 28 / 5, 528 / 5, 528 / 5], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(output, means[None, :, None].expand(1, 4, 8)))

    def run_driver(self, directory, *, corrupt=None):
        """Run the actual _run, submit_request, drive_main_request and capture hooks."""
        directory.mkdir()
        model, runner, _ = self.make(corrupt=corrupt, total=29)
        runner.req_states = NS(req_id_to_index={})
        timeline = json.loads((REPO / "configs/p2-masked-prep/crossblock.json").read_text())
        timeline["block_size"] = 4
        config = directory / "timeline.json"
        config.write_text(json.dumps(timeline))
        # The real tokenizer supplies the declaration boundaries and fixed tokens.
        tokenizer = self.tokenizer
        tokens = [t for p in timeline["generation_script"] for t in tokenizer.encode(p["text"], add_special_tokens=False)]
        tokens.append(tokens[-1])
        trajectory = directory / "trajectory.json"
        prompt_ids = list(range(6))
        trajectory.write_text(json.dumps({"tokens": [[[t]] for t in tokens], "prompt_len": len(prompt_ids),
            "prompt_token_ids_sha256": self.driver.sha256_token_ids(prompt_ids)}))
        args = self.driver.parse_args(["--arm", "patched-reference", "--out", str(directory),
            "--max-tokens", "29", "--force-trajectory", str(trajectory), "--cleanup-check",
            "--timeline-config", str(config)])
        payload = {"prompt_len": 6, "sink_span": [0, 1], "local_window_span": [4, 6], "segment_spans": [[1, 2], [2, 4]]}
        manifest = {"config": {}}
        outer = self

        class Engine:
            def __init__(self):
                self.scheduler = NS(requests={})
                self.model_executor = NS(collective_rpc=lambda *a, **kw: {})
                self.calls, self.aborted, self.logits = 0, [], []

            def get_num_unfinished_requests(self): return len(self.scheduler.requests)

            def add_request(self, external, ids, params):
                if params.extra_args is not None:
                    raise AssertionError("reference payload enabled")
                self.external, self.internal = external, external + "-internal"
                self.scheduler.requests[self.internal] = NS(prompt_token_ids=list(ids))
                runner.req_ids = [self.internal]
                runner.req_states.req_id_to_index[self.internal] = 0
                return self.internal

            def abort_request(self, ids):
                self.aborted.extend(ids)
                self.scheduler.requests.clear()
                runner.req_states.req_id_to_index.clear()

            def step(self):
                if not self.scheduler.requests:
                    return []
                self.calls += 1
                positions = list(range(6)) if self.calls == 1 else [self.calls + 4]
                runner.next_step(positions)
                model.forward(positions=torch.tensor(positions))
                arm = json.loads((directory / "arm.json").read_text())
                if arm["target_req_id"] != self.internal:
                    raise AssertionError("wrong control-file request")
                token = arm["tokens"][self.calls - 1][0][0]
                with (directory / "force.jsonl").open("a") as f:
                    f.write(json.dumps({"step": self.calls, "req_id": self.internal, "req_ids": [self.internal],
                                        "raw_sampled": [[0]], "forced": [token]}) + "\n")
                self.logits.append({"step": self.calls, "req_id": self.internal, "req_ids": [self.internal],
                                    "logits": model.state.flatten()[None].clone()})
                torch.save(self.logits, directory / "logits.pt")
                finished = self.calls == 29
                if finished:
                    self.scheduler.requests.clear()
                    runner.req_states.req_id_to_index.clear()
                return [NS(request_id=self.external, outputs=[NS(token_ids=[token])], finished=finished)]

        engine = Engine()
        llm = NS(llm_engine=engine)
        originals = [i.forward for i in model.impls] + [model.forward, runner.prepare_inputs]
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "vllm": NS(LLM=lambda **k: llm),
                "vllm.config.attention": NS(AttentionConfig=lambda **k: k),
            }))
            def dump(_llm, path):
                path.write_text("[]")
                return {"observable": True}
            def cleanup(*a, **k):
                # Cleanup starts after reference/capture methods have been restored.
                outer.assertEqual([i.forward for i in model.impls] + [model.forward, runner.prepare_inputs], originals)
                return {"ok": not engine.scheduler.requests and not runner.req_states.req_id_to_index}
            overrides = {
                "_model_and_runner": lambda llm: (model, runner),
                "read_max_num_batched_tokens": lambda *a, **k: 8192,
                "sampling_params": lambda *a, **k: NS(extra_args=k.get("extra_args")),
                "protocol_state": lambda *a: {"observable": True},
                "dump_engine_traces": dump, "run_cleanup_check": cleanup,
            }
            for name, value in overrides.items():
                stack.enter_context(patch.object(self.driver, name, value))
            try:
                code = self.driver._run(args, manifest, directory / "manifest.json", None, payload,
                                        prompt_ids, None, directory / "arm.json")
            finally:
                self.assertEqual([i.forward for i in model.impls] + [model.forward, runner.prepare_inputs], originals)
        return code, manifest, model, runner, engine

    def test_real_run_reference_manifest_and_independent_new_requests(self):
        code, manifest, first, _, _ = self.run_driver(self.root / "first")
        self.assertEqual(code, 0, manifest.get("failures"))
        self.assertTrue(manifest["reference"]["ok"])
        self.assertEqual(len(manifest["reference"]["ledger"]), 58)
        self.assertEqual(manifest["reference"]["restored_layers"], [0, 1])
        first_outputs = [x.clone() for x in first.outputs]
        for cache in first.caches:
            cache.fill_(42)
        first.state.fill_(99)
        code2, manifest2, second, _, _ = self.run_driver(self.root / "second")
        self.assertEqual(code2, 0, manifest2.get("failures"))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first_outputs, second.outputs, strict=True)))
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(first.caches, second.caches, strict=True)))

    def test_real_run_failure_keeps_partial_evidence_and_restores(self):
        def corrupt(model, index, metadata, *rest):
            if model.calls == 3 and index == 1:
                metadata.seq_lens.sub_(1)
        out = self.root / "failed"
        with self.assertRaises(ReferenceError):
            self.run_driver(out, corrupt=corrupt)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertFalse(manifest["reference"]["ok"])
        self.assertTrue(manifest["reference"]["incomplete_request"])
        self.assertEqual(manifest["reference"]["completed_forwards"], [1, 2])
        self.assertEqual(manifest["reference"]["restored_layers"], [0, 1])
        self.assertFalse(manifest["failure_cleanup"]["scheduler_has_request"])

    def test_reference_cli_rejects_default_short_prompt(self):
        with self.assertRaisesRegex(RuntimeError, "fixed fixture"):
            self.driver.main(["--arm", "patched-reference", "--out", str(self.root / "invalid")])

    def test_accepted_fixture_binds_script_and_terminal_sample(self):
        payload = json.loads((REPO / "evidence/p3-calib/masked-prep/crossblock-note.json").read_text())
        trajectory = json.loads((REPO / "evidence/p3-masked-smoke/traj-7834-generated.json").read_text())
        tokens = self.driver.trajectory_token_sequence(trajectory)
        args = NS(timeline_config=REPO / "configs/p2-masked-prep/crossblock.json", max_tokens=29)
        reference = self.driver.build_reference_run(args, payload, tokens)
        bridge = reference.bridge
        self.assertEqual([bridge.mode_at(d - 1)[0] for d in (5, 6, 7, 12, 20, 25)],
                         ["global", "local", "local", "global", "focus", "global"])
        self.assertEqual([len(bridge.mask_at(d - 1).read_positions) for d in (6, 7, 20, 25)],
                         [2352, 2353, 4718, 7859])
        for bad in (tokens[:-1], tokens[:-1] + [0], [0] + tokens[1:]):
            with self.assertRaisesRegex(RuntimeError, "trajectory differs"):
                self.driver.build_reference_run(args, payload, bad)


if __name__ == "__main__":
    unittest.main()
