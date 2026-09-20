"""Real capture lifecycle plus individual corruption controls, without GPU."""
import copy
import json
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from _support import REPO
from attnview.smoke_structure import StructureError, check_capture_structure, expectations_from_config


class FakeFullAttentionSpec:
    block_size = 4


class StructureCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        from tests.test_p2_smoke_bounded_capture import load_runner
        cls.driver = load_runner()
        cls.tokenizer = AutoTokenizer.from_pretrained(str(cls.driver.SNAPSHOT), trust_remote_code=False)

    def setUp(self):
        from tests.test_p2_calib_hooks import LayerCaptureTest
        self.h = LayerCaptureTest("test_steps_layers_positions_and_metadata_are_real")
        self.h.setUp()
        self.addCleanup(self.h.tearDown)
        self.payload = {
            "prompt_len": 34, "sink_span": (0, 2), "local_window_span": (30, 34),
            "segment_spans": ((4, 8), (8, 12), (12, 30)),
        }
        self.expected = expectations_from_config(
            timeline_config=REPO / "configs/p2-masked-prep/crossblock.json", doc_fixture=None,
            tokenizer=self.tokenizer, block_size=4, total_steps=29, **self.payload)

    def lifecycle(self):
        from tests.test_p2_calib_hooks import FakeFlashAttentionImpl, FakeModel
        impls = [FakeFlashAttentionImpl(), FakeFlashAttentionImpl()]
        model = FakeModel(impls, prompt_len=34)
        capture, runner = self.h._armed(model, impls, prompt_len=34)
        names = list(capture.layer_names.values())
        fa = NS(kv_cache_spec=FakeFullAttentionSpec(), layer_names=names)
        gdn = NS(kv_cache_spec=object(), layer_names=["gdn"])
        runner.kv_cache_config = NS(kv_cache_groups=[gdn, fa])
        runner.attn_groups = [[gdn], [fa]]
        canonical = torch.tensor([[31 + 3 * i for i in range(20)]], dtype=torch.int32)
        runner.block_tables = NS(input_block_tables=[torch.full_like(canonical, 999), canonical],
                                block_sizes=[4, 4], kernel_block_sizes=[4, 4])
        context = NS(slot_mapping={})
        modules = {"vllm.v1.kv_cache_interface": NS(FullAttentionSpec=FakeFullAttentionSpec),
                   "vllm.forward_context": NS(get_forward_context=lambda: context)}
        trace = {"override_steps": [], "enforce_global_steps": [], "parsed_steps": []}
        old_metadata = model.metadata_for
        def metadata_for(row, q_len):
            meta = old_metadata(row, q_len)
            d = model.calls - 1
            kv = 34 + d
            mode = "local" if 6 <= d <= 11 else "focus" if 20 <= d <= 24 else "global"
            # Literal fixture arithmetic, independent of checker and candidate helpers.
            blocks = list(range((kv + 3) // 4)) if mode == "global" else sorted(
                {0, 7, 8} | set(range(8, (kv + 3) // 4)) | ({2} if mode == "focus" else set()))
            length = sum(min(4, kv - 4 * b) for b in blocks)
            meta.seq_lens = torch.tensor([length], dtype=torch.int32)
            meta.block_table = canonical[:, blocks].clone()
            if mode != "global":
                trace["override_steps"].append({
                    "kind": "override_step", "step": d, "req_ids": ["r-main"],
                    "num_scheduled_tokens": {"r-main": 1},
                    "override": {
                        "group_index": 1, "rows_with_override": [0], "seqused_k": [length],
                        "max_seq_len": length, "visible_logical_blocks": {"0": blocks},
                    },
                })
            return meta
        model.metadata_for = metadata_for
        with patch.dict(sys.modules, modules):
            self.driver.configure_bounded_capture(capture, "patched-masked")
            self.driver.configure_structure_capture(capture)
            for forward in range(1, 30):
                positions = list(range(34)) if forward == 1 else [34 + forward - 2]
                runner.next_step(positions)
                context.slot_mapping = {name: torch.tensor([int(canonical[0, p // 4]) * 4 + p % 4
                                                           for p in positions]) for name in names}
                runner.execute_model_state = NS(slot_mappings_by_layer={name: torch.tensor([-8]) for name in names})
                model.forward(positions=torch.tensor(positions))
                parse = forward - 1
                mode = "local" if 5 <= parse <= 10 else "focus" if 19 <= parse <= 23 else "global"
                trace["parsed_steps"].append({
                    "req_id": "r-main", "parse_step": parse, "effect_step": forward,
                    "mode": mode, "refs": [2] if mode == "focus" else [],
                })
            canonical.fill_(777)
            for value in context.slot_mapping.values():
                value.fill_(-1)
        capture.disarm()
        return capture, trace

    def check(self, capture, trace):
        return check_capture_structure(capture=capture, expectations=self.expected, trace=trace,
                                       prompt_len=34, block_size=4, representative_decodes=(6, 7, 20, 25), req_id="r-main")

    def test_real_lifecycle_and_buffer_reuse(self):
        capture, trace = self.lifecycle()
        checks, evidence = self.check(capture, trace)
        self.assertTrue(all(c["ok"] for c in checks))
        self.assertEqual(len(evidence["steps"]), 29)
        self.assertEqual([self.expected[d + 1].mode for d in (6, 7, 12, 20, 25)],
                         ["local", "local", "global", "focus", "global"])
        self.assertTrue((capture.stream_dir / "structure.json").exists())

    def test_each_corruption_fails(self):
        capture, trace = self.lifecycle()
        name = capture.layer_names[1]
        changes = {
            "length": lambda c, t: c.structure[7][name].update(seq_len=999),
            "physical": lambda c, t: c.structure[7][name]["physical"].__setitem__(0, 999),
            "slot": lambda c, t: c.structure[7][name]["slots"].__setitem__(0, -1),
            "position": lambda c, t: c.structure[7][name]["positions"].__setitem__(0, 0),
            "request": lambda c, t: c.structure[7][name].update(req_id="other"),
            "layer": lambda c, t: c.structure[7].pop(name),
            "step": lambda c, t: c.structure.pop(7),
            "trace": lambda c, t: t.clear(),
            "missing_override": lambda c, t: t["override_steps"].pop(0),
            "trace_request": lambda c, t: t["override_steps"][0].update(req_ids=["other"]),
            "trace_step": lambda c, t: t["override_steps"][0].update(step=7),
            "trace_blocks": lambda c, t: t["override_steps"][0]["override"].update(visible_logical_blocks={"0": [0]}),
            "forced_global": lambda c, t: t["enforce_global_steps"].append({"step": 6}),
            "wrong_mode": lambda c, t: t["parsed_steps"][5].update(mode="global"),
            "extra_layer": lambda c, t: c.structure[7].update(unexpected={}),
        }
        for kind, change in changes.items():
            with self.subTest(kind=kind):
                c, t = copy.copy(capture), copy.deepcopy(trace)
                c.structure = copy.deepcopy(capture.structure)
                change(c, t)
                with self.assertRaises(StructureError):
                    self.check(c, t)

    def test_true_fixture_expectations(self):
        fixture = json.loads((REPO / "evidence/p3-calib/masked-prep/crossblock-note.json").read_text())
        expected = expectations_from_config(
            timeline_config=REPO / "configs/p2-masked-prep/crossblock.json", doc_fixture=None,
            tokenizer=self.tokenizer, block_size=784, total_steps=29,
            **{key: fixture[key] for key in self.payload})
        self.assertEqual(expected[8].kv_len, 7841)
        self.assertEqual(expected[8].mode, "local")
        self.assertIn(10, expected[8].blocks)
        self.assertEqual(expected[8].effective_per_block[-1], 1)
        for d, mode in ((5, "global"), (6, "local"), (11, "local"), (12, "global"),
                        (19, "global"), (20, "focus"), (24, "focus"), (25, "global")):
            self.assertEqual(expected[d + 1].mode, mode)

    def test_driver_gate_records_failure(self):
        capture, trace = self.lifecycle()
        args = NS(out=self.h.dir, timeline_config=REPO / "configs/p2-masked-prep/crossblock.json",
                  doc_fixture=None, max_tokens=29)
        (args.out / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in trace["override_steps"]))
        log, manifest = self.driver.CheckLog(), {}
        self.driver.check_masked_run(capture=capture, args=args, payload=self.payload, req_id="r-main",
                                     engine_traces=[], manifest=manifest, log=log)
        self.assertEqual(log.failed, ["masked_structure"])
        self.assertFalse(manifest["masked_structure"]["ok"])

    def test_real_run_calls_gate_and_failure_exits_nonzero_after_cleanup(self):
        capture, trace = self.lifecycle()
        timeline = json.loads((REPO / "configs/p2-masked-prep/crossblock.json").read_text())
        timeline["block_size"] = 4
        config_path = self.h.dir / "timeline.json"
        config_path.write_text(json.dumps(timeline))
        for corrupt in (False, True, "request_error"):
            with self.subTest(corrupt=corrupt), ExitStack() as stack:
                out = self.h.dir / str(corrupt)
                out.mkdir()
                args = self.driver.parse_args(["--arm", "patched-masked", "--out", str(out),
                    "--max-tokens", "29", "--cleanup-check", "--timeline-config", str(config_path)])
                (out / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in trace["override_steps"]))
                torch.save([{"step": i, "req_id": "r-main", "req_ids": ["r-main"], "logits": torch.zeros(1, 8)}
                            for i in range(1, 30)], out / "logits.pt")
                if corrupt:
                    capture.structure[7][capture.layer_names[0]]["seq_len"] += 1
                aborted = []
                llm = NS(llm_engine=NS(model_executor=NS(collective_rpc=lambda *a, **k: {}),
                                      abort_request=lambda ids, _aborted=aborted: _aborted.extend(ids),
                                      step=lambda: []))
                stack.enter_context(patch.dict(sys.modules, {"vllm": NS(LLM=lambda _llm=llm, **k: _llm),
                    "vllm.config.attention": NS(AttentionConfig=lambda **k: k)}))
                # 闭包一律用默认参数绑定**本轮**的值（B023）：循环变量在下一轮会被重新赋值，
                # 直接引用会让"这一轮的闭包"读到"下一轮的值"。
                def drive(*a, on_first_step, _corrupt=corrupt, **k):
                    on_first_step()
                    if _corrupt == "request_error":
                        raise RuntimeError("inference trace failure")
                    return {"tokens": [1] * 29, "steps": 29, "consuming_steps": 29, "other_request_outputs": []}
                def dump(llm, path):
                    path.write_text(json.dumps(trace["parsed_steps"]))
                    return {"observable": True}
                mocks = {
                    "install_capture": lambda llm: capture,
                    "configure_structure_capture": lambda c: None,
                    "read_max_num_batched_tokens": lambda *a, **k: 8192,
                    "sampling_params": lambda *a, **k: None,
                    "output_mode_of": lambda p: "cumulative",
                    "submit_request": lambda *a: {"internal_id": "r-main"},
                    "drive_main_request": drive,
                    "scheduler_requests": lambda e: {"r-main": NS(prompt_token_ids=list(range(34)))},
                    "_scheduler_has": lambda e, req, _aborted=aborted: req == "r-main" and not _aborted,
                    "protocol_state": lambda *a: {"observable": True},
                    "canonical_blocks": lambda *a: {"observable": False},
                    "dump_engine_traces": dump,
                }
                for name, impl in mocks.items():
                    stack.enter_context(patch.object(self.driver, name, impl))
                cleanup = stack.enter_context(patch.object(self.driver, "run_cleanup_check", return_value={"ok": True}))
                manifest = {"config": {}}
                if corrupt == "request_error":
                    with self.assertRaisesRegex(RuntimeError, "inference trace failure"):
                        self.driver._run(args, manifest, out / "manifest.json", None, self.payload,
                                         list(range(34)), {"attnview": self.payload}, out / "arm.json")
                    self.assertTrue(aborted)
                    self.assertFalse(manifest["failure_cleanup"]["scheduler_has_request"])
                    self.assertTrue(manifest["masked_structure"]["incomplete_request"])
                    self.assertTrue((out / "engine-traces-failure.json").exists())
                    continue
                code = self.driver._run(args, manifest, out / "manifest.json", None, self.payload,
                                        list(range(34)), {"attnview": self.payload}, out / "arm.json")
                self.assertEqual(code, int(corrupt), manifest.get("failures"))
                self.assertEqual(manifest["masked_structure"]["ok"], not corrupt)
                cleanup.assert_called_once()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
