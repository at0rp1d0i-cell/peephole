"""预冻结 masked/reference 数值合同的纯 CPU 行为回归。"""

import copy
import json
import unittest

from _support import REPO
from attnview.numeric_acceptance import (
    NumericAcceptanceError,
    canonical_input_identity,
    evaluate_numeric_acceptance,
)


class TestNumericAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(
            (REPO / "configs/p2-masked-acceptance/numeric-contract.json").read_text(encoding="utf-8")
        )

    def make_summary(self):
        layers = list(range(3, 64, 4))
        decodes = [6, 7, 20, 25]
        rows = [
            {
                "layer": layer,
                "decode": decode,
                "kind": kind,
                "max_abs": 0.1,
                "rms": 0.01,
                "relative_l2": 0.01,
                "nonfinite": 0,
            }
            for decode in decodes
            for layer in layers
            for kind in ("q", "out")
        ]
        logits_steps = [
            {
                "step": step,
                "max_abs": 0.1,
                "rms": 0.01,
                "relative_l2": 0.01,
                "argmax_equal": True,
                "nonfinite": 0,
            }
            for step in range(1, 30)
        ]
        return {
            "schema": "attnview.p3-reference-summary/v1",
            "reference_run": "/tmp/reference/run",
            "masked_run": "/tmp/masked/run",
            "head": "a" * 40,
            "decodes": decodes,
            "reference_exit_code": 0,
            "reference_manifest_failures": [],
            "reference_completed_forwards": list(range(1, 30)),
            "reference_overrides": 448,
            "reference_layers": 16,
            "reference_nonfinite": 0,
            "gdn_state": {
                "enabled": True,
                "required": True,
                "layer_count": 48,
                "missing_steps": [],
                "forward_steps": 29,
                "state_nonfinite": 0,
                "unique_digest_sample": {f"layer.{index}": 29 for index in range(5)},
            },
            "attention_candidate_vs_reference": {
                "decode_q_step": {"count": 64, "max_relative_l2": 0.01, "nonfinite": 0},
                "decode_out_step": {"count": 64, "max_relative_l2": 0.01, "nonfinite": 0},
                "per_array": rows,
            },
            "logits_candidate_vs_reference": {
                "count": 29,
                "max_relative_l2": 0.01,
                "argmax_mismatch": 0,
                "nonfinite": 0,
            },
            "logits_steps": logits_steps,
            "provenance": {"input_hashes": {"heldout.json": "b" * 64}},
        }

    def contract_for(self, summary):
        contract = copy.deepcopy(self.contract)
        contract["heldout_input_identity_sha256"] = canonical_input_identity(
            summary["provenance"]["input_hashes"]
        )
        return contract

    def assert_rejected(self, summary, code):
        decision = evaluate_numeric_acceptance(summary, self.contract_for(summary))
        self.assertFalse(decision["passed"])
        self.assertIn(code, {failure["code"] for failure in decision["failures"]})

    def test_held_out_summary_with_every_hard_gate_and_metric_within_limit_passes(self):
        summary = self.make_summary()
        decision = evaluate_numeric_acceptance(summary, self.contract_for(summary))
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["failures"], [])
        self.assertEqual(decision["observed"]["decode_q_max_relative_l2"], 0.01)
        self.assertEqual(decision["observed"]["logits_argmax_mismatch"], 0)

    def test_each_metric_family_is_checked_per_array_not_only_by_aggregate(self):
        for kind, aggregate, code in (
            ("q", "decode_q_step", "attention_q_relative_l2[0]"),
            ("out", "decode_out_step", "attention_out_relative_l2[1]"),
        ):
            with self.subTest(kind=kind):
                summary = self.make_summary()
                row = next(row for row in summary["attention_candidate_vs_reference"]["per_array"] if row["kind"] == kind)
                row["relative_l2"] = 0.020001
                summary["attention_candidate_vs_reference"][aggregate]["max_relative_l2"] = 0.020001
                self.assert_rejected(summary, code)

        summary = self.make_summary()
        summary["logits_steps"][7]["relative_l2"] = 0.020001
        summary["logits_candidate_vs_reference"]["max_relative_l2"] = 0.020001
        self.assert_rejected(summary, "logits_relative_l2[7]")

    def test_nonfinite_argmax_missing_identity_and_aggregate_tampering_fail(self):
        cases = []

        nonfinite = self.make_summary()
        nonfinite["attention_candidate_vs_reference"]["per_array"][0]["nonfinite"] = 1
        nonfinite["attention_candidate_vs_reference"]["decode_q_step"]["nonfinite"] = 1
        cases.append(("nonfinite", nonfinite, "attention_q_nonfinite[0]"))

        argmax = self.make_summary()
        argmax["logits_steps"][0]["argmax_equal"] = False
        argmax["logits_candidate_vs_reference"]["argmax_mismatch"] = 1
        cases.append(("argmax", argmax, "logits_argmax_equal[0]"))

        duplicate = self.make_summary()
        duplicate["attention_candidate_vs_reference"]["per_array"][-1] = copy.deepcopy(
            duplicate["attention_candidate_vs_reference"]["per_array"][0]
        )
        cases.append(("identity", duplicate, "attention_identities"))

        aggregate = self.make_summary()
        aggregate["logits_candidate_vs_reference"]["max_relative_l2"] = 0.009
        cases.append(("aggregate", aggregate, "logits_max_relative_l2"))

        for label, summary, code in cases:
            with self.subTest(label=label):
                self.assert_rejected(summary, code)

    def test_calibration_identity_cannot_equal_held_out_identity(self):
        summary = self.make_summary()
        contract = self.contract_for(summary)
        contract["calibration_input_identity_sha256"] = contract["heldout_input_identity_sha256"]
        with self.assertRaises(NumericAcceptanceError):
            evaluate_numeric_acceptance(summary, contract)

    def test_unexpected_third_input_identity_cannot_impersonate_held_out(self):
        summary = self.make_summary()
        contract = self.contract_for(summary)
        summary["provenance"]["input_hashes"]["unexpected.json"] = "c" * 64
        decision = evaluate_numeric_acceptance(summary, contract)
        self.assertFalse(decision["passed"])
        self.assertIn("held_out_input_identity", {failure["code"] for failure in decision["failures"]})

    def test_unknown_contract_key_is_rejected_before_decision(self):
        summary = self.make_summary()
        contract = self.contract_for(summary)
        contract["unconsumed_knob"] = True
        with self.assertRaises(NumericAcceptanceError):
            evaluate_numeric_acceptance(summary, contract)


if __name__ == "__main__":
    unittest.main()
