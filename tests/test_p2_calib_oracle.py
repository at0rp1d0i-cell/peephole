"""阶段 05 SUP-004 单请求校准：`tools/p2-calib-oracle.py` 的 CPU 确定性用例。

覆盖（与工作单逐条对应）：

1. 解析可验证的单头用例：`prompt_len=4`、单 query，期望值用**纯 Python `math`** 手算（1e-5 内）；
2. GQA：`num_heads=4, num_kv_heads=2`，共享同一 kv head 的 query head 结果逐位相同，
   且与"逐 head 显式展开连续分组"的实现按元素一致、与交错分组明确不同；
3. 因果掩码：`q_len=2` 时把未来 key 改成 1e4 结果放不变，而改动可见 key 只影响能看到它的那个位置；
4. 非有限值：`out` 注入 NaN → 报告把它单列进 `non_finite`（比较项 `finite=false`、指标为 null），
   同批次的有限比较仍给出完整指标，退出码 3；
5. 必需元数据缺失（删 `scale` / 删多个）→ 退出码 2、stderr 指名、不产出报告；
6. 独立性结构守卫：oracle 源码不出现候选侧符号，也不 import 项目内包模块。

独立性说明：本文件的期望值都来自独立路径——解析用例是纯 Python 手算，GQA 用逐 head 循环，
端到端用例用 float64 逐 head 实现并只借用 BF16 舍入模拟模型侧观测；oracle 只作为被测对象。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO / "tools" / "p2-calib-oracle.py"

FORBIDDEN_TOKENS = ("readview", "gpukv", "step_plan", "visible")
FORBIDDEN_IMPORT_ROOTS = ("attnview", "vllm")


def load_oracle():
    spec = importlib.util.spec_from_file_location("p2_calib_oracle_under_test", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = load_oracle()


# ---------------------------------------------------------------- 夹具与工具


def build_arrays(
    *,
    prompt_len: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q: dict,
    out: dict,
    positions: dict,
    k: dict,
    v: dict,
    layer: int = 3,
    scale_source: str | None = None,
) -> dict:
    """按捕获键名约定组装 npz 载荷。

    `q`/`out`/`positions` 以步号（从 1 起）为键；`k`/`v` 以层号为键。
    """
    arrays: dict = {}
    for step, positions_ in positions.items():
        arrays[f"positions_step{step}"] = np.asarray(positions_, dtype=np.int64)
    for step, tensor in q.items():
        arrays[f"q_step{step}_L{layer}"] = np.asarray(tensor, dtype=np.float32)
    for step, tensor in out.items():
        arrays[f"out_step{step}_L{layer}"] = np.asarray(tensor, dtype=np.float32)
    for layer_, tensor in k.items():
        arrays[f"k_prefill_L{layer_}"] = np.asarray(tensor, dtype=np.float32)
    for layer_, tensor in v.items():
        arrays[f"v_prefill_L{layer_}"] = np.asarray(tensor, dtype=np.float32)
    arrays["prompt_len"] = np.array(prompt_len, dtype=np.int64)
    arrays["scale"] = np.array(scale, dtype=np.float64)
    arrays["num_heads"] = np.array(num_heads, dtype=np.int64)
    arrays["num_kv_heads"] = np.array(num_kv_heads, dtype=np.int64)
    arrays["head_dim"] = np.array(head_dim, dtype=np.int64)
    arrays["layer_index"] = np.array(layer, dtype=np.int64)
    arrays["dtype_name"] = np.array("bfloat16")
    if scale_source is not None:
        arrays["scale_source"] = np.array(scale_source)
    return arrays


def write_npz(path: Path, arrays: dict, drop: tuple[str, ...] = ()) -> Path:
    np.savez(path, **{name: value for name, value in arrays.items() if name not in drop})
    return path


def run_cli(capture: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ORACLE_PATH), "--capture", str(capture), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )


def hand_computed_dense(query: list[float], keys: list[list[float]], values: list[list[float]], scale: float) -> list[float]:
    """纯 Python（`math`）手算单头单 query 的 dense softmax 加权和；不依赖 torch。"""
    logits = [scale * sum(qi * ki for qi, ki in zip(query, key)) for key in keys]
    shifted = [math.exp(value - max(logits)) for value in logits]
    total = sum(shifted)
    weights = [value / total for value in shifted]
    return [sum(w * value[d] for w, value in zip(weights, values)) for d in range(len(query))]


def explicit_expansion_dense(q, k, v, positions, scale: float, *, kv_head_of) -> torch.Tensor:
    """逐 head 显式展开的 dense 参考（独立于 oracle 的批量实现），按输入 dtype 计算。"""
    dtype = q.dtype
    heads, head_dim = int(q.shape[0]), int(q.shape[2])
    out = torch.empty((heads, len(positions), head_dim), dtype=dtype)
    for head in range(heads):
        kv_head = int(kv_head_of(head))
        for index, position in enumerate(positions):
            keys_used = int(position) + 1
            scores = (k[kv_head, :keys_used].to(dtype) @ q[head, index].to(dtype)) * scale
            weights = torch.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[head, index] = weights @ v[kv_head, :keys_used].to(dtype)
    return out


def bf16_round(tensor: torch.Tensor) -> torch.Tensor:
    """按 BF16 落位再回到 FP32，模拟模型侧 BF16 输出。"""
    return tensor.to(torch.bfloat16).float()


# ---------------------------------------------------------------- 1. 解析用例


class AnalyticSingleHeadTest(unittest.TestCase):
    def test_single_head_matches_hand_computed_dense_softmax(self) -> None:
        prompt_len, head_dim, position = 4, 2, 2
        scale = 1.0 / math.sqrt(head_dim)
        query = [1.0, 0.5]
        keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
        values = [[1.0, 2.0], [3.0, 4.0], [0.5, 0.25], [-1.0, 1.0]]

        expected = hand_computed_dense(query, keys[: position + 1], values[: position + 1], scale)
        all_keys = hand_computed_dense(query, keys, values, scale)
        # 因果上界确实起作用：把第 4 个 key 也算进去会明显改变结果。
        self.assertGreater(max(abs(a - b) for a, b in zip(expected, all_keys)), 1e-3)

        ref = ORACLE.dense_reference(
            q=torch.tensor([[query]], dtype=torch.float32),
            k=torch.tensor([keys], dtype=torch.float32),
            v=torch.tensor([values], dtype=torch.float32),
            positions=[position],
            scale=scale,
            num_heads=1,
            num_kv_heads=1,
        )
        self.assertEqual(tuple(ref.shape), (1, 1, head_dim))
        for got, want in zip(ref[0, 0].tolist(), expected):
            self.assertAlmostEqual(got, want, delta=1e-5)


# ---------------------------------------------------------------- 2. GQA


class GqaExpansionTest(unittest.TestCase):
    def test_heads_sharing_one_kv_head_produce_identical_rows(self) -> None:
        num_heads, num_kv_heads, head_dim, prompt_len = 4, 2, 3, 5
        query_a, query_b = [0.5, -1.0, 0.25], [-0.75, 0.5, 1.5]
        q = torch.tensor([[query_a], [query_a], [query_b], [query_b]], dtype=torch.float32)
        generator = torch.Generator().manual_seed(20260918)
        k = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)
        v = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)

        ref = ORACLE.dense_reference(
            q, k, v, [prompt_len - 1], 1.0 / math.sqrt(head_dim), num_heads, num_kv_heads
        )
        # head 0/1 共享 kv head 0、head 2/3 共享 kv head 1（连续分组）。
        self.assertTrue(torch.equal(ref[0], ref[1]))
        self.assertTrue(torch.equal(ref[2], ref[3]))
        self.assertFalse(torch.equal(ref[0], ref[2]))

    def test_grouping_matches_explicit_consecutive_expansion(self) -> None:
        num_heads, num_kv_heads, head_dim, prompt_len = 4, 2, 3, 6
        positions = [4, 5]
        scale = 1.0 / math.sqrt(head_dim)
        generator = torch.Generator().manual_seed(41)
        q = torch.randn(num_heads, len(positions), head_dim, generator=generator)
        k = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)
        v = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)

        ref = ORACLE.dense_reference(q, k, v, positions, scale, num_heads, num_kv_heads)
        repeat = num_heads // num_kv_heads
        consecutive = explicit_expansion_dense(
            q, k, v, positions, scale, kv_head_of=lambda head: head // repeat
        )
        interleaved = explicit_expansion_dense(
            q, k, v, positions, scale, kv_head_of=lambda head: head % repeat
        )
        torch.testing.assert_close(ref, consecutive, rtol=1e-6, atol=1e-6)
        # 反例：交错分组（h % repeat）不是本 oracle 的约定，差异必须是量级可见的。
        self.assertGreater(float((ref - interleaved).abs().max()), 1e-3)


# ---------------------------------------------------------------- 3. 因果掩码


class CausalMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.head_dim, self.prompt_len = 2, 6
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.positions = [2, 3]
        self.q = torch.tensor([[[1.0, 0.5], [0.75, -0.25]]], dtype=torch.float32)
        self.k = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5], [2.0, 0.5], [0.25, 3.0]]],
            dtype=torch.float32,
        )
        self.v = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [0.5, 0.25], [-1.0, 1.0], [0.3, 0.7], [2.0, 2.0]]],
            dtype=torch.float32,
        )
        self.base = ORACLE.dense_reference(
            self.q, self.k, self.v, self.positions, self.scale, 1, 1
        )

    def _rerun(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return ORACLE.dense_reference(self.q, k, v, self.positions, self.scale, 1, 1)

    def test_future_keys_never_change_the_result(self) -> None:
        k_future = self.k.clone()
        v_future = self.v.clone()
        k_future[:, 4:, :] = 1e4
        v_future[:, 4:, :] = 1e4
        torch.testing.assert_close(
            self._rerun(k_future, v_future), self.base, rtol=0.0, atol=0.0
        )

    def test_only_positions_that_can_see_the_key_change(self) -> None:
        k_touched = self.k.clone()
        k_touched[:, 3, :] = 1e4  # 位置 3 的 key：位置 2 看不到，位置 3 看得到
        changed = self._rerun(k_touched, self.v)
        # (head 0, query 0) = 绝对位置 2 的输出必须逐位不变；(head 0, query 1) = 位置 3 必须变。
        torch.testing.assert_close(changed[0, 0], self.base[0, 0], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(changed[0, 1], self.base[0, 1]))


# ---------------------------------------------------------------- 4. 非有限值


class NonFiniteReportTest(unittest.TestCase):
    def test_nan_output_is_listed_instead_of_silently_dropped(self) -> None:
        head_dim, prompt_len = 3, 5
        scale = 1.0 / math.sqrt(head_dim)
        generator = torch.Generator().manual_seed(7)
        k = torch.randn(1, prompt_len, head_dim, generator=generator)
        v = torch.randn(1, prompt_len, head_dim, generator=generator)
        q = torch.randn(1, 1, head_dim, generator=generator)
        good = ORACLE.dense_reference(q, k, v, [4], scale, 1, 1)
        bad_out = good.clone()
        bad_out[0, 0, 0] = float("nan")

        with tempfile.TemporaryDirectory() as tmp:
            capture = write_npz(
                Path(tmp) / "capture.npz",
                build_arrays(
                    prompt_len=prompt_len,
                    scale=scale,
                    num_heads=1,
                    num_kv_heads=1,
                    head_dim=head_dim,
                    q={1: q, 2: q},
                    out={1: bad_out, 2: good},
                    positions={1: [3], 2: [4]},
                    k={3: k},
                    v={3: v},
                ),
            )
            report_path = Path(tmp) / "report.json"
            proc = run_cli(capture, report_path)
            self.assertEqual(proc.returncode, 3, proc.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        failures = [entry for entry in report["non_finite"] if entry["tensor"] == "out"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            {key: failures[0][key] for key in ("scope", "layer", "step", "position")},
            {"scope": "comparison", "layer": 3, "step": 1, "position": 3},
        )
        self.assertEqual(failures[0]["non_finite"], 1)

        by_step = {item["step"]: item for item in report["comparisons"]}
        self.assertFalse(by_step[1]["finite"])
        for metric in ("max_abs_err", "rms_err", "out_norm", "ref_norm", "rel_err"):
            self.assertIsNone(by_step[1][metric])
        # 同批次的有限比较不能被非有限值掩盖：仍给出完整指标与统计。
        self.assertTrue(by_step[2]["finite"])
        self.assertIsInstance(by_step[2]["max_abs_err"], float)
        # 该步观测就是本实现算出的参考 → 误差恰为 0，且 0 必须仍按有限值处理。
        self.assertEqual(by_step[2]["max_abs_err"], 0.0)
        self.assertEqual(report["summary"]["status"], "non_finite")
        self.assertEqual(report["summary"]["non_finite_count"], 1)
        self.assertEqual(report["summary"]["finite_comparisons"], 1)
        self.assertEqual(report["per_layer"]["3"]["finite"], 1)
        self.assertEqual(report["per_step"]["1"]["finite"], 0)
        self.assertEqual(report["per_step"]["2"]["finite"], 1)


# ---------------------------------------------------------------- 5. 元数据缺失


class MissingMetadataTest(unittest.TestCase):
    def _capture(self, tmp: str, drop: tuple[str, ...]) -> Path:
        head_dim, prompt_len = 2, 3
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.ones(1, 1, head_dim)
        out = ORACLE.dense_reference(q, torch.ones(1, prompt_len, head_dim), torch.ones(1, prompt_len, head_dim), [2], scale, 1, 1)
        return write_npz(
            Path(tmp) / "capture.npz",
            build_arrays(
                prompt_len=prompt_len,
                scale=scale,
                num_heads=1,
                num_kv_heads=1,
                head_dim=head_dim,
                q={1: q},
                out={1: out},
                positions={1: [2]},
                k={3: torch.ones(1, prompt_len, head_dim)},
                v={3: torch.ones(1, prompt_len, head_dim)},
            ),
            drop=drop,
        )

    def test_missing_scale_exits_2_with_clear_message_and_no_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = self._capture(tmp, ("scale",))
            report_path = Path(tmp) / "report.json"
            proc = run_cli(capture, report_path)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("scale", proc.stderr)
            self.assertIn("必需元数据缺失", proc.stderr)
            self.assertFalse(report_path.exists())
            self.assertEqual(proc.stdout, "")

    def test_multiple_missing_keys_are_all_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = self._capture(tmp, ("scale", "prompt_len"))
            proc = run_cli(capture, Path(tmp) / "report.json")
            self.assertEqual(proc.returncode, 2)
            self.assertIn("scale", proc.stderr)
            self.assertIn("prompt_len", proc.stderr)


# ---------------------------------------------------------------- 6. 独立性守卫


class IndependenceGuardTest(unittest.TestCase):
    """结构守卫（不作为数值证据）：oracle 不得依赖候选侧的代码或符号。"""

    def setUp(self) -> None:
        self.source = ORACLE_PATH.read_text(encoding="utf-8")

    def test_source_avoids_candidate_symbols(self) -> None:
        lowered = self.source.lower()
        for token in FORBIDDEN_TOKENS:
            self.assertNotIn(token, lowered, f"oracle 源码出现了候选侧符号 {token!r}")

    def test_source_imports_no_project_package(self) -> None:
        imported: list[str] = []
        for node in ast.walk(ast.parse(self.source)):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "oracle 不应使用相对 import")
                imported.append(node.module or "")
        self.assertIn("torch", imported)
        for name in imported:
            self.assertNotIn(name.split(".")[0], FORBIDDEN_IMPORT_ROOTS, f"oracle 不应 import {name}")


# ---------------------------------------------------------------- 7. 端到端


class CliEndToEndTest(unittest.TestCase):
    def test_report_metrics_and_single_line_stdout(self) -> None:
        num_heads, num_kv_heads, head_dim, prompt_len = 4, 2, 8, 6
        positions = [5]
        scale = 1.0 / math.sqrt(head_dim)
        generator = torch.Generator().manual_seed(2026)
        q = torch.randn(num_heads, len(positions), head_dim, generator=generator)
        k = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)
        v = torch.randn(num_kv_heads, prompt_len, head_dim, generator=generator)

        # 独立实现（float64 逐 head）算参考，再按 BF16 落位当作模型侧观测。
        ref64 = explicit_expansion_dense(
            q.double(),
            k.double(),
            v.double(),
            positions,
            scale,
            kv_head_of=lambda head: head // (num_heads // num_kv_heads),
        )
        observed = bf16_round(ref64.float())

        with tempfile.TemporaryDirectory() as tmp:
            arrays = build_arrays(
                prompt_len=prompt_len,
                scale=scale,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q={1: q},
                out={1: observed},
                positions={1: positions},
                k={3: k},
                v={3: v},
            )
            # 张量按 FP16 存储 → oracle 必须先升到 FP32；元数据/positions 不在该清单里。
            for name in list(arrays):
                if name.startswith(("k_prefill_", "v_prefill_", "q_step", "out_step")):
                    arrays[name] = arrays[name].astype(np.float16)
            capture = write_npz(Path(tmp) / "capture.npz", arrays)
            report_path = Path(tmp) / "report.json"
            proc = run_cli(capture, report_path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        lines = proc.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("max_abs_err_max=", lines[0])
        self.assertIn("non_finite=0", lines[0])

        self.assertEqual(report["summary"]["status"], "ok")
        self.assertEqual(report["summary"]["comparisons"], 1)
        comparison = report["comparisons"][0]
        self.assertEqual(
            {key: comparison[key] for key in ("layer", "step", "position", "keys_used", "finite")},
            {"layer": 3, "step": 1, "position": 5, "keys_used": 6, "finite": True},
        )
        for metric in ("max_abs_err", "rms_err", "out_norm", "ref_norm", "rel_err"):
            self.assertIsInstance(comparison[metric], float)
        # BF16 舍入的误差量级：大于 0 且远小于 1（不是"全零假通过"也不是离谱量级）。
        self.assertGreater(comparison["max_abs_err"], 0.0)
        self.assertLess(comparison["max_abs_err"], 0.05)
        self.assertAlmostEqual(
            comparison["rel_err"],
            comparison["max_abs_err"] / max(comparison["out_norm"], 1e-12),
            places=12,
        )
        self.assertEqual(report["per_layer"]["3"]["max_abs_err"]["max"], comparison["max_abs_err"])
        self.assertEqual(report["metadata"]["scale"], scale)
        self.assertEqual(report["metadata"]["scale_source"], "capture")
        self.assertEqual(report["numerics"]["scale_source"], "capture")
        self.assertFalse(report["numerics"]["scale_is_approximate"])
        self.assertFalse(report["numerics"]["rope_scale_reapplied"])
        self.assertEqual(report["numerics"]["compute_dtype"], "float32")
        self.assertEqual(report["numerics"]["dtype_name"], "bfloat16")
        self.assertEqual(
            report["numerics"]["upcast_to_float32"],
            ["k_prefill_L3", "out_step1_L3", "q_step1_L3", "v_prefill_L3"],
        )

    def test_declared_derived_scale_is_flagged_as_known_approximation(self) -> None:
        head_dim, prompt_len = 2, 3
        scale = 1.0 / math.sqrt(head_dim)
        ones = torch.ones(1, prompt_len, head_dim)
        q = torch.ones(1, 1, head_dim)
        arrays = build_arrays(
            prompt_len=prompt_len,
            scale=scale,
            num_heads=1,
            num_kv_heads=1,
            head_dim=head_dim,
            q={1: q},
            out={1: ORACLE.dense_reference(q, ones, ones, [2], scale, 1, 1)},
            positions={1: [2]},
            k={3: ones},
            v={3: ones},
            scale_source="derived_head_dim**-0.5",
        )
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            proc = run_cli(write_npz(Path(tmp) / "capture.npz", arrays), report_path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        # derived 的 scale 仍可用（不拒绝），但必须在报告里标成已知近似来源。
        self.assertEqual(report["metadata"]["scale_source"], "derived_head_dim**-0.5")
        self.assertEqual(report["numerics"]["scale_source"], "derived_head_dim**-0.5")
        self.assertTrue(report["numerics"]["scale_is_approximate"])
        self.assertTrue(
            any("scale" in warning and "derived" in warning for warning in report["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
