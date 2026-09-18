"""worker 侧覆写入口（`fa_override_for_step`）的真实路径 CPU 测试。

要点（本地复核要求）：
- **不能**只测纯函数与 AST：必须调用真实入口 `fa_override_for_step`，用 CPU 张量跑通
  「镜像 canonical 行 → 设备侧 gather 可见块 → 右侧填充 → 写 seq_lens」；
- **配置门禁必须在分配覆写缓冲之前**生效：同步调度但开启前缀缓存/图捕获/投机时，
  带载荷的请求必须被拒绝，且不得留下任何缓冲。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from attnview.step_plan import (  # noqa: E402
    DaRequestConfig,
    Geometry,
    UnsupportedConfig,
    build_step_plan,
)

ADAPTER_PATH = REPO / "vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py"

B = 784
PROMPT_LEN = 6272
FA_PHYSICAL = [41, 7, 90, 12, 63, 28, 55, 88, 102]  # 逻辑块 0..8 的物理块
MAX_REQS = 2  # 批内请求行数（与真实 `gather_block_tables` 的输出行数一致）


def load_adapter():
    spec = importlib.util.spec_from_file_location("attnview_adapter_under_test", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` 需要模块已在 sys.modules 里（否则 dataclasses 取 cls.__module__ 会失败）
    sys.modules["attnview_adapter_under_test"] = module
    spec.loader.exec_module(module)
    return module


def make_config(*, prefix_caching=False, async_scheduling=False, cudagraph=None, spec=None):
    from vllm.config import CUDAGraphMode  # noqa: PLC0415

    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        max_concurrent_batches=1,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE if cudagraph is None else cudagraph),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        speculative_config=spec,
        model_config=SimpleNamespace(max_model_len=8192),
    )


def make_kv_cache_config():
    from vllm.v1.kv_cache_interface import MambaSpec  # noqa: PLC0415

    mamba = MambaSpec(block_size=B, shapes=(), dtypes=())
    attention_like = SimpleNamespace(block_size=B)  # 非 MambaSpec ⇒ 全注意力组
    groups = [SimpleNamespace(kv_cache_spec=s) for s in (mamba, mamba, mamba, attention_like)]
    return SimpleNamespace(kv_cache_groups=groups)


def plan_payload() -> dict:
    payload = {
        "protocol": "v1.0",
        "prompt_len": PROMPT_LEN,
        "segment_spans": [[784, 2352], [2352, 3920], [3920, 5488]],
        "local_window_span": [4000, 6272],
        "sink_span": [0, 16],
    }
    geometry = Geometry(kernel_block_size=B, num_kv_groups=4, fa_group_index=3,
                        blocks_per_kv_block=1, max_model_len=8192)
    plan = build_step_plan(
        req_id="r1",
        mode="local",
        refs=(),
        config=DaRequestConfig.from_extra_args({"attnview": payload}),
        geometry=geometry,
        canonical_blocks=FA_PHYSICAL,
        attention_kv_len=PROMPT_LEN + 1,  # post：含本步写入的当前 token
        effect_step=1,
    )
    return plan.as_payload()


class OverrideEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_adapter()
        cls.mod.reset_metering()

    def make_runner(self, config):
        canonical_fa = torch.zeros((MAX_REQS, 11), dtype=torch.int32)
        canonical_fa[0, : len(FA_PHYSICAL)] = torch.tensor(FA_PHYSICAL, dtype=torch.int32)
        canonical_fa[1, :2] = torch.tensor([903, 904], dtype=torch.int32)  # 非 DA 行（普通请求）
        block_tables = (canonical_fa.clone(), canonical_fa.clone(), canonical_fa.clone(), canonical_fa.clone())
        num_blocks = np.zeros((4, MAX_REQS), dtype=np.int32)
        num_blocks[3, :] = len(FA_PHYSICAL)
        runner = SimpleNamespace(
            vllm_config=config,
            kv_cache_config=make_kv_cache_config(),
            block_tables=SimpleNamespace(num_blocks=SimpleNamespace(np=num_blocks)),
        )
        return runner, block_tables

    def make_input_batch(self):
        # 两行：row 0 = DA 请求 r1，row 1 = 普通请求 r2（无计划 → 必须保持 canonical）
        return SimpleNamespace(
            req_ids=["r1", "r2"],
            idx_mapping_np=np.array([0, 1], dtype=np.intp),
            seq_lens=torch.tensor([PROMPT_LEN + 1, 500], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([PROMPT_LEN + 1, 500], dtype=torch.int32),
        )

    def scheduler_output(self, plans=None):
        return SimpleNamespace(da_step_plans=plans)

    def test_valid_config_builds_override_on_cpu_tensors(self) -> None:
        self.mod.reset_metering()
        runner, block_tables = self.make_runner(make_config())
        override = self.mod.fa_override_for_step(
            runner, self.scheduler_output({"r1": plan_payload()}), self.make_input_batch(), block_tables
        )
        self.assertIsNotNone(override)
        self.assertEqual(override.group_index, 3)
        self.assertEqual(override.rows_with_override, (0,))
        row = override.block_table[0].tolist()
        # 可见逻辑块 [0,5,6,7,8] → 物理 [41,28,55,88,102]，右侧填充复用最后一块（绝不放 -1）
        self.assertEqual(row[:5], [41, 28, 55, 88, 102])
        self.assertEqual(row[5:], [102] * 6)
        self.assertNotIn(-1, row)
        self.assertEqual(int(override.seq_lens[0]), 3137)  # 压缩可见长度（不含被排除块）
        self.assertEqual(int(override.seq_lens_cpu_upper_bound[0]), 3137)
        self.assertEqual(override.max_seq_len, PROMPT_LEN + 1)  # 并入 canonical 上界
        # 非 DA 行（r2）保持 canonical 语义：块表与 seq_lens 都未被改写
        self.assertEqual(override.block_table[1].tolist()[:2], [903, 904])
        self.assertEqual(int(override.seq_lens[1]), 500)
        self.assertEqual(int(override.seq_lens_cpu_upper_bound[1]), 500)
        meter = self.mod.metering_snapshot()
        self.assertGreaterEqual(meter["h2d_calls"], 1)
        self.assertGreaterEqual(meter["device_index_select_calls"], 1)
        self.assertEqual(meter["host_reads_of_device_tensors"], 0, "稳态不得读回 GPU 张量")

    def test_gate_refuses_before_allocating_buffers(self) -> None:
        cases = {
            "prefix_caching": make_config(prefix_caching=True),
            "async_scheduling": make_config(async_scheduling=True),
            "cudagraph_full": make_config(cudagraph="FULL"),
            "speculative": make_config(spec=SimpleNamespace(method="ngram")),
        }
        for name, config in cases.items():
            with self.subTest(case=name):
                runner, block_tables = self.make_runner(config)
                with self.assertRaises(UnsupportedConfig):
                    self.mod.fa_override_for_step(
                        runner, self.scheduler_output({"r1": plan_payload()}),
                        self.make_input_batch(), block_tables,
                    )
                self.assertFalse(
                    hasattr(runner, "_attnview_buffers"),
                    "拒绝必须发生在分配覆写缓冲之前",
                )

    def test_no_plans_is_passthrough_without_gate(self) -> None:
        # 普通请求（无计划）不应触碰门禁或分配缓冲：原版路径零开销
        runner, block_tables = self.make_runner(make_config(prefix_caching=True))
        self.assertIsNone(
            self.mod.fa_override_for_step(
                runner, self.scheduler_output(None), self.make_input_batch(), block_tables
            )
        )
        self.assertFalse(hasattr(runner, "_attnview_buffers"))

    def test_visible_block_beyond_allocated_is_refused(self) -> None:
        runner, block_tables = self.make_runner(make_config())
        runner.block_tables.num_blocks.np[3, :] = 8  # 只分配 0..7，而可见块含 8
        with self.assertRaises(RuntimeError):
            self.mod.fa_override_for_step(
                runner, self.scheduler_output({"r1": plan_payload()}), self.make_input_batch(), block_tables
            )

    def test_persistent_buffer_shape_is_enforced(self) -> None:
        runner, block_tables = self.make_runner(make_config())
        override = self.mod.fa_override_for_step(
            runner, self.scheduler_output({"r1": plan_payload()}), self.make_input_batch(), block_tables
        )
        self.assertIsNotNone(override)
        # 第二次调用沿用同一缓冲（地址/形状稳定，I5）：形状变化必须报错而不是静默重建
        smaller = torch.zeros((MAX_REQS, 5), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            self.mod.fa_override_for_step(
                runner,
                self.scheduler_output({"r1": plan_payload()}),
                self.make_input_batch(),
                (smaller, smaller, smaller, smaller),
            )


if __name__ == "__main__":
    unittest.main()
