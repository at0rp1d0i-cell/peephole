"""worker 侧覆写入口（`fa_override_for_step`）的真实路径 CPU 测试。

要点（SUP-004-R2 §3/§4 与顺手项）：
- 几何必须来自 **runner 的实际对象**（`block_tables.block_sizes` / `kernel_block_sizes` /
  `blocks_per_kv_block` + `runner.kernel_block_sizes`），并互相核对；"各组 manager 块大小相等"
  之类的代理推断必须消失：split 例（manager 784 / kernel 16 / ratio 49）要**拒绝**，不得报 784/1；
- 只接受真实 `FullAttentionSpec` 目标组 + 支持集内的后端（FA2）+ 单卡 TP1；
- 门禁按 **runner 实例**隔离：同一进程内先校验合法 A、再校验非法 B，B 必须抛错（不重载模块）；
- 消费边界显式核对：批内单活跃请求、单 query(decode) 步；
- metadata 对照用**真实调用 + 捕获 builder 输入**，并按对象身份断言"只有 FA 组换、其它组不换"。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from _support import REPO
from _support import load_module as _load_module
from attnview.step_plan import (  # noqa: E402
    DaRequestConfig,
    Geometry,
    UnsupportedConfig,
    build_step_plan,
)

ADAPTER_PATH = REPO / "vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py"
ATTN_UTILS_PATH = REPO / "vllm-patch/patched/v1/worker/gpu/attn_utils.py"

B = 784  # manager 块大小（该配置实测值，不作为常量假设；kernel 同值 ⇒ ratio=1）
K = 784
PROMPT_LEN = 6272
FA_PHYSICAL = [41, 7, 90, 12, 63, 28, 55, 88, 102]  # 逻辑块 0..8 的物理块
MAX_REQS = 1  # 本阶段单活跃请求


def load_module(name: str, path: Path):
    """同名包装：`_support.load_module` 的名字与本文件被包装者相同，故以别名导入。"""
    return _load_module(name, path)


def make_config(*, prefix_caching=False, async_scheduling=False, cudagraph=None, spec=None, tp=1,
                flash_attn_version=2):
    from vllm.config import CUDAGraphMode  # noqa: PLC0415

    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        max_concurrent_batches=1,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE if cudagraph is None else cudagraph
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        speculative_config=spec,
        model_config=SimpleNamespace(max_model_len=8192),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, world_size=tp),
        attention_config=SimpleNamespace(flash_attn_version=flash_attn_version),
    )


def make_kv_cache_config(*, manager_sizes=(B, B, B, B), spec_sizes=None):
    """3×MambaSpec(GDN) + 1×FullAttentionSpec —— 目标模型的真实组结构。"""
    from vllm.v1.kv_cache_interface import (  # noqa: PLC0415
        FullAttentionSpec,
        MambaSpec,
    )

    sizes = tuple(manager_sizes if spec_sizes is None else spec_sizes)
    mamba = [MambaSpec(block_size=s, shapes=(), dtypes=()) for s in sizes[:3]]
    fa = FullAttentionSpec(
        block_size=sizes[3], num_kv_heads=4, head_size=128, dtype=torch.float16
    )
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in (*mamba, fa)]
    )


class _FakeBackend:
    def __init__(self, name: str = "FLASH_ATTN") -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


def plan_payload() -> dict:
    payload = {
        "protocol": "v1.0",
        "prompt_len": PROMPT_LEN,
        "segment_spans": [[784, 2352], [2352, 3920], [3920, 5488]],
        "local_window_span": [4000, 6272],
        "sink_span": [0, 16],
    }
    geometry = Geometry(
        kernel_block_size=K, num_kv_groups=4, fa_group_index=3,
        blocks_per_kv_block=1, max_model_len=8192,
    )
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
        cls.mod = load_module("attnview_adapter_under_test", ADAPTER_PATH)
        cls.mod.reset_metering()

    # --- 夹具 ------------------------------------------------------------- #

    def make_runner(
        self,
        config,
        *,
        manager_sizes=(B, B, B, B),
        spec_sizes=None,
        kernel_sizes=(K, K, K, K),
        ratios=None,
        backend_name="FLASH_ATTN",
    ):
        ratios = (
            [m // k for m, k in zip(manager_sizes, kernel_sizes, strict=True)] if ratios is None else ratios
        )
        canonical_fa = torch.zeros((MAX_REQS, 11), dtype=torch.int32)
        canonical_fa[0, : len(FA_PHYSICAL)] = torch.tensor(FA_PHYSICAL, dtype=torch.int32)
        block_tables = tuple(canonical_fa.clone() for _ in range(4))
        num_blocks = np.zeros((4, MAX_REQS), dtype=np.int32)
        num_blocks[3, :] = len(FA_PHYSICAL)
        kv_cache_config = make_kv_cache_config(
            manager_sizes=manager_sizes, spec_sizes=spec_sizes
        )
        runner = SimpleNamespace(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            kernel_block_sizes=list(kernel_sizes),
            block_tables=SimpleNamespace(
                block_sizes=list(manager_sizes),
                kernel_block_sizes=list(kernel_sizes),
                blocks_per_kv_block=list(ratios),
                num_blocks=SimpleNamespace(np=num_blocks),
            ),
            attn_groups=[
                [SimpleNamespace(backend=_FakeBackend(backend_name),
                                 kv_cache_spec=g.kv_cache_spec)]
                for g in kv_cache_config.kv_cache_groups
            ],
        )
        return runner, block_tables

    def make_input_batch(self, *, req_ids=("r1",), seq_lens=(PROMPT_LEN + 1,)):
        rows = len(req_ids)
        return SimpleNamespace(
            req_ids=list(req_ids),
            idx_mapping_np=np.arange(rows, dtype=np.intp),
            seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(list(seq_lens), dtype=torch.int32),
            is_prefilling_np=np.zeros(rows, dtype=np.bool_),
        )

    def scheduler_output(self, plans=None, *, scheduled=None):
        return SimpleNamespace(
            da_step_plans=plans,
            num_scheduled_tokens={"r1": 1} if scheduled is None else scheduled,
        )

    # --- 正常落位 --------------------------------------------------------- #

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
        # 顺手项：max_seq_len 取**覆写后 CPU 长度向量**的最大值（不再并入 canonical 上界），
        # 单请求压缩视图下确实缩短（6273 → 3137），便于后续耗时归因。
        self.assertEqual(override.max_seq_len, 3137)
        meter = self.mod.metering_snapshot()
        self.assertGreaterEqual(meter["h2d_calls"], 1)
        self.assertGreaterEqual(meter["device_index_select_calls"], 1)
        self.assertGreaterEqual(meter["scalar_assignments"], 2)
        self.assertNotIn("host_reads_of_device_tensors", meter, "不得再留从未更新的设备读计数")

    def test_geometry_uses_real_kernel_block_size(self) -> None:
        # manager 784 / kernel 16 且 ratio 49 → 显式拒绝（不得报 784/1）
        runner, _ = self.make_runner(
            make_config(), manager_sizes=(784, 784, 784, 784),
            kernel_sizes=(16, 16, 16, 16), ratios=[49, 49, 49, 49],
        )
        with self.assertRaises(UnsupportedConfig) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("49", str(ctx.exception))

    def test_ratio_vector_mismatch_is_rejected(self) -> None:
        runner, _ = self.make_runner(make_config(), ratios=[1, 1, 1, 7])
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("blocks_per_kv_block", str(ctx.exception))

    def test_manager_block_size_mismatch_with_spec_is_rejected(self) -> None:
        runner, _ = self.make_runner(
            make_config(), manager_sizes=(784, 784, 784, 512), spec_sizes=(784, 784, 784, 784)
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("manager 块大小不一致", str(ctx.exception))

    def test_kernel_block_size_mismatch_between_tables_and_runner_is_rejected(self) -> None:
        runner, _ = self.make_runner(make_config(), kernel_sizes=(16, 16, 16, 16))
        runner.kernel_block_sizes = [784, 784, 784, 784]
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("kernel 块大小不一致", str(ctx.exception))

    def test_unsupported_backend_is_rejected(self) -> None:
        runner, _ = self.make_runner(make_config(), backend_name="TRITON_ATTN")
        with self.assertRaises(UnsupportedConfig) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("TRITON_ATTN", str(ctx.exception))

    def test_flash_attn_version_other_than_2_is_rejected(self) -> None:
        """同名 `FLASH_ATTN` 可能是 FA3/FA4（Blackwell 默认 FA4）—— 必须要求显式 FA2。"""
        for version, label in ((3, "fa3"), (4, "fa4"), (None, "platform_default")):
            with self.subTest(case=label):
                runner, _ = self.make_runner(make_config(flash_attn_version=version))
                with self.assertRaises(UnsupportedConfig) as ctx:
                    self.mod.derive_geometry(runner)
                self.assertIn("flash_attn_version", str(ctx.exception))
                self.assertIn("FA2", str(ctx.exception))

    def test_flash_attn_version_2_is_accepted(self) -> None:
        runner, _ = self.make_runner(make_config(flash_attn_version=2))
        geometry, _sizes = self.mod.derive_geometry(runner)
        self.assertEqual(geometry.fa_group_index, 3)

    def test_last_prefill_chunk_with_single_query_is_rejected(self) -> None:
        """`scheduled == 1` 不能证明 decode：最后一个 prefill chunk 也可能只有 1 个 token。"""
        runner, block_tables = self.make_runner(make_config())
        batch = self.make_input_batch()
        batch.is_prefilling_np = np.ones(1, dtype=np.bool_)
        with self.assertRaises(UnsupportedConfig) as ctx:
            self.mod.fa_override_for_step(
                runner, self.scheduler_output({"r1": plan_payload()}), batch, block_tables
            )
        self.assertIn("prefill", str(ctx.exception))
        self.assertFalse(hasattr(runner, "_attnview_buffers"), "拒绝必须早于缓冲分配")

    def test_missing_prefilling_flag_is_rejected(self) -> None:
        runner, block_tables = self.make_runner(make_config())
        batch = self.make_input_batch()
        del batch.is_prefilling_np
        with self.assertRaises(UnsupportedConfig):
            self.mod.fa_override_for_step(
                runner, self.scheduler_output({"r1": plan_payload()}), batch, block_tables
            )

    def test_multi_gpu_is_rejected(self) -> None:
        runner, _ = self.make_runner(make_config(tp=2))
        with self.assertRaises(UnsupportedConfig) as ctx:
            self.mod.derive_geometry(runner)
        self.assertIn("TP1", str(ctx.exception))

    def test_multiple_active_requests_are_rejected(self) -> None:
        runner, block_tables = self.make_runner(make_config())
        batch = self.make_input_batch(req_ids=("r1", "r2"), seq_lens=(PROMPT_LEN + 1, 500))
        with self.assertRaises(UnsupportedConfig):
            self.mod.fa_override_for_step(
                runner, self.scheduler_output({"r1": plan_payload()}), batch, block_tables
            )

    def test_prefill_step_is_rejected(self) -> None:
        runner, block_tables = self.make_runner(make_config())
        with self.assertRaises(UnsupportedConfig) as ctx:
            self.mod.fa_override_for_step(
                runner,
                self.scheduler_output({"r1": plan_payload()}, scheduled={"r1": 32}),
                self.make_input_batch(),
                block_tables,
            )
        self.assertIn("单 query", str(ctx.exception))

    # --- 门禁 ------------------------------------------------------------- #

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

    def test_gate_is_per_runner_instance(self) -> None:
        """同一进程内：先校验合法 runner A，再校验非法 runner B —— B 必须仍被拒绝。

        旧实现把校验结果放在函数的可变默认参数（模块级共享字典）里，第二次调用直接命中缓存、
        不再校验；这里**不重载模块**，直接暴露该缺陷。
        """
        valid, _ = self.make_runner(make_config())
        invalid, _ = self.make_runner(make_config(prefix_caching=True))
        self.mod.assert_supported_config(valid)          # A：合法 → 通过
        self.assertFalse(hasattr(invalid, "_attnview_config_checked"))
        with self.assertRaises(UnsupportedConfig):       # B：非法 → 必须抛错
            self.mod.assert_supported_config(invalid)
        self.mod.assert_supported_config(valid)          # A 仍可复用自身结果
        self.assertTrue(valid._attnview_config_checked)

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


class MetadataOwnershipTest(unittest.TestCase):
    """真实调用 `build_attn_metadata`，捕获各组的 builder 输入，按对象身份断言所有权。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("attnview_adapter_under_test_m", ADAPTER_PATH)
        cls.attn_utils = load_module("attnview_attn_utils_under_test", ATTN_UTILS_PATH)

    def _capture_builders(self):
        captured: dict[int, object] = {}

        class _Builder:
            def __init__(self, group_index: int) -> None:
                self.group_index = group_index

            def build(self, common_attn_metadata, **kwargs):
                captured[self.group_index] = common_attn_metadata
                return {"group": self.group_index}

        groups = []
        for i in range(4):
            builder = _Builder(i)
            groups.append(
                [
                    SimpleNamespace(
                        get_metadata_builder=lambda _i=i, _b=builder: _b,
                        layer_names=[f"layer{i}"],
                        layer_names_set={f"layer{i}"},
                    )
                ]
            )
        return groups, captured

    def test_override_replaces_only_fa_group_tensors(self) -> None:
        runner, block_tables = OverrideEntryTest().make_runner(make_config())
        override = self.mod.fa_override_for_step(
            runner,
            OverrideEntryTest().scheduler_output({"r1": plan_payload()}),
            OverrideEntryTest().make_input_batch(),
            block_tables,
        )
        self.assertIsNotNone(override)
        attn_groups, captured = self._capture_builders()
        seq_lens = torch.tensor([PROMPT_LEN + 1], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([PROMPT_LEN + 1], dtype=torch.int32)
        canonical_seq_lens_before = seq_lens.clone()
        canonical_seq_lens_cpu_before = seq_lens_cpu.clone()
        positions = torch.zeros((1,), dtype=torch.int64)
        slot_mappings = tuple(torch.zeros((1,), dtype=torch.int64) for _ in range(4))
        meta = self.attn_utils.build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=1,
            num_tokens=1,
            query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            max_query_len=1,
            seq_lens=seq_lens,
            max_seq_len=PROMPT_LEN + 1,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * 4),
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            dcp_local_seq_lens=None,
            positions=positions,
            is_prefilling=torch.zeros((1,), dtype=torch.bool),
            for_cudagraph_capture=False,
            da_fa_override=override,
        )
        self.assertEqual(len(meta), 4)
        # FA 组（下标 3）：读到的是覆写张量（对象身份）
        fa_meta = captured[3]
        self.assertIs(fa_meta.block_table_tensor, override.block_table)
        self.assertIs(fa_meta.seq_lens, override.seq_lens)
        self.assertIs(fa_meta.seq_lens_cpu_upper_bound, override.seq_lens_cpu_upper_bound)
        self.assertEqual(fa_meta.max_seq_len, override.max_seq_len)
        # 其它组（GDN/Mamba）：**没有**被换成覆写张量，仍是 canonical 值、未缩短的 max_seq_len。
        # 注意：`build_attn_metadata` 开头会做 `seq_lens = seq_lens[:num_reqs]`（pin 的切片 ⇒ 新对象），
        # 所以这里对 canonical 值断言"值相等 + 不是覆写对象"，对块表/槽位断言对象身份。
        for i in (0, 1, 2):
            other = captured[i]
            self.assertIsNot(other.seq_lens, override.seq_lens)
            self.assertTrue(torch.equal(other.seq_lens, canonical_seq_lens_before))
            self.assertIsNot(other.seq_lens_cpu_upper_bound, override.seq_lens_cpu_upper_bound)
            self.assertTrue(torch.equal(other.seq_lens_cpu_upper_bound, canonical_seq_lens_cpu_before))
            self.assertIs(other.block_table_tensor, block_tables[i])
            self.assertEqual(other.max_seq_len, PROMPT_LEN + 1)
            self.assertIs(other.slot_mapping, slot_mappings[i])
        # canonical 输入本身没有被就地改写
        self.assertTrue(torch.equal(seq_lens, canonical_seq_lens_before))
        self.assertTrue(torch.equal(seq_lens_cpu, canonical_seq_lens_cpu_before))

    def test_no_override_keeps_canonical_for_every_group(self) -> None:
        attn_groups, captured = self._capture_builders()
        seq_lens = torch.tensor([10], dtype=torch.int32)
        block_tables = tuple(torch.zeros((1, 2), dtype=torch.int32) for _ in range(4))
        slot_mappings = tuple(torch.zeros((1,), dtype=torch.int64) for _ in range(4))
        self.attn_utils.build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=1,
            num_tokens=1,
            query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            max_query_len=1,
            seq_lens=seq_lens,
            max_seq_len=10,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * 4),
            seq_lens_cpu_upper_bound=seq_lens,
            dcp_local_seq_lens=None,
            positions=torch.zeros((1,), dtype=torch.int64),
            is_prefilling=torch.zeros((1,), dtype=torch.bool),
            for_cudagraph_capture=False,
            da_fa_override=None,
        )
        for i in range(4):
            # 无覆写时每组都用 canonical（`seq_lens` 经 pin 的 `[:num_reqs]` 切片 ⇒ 值相等）
            self.assertTrue(torch.equal(captured[i].seq_lens, seq_lens))
            self.assertIs(captured[i].block_table_tensor, block_tables[i])
            self.assertEqual(captured[i].max_seq_len, 10)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
