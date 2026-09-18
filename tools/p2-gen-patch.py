#!/usr/bin/env python3
"""阶段 05 窄 patch 生成器：从 pin 源码生成可部署、可撤销的 patched 文件与清单。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-gen-patch.py            # 生成 vllm-patch/patched + manifest.json
    "$ATTNVIEW_PYTHON" tools/p2-gen-patch.py --check     # 只校验（不改写）

约束：只读 pin 源码；每个替换必须**恰好命中一次**，否则报错退出（不做模糊匹配）。
新文件（adapter/engine 模块）不进替换表，由 `files/` 目录原样部署。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN_ROOT = REPO / "vllm"  # vLLM 源码 checkout（pin 98dff2a8）
PATCH_ROOT = REPO / "vllm-patch"
NEW_FILES_DIR = PATCH_ROOT / "files"

# 需要部署的新文件（相对 vLLM 包根与安装根的路径一致）：(files 下的相对路径, 部署目标相对路径)
#: `dest` 是相对**包根**（即相对 `.../site-packages/vllm/`）的路径，不含 `vllm/` 前缀。
NEW_FILES = (
    ("vllm/v1/worker/gpu/attnview_adapter.py", "v1/worker/gpu/attnview_adapter.py"),
    ("vllm/v1/engine/attnview_engine.py", "v1/engine/attnview_engine.py"),
)

# 需要随部署安装的纯逻辑包文件（源码真身在 src/attnview/，部署时拷贝到 site-packages/attnview/）
PACKAGE_FILES = tuple(
    f"src/attnview/{name}"
    for name in (
        "__init__.py",
        "decode.py",
        "parser.py",
        "readview.py",
        "gpukv.py",
        "state.py",
        "step_plan.py",
    )
)

# --------------------------------------------------------------------------- #
# 替换表：(相对 vLLM 包根的路径, 旧文本, 新文本)
# --------------------------------------------------------------------------- #

EDITS: list[tuple[str, str, str]] = [
    # --- 1) SchedulerOutput：新增可选的原生类型计划字段 ----------------------- #
    (
        "v1/core/sched/output.py",
        """    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False
""",
        """    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # attnview: 每步读取视图计划（req_id -> 原生类型载荷）。None 时行为与原版一致。
    # 由 EngineCore 侧 AttnViewEngine 在输出解析后构造、下次 schedule 后注入；
    # 仅内部请求携带，公共 API 不暴露该字段。
    da_step_plans: dict[str, dict] | None = None
""",
    ),
    # --- 2) EngineCore：构造适配层并接合每步 ---------------------------------- #
    (
        "v1/engine/core.py",
        """        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
""",
        """        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # attnview: EngineCore 侧解析层。普通请求（无载荷）完全不经过它；
        # 只有携带 extra_args["attnview"] 的内部请求才会启用，并在不支持配置下显式拒绝。
        from vllm.v1.engine.attnview_engine import AttnViewEngine, make_token_text_of

        self.attnview = AttnViewEngine(
            vllm_config,
            self.scheduler,
            geometry_fetcher=lambda: self.model_executor.collective_rpc(
                "attnview_geometry", single_value=True
            ),
            # 真实 tokenizer 的懒加载取值函数：普通请求不会触发，DA 请求首 token 即可解析。
            token_text_of=make_token_text_of(vllm_config),
        )
""",
    ),
    (
        "v1/engine/core.py",
        """            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
""",
        """            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            # attnview: 批次队列路径下本适配层不解析也不下推计划 —— 出现内部请求即拒绝，
            # 绝不静默退化为原版读取。
            self.attnview.refuse_unsupported_step(scheduler_output)
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
""",
    ),
    (
        "v1/engine/core.py",
        """        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
""",
        """        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        # attnview: 在执行**之前**处理本步抢占、登记新请求（载荷/布局校验与门禁）、并注入本步读取计划。
        # 位置必须在 execute_model 之前：配置/布局错误要 fail-fast，且"首步即终结"的请求此刻仍在账本里。
        self.attnview.on_step_scheduled(scheduler_output)
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
""",
    ),
    (
        "v1/engine/core.py",
        """    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        self.structured_output_manager.clear_backend()""",
        """    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        # attnview: flush 增量 UTF-8 残留字节并释放请求级协议状态（正常/异常退出都要走到）。
        if getattr(self, "attnview", None) is not None:
            self.attnview.shutdown()
        self.structured_output_manager.clear_backend()""",
    ),
    (
        "v1/engine/core.py",
        """        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
""",
        """        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)
        # attnview: 本步解析（第 t 步解析、第 t+1 步生效）。位置固定在 update_from_output
        # 之后、下一次 schedule() 之前：此时已终结/取消的请求已从调度器移除，且本步输出
        # 已并入账本；因此不会被同一步输出重建状态。
        self.attnview.on_step_outputs(scheduler_output, model_output, engine_core_outputs)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
""",
    ),
    # --- 3) build_attn_metadata：只对全注意力组替换读 metadata ----------------- #
    (
        "v1/worker/gpu/attn_utils.py",
        """    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
) -> dict[str, Any]:""",
        """    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
    da_fa_override: Any | None = None,
) -> dict[str, Any]:""",
    ),
    (
        "v1/worker/gpu/attn_utils.py",
        """    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
""",
        """    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # attnview: 只对**全注意力组**替换读取侧 metadata（块表与有效读长度）。
        # 构造新对象、不改写跨组共享的张量；写入 slot、query_start_loc、GDN/Mamba 字段不动。
        da_group_active = (
            da_fa_override is not None
            and i == da_fa_override.group_index
            and not for_cudagraph_capture
        )
        if da_group_active:
            block_table = da_fa_override.block_table
            group_seq_lens = da_fa_override.seq_lens
            group_seq_lens_cpu_upper_bound = da_fa_override.seq_lens_cpu_upper_bound
            group_max_seq_len = da_fa_override.max_seq_len
        else:
            group_seq_lens = seq_lens
            group_seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound
            group_max_seq_len = max_seq_len
""",
    ),
    (
        "v1/worker/gpu/attn_utils.py",
        """        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,""",
        """        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=group_seq_lens,
            seq_lens_cpu_upper_bound=group_seq_lens_cpu_upper_bound,
            max_seq_len=group_max_seq_len,""",
    ),
    # --- 4) 参数通道：model_state.prepare_attn 各实现 -------------------------- #
    (
        "v1/worker/gpu/model_states/interface.py",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:""",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        da_fa_override: Any | None = None,
    ) -> dict[str, Any]:""",
    ),
    (
        "v1/worker/gpu/model_states/mamba_hybrid.py",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:""",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        da_fa_override: Any | None = None,
    ) -> dict[str, Any]:""",
    ),
    (
        "v1/worker/gpu/model_states/mamba_hybrid.py",
        """            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )""",
        """            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            da_fa_override=da_fa_override,
        )""",
    ),
    (
        "v1/worker/gpu/model_states/default.py",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:""",
        """        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        da_fa_override: Any | None = None,
    ) -> dict[str, Any]:
        # attnview: 本阶段只在 MambaHybridModelState（目标模型的真实入口）接线读取覆写；
        # 纯注意力路径不接受该覆写 —— 出现即显式拒绝，绝不留"接收但不消费"的静默分支。
        if da_fa_override is not None:
            raise RuntimeError(
                "attnview: DefaultModelState 不在本阶段支持范围（目标模型走 MambaHybridModelState）："
                "拒绝该请求的 DA 读取覆写，不静默吞载荷"
            )""",
    ),
    # --- 5) model_runner：本步构造覆写并传入（dummy 一律不落位） --------------- #
    (
        "v1/worker/gpu/model_runner.py",
        """            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                self.kv_cache_config,
                # FULL replay reads capture-time metadata buffers. Re-stage them
                # from the zeroed dummy block tables instead of retaining state
                # indices from the previous real batch.
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )""",
        """            # attnview: 只有内部请求（载荷）且非 dummy 时才构造读取覆写；否则完全走原版。
            da_fa_override = (
                attnview_adapter.fa_override_for_step(
                    self, scheduler_output, input_batch, block_tables
                )
                if (not dummy_run and getattr(scheduler_output, "da_step_plans", None))
                else None
            )
            prepare_attn_kwargs = (
                {"da_fa_override": da_fa_override} if da_fa_override is not None else {}
            )
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                self.kv_cache_config,
                # FULL replay reads capture-time metadata buffers. Re-stage them
                # from the zeroed dummy block tables instead of retaining state
                # indices from the previous real batch.
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
                **prepare_attn_kwargs,
            )""",
    ),
    (
        "v1/worker/gpu/model_runner.py",
        """        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.""",
        """        # attnview 校准: 测试专用完整 logits 捕获 —— 必须在 `compute_logits` **之后**、
        # grammar/sampler 就地改写 **之前**（否则拿到的是被掩码/采样器改过的张量；
        # 归一化/截断后的 top-k 无法用于全词表误差与尾部非有限值检查）。
        # 未设置 ATTNVIEW_CALIB_LOGITS 时完全不介入（普通请求零影响）。
        attnview_adapter.calibration_capture_logits(logits, input_batch)

        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.""",
    ),
    (
        "v1/worker/gpu/model_runner.py",
        """        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
""",
        """        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        # attnview 校准: 测试专用 token 强制点 —— 必须在**采样之后**、
        # PP broadcast / AsyncOutput / postprocess_sampled **之前**：这样 worker 历史
        # （postprocess_sampled 读的就是这块内存）与送往宿主的 token 是同一个值，不会分叉。
        # 未设置 ATTNVIEW_CALIB_FORCE 时完全不介入（普通请求零影响）。
        attnview_adapter.calibration_force_tokens(
            sampler_output, input_batch.req_ids, num_sampled
        )
""",
    ),
    (
        "v1/worker/gpu/model_runner.py",
        """from vllm.v1.worker.gpu.block_table import BlockTables""",
        """from vllm.v1.worker.gpu import attnview_adapter
from vllm.v1.worker.gpu.block_table import BlockTables""",
    ),
    # --- 6) worker RPC：让 EngineCore 一次性核对实际几何 ---------------------- #
    (
        "v1/worker/gpu_worker.py",
        """        if not self._weight_update_is_draft:
            self.model_runner.reset_lora_state()

    def shutdown(self) -> None:""",
        """        if not self._weight_update_is_draft:
            self.model_runner.reset_lora_state()

    def attnview_geometry(self) -> dict:
        \"\"\"attnview: 返回运行期 KV 几何，供 EngineCore 一次性核对（不使用默认块大小）。\"\"\"
        from vllm.v1.worker.gpu import attnview_adapter

        geometry, group_block_sizes = attnview_adapter.derive_geometry(self.model_runner)
        return {
            "kernel_block_size": geometry.kernel_block_size,
            "num_kv_groups": geometry.num_kv_groups,
            "fa_group_index": geometry.fa_group_index,
            "blocks_per_kv_block": geometry.blocks_per_kv_block,
            "max_model_len": geometry.max_model_len,
            "group_block_sizes": group_block_sizes,
            "flash_attn_version": getattr(
                getattr(self.model_runner.vllm_config, "attention_config", None),
                "flash_attn_version",
                None,
            ),
        }

    def shutdown(self) -> None:""",
    ),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    if not PIN_ROOT.is_dir():
        print(f"缺少 vLLM 源码 checkout：{PIN_ROOT}", file=sys.stderr)
        return 2

    patched_root = PATCH_ROOT / "patched"
    if not args.check:
        patched_root.mkdir(parents=True, exist_ok=True)

    per_file: dict[str, list[tuple[str, str]]] = {}
    for rel, old, new in EDITS:
        per_file.setdefault(rel, []).append((old, new))

    manifest: dict = {
        "generated_cst": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800"),
        "pin_commit": (PIN_ROOT / ".git" / "HEAD").read_text().strip()
        if (PIN_ROOT / ".git" / "HEAD").exists()
        else None,
        "edits": {},
        "new_files": [],
        "package_files": [],
    }

    for rel, pairs in sorted(per_file.items()):
        src = PIN_ROOT / "vllm" / rel
        if not src.is_file():
            print(f"缺少源文件：{src}", file=sys.stderr)
            return 2
        text = src.read_text()
        for idx, (old, new) in enumerate(pairs):
            hits = text.count(old)
            if hits != 1:
                print(
                    f"替换未恰好命中一次：{rel} 第 {idx + 1} 条 命中 {hits} 次",
                    file=sys.stderr,
                )
                return 3
            text = text.replace(old, new, 1)
        dest = patched_root / rel
        if not args.check:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
        manifest["edits"][rel] = {
            "replacements": len(pairs),
            "pre_sha256": sha256_file(src),
            "post_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "patched_path": str(dest.relative_to(REPO)),
        }

    for src_rel, dest_rel in NEW_FILES:
        src = NEW_FILES_DIR / src_rel
        if not src.is_file():
            print(f"缺少新文件：{src}", file=sys.stderr)
            return 2
        manifest["new_files"].append(
            {
                "src": str(src.relative_to(REPO)),
                "dest": dest_rel,
                "sha256": sha256_file(src),
            }
        )
    for rel in PACKAGE_FILES:
        src = REPO / rel
        if not src.is_file():
            print(f"缺少包文件：{src}", file=sys.stderr)
            return 2
        manifest["package_files"].append({"file": rel, "sha256": sha256_file(src)})

    out = PATCH_ROOT / "manifest.json"
    if not args.check:
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"生成完成：{out.relative_to(REPO)}（{len(manifest['edits'])} 个文件被改、"
              f"{len(manifest['new_files'])} 个新文件、{len(manifest['package_files'])} 个包文件）")
    else:
        print("校验通过（未写文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
