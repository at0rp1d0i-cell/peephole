"""只读探针：把 GPUModelRunner 算出的 `_kernel_block_sizes` 抄一份到文件。

用途：vLLM v0.29.0 的日志与 /metrics 都不输出 kernel block size，而它由
`vllm/v1/worker/utils.py: prepare_kernel_block_sizes` → `select_common_block_size` 决定：
**Case 1：manager 块大小被所有 backend 支持时直接返回它**（不是取 backend 声明的最小值）。
因此不能从 `FLASH_ATTN.get_supported_kernel_block_sizes() == [MultipleOf(16)]` 推断出 16——
784 本身是 16 的倍数，Case 1 命中，实际取 784、不做拆分。

为什么用 sitecustomize：EngineCore 跑在**子进程**里（父进程里的 monkeypatch 不会被子进程继承）。
本模块由解释器启动时自动导入，因此在任何进程（含子进程）里都能装上钩子。
行为只读：调用原方法后读取属性并写文件，不改任何返回值、不触碰 vLLM 源码。

用法见 tools/e4-kernel-block-probe.sh（设置 KBS_PROBE_OUT 后运行一次原版 serve）。
"""

from __future__ import annotations

import json
import os
import traceback

_OUT = os.environ.get("KBS_PROBE_OUT")

if _OUT:  # 只在显式请求时装钩子
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        _orig = GPUModelRunner.initialize_kv_cache

        def _initialize_kv_cache(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = _orig(self, *args, **kwargs)
            try:
                kv_cfg = getattr(self, "kv_cache_config", None)
                # V2 runner（v1/worker/gpu/model_runner.py:587）用 `kernel_block_sizes`（无下划线）；
                # 旧 runner（v1/worker/gpu_model_runner.py）用 `_kernel_block_sizes`。两者都读。
                kbs = getattr(self, "kernel_block_sizes", None)
                if kbs is None:
                    kbs = getattr(self, "_kernel_block_sizes", None)
                payload: dict[str, object] = {
                    "runner_class": type(self).__name__,
                    "kernel_block_sizes": list(kbs or []),
                }
                if kv_cfg is not None:
                    payload["kv_manager_block_sizes"] = [
                        getattr(g.kv_cache_spec, "block_size", None) for g in kv_cfg.kv_cache_groups
                    ]
                    payload["kv_group_spec_types"] = [
                        type(g.kv_cache_spec).__name__ for g in kv_cfg.kv_cache_groups
                    ]
                    payload["num_blocks"] = getattr(kv_cfg, "num_blocks", None)
                mgr = payload.get("kv_manager_block_sizes") or []
                kern = payload["kernel_block_sizes"] or []
                payload["hybrid_splitting_used"] = (
                    None if not kern else any(k != m for k, m in zip(kern, mgr))
                )
                # 允许拿到全部后端能力（用于复核 Case 1 判据）
                groups = getattr(self, "attn_groups", None)
                if groups:
                    payload["backends"] = [
                        {
                            "group_id": gid,
                            "backend": getattr(g.backend, "__name__", str(g.backend)),
                            "supported_kernel_block_sizes": [
                                f"MultipleOf({s.base})" if hasattr(s, "base") else s
                                for s in g.backend.get_supported_kernel_block_sizes()
                            ],
                            "supports_manager_block_size": [
                                g.backend.supports_block_size(m) for m in mgr
                            ],
                        }
                        for gid, gs in enumerate(groups)
                        for g in gs
                    ]
                with open(_OUT, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
            except Exception:  # 探针失败不能影响服务
                with open(_OUT + ".error", "w", encoding="utf-8") as fh:
                    traceback.print_exc(file=fh)
            return result

        GPUModelRunner.initialize_kv_cache = _initialize_kv_cache
    except Exception:
        try:
            with open(_OUT + ".error", "w", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
        except Exception:
            pass
