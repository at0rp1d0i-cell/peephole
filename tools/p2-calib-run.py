#!/usr/bin/env python3
"""SUP-004 校准步骤③–⑤：单请求 ≤8192 的**原版重复**与**全 global**对照。

设计要点（对齐 `inbox/SUP-004-calibration.md`）：

- **同一 prompt、同一 token 序列**：三个臂用同一个已渲染 prompt（含协议指令）与同一条强制轨迹，
  只改变执行路径（`vanilla` = 无载荷直通；`da-global` = 真实带载荷 + `enforce_global=True`）。
  强制点在 worker 的 `self.sample()` 之后、`AsyncOutput`/`postprocess_sampled` 之前（两侧共用同一 token）。
- **完整 logits**（非 top-k）：由 worker 侧钩子保存，驱动只负责落盘与校验。
- 预算：启动 ≤15 分钟、单请求 ≤180 秒；超时保存证据并退出，不擅自加长/加批。
- 不放开 masked、自由生成与性能结论；`max_num_seqs=1`、同步调度、eager、无 prefix/投机/图。

用法：
    source env.sh && CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-calib-run.py \
        --arm vanilla --out evidence/p3-calib/run-vanilla-1 --emit-trajectory evidence/p3-calib/traj.json
    ... --arm da-global --force-trajectory evidence/p3-calib/traj.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SNAPSHOT = (
    REPO
    / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
MAX_MODEL_LEN = 8192
STARTUP_BUDGET_S = 15 * 60
REQUEST_BUDGET_S = 180
SEED = 20260918

CONTEXT = (
    "Declarative attention lets a model name the context regions it wants to read. "
    "The runtime turns those declarations into a paged KV read view, so the same weights can "
    "attend globally, focus a segment, or keep a local window plus content-free sink tokens. "
    "This paragraph exists only to give the calibration run a fixed, non-trivial context."
)
QUESTION = "In one short sentence, state what a read view is."


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt() -> tuple[object, dict]:
    """用阶段 03 的既有入口渲染协议 prompt 与其 token span（不改 prompt 合同）。"""
    from transformers import AutoTokenizer

    from attnview.prompt import render_arm
    from attnview.segmenter import build_offsets_index, segment_context

    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    token_ids, offsets = None, None
    from attnview.prompt import _tokenize_with_offsets  # noqa: PLC0415

    token_ids, offsets = _tokenize_with_offsets(tokenizer, CONTEXT)
    segments = segment_context(CONTEXT, build_offsets_index(offsets))
    arm = render_arm(
        "da",
        segments,
        QUESTION,
        CONTEXT,
        tokenizer,
        tokenizer_hash=sha256_file(SNAPSHOT / "tokenizer.json") if (SNAPSHOT / "tokenizer.json").exists() else "",
        template_hash=sha256_file(SNAPSHOT / "chat_template.jinja") if (SNAPSHOT / "chat_template.jinja").exists() else "",
        enable_thinking=False,
    )
    payload = {
        "protocol": "v1.0",
        "prompt_len": len(arm.token_ids),
        "segment_spans": [list(s) for s in arm.segment_spans],
        "local_window_span": list(arm.scaffold.local_window_span),
        "sink_span": list(arm.scaffold.sink_span),
    }
    return arm, payload


def find_fa_layers(llm) -> list:
    """定位全注意力层的 backend impl（观测点：FA 的 q/k/v 与输出，post-RoPE）。"""
    candidates = []
    for path in (
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
    ):
        try:
            candidates.append(path())
        except Exception:
            continue
    model = next((m for m in candidates if m is not None), None)
    if model is None:
        raise RuntimeError("校准: 找不到模型对象（需要 VLLM_ENABLE_V1_MULTIPROCESSING=0 的单进程模式）")
    layers = list(getattr(model, "layers", ()) or ())
    if not layers:
        raise RuntimeError("校准: 模型没有 .layers，无法安装层观测钩子（拒绝静默跳过）")
    fa_layers = []
    for index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        impl = getattr(attn, "impl", None)
        if impl is not None and "flash" in repr(type(impl)).lower():
            fa_layers.append((index, impl))
    if not fa_layers:
        raise RuntimeError("校准: 没有找到 FLASH_ATTN 实现层（目标模型应为 3×GDN + 1×FA）")
    return fa_layers


def install_hooks(fa_layers: list, capture: dict) -> None:
    def make_hook(index: int):
        def hook(module, args, output):
            q, k, v = args[0], args[1], args[2]
            step = int(capture.get("step", 0))
            record = capture.setdefault("records", {})
            key = f"step{step}"
            record.setdefault(key, {})[index] = {
                "q": q.detach().to("cpu", dtype=__import__("torch").float32),
                "k": k.detach().to("cpu", dtype=__import__("torch").float32),
                "v": v.detach().to("cpu", dtype=__import__("torch").float32),
                "out": output.detach().to("cpu", dtype=__import__("torch").float32)
                if hasattr(output, "detach")
                else None,
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
            }

        return hook

    for index, impl in fa_layers:
        impl.register_forward_hook(make_hook(index))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("vanilla", "da-global"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--emit-trajectory", type=Path)
    ap.add_argument("--force-trajectory", type=Path)
    ap.add_argument("--record-layers", action="store_true")
    ap.add_argument("--cleanup-check", action="store_true", help="额外跑一个正常终结 + 一个取消请求")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["ATTNVIEW_CALIB_LOGITS"] = str(args.out / "logits.pt")
    os.environ["ATTNVIEW_CALIB_TRACE"] = str(args.out / "steps.jsonl")
    if args.force_trajectory is not None:
        os.environ["ATTNVIEW_CALIB_FORCE"] = str(args.force_trajectory)
        os.environ["ATTNVIEW_CALIB_FORCE_LOG"] = str(args.out / "force.jsonl")

    arm_prompt, payload = build_prompt()
    payload = dict(payload)
    payload["enforce_global"] = args.arm == "da-global"
    extra_args = {"attnview": payload} if args.arm == "da-global" else None

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=str(SNAPSHOT),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )
    try:
        from vllm.config.attention import AttentionConfig

        llm_kwargs["attention_config"] = AttentionConfig(flash_attn_version=2)
    except Exception as exc:  # 无法构造 ⇒ 让门禁明确拒绝，而不是偷偷用默认值
        print(f"警告: 无法构造 AttentionConfig(flash_attn_version=2): {exc}", file=sys.stderr)

    started = time.time()
    llm = LLM(**llm_kwargs)
    startup_s = time.time() - started

    geometry = llm.llm_engine.model_executor.collective_rpc(
        "attnview_geometry", single_value=True
    )
    capture: dict = {"step": 0, "records": {}}
    if args.record_layers:
        install_hooks(find_fa_layers(llm), capture)

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=SEED,
        extra_args=extra_args,
    )

    def run_one(prompt: str, sampling_params: SamplingParams, *, capture_steps: bool) -> dict:
        before = capture["step"] if capture_steps else 0
        started_req = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - started_req
        out = outputs[0]
        return {
            "elapsed_s": elapsed,
            "token_ids": list(out.outputs[0].token_ids),
            "finish_reason": str(out.outputs[0].finish_reason),
            "text_sha256": sha256_text(out.outputs[0].text),
            "steps_captured": capture["step"] - before if capture_steps else 0,
        }

    result = run_one(arm_prompt.rendered, params, capture_steps=True)
    if result["elapsed_s"] > REQUEST_BUDGET_S:
        print(f"请求超时预算（{result['elapsed_s']:.1f}s > {REQUEST_BUDGET_S}s）：保存证据后退出", file=sys.stderr)

    cleanup: dict = {}
    if args.cleanup_check:
        cleanup["normal"] = run_one(arm_prompt.rendered, SamplingParams(max_tokens=4, temperature=0.0, seed=SEED), capture_steps=False)
        try:
            llm.llm_engine.abort_request(llm.llm_engine.get_request_id(0))
        except Exception as exc:
            cleanup["abort_note"] = f"abort 调用不可用（记录，不改行为）: {exc}"

    if args.emit_trajectory is not None:
        # 格式：tokens[step][row] = [token_ids...]；本驱动是单请求、每步 1 个 token
        args.emit_trajectory.write_text(
            json.dumps({"tokens": [[[t]] for t in result["token_ids"]]})
        )

    manifest = {
        "arm": args.arm,
        "head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "pin_commit": (REPO / "vllm/.git/HEAD").read_text().strip() if (REPO / "vllm/.git/HEAD").exists() else None,
        "model_snapshot": str(SNAPSHOT),
        "model_revision": SNAPSHOT.name,
        "prompt_sha256": sha256_text(arm_prompt.rendered),
        "prompt_len": len(arm_prompt.token_ids),
        "payload": payload,
        "extra_args": extra_args,
        "startup_s": startup_s,
        "startup_budget_s": STARTUP_BUDGET_S,
        "geometry_rpc": geometry,
        "sampling": {"max_tokens": args.max_tokens, "temperature": 0.0, "seed": SEED},
        "result": result,
        "cleanup": cleanup,
        "llm_kwargs": {k: str(v) for k, v in llm_kwargs.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if capture["records"]:
        import torch

        torch.save(capture, args.out / "layers.pt")
    print(json.dumps({"arm": args.arm, "startup_s": round(startup_s, 1),
                      "request_s": round(result["elapsed_s"], 2), "tokens": result["token_ids"],
                      "finish": result["finish_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
