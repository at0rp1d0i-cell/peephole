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
import threading
import time

import torch
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


class BudgetExceeded(RuntimeError):
    """启动/请求超出工作单明确上限（15 分钟 / 180 秒）。"""


class Watchdog:
    """硬期限：到期时**先落盘证据再退出**（`os._exit(3)`），不擅自加长/加批。

    为什么用线程而不是 `signal.alarm`：模型加载与 `generate` 会长时间阻塞在 C++ 侧，
    Python 信号处理器要等控制权回到解释器才执行；看门狗线程在 C 阻塞期间同样能生效。
    """

    def __init__(self, seconds: float, label: str, out_dir: Path) -> None:
        self.seconds = float(seconds)
        self.label = label
        self.out_dir = out_dir
        self._timer: threading.Timer | None = None
        self.started = time.time()

    def _fire(self) -> None:
        payload = {
            "kind": "budget_exceeded",
            "label": self.label,
            "budget_s": self.seconds,
            "elapsed_s": round(time.time() - self.started, 2),
            "action": "保存证据后退出（不提高长度/批次，不继续加载）",
        }
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"budget_exceeded.{self.label}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
            sys.stderr.write(f"超出预算：{json.dumps(payload, ensure_ascii=False)}\n")
            sys.stderr.flush()
        finally:
            os._exit(3)

    def __enter__(self) -> "Watchdog":
        self._timer = threading.Timer(self.seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._timer is not None:
            self._timer.cancel()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started


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


class LayerCapture:
    """按**真实契约**接入层观测：`FlashAttentionImpl.forward` 是 ABC 的普通方法（**不是** nn.Module），
    签名 `(layer, query, key, value, kv_cache, attn_metadata, output, ...)`，返回注意力输出
    （pin `v1/attention/backends/flash_attn.py:947-958`）。步边界用 `model.forward` 包裹得到
    （每步一次），避免用层调用次数推步号。"""

    def __init__(self) -> None:
        self.step = 0
        self.records: dict[int, dict[int, dict]] = {}
        #: 索引 → 真实层名（如 `language_model.model.layers.3.self_attn.attn`），写进捕获便于追溯
        self.layer_names: dict[int, str] = {}

    def wrap_impl(self, index: int, impl: object) -> None:
        original = impl.forward

        def wrapper(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            result = original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            positions = getattr(attn_metadata, "positions", None)
            self.records.setdefault(self.step, {})[index] = {
                "q": query.detach().to("cpu", dtype=torch.float32),
                "k": key.detach().to("cpu", dtype=torch.float32),
                "v": value.detach().to("cpu", dtype=torch.float32),
                "out": (result if result is not None else output).detach().to("cpu", dtype=torch.float32),
                "q_len": int(query.shape[0]),
                "layer_index": index,
                # 真实 softmax scale（impl 自己算的那个）；读不到就记 None，由 dump 明确标注来源
                "impl_scale": float(getattr(impl, "scale")) if getattr(impl, "scale", None) is not None else None,
                "positions": None if positions is None else [int(v) for v in positions[: int(query.shape[0])].tolist()],
            }
            return result

        impl.forward = wrapper

    def wrap_model(self, model: object) -> None:
        original = model.forward

        def wrapper(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                self.step += 1

        model.forward = wrapper


def run_cleanup_check(llm, arm_prompt, payload: dict, patched: bool) -> dict:
    """校准 #5 的清理验收：**真实提交 → 逐 step 驱动 → 执行中取消 → 清理轮 → 新请求**。

    用 pin 的真实 API（`LLMEngine.add_request(request_id, prompt, params)` `:218`、
    `abort_request([id])` `:212`、`step()` `:299`），不走 `LLM.generate`（那会一路跑到结束，
    无法覆盖"执行中取消"）。任何取消路径未真正执行 ⇒ `ok=False`（由调用方判失败，不吞成备注）。
    """
    from vllm import SamplingParams

    engine = llm.llm_engine
    result: dict = {"ok": False, "events": []}
    if not hasattr(engine, "add_request") or not hasattr(engine, "abort_request"):
        result["error"] = "引擎没有 add_request/abort_request，无法做执行中取消验收"
        return result

    prompt_ids = list(arm_prompt.token_ids)
    params = SamplingParams(max_tokens=32, temperature=0.0, seed=SEED,
                            extra_args=({"attnview": payload} if patched else None))

    def drive(rounds: int, label: str) -> None:
        for _ in range(rounds):
            engine.step()
            result["events"].append(f"{label}:step")

    # 1) 提交带载荷请求（patched 臂）或普通请求，驱动若干步
    cancel_id = f"calib-cancel-{int(time.time())}"
    engine.add_request(cancel_id, prompt_ids, params)
    result["submitted_cancel_id"] = cancel_id
    drive(3, "cancel-req")
    # 2) 执行中取消
    engine.abort_request([cancel_id])
    result["abort_called"] = True
    drive(3, "cleanup")
    result["cancel_path_executed"] = True

    # 3) 正常结束的请求，再提交新请求并跑到结束（状态重置）
    normal_id = f"calib-normal-{int(time.time())}"
    engine.add_request(normal_id, prompt_ids, SamplingParams(max_tokens=4, temperature=0.0, seed=SEED))
    drive(12, "normal-req")
    follow_id = f"calib-follow-{int(time.time())}"
    engine.add_request(follow_id, prompt_ids, SamplingParams(max_tokens=4, temperature=0.0, seed=SEED))
    drive(12, "follow-req")
    result["normal_then_new_request"] = True
    result["ok"] = bool(result["abort_called"] and result["cancel_path_executed"])
    return result


def wrap_host_logits(llm, sink: list) -> None:
    """宿主级完整 logits 捕获：包裹 `model.compute_logits`（两个部署身份下都存在）。

    与补丁钩子的区别：这里在**模型对象**上包裹，不依赖补丁；用于 original 臂取完整 logits。
    含显式 D2H 同步（校准专用），需在报告里标注。
    """
    model, _ = find_fa_layers(llm)
    original = model.compute_logits

    def wrapper(hidden_states, *args, **kwargs):
        result = original(hidden_states, *args, **kwargs)
        sink.append(result.detach().to("cpu", dtype=torch.float32))
        return result

    model.compute_logits = wrapper


def install_capture(llm) -> LayerCapture:
    capture = LayerCapture()
    model, fa_layers = find_fa_layers(llm)
    for index, (name, impl) in enumerate(fa_layers):
        capture.layer_names[index] = name
        capture.wrap_impl(index, impl)
    capture.wrap_model(model)
    return capture


def _model_and_runner(llm) -> tuple[object, object]:
    """取 (模型, model_runner)。真实层级：`Qwen3_5ForConditionalGeneration` 把语言模型放在
    `.language_model`（pin `qwen3_5.py:522`），层内是 `.self_attn.attn`（`qwen3_next.py:341`）——
    因此**不靠硬编码层级**，而是用 runner 的真实 `attn_groups` 里的 `layer_names` 定位。"""
    executor = llm.llm_engine.model_executor
    worker = getattr(executor, "driver_worker", None)
    candidates = []
    for attr in ("model_runner",):
        candidates.append(getattr(worker, attr, None))
    inner = getattr(worker, "worker", None)
    if inner is not None:
        candidates.append(getattr(inner, "model_runner", None))
    runner = next((c for c in candidates if c is not None and hasattr(c, "attn_groups")), None)
    if runner is None:
        raise RuntimeError("校准: 找不到 model_runner（需要 VLLM_ENABLE_V1_MULTIPROCESSING=0 单进程模式）")
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("校准: model_runner 上没有 model")
    return model, runner


def find_fa_layers(llm) -> tuple[object, list]:
    """返回 (模型, [(层名, FLASH_ATTN impl)])。

    目标全注意力层集合取自 runner.attn_groups 里 `FullAttentionSpec` 组的 `layer_names`
    （这是运行期真实事实，不是层级猜测），再按名字在 `model.named_modules()` 里解析出
    `Attention` 模块与其 `.impl`；任何一步解析不到都**明确报错**，不静默跳过。
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    model, runner = _model_and_runner(llm)
    groups = list(getattr(runner, "attn_groups", ()) or ())
    if not groups:
        raise RuntimeError("校准: runner.attn_groups 为空，无法定位全注意力层")
    named = dict(model.named_modules())
    found: list[tuple[str, object]] = []
    for layer_groups in groups:
        for group in layer_groups:
            if not isinstance(getattr(group, "kv_cache_spec", None), FullAttentionSpec):
                continue
            for name in list(getattr(group, "layer_names", ()) or ()):
                module = named.get(name)
                if module is None:
                    raise RuntimeError(f"校准: 全注意力层 {name!r} 不在 model.named_modules() 里")
                impl = getattr(module, "impl", None)
                if impl is None:
                    raise RuntimeError(f"校准: 层 {name!r} 没有 .impl（无法观测 q/k/v/out）")
                if "flash" not in repr(type(impl)).lower():
                    raise RuntimeError(f"校准: 层 {name!r} 的 impl 不是 FLASH_ATTN：{type(impl).__name__}")
                found.append((name, impl))
    if not found:
        raise RuntimeError("校准: runner.attn_groups 里没有 FullAttentionSpec 组（目标模型应为 3×GDN + 1×FA）")
    return model, found


def dump_capture(capture: LayerCapture, path: Path) -> None:
    """把层观测导成 oracle 约定的 npz。

    口径（与 oracle 子代理确认后收紧）：

    - **只有 prefill 步可做完整全局参考**：其可见 key = 整段 prefill（已捕获），绝对位置 < prompt_len。
      因此把 prefill 步导出为 `q_step{step}_L{i}` / `out_step{step}_L{i}` / `positions_step{step}`。
    - decode 步的可见 key 还包含"已生成 token + 当前 token"的历史，捕获里没有 ⇒ 用
      `decode_q_step{step}_L{i}` / `decode_out_step{step}_L{i}` 前缀单独导出，**不**冒充可参考步。
    - `scale` 优先用运行时 impl 的真实值；取不到时回退 `head_dim**-0.5` 并用 `scale_source` 标明来源。
    """
    import numpy as np

    arrays: dict[str, np.ndarray] = {}
    for step in sorted(capture.records):
        for index, rec in sorted(capture.records[step].items()):
            q, k, v, out = rec["q"], rec["k"], rec["v"], rec["out"]
            num_heads, head_dim = int(q.shape[1]), int(q.shape[2])
            num_kv_heads = int(k.shape[1])
            positions = rec.get("positions")
            if rec["q_len"] > 1:  # prefill 步：整段 K/V + 可完整参考的 query/输出
                arrays[f"k_prefill_L{index}"] = k.transpose(0, 1).contiguous().numpy()
                arrays[f"v_prefill_L{index}"] = v.transpose(0, 1).contiguous().numpy()
                arrays[f"q_step{step}_L{index}"] = q.transpose(0, 1).contiguous().numpy()
                arrays[f"out_step{step}_L{index}"] = out.transpose(0, 1).contiguous().numpy()
                arrays[f"positions_step{step}"] = np.asarray(
                    positions if positions is not None else list(range(rec["q_len"])), dtype=np.int64
                )
                arrays["prompt_len"] = np.array(k.shape[0])
            else:  # decode 步：历史 KV 不全 ⇒ 只作观测记录，不冒充可参考步
                arrays[f"decode_q_step{step}_L{index}"] = q.transpose(0, 1).contiguous().numpy()
                arrays[f"decode_out_step{step}_L{index}"] = out.transpose(0, 1).contiguous().numpy()
            arrays["layer_index"] = np.array(index)
            arrays["num_heads"] = np.array(num_heads)
            arrays["num_kv_heads"] = np.array(num_kv_heads)
            arrays["head_dim"] = np.array(head_dim)
            real_scale = rec.get("impl_scale")
            arrays["scale"] = np.array(real_scale if real_scale is not None else head_dim ** -0.5)
            arrays["scale_source"] = np.array(
                "impl.scale" if real_scale is not None else "derived_head_dim**-0.5"
            )
            arrays["dtype_name"] = np.array("captured_fp32")
            arrays[f"layer_name_L{index}"] = np.array(
                capture.layer_names.get(index, f"index{index}")
            )
    if not any(key.startswith("k_prefill_") for key in arrays):
        raise RuntimeError("校准: 捕获里没有 prefill 步（无法产出可完整参考的观测）")
    np.savez(path, **arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=("original", "patched-disabled", "patched-global"),
        required=True,
        help="original = 未打补丁的原版；patched-disabled = 打补丁但请求无载荷（插入层关闭）；"
        "patched-global = 打补丁 + 真实带载荷且 enforce_global=True",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--emit-trajectory", type=Path)
    ap.add_argument("--force-trajectory", type=Path)
    ap.add_argument("--record-layers", action="store_true")
    ap.add_argument(
        "--capture-host-logits",
        action="store_true",
        help="宿主级完整 logits 捕获（包裹 model.compute_logits）：original 臂唯一的完整 logits 来源",
    )
    ap.add_argument("--cleanup-check", action="store_true", help="额外跑一个正常终结 + 一个取消请求")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patched = args.arm != "original"



    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if patched:  # 这两个钩子由补丁提供；original 臂用宿主级钩子（下方 capture_logits_host）
        os.environ["ATTNVIEW_CALIB_LOGITS"] = str(args.out / "logits.pt")
        os.environ["ATTNVIEW_CALIB_TRACE"] = str(args.out / "steps.jsonl")
    if args.force_trajectory is not None:
        if not patched:
            raise RuntimeError("校准: original 臂没有强制钩子（补丁未部署）——轨迹由它产生，不由它回放")
        os.environ["ATTNVIEW_CALIB_FORCE"] = str(args.force_trajectory)
        os.environ["ATTNVIEW_CALIB_FORCE_LOG"] = str(args.out / "force.jsonl")

    arm_prompt, payload = build_prompt()
    arm_prompt, payload = build_prompt()
    payload = dict(payload)
    payload["enforce_global"] = args.arm == "patched-global"
    extra_args = {"attnview": payload} if args.arm == "patched-global" else None
    installed = REPO / "venvs/attnview/lib/python3.12/site-packages/vllm"
    deployment = {
        "identity": "patched" if patched else "original",
        "installed_core_sha256": sha256_file(installed / "v1/engine/core.py")[:16],
        "installed_engine_sha256": sha256_file(installed / "v1/engine/attnview_engine.py")[:16]
        if (installed / "v1/engine/attnview_engine.py").exists()
        else None,
        "note": "original 必须在**未部署**的原始副本上运行；patched-* 必须在 apply 后的副本上运行",
    }
    if not patched and deployment["installed_engine_sha256"] is not None:
        raise RuntimeError("校准: --arm original 必须在未部署（无 attnview_engine.py）的原始副本上运行")
    if patched and deployment["installed_engine_sha256"] is None:
        raise RuntimeError("校准: --arm patched-* 需要已部署的副本（先 apply）")

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
        # 同步调度：与 worker 侧门禁一致（`config/vllm.py:1263-1311` 未指定会默认 True）
        async_scheduling=False,
    )
    try:
        from vllm.config.attention import AttentionConfig

        llm_kwargs["attention_config"] = AttentionConfig(flash_attn_version=2)
    except Exception as exc:  # 无法构造 ⇒ 让门禁明确拒绝，而不是偷偷用默认值
        print(f"警告: 无法构造 AttentionConfig(flash_attn_version=2): {exc}", file=sys.stderr)

    with Watchdog(STARTUP_BUDGET_S, "startup", args.out) as startup_watch:
        llm = LLM(**llm_kwargs)
    startup_s = startup_watch.elapsed
    if startup_s > STARTUP_BUDGET_S:  # 双保险（看门狗正常路径下已退出）
        raise BudgetExceeded(f"启动耗时 {startup_s:.1f}s 超过预算 {STARTUP_BUDGET_S}s")

    geometry = (
        llm.llm_engine.model_executor.collective_rpc("attnview_geometry", single_value=True)
        if patched
        else {"skipped": "original 臂没有该 RPC（补丁未部署）"}
    )
    capture = install_capture(llm) if args.record_layers else None
    host_logits: list = []
    if args.capture_host_logits:
        wrap_host_logits(llm, host_logits)

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=SEED,
        extra_args=extra_args,
    )

    def run_one(prompt: str, sampling_params: SamplingParams, *, capture_steps: bool) -> dict:
        before = capture.step if (capture is not None and capture_steps) else 0
        with Watchdog(REQUEST_BUDGET_S, "request", args.out) as req_watch:
            outputs = llm.generate([prompt], sampling_params)
        elapsed = req_watch.elapsed
        out = outputs[0]
        return {
            "elapsed_s": elapsed,
            "token_ids": list(out.outputs[0].token_ids),
            "finish_reason": str(out.outputs[0].finish_reason),
            "text_sha256": sha256_text(out.outputs[0].text),
            "steps_captured": (capture.step - before) if (capture is not None and capture_steps) else 0,
        }

    result = run_one(arm_prompt.rendered, params, capture_steps=True)
    if result["elapsed_s"] > REQUEST_BUDGET_S:  # 双保险
        raise BudgetExceeded(f"请求耗时 {result['elapsed_s']:.1f}s 超过预算 {REQUEST_BUDGET_S}s")

    cleanup: dict = {}
    if args.cleanup_check:
        cleanup = run_cleanup_check(llm, arm_prompt, payload, patched)

    if args.emit_trajectory is not None:
        # 格式：tokens[step][row] = [token_ids...]；本驱动是单请求、每步 1 个 token
        args.emit_trajectory.write_text(
            json.dumps({"tokens": [[[t]] for t in result["token_ids"]]})
        )

    manifest = {
        "arm": args.arm,
        "deployment": deployment,
        "host_logits_saved": str(args.out / "logits-host.pt") if args.capture_host_logits else None,
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
    if capture is not None and capture.records:
        dump_capture(capture, args.out / "layers.npz")
    if host_logits:
        torch.save(
            [{"step": i + 1, "logits": t} for i, t in enumerate(host_logits)],
            args.out / "logits-host.pt",
        )
    print(json.dumps({"arm": args.arm, "startup_s": round(startup_s, 1),
                      "request_s": round(result["elapsed_s"], 2), "tokens": result["token_ids"],
                      "finish": result["finish_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
