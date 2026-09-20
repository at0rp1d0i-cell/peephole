#!/usr/bin/env python3
"""SUP-004 校准步骤③–⑤：**同一 token 轨迹**下的三身份对照（模型级运行入口）。

身份（唯一入口，见 `--arm`）：

- `original`：未打补丁的原版（**必须**跑在未部署的副本上；强制轨迹由它自然 greedy 产生）；
- `patched-disabled`：已打补丁但请求**无载荷**（插入层不参与，走原版读 metadata）；
- `patched-global`：已打补丁 + 真实带载荷且 `enforce_global=True`。

钩子绑定与 id 流程（本地复核 R1/R2 后收紧）：

① 只设置 `ATTNVIEW_CALIB_ARM=<out>/arm.json`（**不创建文件**，warmup 期间钩子未 armed）
→ ② `LLM(...)`（内部真实 warmup）→ ③ `LLMEngine.add_request(<外部 id>, prompt_token_ids, params)`
（`prompt_token_ids` 直接用阶段 03 `render_arm` 的最终 token_ids，**不重新 tokenize**；返回值是**内部 id**）
→ ④ **首次 `step()` 之前**写 `arm.json`（`target_req_id` = **内部 id**）+ `tokens`/`force_log`/`logits_path`/`trace_path`
→ ⑤ `engine.step()` 驱动到该请求完成（看门狗约束）。

id 三层语义（pin 事实，混用即错配）：

- **内部 id**（`add_request` 返回值，`input_processor.py:262-278` 追加 8 位随机后缀）：`scheduler.requests`、
  worker `input_batch.req_ids`、校准钩子绑定（`arm.json.target_req_id`）、engine 侧协议状态；
- **外部 id**（驱动传入的那个）：宿主侧 `RequestOutput.request_id`（`output_processor.py:380-381`）
  ⇒ 宿主消费计数与轨迹匹配用它；
- `abort_request` 默认按**外部 id** 查 external→internal 映射（`llm_engine.py:212`、
  `output_processor.py:494-524`），因此清理段用外部 id 取消、用内部 id 核对"确实不再 active"。

warmup 请求、未完成 prefill 的丢弃步、cleanup/其它请求都不进入主对照：强制日志只记 `target_req_id`
的消费步且 `step` 从 1 连续，logits / 层捕获同样只覆盖主请求窗口。

步数/消费账本：消费计数取**宿主观测**（每轮新增 token ≥ 1 的轮数），decode 捕获步数 = 消费计数 − 1；
相位判定的**唯一权威证据**是本步 `InputBatch.is_prefilling_np[目标行]`（pin `model_runner.py:1138,1345`，
由 `LayerCapture.wrap_runner_inputs` 包裹 `prepare_inputs` 取得）：取不到即报错，**不得**用
`q_len`/`max_query_len` 代替（末尾单 token 的 prefill chunk 也满足 q_len==1）；`num_prefill_*`/
`num_decode_*` 计数器只在有值时作交叉核对（本 pin 实测可为全 0），并与真实绝对位置互核。
本阶段三臂执行视图都必须是 canonical/global（FA metadata `seq_lens[0] == 本步最后位置 + 1`），
**不应**出现受限读视图覆写 trace。

执行证据：**起时就落** `<out>/manifest.json`（HEAD、源码 dirty、完整安装指纹、生效配置、prompt token ids
与哈希、起时时间戳），结束（成功/异常/超时）补 `ended_cst` 与 `exit_code`。

预算：启动 ≤15 分钟、单请求 ≤180 秒、清理段 ≤120 秒；超时先落盘证据（含 manifest 结束字段）再退出。

用法：
    source env.sh && CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-calib-run.py \\
        --arm original --out evidence/p3-calib/run-original-1 \\
        --emit-trajectory evidence/p3-calib/traj-original-1.json --cleanup-check
    ...
    --arm patched-global --out evidence/p3-calib/run-global-1 \\
        --force-trajectory evidence/p3-calib/traj-original-1.json --cleanup-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

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
CLEANUP_BUDGET_S = 120
SEED = 20260918
#: 三身份（唯一入口；`vanilla` / `da-global` 等旧名已删除）
ARMS = ("original", "patched-disabled", "patched-global", "patched-masked")
SCHEMA = "attnview.p2-calib-run/v2"

INSTALLED_VLLM = REPO / "venvs/attnview/lib/python3.12/site-packages/vllm"
PIN_VLLM = REPO / "vllm/vllm"
PATCH_ROOT = REPO / "vllm-patch"

#: 按层分键的元数据（不被最后一层覆盖；跨步必须一致）
_PER_LAYER_KEY = re.compile(r"^(?:scale|capture_dtype|layer_name)_L\d+$")

CONTEXT = (
    "Declarative attention lets a model name the context regions it wants to read. "
    "The runtime turns those declarations into a paged KV read view, so the same weights can "
    "attend globally, focus a segment, or keep a local window plus content-free sink tokens. "
    "This paragraph exists only to give the calibration run a fixed, non-trivial context."
)
QUESTION = "In one short sentence, state what a read view is."


class BudgetExceeded(RuntimeError):
    """启动/请求超出工作单明确上限（15 分钟 / 180 秒 / 清理 120 秒）。"""


def now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: list[int]) -> str:
    return sha256_text(json.dumps([int(t) for t in token_ids]))


def _sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _git(*args: str, cwd: Path = REPO) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        return f"<git {' '.join(args)} 失败: {proc.stderr.strip()}>"
    return proc.stdout.strip()


def save_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def mark_manifest_ended(manifest_path: Path | None, *, exit_code: int, note: str | None = None) -> None:
    """尽力补写 manifest 结束字段（看门狗走 `os._exit`，只能在这里补）。"""
    if manifest_path is None:
        return
    try:
        data = json.loads(Path(manifest_path).read_text())
        data["ended_cst"] = now_cst()
        data["exit_code"] = int(exit_code)
        if note:
            data.setdefault("notes", []).append(note)
        save_manifest(Path(manifest_path), data)
    except Exception as exc:  # 证据补写失败不得掩盖原始失败原因
        sys.stderr.write(f"警告: 无法补写 manifest 结束字段：{exc}\n")


class CheckLog:
    """判据账本：每条判据都带可观察细节，失败项构成非零退出。"""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str) -> bool:
        ok = bool(passed)
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        return ok

    @property
    def failed(self) -> list[str]:
        return [c["check"] for c in self.checks if not c["ok"]]


class Watchdog:
    """硬期限：到期时**先落盘证据再退出**（`os._exit(3)`），不擅自加长/加批。

    为什么用线程而不是 `signal.alarm`：模型加载与 `engine.step()` 会长时间阻塞在 C++ 侧，
    Python 信号处理器要等控制权回到解释器才执行；看门狗线程在 C 阻塞期间同样能生效。
    """

    def __init__(
        self, seconds: float, label: str, out_dir: Path, manifest_path: Path | None = None
    ) -> None:
        self.seconds = float(seconds)
        self.label = label
        self.out_dir = Path(out_dir)
        self.manifest_path = manifest_path
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
            mark_manifest_ended(
                self.manifest_path, exit_code=3, note=f"看门狗超时（{self.label}，预算 {self.seconds:g}s）"
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


# --------------------------------------------------------------------------- #
# 部署指纹（original 必须与 pin checkout 逐字节一致；patched-* 必须与补丁树一致）
# --------------------------------------------------------------------------- #


def patch_targets(patch_root: Path | None = None) -> tuple[list[str], list[str]]:
    """从补丁 manifest 读**本次补丁涉及的全部目标文件**（相对 vllm 包根）。"""
    manifest = json.loads(((patch_root or PATCH_ROOT) / "manifest.json").read_text())
    edited = sorted(manifest["edits"])
    added = sorted(entry["dest"] for entry in manifest["new_files"])
    return edited, added


def deployment_fingerprint(arm: str, *, installed_root: Path | None = None, pin_root: Path | None = None,
                           patch_root: Path | None = None) -> dict:
    """按**臂**核对部署身份（对象不同、期望值不同；失败信息里同时给出双方实际值与双方期望值）。

    - `original`：安装副本**与** pin checkout 都必须 == manifest 的 **pre**（未部署态）；
    - `patched-*`：安装副本**与** checkout 都必须 == manifest 的 **post** —— 两者都是部署目标
      （`tools/p2-apply-patch.py apply` 同时改写，本地复核反例已确认），任一不等于 post 即拒绝；
    - 补丁树与 manifest 声明必须自洽（post / new_files.sha256）；
    - 失败信息同时给出「对象（installed / checkout）× 期望（pre / post）× 实际值」。
    """
    installed_root = Path(installed_root or INSTALLED_VLLM)
    pin_root = Path(pin_root or PIN_VLLM)
    patch_root = Path(patch_root or PATCH_ROOT)
    manifest = json.loads((patch_root / "manifest.json").read_text())
    edited, added = patch_targets(patch_root)
    declared_added = {entry["dest"]: entry["sha256"] for entry in manifest["new_files"]}
    evidence: dict = {
        "arm": arm,
        "installed_root": str(installed_root),
        "pin_root": str(pin_root),
        "patch_root": str(patch_root),
        "patch_manifest_generated_cst": manifest.get("generated_cst"),
        "patch_manifest_pin_commit": manifest.get("pin_commit"),
        "pin_checkout_commit": _git("rev-parse", "HEAD", cwd=REPO / "vllm"),
        "expectation": (
            "original：installed == checkout == manifest.pre（未部署）"
            if arm == "original" else
            "patched-*：installed == checkout == manifest.post（两副本都是部署目标）"
        ),
        "edited": {},
        "added": {},
    }
    for rel in edited:
        rule = manifest["edits"][rel]
        pre, post = rule.get("pre_sha256"), rule.get("post_sha256")
        installed = _sha_or_none(installed_root / rel)
        checkout = _sha_or_none(pin_root / rel)
        patch_tree = _sha_or_none(patch_root / "patched" / rel)
        expectation = pre if arm == "original" else post
        evidence["edited"][rel] = {
            "installed_sha256": installed,
            "checkout_sha256": checkout,
            "patch_tree_sha256": patch_tree,
            "manifest_pre_sha256": pre,
            "manifest_post_sha256": post,
            "expected_for_arm": expectation,
            "checkout_checked": True,
        }
        if patch_tree != post:
            raise RuntimeError(
                f"校准: 补丁 manifest 与补丁树不一致（{rel}: manifest post={post} tree={patch_tree}）"
                "—— 先重新生成补丁 manifest 再运行"
            )
        if arm == "original":
            if installed != pre or checkout != pre:
                raise RuntimeError(
                    f"校准: original 臂要求未部署状态（{rel}）：installed={installed} checkout={checkout} "
                    f"期望双方都 == manifest.pre={pre}"
                )
        else:
            if installed != post or checkout != post:
                raise RuntimeError(
                    f"校准: patched 臂要求 installed 与 checkout **都**为部署后状态（{rel}）："
                    f"installed={installed} checkout={checkout} 期望双方都 == manifest.post={post}"
                )
    for rel in added:
        installed = _sha_or_none(installed_root / rel)
        patch_tree = _sha_or_none(patch_root / "files/vllm" / rel)
        checkout = _sha_or_none(pin_root / rel)
        declared = declared_added.get(rel)
        evidence["added"][rel] = {
            "installed_sha256": installed,
            "checkout_sha256": checkout,
            "patch_tree_sha256": patch_tree,
            "manifest_sha256": declared,
            "expected_for_arm": None if arm == "original" else declared,
            "checkout_checked": True,
        }
        if patch_tree != declared:
            raise RuntimeError(
                f"校准: 补丁 manifest 与新增文件不一致（{rel}: manifest={declared} tree={patch_tree}）"
                "—— 先重新生成补丁 manifest 再运行"
            )
        if arm == "original":
            if installed is not None:
                raise RuntimeError(
                    f"校准: original 臂要求 {rel} **不存在**（补丁未部署）：installed={installed} "
                    f"checkout={checkout}（期望双方都不存在）"
                )
        else:
            if installed != declared or checkout != declared:
                raise RuntimeError(
                    f"校准: patched 臂要求新增文件在两副本都已部署（{rel}）：installed={installed} "
                    f"checkout={checkout} 期望双方都 == manifest 声明 {declared}"
                )
    return evidence


# --------------------------------------------------------------------------- #
# prompt（阶段 03 的单一入口；token_ids 直接交给引擎，不重新 tokenize）
# --------------------------------------------------------------------------- #


def build_prompt_from_fixture(doc_fixture: Path, expect_fixture: Path, timeline_config: Path):
    """按已验收夹具重建 prompt(与轨迹生成器**共享** `attnview.fixture_rebuild`),逐项校验。"""
    from attnview.fixture_rebuild import rebuild_prompt

    prompt, payload, evidence = rebuild_prompt(
        doc_fixture=doc_fixture, expect_fixture=expect_fixture, timeline_config=timeline_config,
        snapshot=SNAPSHOT,
        tokenizer_hash=_sha_or_none(SNAPSHOT / "tokenizer.json") or "",
        template_hash=_sha_or_none(SNAPSHOT / "chat_template.jinja") or "",
    )
    return prompt, payload, evidence


def build_prompt() -> tuple[object, dict]:
    """用阶段 03 的既有入口渲染协议 prompt 与其 token span（不改 prompt 合同）。"""
    from transformers import AutoTokenizer

    from attnview.prompt import _tokenize_with_offsets, render_arm
    from attnview.segmenter import build_offsets_index, segment_context

    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=False)
    _token_ids, offsets = _tokenize_with_offsets(tokenizer, CONTEXT)
    segments = segment_context(CONTEXT, build_offsets_index(offsets))
    prompt = render_arm(
        "da",
        segments,
        QUESTION,
        CONTEXT,
        tokenizer,
        tokenizer_hash=_sha_or_none(SNAPSHOT / "tokenizer.json") or "",
        template_hash=_sha_or_none(SNAPSHOT / "chat_template.jinja") or "",
        enable_thinking=False,
    )
    payload = {
        "protocol": "v1.0",
        "prompt_len": len(prompt.token_ids),
        "segment_spans": [list(s) for s in prompt.segment_spans],
        "local_window_span": list(prompt.scaffold.local_window_span),
        "sink_span": list(prompt.scaffold.sink_span),
    }
    return prompt, payload


# --------------------------------------------------------------------------- #
# 模型 / 层定位
# --------------------------------------------------------------------------- #


def _model_and_runner(llm) -> tuple[object, object]:
    """取 (模型, model_runner)。真实层级：`Qwen3_5ForConditionalGeneration` 把语言模型放在
    `.language_model`（pin `qwen3_5.py:522`），层内是 `.self_attn.attn`（`qwen3_next.py:341`）——
    因此**不靠硬编码层级**，而是用 runner 的真实 `attn_groups` 里的 `layer_names` 定位。"""
    executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
    worker = getattr(executor, "driver_worker", None)
    candidates = [getattr(worker, "model_runner", None)]
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
    """返回 (模型, [(层名, FLASH_ATTN impl)])——**唯一定义**。

    目标全注意力层集合取自 runner.attn_groups 里 `FullAttentionSpec` 组的 `layer_names`
    （运行期真实事实，不是层级猜测），再按名字在 `model.named_modules()` 里解析出
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
                    raise RuntimeError(
                        f"校准: 层 {name!r} 的 impl 不是 FLASH_ATTN：{type(impl).__name__}"
                    )
                found.append((name, impl))
    if not found:
        raise RuntimeError(
            "校准: runner.attn_groups 里没有 FullAttentionSpec 组（目标模型应为 3×GDN + 1×FA）"
        )
    return model, sorted(found)


# --------------------------------------------------------------------------- #
# 层观测：目标请求窗口内逐层逐步（真实位置 / 真实 dtype / 真实 scale / 逐步落盘）
# --------------------------------------------------------------------------- #


class LayerCapture:
    """按 pin 的**真实契约**接入层观测：`FlashAttentionImpl.forward` 是 ABC 的普通方法（**不是**
    nn.Module），签名 `(layer, query, key, value, kv_cache, attn_metadata, output, ...)`，
    pin `v1/attention/backends/flash_attn.py:969-1217`（返回的 output 是
    `Attention.forward` 里 `output.view(-1, num_heads, head_size_v)`，pin
    `model_executor/layers/attention/attention.py:519-525`）。步边界用 `model.forward` 包裹
    （每步一次），避免用层调用次数推步号。

    本地复核 R1 后收紧（缺关键值即失败，不做替代品）：

    - 绝对位置取自**真实 model inputs**（`model.forward(positions=...)`，即 `input_batch.positions`，
      pin `worker/gpu/model_runner.py:1701-1702`）；取不到 / 形状不可用即**报错**，**不用**
      `range(q_len)` 代替；单请求内逐步核验位置连续（prefill = 0..q_len-1，decode = prompt_len+k）；
    - 设备 dtype 取 `query.dtype`，scale 取 `impl.scale`（读不到即报错，**不**用 `head_dim**-0.5`）；
    - 元数据按层分键（`*_L{L}`），不被最后一层覆盖；
    - **逐步落盘**：每步结束写出 `<dir>/step{N}.npz`（异常也留有可审证据）；
    - 只观测 **armed** 的目标请求：warmup / 其它 req_id 的步不进捕获。
    """

    def __init__(
        self, *, runner: object | None = None, expected_layers: dict[int, str] | None = None
    ) -> None:
        self.runner = runner
        self.step = 0  # 全部 forward 计数（含未 armed 步）
        self.request_step = 0  # 目标请求内计数（1-based；1 = prefill 步）
        self.records: dict[int, dict[int, dict]] = {}
        self.positions: dict[int, dict] = {}
        self.markers: dict[int, dict] = {}
        self.layer_names: dict[int, str] = dict(expected_layers or {})
        self.scales: dict[int, float] = {}
        self.armed_req_id: str | None = None
        self.prompt_len: int | None = None
        self.stream_dir: Path | None = None
        self.prev_req_ids: dict[int, list[str]] = {}
        self._active = False
        self._step_open = False  # 只有 `_begin_step` 与 `_end_step` 之间才记录（防止跨步/嵌套误写）
        self._step_positions = None
        self._step_positions_shape: list[int] = []
        self._step_mrope_axes = None
        self._step_metadata = None
        self._step_view: dict = {"observable": False}
        #: 逻辑位置来源（供审计）：本步取到的一维缓冲说明
        self.logical_positions_source: str | None = None
        #: 本步 InputBatch（`prepare_inputs` 包裹得到）与逐步相位证据快照
        self.current_input_batch = None
        self.batch_phase: dict[int, dict] = {}

    # --- arming（由驱动在 engine 就绪后、提交主请求前调用） ------------------ #

    def arm(self, req_id: str, stream_dir: Path, *, prompt_len: int) -> None:
        self.armed_req_id = str(req_id)
        self.prompt_len = int(prompt_len)
        self.stream_dir = Path(stream_dir)
        self.stream_dir.mkdir(parents=True, exist_ok=True)

    def disarm(self) -> None:
        self.armed_req_id = None
        self._active = False

    # --- 安装 -------------------------------------------------------------- #

    def wrap_impl(self, index: int, impl: object) -> None:
        scale = getattr(impl, "scale", None)
        if scale is None:
            raise RuntimeError(
                f"校准: 层 L{index} 的 impl 没有 .scale（本 pin 可读）—— 拒绝用 head_dim 推导替代"
            )
        self.scales[index] = float(scale)
        original = impl.forward

        def wrapper(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            result = original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            self._record(index, query, key, value, result if result is not None else output, attn_metadata)
            return result

        impl.forward = wrapper

    def wrap_model(self, model: object) -> None:
        original = model.forward

        def wrapper(*args, **kwargs):
            self.step += 1
            self._begin_step(kwargs)
            try:
                return original(*args, **kwargs)
            finally:
                self._end_step()

        model.forward = wrapper

    # --- 记录 -------------------------------------------------------------- #

    def _prev_req_ids(self) -> list[str]:
        """上一步 `ExecuteModelState.input_batch.req_ids`（**只作证据**，不代替真实输入）。"""
        state = getattr(self.runner, "execute_model_state", None)
        batch = getattr(state, "input_batch", None)
        return [str(r) for r in (getattr(batch, "req_ids", ()) or ())]

    def _begin_step(self, kwargs: dict) -> None:
        self._active = self.armed_req_id is not None
        self._step_open = False
        self._step_positions = None
        self._step_metadata = None
        if not self._active:
            return
        positions = kwargs.get("positions")
        if positions is None:
            raise RuntimeError(
                "校准: 本步 model inputs 没有 positions（真实绝对位置）—— 拒绝用 range(q_len) 代替"
            )
        dim = getattr(positions, "dim", lambda: 0)()
        if dim not in (1, 2):
            raise RuntimeError(
                f"校准: 本步 positions 维数 {dim}（形状 {tuple(getattr(positions, 'shape', ()))}）不可用作"
                "一维/每请求一行的绝对位置 —— 拒绝推断"
            )
        self.request_step += 1
        prev = self._prev_req_ids()
        self.prev_req_ids[self.request_step] = prev
        # 单活跃请求是本阶段的硬前提：2-D 批张量只取第 0 行，多请求会取错行 ⇒ 明确报错。
        # `prev` 为上一步 `execute_model_state.input_batch.req_ids`（可观察时才有值）：非空且不含目标请求
        # 即视为批内多请求/绑定错配；不可观察（空）时只记录，不凭空断言。
        if self.request_step > 1 and prev and prev != [self.armed_req_id]:
            raise RuntimeError(
                f"校准: 第 {self.request_step} 步上一步批次的 req_ids={prev} 不是目标请求 "
                f"{self.armed_req_id!r}（批内多请求/绑定错配）—— 拒绝继续"
            )
        self.batch_phase[self.request_step] = self._batch_phase_evidence()
        # 模型位置输入**原样保留**（真实入口实测为 (3, max_num_tokens) 的 mRoPE 三轴张量）：
        # 只作记录，不改模型位置、不假设三轴恒相等。
        self._step_positions_shape = [int(d) for d in getattr(positions, "shape", ())]
        if dim == 2:
            if any(d < 1 for d in self._step_positions_shape):
                raise RuntimeError(f"校准: 本步 positions 张量形状 {tuple(positions.shape)} 不可用")
            self._step_mrope_axes = positions
        else:
            self._step_mrope_axes = None
        # 供 KV/oracle 用的 **canonical 逻辑位置**另有来源：runner 本次真实的一维逻辑位置缓冲
        # （pin `input_batch.py:29` `InputBuffers.positions`，由 `prepare_pos_seq_lens` 每步写入）。
        self.records.setdefault(self.request_step, {})
        self._step_open = True

    def _logical_positions_from_runner(self, q_len: int) -> tuple[list[int], str]:
        """取本步**一维逻辑位置**（真实缓冲，按实际 token 数截取）。

        - 来源：`runner.input_buffers.positions[:num_tokens]`（pin `input_batch.py:29`，
          由 `prepare_pos_seq_lens`（`input_batch.py:367-385`）每步写入的真实逻辑位置）；
        - `num_tokens` 取本步 `InputBatch.num_tokens`（`prepare_inputs` 包裹所得）；
        - **不得**用 `range(q_len)`，**不得**由 mRoPE 三轴推导：两者都取不到即报错。
        """
        runner = self.runner
        buffers = getattr(runner, "input_buffers", None)
        buffer = getattr(buffers, "positions", None)
        batch = self.current_input_batch
        raw_num_tokens = getattr(batch, "num_tokens", None) if batch is not None else None
        buffer_len = int(buffer.shape[0]) if buffer is not None and hasattr(buffer, "shape") else None
        # 三者必须齐备且一致：本步 InputBatch.num_tokens（真实字段）↔ 逻辑位置缓冲长度 ↔ 捕获 q_len。
        # 任何缺失/不一致都直接报错：**不回退 q_len、不取 max、不用 range(q_len)、不由三轴推导**。
        if buffer is None:
            raise RuntimeError(
                f"校准: 取不到 runner.input_buffers.positions（一维逻辑位置缓冲）"
                f"（num_tokens={raw_num_tokens} buffer_len={buffer_len} q_len={q_len}）—— 拒绝用 "
                "range(q_len) 或 mRoPE 三轴推导冒充逻辑位置"
            )
        if raw_num_tokens is None or int(raw_num_tokens) <= 0:
            raise RuntimeError(
                f"校准: 本步 InputBatch.num_tokens 缺失或非正（num_tokens={raw_num_tokens} "
                f"buffer_len={buffer_len} q_len={q_len}）—— 不得用 q_len 回退"
            )
        num_tokens = int(raw_num_tokens)
        if num_tokens != q_len:
            raise RuntimeError(
                f"校准: 本步 token 数不一致（num_tokens={num_tokens} buffer_len={buffer_len} "
                f"q_len={q_len}）—— 拒绝取 max 或回退，请核查位置来源"
            )
        if buffer_len is None or buffer_len < num_tokens:
            raise RuntimeError(
                f"校准: 逻辑位置缓冲不足以覆盖本步 token（num_tokens={num_tokens} "
                f"buffer_len={buffer_len} q_len={q_len}）"
            )
        values = [int(v) for v in buffer[:num_tokens].detach().to("cpu").tolist()]
        return values, "runner.input_buffers.positions[:num_tokens]（真实一维逻辑位置缓冲）"

    def wrap_runner_inputs(self, runner: object) -> None:
        """包裹 runner 里构造**本步** `InputBatch` 的方法（pin `worker/gpu/model_runner.py:1159`
        `prepare_inputs(...) -> InputBatch`）。

        目的：拿到本步真实的 `input_batch.is_prefilling_np[目标行]`（适配层同样以它为准）作为相位判定的
        **首选证据**。本 pin 实测 `num_prefill_*`/`num_decode_*` 在该步**未被填充（全 0）**，不能用它们反推。
        包裹失败/方法缺失时不报错，但相位判定会退化到 `max_query_len`，两者都不可用时**明确报错**。
        """
        original = getattr(runner, "prepare_inputs", None)
        if original is None:
            return

        def wrapper(*args, **kwargs):
            batch = original(*args, **kwargs)
            self.current_input_batch = batch
            return batch

        try:
            runner.prepare_inputs = wrapper
        except Exception:  # 只读替身/不可写对象：退化为 max_query_len 证据
            return

    def _batch_phase_evidence(self) -> dict:
        """本步 `InputBatch` 里的相位证据（只在批次确实包含目标请求时可用）。"""
        batch = self.current_input_batch
        if batch is None:
            return {"available": False, "reason": "本步 InputBatch 不可得（prepare_inputs 未包裹/未调用）"}
        req_ids = [str(r) for r in (getattr(batch, "req_ids", ()) or ())]
        flags = getattr(batch, "is_prefilling_np", None)
        try:
            values = [bool(v) for v in (flags.tolist() if hasattr(flags, "tolist") else list(flags or []))]
        except Exception as exc:
            return {"available": False, "reason": f"is_prefilling_np 不可读：{type(exc).__name__}: {exc}"}
        if self.armed_req_id in req_ids:
            row = req_ids.index(self.armed_req_id)
            if row < len(values):
                return {
                    "available": True,
                    "row": row,
                    "is_prefilling_row": values[row],
                    "req_ids": req_ids,
                    "is_prefilling_np": values,
                }
        return {
            "available": False,
            "is_prefilling_row": None,
            "req_ids": req_ids,
            "is_prefilling_np": values,
            "reason": "本步批次里没有目标请求（或行号越界）",
        }

    def _record(self, index: int, query, key, value, out, attn_metadata=None) -> None:
        if not self._active or not self._step_open or self.armed_req_id is None:
            return
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise RuntimeError(
                f"校准: L{index} 的 q/k/v dtype 不一致（{query.dtype}/{key.dtype}/{value.dtype}）"
            )
        marker = self._step_marker(attn_metadata)
        if marker is not None:
            self.markers.setdefault(self.request_step, marker)
            if self.markers[self.request_step] != marker:
                raise RuntimeError(
                    f"校准: 第 {self.request_step} 步各层的 attn_metadata 真实标记不一致："
                    f"{self.markers[self.request_step]} vs {marker}"
                )
            self._step_metadata = attn_metadata
        self.records[self.request_step][index] = {
            "q": query.detach().to("cpu", dtype=torch.float32),
            "k": key.detach().to("cpu", dtype=torch.float32),
            "v": value.detach().to("cpu", dtype=torch.float32),
            "out": out.detach().to("cpu", dtype=torch.float32),
            "device_dtype": str(query.dtype),
            "scale": self.scales[index],
            "layer_name": self.layer_names.get(index, f"index{index}"),
        }

    @staticmethod
    def _step_marker(attn_metadata) -> dict | None:
        """本步 `attn_metadata` 上的相位相关字段（**只作记录**）。

        本 pin 实测 `num_prefill_reqs`/`num_decode_reqs`/`num_prefill_tokens`/`num_decode_tokens`
        在该步可能**全为 0（未填充）**，因此它们只在**有值时**作交叉核对；相位判定的首选证据是本步
        `InputBatch.is_prefilling_np[目标行]`，退化顺序见 `_end_step`。
        """
        if attn_metadata is None:
            return None
        fields = ("num_prefill_reqs", "num_decode_reqs", "num_prefill_tokens", "num_decode_tokens")
        if not any(hasattr(attn_metadata, name) for name in fields):
            return None
        marker = {name: int(getattr(attn_metadata, name, 0)) for name in fields}
        max_query_len = getattr(attn_metadata, "max_query_len", None)
        marker["max_query_len"] = int(max_query_len) if max_query_len is not None else None
        return marker

    @staticmethod
    def _step_view_evidence(attn_metadata) -> dict:
        """本步 FA metadata 的**读取视图证据**（用于判定执行视图是否就是 canonical/global）。

        `seq_lens` 是 FA 组实际使用的有效读长度：受限读视图会把 `seqused_k` 缩短（adapter 覆写时
        正是替换 `group_seq_lens`，见 `vllm-patch/.../attn_utils.py:288-292`），因此
        `seq_lens[0] == 本步最后位置 + 1` 是"未走受限视图"的直接证据。块表同时记录首行前缀
        （块号单位在本轮未独立核验，故只作记录）。
        """
        if attn_metadata is None:
            return {"observable": False, "reason": "本步没有 attn_metadata（无法给出视图证据）"}
        evidence: dict = {"observable": True}
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is not None and getattr(seq_lens, "numel", lambda: 0)() > 0:
            evidence["seq_lens_first"] = int(seq_lens.reshape(-1)[0].detach().to("cpu").item())
        else:
            evidence["seq_lens_first"] = None
        evidence["max_seq_len"] = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
        evidence["num_actual_tokens"] = int(getattr(attn_metadata, "num_actual_tokens", 0) or 0)
        table = getattr(attn_metadata, "block_table", None)
        if table is not None and getattr(table, "numel", lambda: 0)() > 0:
            evidence["block_table_shape"] = list(table.shape)
            evidence["block_table_head"] = [
                int(v) for v in table.reshape(table.shape[0], -1)[0][:16].detach().to("cpu").tolist()
            ]
        return evidence

    def _end_step(self) -> None:
        if not self._active or not self._step_open:
            self._step_open = False
            return
        self._step_open = False
        step = self.request_step
        layers = self.records.get(step) or {}
        self._step_positions = None
        if not layers:
            raise RuntimeError(f"校准: 目标请求第 {step} 步没有任何全注意力层被观测到（无法给出参考）")
        q_lens = {int(rec["q"].shape[0]) for rec in layers.values()}
        if len(q_lens) != 1:
            raise RuntimeError(f"校准: 目标请求第 {step} 步各层 q_len 不一致：{sorted(q_lens)}")
        q_len = q_lens.pop()
        positions, logical_source = self._logical_positions_from_runner(q_len)
        self.logical_positions_source = logical_source
        raw_len = len(positions)
        view = self._step_view_evidence(self._step_metadata)
        self._step_metadata = None
        if positions != list(range(positions[0], positions[0] + q_len)):
            raise RuntimeError(
                f"校准: 第 {step} 步 positions 前若干项 {positions[:8]} 不是本步 token 的真实绝对位置"
            )
        if step == 1:
            if positions[0] != 0:
                raise RuntimeError(f"校准: prefill 步的起点位置是 {positions[0]}，不是 0")
            if self.prompt_len is not None and q_len != self.prompt_len:
                raise RuntimeError(
                    f"校准: prefill 步 q_len={q_len} 与 prompt 长度 {self.prompt_len} 不一致"
                    "（prefill 被分块或不是同一 prompt）"
                )
            phase = "prefill"
        else:
            if self.prompt_len is None:
                raise RuntimeError("校准: 未记录 prefill 步，无法核验 decode 位置")
            expected = self.prompt_len + (step - 2)
            if positions != [expected]:
                raise RuntimeError(
                    f"校准: 第 {step} 步 decode 位置 {positions} != 期望 [{expected}]"
                    "（目标请求的消费 token 序列不连续）"
                )
            phase = "decode"
        marker = dict(self.markers.get(step) or {})
        batch = self.batch_phase.get(step) or {"available": False, "reason": "本步未取到 InputBatch 证据"}
        counters = (
            int(marker.get("num_prefill_reqs", 0)) + int(marker.get("num_decode_reqs", 0))
            + int(marker.get("num_prefill_tokens", 0)) + int(marker.get("num_decode_tokens", 0))
        )
        marker.update({
            "is_prefilling_available": bool(batch.get("available")),
            "is_prefilling_row": batch.get("is_prefilling_row"),
            "batch_req_ids": batch.get("req_ids"),
            "batch_evidence_reason": batch.get("reason"),
            "counters_populated": bool(counters),
            "counters_note": (
                "计数器有值，作交叉核对" if counters else
                "本 pin 该步未填充 num_prefill_*/num_decode_*（全 0）⇒ **不作相位证据**"
            ),
        })
        # 相位判定的**唯一权威证据**：本步 `InputBatch.is_prefilling_np[目标行]`
        # （语义 = `num_computed_prefill_tokens_np < prefill_len_np`，pin `model_runner.py:1138,1345`）。
        # 取不到即失败：**不得**用 `q_len` / `max_query_len` 代替（末尾单 token 的 prefill chunk 同样
        # 满足 q_len==1，按 q_len 分类是明令禁止的）；计数器只在有值时作交叉核对。
        if batch.get("available"):
            marker_phase = "prefill" if batch["is_prefilling_row"] else "decode"
            decided_by = "is_prefilling_np"
        else:
            raise RuntimeError(
                f"校准: 第 {step} 步相位权威证据不可用（本步 InputBatch.is_prefilling_np[目标行] 取不到："
                f"{marker.get('batch_evidence_reason')}）—— 拒绝用 q_len/max_query_len 代替"
            )
        marker["decided_by"] = decided_by
        marker["decided_phase"] = marker_phase
        if marker_phase != phase:
            raise RuntimeError(
                f"校准: 第 {step} 步的相位判定不一致：按真实位置是 {phase}，"
                f"按 {decided_by} 是 {marker_phase}（{marker}）"
            )
        if counters:
            prefill_rows = int(marker["num_prefill_reqs"]) + int(marker["num_prefill_tokens"])
            decode_rows = int(marker["num_decode_reqs"]) + int(marker["num_decode_tokens"])
            if prefill_rows and decode_rows:
                raise RuntimeError(
                    f"校准: 第 {step} 步是 prefill/decode 混合批（{marker}）—— 本阶段只支持单请求单相位"
                )
            counters_phase = "prefill" if prefill_rows else "decode"
            if counters_phase != phase:
                raise RuntimeError(
                    f"校准: 第 {step} 步计数器与相位不一致：计数器判为 {counters_phase}，真实位置/证据为 "
                    f"{phase}（{marker}）"
                )
        phase_source = f"positions+{decided_by}"
        self.positions[step] = {
            "positions": positions,
            "q_len": q_len,
            "positions_raw_len": raw_len,
            "positions_source": logical_source,
            "mrope_axes_shape": self._step_positions_shape if self._step_mrope_axes is not None else None,
            "mrope_axes_head": (
                [int(v) for v in self._step_mrope_axes[:, :2].detach().to("cpu").reshape(-1).tolist()]
                if self._step_mrope_axes is not None else None
            ),
            "tail_columns_ignored": 0,
            "phase": phase,
            "phase_source": phase_source,
            "phase_markers": marker,
            "view": view,
        }
        if self.stream_dir is not None:
            self.flush(self.stream_dir)

    # --- 导出 -------------------------------------------------------------- #

    def step_arrays(self, step: int) -> dict:
        """单步 npz 数组。

        - prefill 步（内部步 1）：完整 `k_prefill_L{L}` / `v_prefill_L{L}` + `q_step1` / `out_step1`；
        - decode 步（内部步 `step`）：键里的 `i` 是**生成 token 序号**（1-based）= `step - 1`，
          导出 `decode_q_step{i}` / `decode_out_step{i}` / `decode_pos_step{i}` /
          `k_current_step{i}` / `v_current_step{i}`（该步**当前写入 token** 的 canonical K/V）。

        所有 decode 步共用**同一个** prefill 前缀（prefill 只导一次，重复键会被 `dump_capture` 拒绝）。
        """
        import numpy as np

        meta = self.positions.get(step)
        if meta is None:
            return {}
        arrays: dict[str, object] = {}
        decode_index = int(step) - 1
        for index, rec in sorted((self.records.get(step) or {}).items()):
            q, k, v, out = rec["q"], rec["k"], rec["v"], rec["out"]
            if step == 1:
                arrays[f"k_prefill_L{index}"] = k.transpose(0, 1).contiguous().numpy()
                arrays[f"v_prefill_L{index}"] = v.transpose(0, 1).contiguous().numpy()
                arrays[f"q_step{step}_L{index}"] = q.transpose(0, 1).contiguous().numpy()
                arrays[f"out_step{step}_L{index}"] = out.transpose(0, 1).contiguous().numpy()
            else:
                arrays[f"decode_q_step{decode_index}_L{index}"] = q.transpose(0, 1).contiguous().numpy()
                arrays[f"decode_out_step{decode_index}_L{index}"] = out.transpose(0, 1).contiguous().numpy()
                arrays[f"k_current_step{decode_index}_L{index}"] = k.transpose(0, 1).contiguous().numpy()
                arrays[f"v_current_step{decode_index}_L{index}"] = v.transpose(0, 1).contiguous().numpy()
            arrays[f"scale_L{index}"] = np.array(float(rec["scale"]))
            arrays[f"capture_dtype_L{index}"] = np.array(str(rec["device_dtype"]))
            arrays[f"layer_name_L{index}"] = np.array(str(rec["layer_name"]))
        key = f"positions_step{step}" if step == 1 else f"decode_pos_step{decode_index}"
        arrays[key] = np.asarray(meta["positions"], dtype=np.int64)
        return arrays

    def flush(self, directory: Path) -> Path:
        """把**当前步**写出 `<dir>/forward{N}.npz`（N = 内部 forward 步号，1 = prefill）。

        逐步落盘：异常也留有可审证据；步号与 `decode_*_step{i}` 里的 i 相差 1（i = N - 1）。
        """
        import numpy as np

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"forward{self.request_step}.npz"
        np.savez(target, **self.step_arrays(self.request_step))
        return target

    def decode_steps(self) -> list[int]:
        """选定 decode 步清单（`decode_*_step{i}` 里的 i）：i = 生成 token 序号（1-based）。

        本捕获覆盖**每一个** decode 步 —— prefill 之后的每次 forward 都测，因此在
        `prompt_len + i - 1` 位置上的 current K/V 与 prefill 前缀拼起来就是该步完整 canonical 历史。
        """
        return [step - 1 for step in sorted(self.positions) if step >= 2]

    def decode_forwards(self) -> list[int]:
        """选定 decode 步对应的**内部 forward 步号**（1 = prefill），便于与逐文件/驱动步数对齐。"""
        return [step for step in sorted(self.positions) if step >= 2]

    def summary(self) -> dict:
        """捕获摘要（写进 manifest；不含大张量）。"""
        return {
            "armed_req_id": self.armed_req_id,
            "prompt_len": self.prompt_len,
            "forward_steps_total": self.step,
            "request_steps": sorted(self.records),
            "decode_steps": self.decode_steps(),
            "decode_forwards": self.decode_forwards(),
            "expected_layers": {str(i): self.layer_names[i] for i in sorted(self.layer_names)},
            "expected_layer_count": len(self.layer_names),
            "scales": {str(i): self.scales[i] for i in sorted(self.scales)},
            "positions": {str(s): self.positions[s] for s in sorted(self.positions)},
            "prev_req_ids": {str(s): v for s, v in sorted(self.prev_req_ids.items())},
        }


def install_capture(llm) -> LayerCapture:
    """安装层观测：层集合取自 runner.attn_groups 的 FullAttentionSpec 组（运行期事实）。"""
    model, fa_layers = find_fa_layers(llm)
    _model, runner = _model_and_runner(llm)
    expected = {index: name for index, (name, _impl) in enumerate(fa_layers)}
    capture = LayerCapture(runner=runner, expected_layers=expected)
    for index, (_name, impl) in enumerate(fa_layers):
        capture.wrap_impl(index, impl)
    capture.wrap_runner_inputs(runner)  # 本步 InputBatch（is_prefilling_np = 相位首选证据）
    capture.wrap_model(model)
    return capture


def dump_capture(capture: LayerCapture, path: Path) -> None:
    """把层观测导成 oracle 约定的 npz（**单一合并文件**；逐步文件已在 `step{N}.npz`）。

    键名（固定合同，供 oracle 消费）：

    - prefill 步（内部步 1）：`k_prefill_L{L}` / `v_prefill_L{L}` / `q_step1_L{L}` /
      `out_step1_L{L}` / `positions_step1`（**只导一次**，是全部 decode 步共用的 canonical 前缀）；
    - decode 步 i（i = 生成 token 序号，1-based，位置 = prompt_len + i - 1）：
      `decode_q_step{i}_L{L}` / `decode_out_step{i}_L{L}` / `decode_pos_step{i}`（int64）/
      `k_current_step{i}_L{L}` / `v_current_step{i}_L{L}`（该步**当前写入 token** 的 canonical K/V，
      `[KVH, q_len, D]`）；每步都导，prefill + 步 1..i 的 current 即第 i 步的完整 canonical 历史；
    - 选定 decode 步清单：`decode_steps`（= i 序列）/ `decode_forwards`（= 内部 forward 步号）/ `decode_note`；
    - 每步每层元数据**按层分键**：`capture_dtype_L{L}`（设备真实 dtype）、`layer_name_L{L}`、
      `scale_L{L}`（真实 `impl.scale`）；
    - 全局标量：`prompt_len` / `num_heads` / `num_kv_heads` / `head_dim` / `scale`（= 各层一致的真实
      scale）/ `scale_source="impl.scale"` / `stored_dtype="float32"`；
    - 先算完整**全局**参考所需的 prefill K/V 与逐步 q/out；decode 步不冒充 prefill 步。
    """
    import numpy as np

    if not capture.positions:
        raise RuntimeError("校准: 捕获里没有任何步（主请求没有被观测到）")
    if 1 not in capture.positions:
        raise RuntimeError("校准: 捕获里没有 prefill 步（无法产出可完整参考的观测）")

    arrays: dict[str, object] = {}
    per_layer: dict[str, object] = {}
    for step in sorted(capture.positions):
        for name, value in capture.step_arrays(step).items():
            if _PER_LAYER_KEY.match(name):
                previous = per_layer.get(name)
                if previous is None:
                    per_layer[name] = value
                elif not np.array_equal(np.asarray(previous), np.asarray(value)):
                    raise RuntimeError(
                        f"校准: 按层键 {name!r} 在不同步之间不一致（{previous!r} vs {value!r}）—— "
                        "元数据必须按层分键且逐步一致"
                    )
                continue
            if name in arrays:
                raise RuntimeError(f"校准: 捕获键 {name!r} 被覆盖（元数据必须按层/按步分键）")
            arrays[name] = value
    arrays.update(per_layer)

    first = capture.records[1]
    any_rec = next(iter(first.values()))
    num_heads, head_dim = int(any_rec["q"].shape[1]), int(any_rec["q"].shape[2])
    num_kv_heads = int(any_rec["k"].shape[1])
    scales = {round(float(rec["scale"]), 12) for step in capture.records for rec in capture.records[step].values()}
    if len(scales) != 1:
        raise RuntimeError(f"校准: 捕获到多个不同 scale：{sorted(scales)}（元数据不得被最后一层覆盖）")
    dtypes = {rec["device_dtype"] for step in capture.records for rec in capture.records[step].values()}
    if len(dtypes) != 1:
        raise RuntimeError(f"校准: 捕获到多个设备 dtype：{sorted(dtypes)}")

    arrays["prompt_len"] = np.array(int(capture.positions[1]["q_len"]))
    arrays["num_heads"] = np.array(num_heads)
    arrays["num_kv_heads"] = np.array(num_kv_heads)
    arrays["head_dim"] = np.array(head_dim)
    arrays["scale"] = np.array(scales.pop())
    arrays["scale_source"] = np.array("impl.scale")
    arrays["stored_dtype"] = np.array("float32")
    arrays["layer_index"] = np.array(sorted(capture.layer_names))
    # 选定 decode 步清单：`decode_*_step{i}` 里的 i = 生成 token 序号（1-based）；
    # `decode_forwards` 给出对应的内部 forward 步号（1 = prefill），便于与逐文件/驱动步数对齐。
    arrays["decode_steps"] = np.asarray(capture.decode_steps(), dtype=np.int64)
    arrays["decode_forwards"] = np.asarray(capture.decode_forwards(), dtype=np.int64)
    arrays["decode_note"] = np.array(
        "prefill（k_prefill/v_prefill 只导一次，步 1）+ 每个 decode 步 i 的 current K/V"
        "（位置 = prompt_len + i - 1）拼成该步完整 canonical 历史；positions 均为真实绝对位置"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


# --------------------------------------------------------------------------- #
# 宿主级完整 logits（original 臂唯一的完整 logits 来源）
# --------------------------------------------------------------------------- #


def wrap_host_logits(llm, sink: list, *, req_id: str) -> object:
    """包裹 `model.compute_logits`，只在**主请求窗口**内记录（窗口外 warmup/cleanup 不记录）。

    返回 `restore()`：窗口结束必须恢复原方法，避免 cleanup 段继续进产物。
    记录里带 `req_id`/`req_ids`（由**窗口**界定：宿主侧看不到真实 req_ids，已在 `req_ids_source` 标注）。
    含显式 D2H 同步（校准专用），需在报告里标注。"""
    model, _runner = _model_and_runner(llm)
    original = model.compute_logits

    def wrapper(hidden_states, *args, **kwargs):
        result = original(hidden_states, *args, **kwargs)
        sink.append(
            {
                "step": len(sink) + 1,
                "req_id": str(req_id),
                "req_ids": [str(req_id)],
                "req_ids_source": "主请求驱动窗口（宿主侧包裹看不到真实 req_ids）",
                "logits": result.detach().to("cpu", dtype=torch.float32),
                "shape": list(result.shape),
                "dtype_on_device": str(result.dtype),
                "source": "host_wrap:model.compute_logits",
                "window": "main-request",
                "sync_note": "本捕获含显式 D2H 同步（保存完整 logits），属校准专用，非稳态行为",
            }
        )
        return result

    model.compute_logits = wrapper

    def restore() -> None:
        model.compute_logits = original

    return restore


def save_host_logits(records: list, path: Path) -> None:
    """宿主侧完整 logits 落盘（与 worker 钩子同构的记录结构；`req_ids` 来源已在记录里标注）。"""
    torch.save([dict(record) for record in records], path)


# --------------------------------------------------------------------------- #
# 引擎侧观察（宿主可达性 + 请求级协议状态）
# --------------------------------------------------------------------------- #


def _engine_core(engine) -> object | None:
    """从宿主侧 `LLMEngine` 取 EngineCore（单进程 `InprocClient` 路径）。

    `LLMEngine.engine_core` 是客户端（`InprocClient`），真正的引擎核心在它的 `.engine_core`；
    若对象本身就直接持有 `scheduler`（部分替身/直连实现），则回退到它自己。
    """
    if engine is None:
        return None
    client = getattr(engine, "engine_core", None)
    if client is None:
        return engine
    return getattr(client, "engine_core", client)


def scheduler_requests(engine) -> dict | None:
    core = _engine_core(engine)
    requests = getattr(getattr(core, "scheduler", None), "requests", None)
    return requests if isinstance(requests, dict) else None


def _scheduler_has(engine, req_id: str) -> bool | None:
    requests = scheduler_requests(engine)
    if requests is None:
        return None
    return str(req_id) in {str(k) for k in requests}


def _unfinished_count(engine) -> int | None:
    getter = getattr(engine, "get_num_unfinished_requests", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def engine_observability(llm) -> dict:
    """记录宿主侧**能观察到什么**（不可观察项必须显式标注，不得静默当成功）。"""
    engine = getattr(llm, "llm_engine", None)
    core = _engine_core(engine)
    scheduler = getattr(core, "scheduler", None)
    attnview = getattr(core, "attnview", None)
    return {
        "engine_core_type": type(core).__name__ if core is not None else None,
        "scheduler_type": type(scheduler).__name__ if scheduler is not None else None,
        "scheduler_requests_observable": isinstance(getattr(scheduler, "requests", None), dict),
        "unfinished_counter_observable": callable(getattr(engine, "get_num_unfinished_requests", None)),
        "attnview_observable": attnview is not None,
        "attnview_type": type(attnview).__name__ if attnview is not None else None,
    }


def protocol_state(llm, req_id: str) -> dict:
    """请求级协议状态（engine 侧 registry/config/detok/pending）；取不到即明确标注不可观察。"""
    engine = getattr(llm, "llm_engine", None)
    core = _engine_core(engine)
    attnview = getattr(core, "attnview", None)
    if attnview is None:
        return {
            "observable": False,
            "reason": "宿主侧取不到 engine 的 attnview 对象（原版无该对象，或层级不同）",
        }
    registry = getattr(attnview, "registry", None)
    ids = list(registry.active_ids()) if hasattr(registry, "active_ids") else None
    configs = getattr(attnview, "_configs", None)
    detok = getattr(attnview, "_detokenizers", None)
    pending = attnview.pending_plans() if hasattr(attnview, "pending_plans") else None
    config = configs.get(req_id) if isinstance(configs, dict) else None
    return {
        "observable": True,
        "registry_ids": ids,
        "in_registry": (str(req_id) in ids) if ids is not None else None,
        "in_configs": (str(req_id) in configs) if isinstance(configs, dict) else None,
        "in_detok": (str(req_id) in detok) if isinstance(detok, dict) else None,
        "in_pending": (str(req_id) in pending) if isinstance(pending, dict) else None,
        "config_enforce_global": getattr(config, "enforce_global", None) if config is not None else None,
        "enforce_global_steps": [
            dict(m) for m in (getattr(attnview, "enforce_global_steps", ()) or ())
        ],
        "traces_count": len(getattr(attnview, "traces", ()) or ()),
        "unsupported": [dict(m) for m in (getattr(attnview, "unsupported", ()) or ())],
    }


def protocol_mode(llm, req_id: str) -> dict:
    """该请求当前的**真实协议模式**（engine 侧 `ProtocolRegistry` 状态对象的 `.mode`）。

    取不到即明确标注不可观察（不构造、不推断）。用途：判断某步是否属于"协议模式非 global、
    执行被 `enforce_global` 覆写"的情形（`attnview_engine.py:541-553`）。
    """
    engine = getattr(llm, "llm_engine", None)
    core = _engine_core(engine)
    attnview = getattr(core, "attnview", None)
    registry = getattr(attnview, "registry", None)
    if registry is None or not hasattr(registry, "get"):
        return {"observable": False, "reason": "宿主侧取不到 engine 的 attnview.registry"}
    try:
        state = registry.get(str(req_id))
    except Exception as exc:
        return {"observable": False, "reason": f"registry 无 {req_id} 的状态：{type(exc).__name__}"}
    mode = getattr(state, "mode", None)
    if mode is None:
        return {"observable": False, "reason": "状态对象没有 mode 字段"}
    return {"observable": True, "mode": str(mode)}


def dump_engine_traces(llm, path: Path) -> dict:
    """把 engine 侧逐步 trace（解析序号 / enforce_global 标记）落盘；不可观察则记录原因。"""
    engine = getattr(llm, "llm_engine", None)
    core = _engine_core(engine)
    attnview = getattr(core, "attnview", None)
    if attnview is None or not hasattr(attnview, "dump_traces"):
        return {"observable": False, "reason": "宿主侧取不到 engine 的 attnview.dump_traces"}
    try:
        attnview.dump_traces(path)
    except Exception as exc:
        return {"observable": True, "error": f"{type(exc).__name__}: {exc}"}
    return {"observable": True, "path": str(path), "count": len(getattr(attnview, "traces", ()) or ())}


def worker_req_ids(llm) -> list[str] | None:
    """worker 侧本步 req_ids（宿主可见的可行性证据）；不可观察返回 None。"""
    try:
        _model, runner = _model_and_runner(llm)
    except Exception:
        return None
    state = getattr(runner, "execute_model_state", None)
    batch = getattr(state, "input_batch", None)
    if batch is None:
        return None
    return [str(r) for r in (getattr(batch, "req_ids", ()) or ())]


# --------------------------------------------------------------------------- #
# 驱动
# --------------------------------------------------------------------------- #


def canonical_blocks(engine, req_id: str) -> dict:
    """engine 侧 canonical 块表（pin `v1/core/kv_cache_manager.py:696` 的 `get_block_ids`）。

    取不到即明确标注不可观察（不静默当成功）；本项用于与 worker 侧 FA metadata 的块表**交叉核对**。
    """
    core = _engine_core(engine)
    manager = getattr(getattr(core, "scheduler", None), "kv_cache_manager", None)
    if manager is None or not hasattr(manager, "get_block_ids"):
        return {"observable": False, "reason": "宿主侧取不到 scheduler.kv_cache_manager.get_block_ids"}
    try:
        groups = [list(group) for group in manager.get_block_ids(req_id)]
    except Exception as exc:
        return {"observable": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"observable": True, "groups": groups}


def block_table_evidence(canonical: dict, capture: "LayerCapture") -> dict:
    """worker 侧 FA metadata 块表首行前缀 vs engine 侧 canonical 块表（执行视图是否为 global 的旁证）。

    口径：若走受限读视图，`attn_utils.py` 会把 FA 组的 `block_table` 换成"按读视图挑选/压缩"的
    另一个张量（adapter 自建 buffer），其首行内容不会等于 canonical 组的前缀。块号单位
    （manager 块 ↔ kernel 块）在本轮未独立核验，故本项**只记录**，硬判据仍是 `seq_lens`。
    """
    if not canonical.get("observable"):
        return {"observable": False, "reason": canonical.get("reason"), "note": "仅记录：块表旁证不可得"}
    candidates = [list(group) for group in canonical.get("groups", [])]
    per_step: dict = {}
    for step in sorted(capture.positions):
        head = capture.positions[step]["view"].get("block_table_head")
        if head is None:
            per_step[str(step)] = {"block_table_head": None, "note": "本步没有块表记录"}
            continue
        matches = [
            index for index, group in enumerate(candidates)
            if group and head[: len(group)] == group[: len(head)]
        ]
        per_step[str(step)] = {
            "block_table_head": head[:8],
            "canonical_groups": [group[:8] for group in candidates],
            "matching_group_indices": matches,
            "prefix_matches_a_canonical_group": bool(matches),
        }
    return {
        "observable": True,
        "canonical_groups": [group[:8] for group in candidates],
        "per_step": per_step,
        "note": "块号单位（manager↔kernel 块）未在本轮独立核验 ⇒ 只作记录；硬判据是 FA metadata 的 seq_lens",
    }


def submit_request(engine, external_id: str, prompt_ids: list[int], params) -> dict:
    """提交一个请求，返回 id 账本（外部 id / 内部 id）——**两者不可混用**。

    pin 事实（本轮独立复核已确认）：

    - `InputProcessor.assign_request_id`（pin `v1/engine/input_processor.py:262-278`）把外部 id 换成
      `f"{external_req_id}-{random_uuid():.8}"` 的**内部 id**（原值存 `request.external_req_id`）；
    - `LLMEngine.add_request`（pin `v1/engine/llm_engine.py:218-296`）**返回内部 id**；
    - `scheduler.requests` / worker `input_batch.req_ids` / 校准钩子绑定 / engine 侧协议状态
      （registry/config/detok）一律以**内部 id** 为键；
    - 宿主侧 `RequestOutput.request_id` 是**外部 id**（pin `v1/engine/output_processor.py:380-381`：
      "request_id is what was provided externally"），而 `abort_request` 默认按**外部 id** 经
      `output_processor` 的 external→internal 映射定位（pin `v1/engine/output_processor.py:494-524`，
      `internal=False` 路径）。
    """
    internal_id = engine.add_request(str(external_id), [int(t) for t in prompt_ids], params)
    if not isinstance(internal_id, str) or not internal_id:
        raise RuntimeError(
            "校准: add_request 没有返回内部 request id（pin 约定返回内部 id；拒绝按外部 id 继续）"
        )
    return {
        "external_id": str(external_id),
        "internal_id": internal_id,
        "randomized": internal_id != str(external_id),
    }


def sampling_params(max_tokens: int, *, extra_args: dict | None = None):
    """统一的采样参数：**显式 DELTA** 输出。

    pin 默认 `output_kind=RequestOutputKind.CUMULATIVE`（`sampling_params.py:317`）⇒ 每轮返回的是
    **累计** token 列表，整段 `extend` 会重复前缀（本地复核反例：3 个 token 被算成 6）。
    四臂一律用本函数构造参数；累积侧另有 `_merge_step_tokens` 兜底，两种形态都只计一次。
    """
    from vllm import SamplingParams

    try:
        from vllm.sampling_params import RequestOutputKind
    except Exception:  # 兼容不同导出位置
        from vllm import RequestOutputKind  # type: ignore[attr-defined]

    return SamplingParams(
        max_tokens=int(max_tokens),
        temperature=0.0,
        seed=SEED,
        extra_args=extra_args,
        output_kind=RequestOutputKind.DELTA,
    )


def output_mode_of(params) -> str:
    """从**实际请求的** `SamplingParams.output_kind` 读出输出形态（`delta` / `cumulative`）。

    pin 默认 `CUMULATIVE`（`sampling_params.py:317`）；四臂一律显式请求 `DELTA`（见 `sampling_params`），
    合并逻辑按这里读出的形态**明确分支**，不靠前缀猜。
    """
    kind = str(getattr(getattr(params, "output_kind", None), "name", "DELTA")).upper()
    return "cumulative" if kind.startswith("CUMUL") else "delta"


def _merge_step_tokens(seen: list[int], row_token_ids, *, mode: str) -> list[int]:
    """把一轮输出行并入 `seen`，返回**新增 token**（按已知 `output_kind` 明确分支）。

    - `mode="delta"`：直接采纳整行（`seen=[11]` + 行 `[11,12]` ⇒ 新增 2 个，**不得**按前缀猜成 1 个）；
    - `mode="cumulative"`（显式声明）：前缀必须与 `seen` 相符（否则报错）、等长 ⇒ **零新增**
      （终止时的重复快照）、变短 ⇒ 报错（来源异常）；
    - 其它值 ⇒ 报错（形态必须显式给出）。
    """
    incoming = [int(t) for t in row_token_ids]
    if mode == "delta":
        return incoming
    if mode != "cumulative":
        raise RuntimeError(f"校准: 未知输出形态 {mode!r}（只支持 delta / cumulative，且必须显式给出）")
    if len(incoming) < len(seen):
        raise RuntimeError(
            f"校准: 累计输出行长度 {len(incoming)} 小于已见 {len(seen)}——来源异常，拒绝继续"
        )
    if incoming[: len(seen)] != seen:
        raise RuntimeError(
            f"校准: 累计输出行的前缀与已见序列不符（seen[:8]={seen[:8]} incoming[:8]={incoming[:8]}）"
        )
    return incoming[len(seen):]


def drive_main_request(engine, req_id: str, *, max_tokens: int, on_first_step=None,
                       mode: str = "delta") -> dict:
    """用 `engine.step()` 循环驱动**单个**主请求到结束（步数上界显式，超界即报错不继续）。

    返回值：

    - `tokens`：宿主侧实际看到的目标请求 token 序列；
    - `consuming_steps`：**宿主侧消费计数** —— `engine.step()` 每轮返回里目标请求**新增 token ≥ 1**
      的轮数（四臂通用口径：original 臂没有强制钩子，不能用 `force_step` 代替）；
    - `steps`：本次驱动的 forward 轮数；
    - `output_mode`：本轮的输出形态（来自实际请求的 `output_kind`，合并逻辑据此明确分支）。
    """
    step_budget = int(max_tokens) + 16
    tokens: list[int] = []
    other_outputs: list[str] = []
    steps = 0
    consuming_steps = 0
    finished = False
    while not finished:
        if steps >= step_budget:
            raise RuntimeError(
                f"校准: 主请求 {req_id} 在 {steps} 步内没有结束（上界 {step_budget}）—— 不继续驱动"
            )
        outputs = engine.step() or ()
        steps += 1
        if steps == 1 and on_first_step is not None:
            on_first_step()
        produced = 0
        for out in outputs:
            rid = str(getattr(out, "request_id", ""))
            if rid != str(req_id):
                other_outputs.append(rid)
                continue
            for row in getattr(out, "outputs", ()) or ():
                fresh = _merge_step_tokens(tokens, row.token_ids, mode=mode)
                produced += len(fresh)
                tokens.extend(fresh)
            finished = bool(getattr(out, "finished", False))
        if produced:
            consuming_steps += 1
    return {
        "tokens": tokens,
        "steps": steps,
        "consuming_steps": consuming_steps,
        "finished": finished,
        "other_request_outputs": other_outputs,
        "output_mode": mode,
    }


def drive_until_token(engine, req_id: str, *, budget_steps: int, min_steps: int = 1,
                      mode: str = "delta") -> dict:
    """驱动到出现 token（且至少走 `min_steps` 步，用于覆盖 prefill 之后的 decode 步）。"""
    tokens: list[int] = []
    steps = 0
    while steps < budget_steps and (not tokens or steps < min_steps):
        outputs = engine.step() or ()
        steps += 1
        for out in outputs:
            if str(getattr(out, "request_id", "")) != str(req_id):
                continue
            for row in getattr(out, "outputs", ()) or ():
                tokens.extend(_merge_step_tokens(tokens, row.token_ids, mode=mode))
    return {"tokens": tokens, "steps": steps}


def drive_until_finished(engine, req_id: str, *, budget_steps: int, on_step=None,
                         mode: str = "delta") -> dict:
    """驱动到该请求自然结束；每步后可回调（用于在请求仍 active 时探针其协议状态）。"""
    tokens: list[int] = []
    steps = 0
    finished = False
    while steps < budget_steps and not finished:
        outputs = engine.step() or ()
        steps += 1
        for out in outputs:
            if str(getattr(out, "request_id", "")) != str(req_id):
                continue
            for row in getattr(out, "outputs", ()) or ():
                tokens.extend(_merge_step_tokens(tokens, row.token_ids, mode=mode))
            finished = bool(getattr(out, "finished", False))
        if on_step is not None:
            on_step(steps)
    return {"tokens": tokens, "steps": steps, "finished": finished}


def drive_cleanup_rounds(llm, req_id: str, *, max_rounds: int = 16) -> dict:
    """finished-only 清理轮：驱动到引擎内不再有该请求（scheduler + 未完成计数）。"""
    engine = getattr(llm, "llm_engine", None)
    rounds = 0
    while rounds < max_rounds:
        engine.step()
        rounds += 1
        if _scheduler_has(engine, req_id) is False and _unfinished_count(engine) in (0, None):
            break
    return {
        "rounds": rounds,
        "scheduler_has_req": _scheduler_has(engine, req_id),
        "unfinished": _unfinished_count(engine),
        "worker_req_ids": worker_req_ids(llm),
    }


def cleanup_enforce_checks(*, enforce_global: bool, config_enforce_global, marks, non_global_steps,
                           observed_modes) -> list[tuple[str, bool, str]]:
    """cleanup 阶段按臂的 enforce_global 预期(纯函数,便于 CPU 控制流测试)。

    - `enforce_global=True`(既有 global 臂):载荷配置必须为 True;出现协议模式非 global 的步时**必须**留下
      `applied_view=global` 的覆写 mark;全程 global 时 `marks=[]` 属正确结果。**语义与原来完全一致**。
    - `enforce_global=False`(masked 臂):载荷配置必须为 **False**;**不要求**任何覆写 mark,并且必须
      **确认无强制全局标记**(`marks == []`),因为该臂不允许经 enforce_global 把执行拉回 global。
    """
    out: list[tuple[str, bool, str]] = []
    name = "new_req_payload_enforce_global"
    if enforce_global:
        out.append((name, config_enforce_global is True,
                    f"B 的请求级配置 enforce_global={config_enforce_global}(期望 True)"))
        if non_global_steps:
            bad = [m for m in marks
                   if str(m.get("applied_view")).lower() != "global"
                   or str(m.get("protocol_mode", "")).lower() == "global"]
            out.append(("new_req_enforce_global_step_recorded", bool(marks) and not bad,
                        f"存在协议模式非 global 的步 {non_global_steps}(真实模式 {observed_modes})⇒ 必须有"
                        f" applied_view=global 且 protocol_mode 非 global 的 mark;实际 marks={marks[:2]} 不合规项={bad[:2]}"))
        elif observed_modes:
            out.append(("new_req_enforce_global_step_recorded", True,
                        f"B 全程协议模式为 global(真实模式 {observed_modes})⇒ 无覆写 mark 属正确结果(marks={marks[:2]})"))
        else:
            out.append(("new_req_enforce_global_step_recorded", False,
                        f"无法从真实状态观察到 B 的协议模式 ⇒ 不得据此断言 marks 是否应存在"))
    else:
        out.append((name, config_enforce_global is False,
                    f"masked 臂载荷配置 enforce_global={config_enforce_global}(期望 False:不得经 enforce_global 拉回 global)"))
        out.append(("masked_no_forced_global_marks", not marks,
                    f"masked 臂不得出现强制全局覆写标记;实际 marks={marks[:2]}(共 {len(marks)})"))
    return out


def run_cleanup_check(llm, prompt_ids: list[int], *, payload: dict, patched: bool, out_dir: Path,
                      enforce_global: bool = True) -> dict:
    """校准 #5 的清理验收：**真实提交 → 驱动到产出 token → 执行中取消 → 清理轮 → 新带载荷请求**。

    判据只看**可观察效果**：是否真的产出 token、取消后是否真的不再 active、协议状态是否释放、
    新请求是否真的带载荷且跑出 token。对 `add_request/abort_request/step` 都无动作的引擎一律
    `ok=False`（旧实现对该假引擎返回 ok=True，已由本地复核反例否定）。

    patched 臂的 cleanup 请求**显式带载荷且 `enforce_global=True`**（不夹带 masked）；original 臂无补丁，
    请求不带载荷，协议状态不可观察时以 scheduler/未完成计数/worker req_ids 作为替代证据并明确标注。
    """
    engine = getattr(llm, "llm_engine", None)
    log = CheckLog()
    result: dict = {
        "ok": False,
        "checks": log.checks,
        "observations": {},
        "extra_args": None,
    }
    for attr in ("add_request", "step", "abort_request"):
        if not callable(getattr(engine, attr, None)):
            log.check("engine_api", False, f"引擎缺少 {attr}：无法做清理验收（不以调用过方法判成功）")
            result["failed_checks"] = log.failed
            return result

    extra_args = ({"attnview": dict(payload, enforce_global=enforce_global)} if patched else None)
    result["extra_args"] = extra_args

    def params(max_tokens: int):
        return sampling_params(max_tokens, extra_args=extra_args)

    output_mode = "delta"  # 四臂显式请求 DELTA；合并逻辑按实际形态分支（见 output_mode_of）

    prompt_ids = [int(t) for t in prompt_ids]
    stamp = int(time.time())
    ids = {
        "cancel": f"calib-cleanup-cancel-{stamp}",
        "new": f"calib-cleanup-new-{stamp}",
        "normal": f"calib-cleanup-normal-{stamp}",
        "follow": f"calib-cleanup-follow-{stamp}",
    }
    result["request_ids"] = ids
    submitted: dict[str, dict] = {}
    result["observations"]["ids"] = submitted

    def submit(label: str, params_obj) -> dict:
        """提交并登记 id 账本；返回**内部 id**（scheduler/worker/协议状态都以它为键）。"""
        ledger = submit_request(engine, ids[label], prompt_ids, params_obj)
        submitted[label] = ledger
        return ledger

    def internal(label: str) -> str:
        return submitted[label]["internal_id"]

    with Watchdog(CLEANUP_BUDGET_S, "cleanup", out_dir):
        try:
            # 1) 真实带载荷请求 A：驱动到**确实 forward 并产出 token**
            #    （宿主输出按 external id 归属；scheduler/协议状态按 internal id 查询）
            submit("cancel", params(8))
            drove = drive_until_token(engine, ids["cancel"], budget_steps=8, mode=output_mode)
            result["observations"]["cancel_req"] = drove
            log.check(
                "cancel_req_produced_token",
                bool(drove["tokens"]),
                f"A 产出 token={drove['tokens'][:4]}（步数 {drove['steps']}）——未见 token 即失败",
            )
            state_a = protocol_state(llm, internal("cancel"))
            result["observations"]["cancel_req_state"] = state_a
            if state_a.get("observable"):
                log.check(
                    "cancel_req_payload_state_created",
                    bool(state_a.get("in_registry")) and bool(state_a.get("in_configs")),
                    f"A 的请求级协议状态：registry={state_a.get('registry_ids')} "
                    f"in_configs={state_a.get('in_configs')} in_detok={state_a.get('in_detok')}",
                )
            else:
                log.check(
                    "cancel_req_state_unobservable_documented",
                    state_a.get("observable") is False,
                    f"协议状态不可从宿主侧观察（已记录）：{state_a.get('reason')}",
                )
            before = _scheduler_has(engine, internal("cancel"))
            result["observations"]["scheduler_has_cancel_before_abort"] = before
            log.check("cancel_req_active_before_abort", before is True, f"取消前 scheduler.requests 含 A：{before}")

            # 2) 执行中取消：pin `LLMEngine.abort_request` 默认 `internal=False` ⇒ 传**外部 id**，
            #    由 output_processor 的 external→internal 映射定位同一请求（pin `llm_engine.py:212`、
            #    `output_processor.py:494-524`）；随后用**内部 id** 核对它确实不再 active。
            engine.abort_request([ids["cancel"]])
            after = _scheduler_has(engine, internal("cancel"))
            result["observations"]["scheduler_has_cancel_after_abort"] = after
            log.check("abort_removed_from_scheduler", after is False, f"abort 后 scheduler.requests 含 A：{after}")

            # 3) finished-only 清理轮 + 状态释放
            rounds = drive_cleanup_rounds(llm, internal("cancel"))
            result["observations"]["cleanup_rounds"] = rounds
            log.check(
                "cancel_req_gone_after_cleanup",
                rounds["scheduler_has_req"] is False and rounds["unfinished"] in (0, None),
                f"清理 {rounds['rounds']} 轮后 scheduler_has_req={rounds['scheduler_has_req']} "
                f"unfinished={rounds['unfinished']}",
            )
            state_a2 = protocol_state(llm, internal("cancel"))
            result["observations"]["cancel_req_state_after_cleanup"] = state_a2
            if state_a2.get("observable"):
                leftovers = [
                    key for key in ("in_registry", "in_configs", "in_detok", "in_pending") if state_a2.get(key)
                ]
                log.check(
                    "cancel_req_protocol_state_released",
                    not leftovers,
                    f"清理后仍存在的协议状态字段：{leftovers or '无'}"
                    f"（registry={state_a2.get('registry_ids')}）",
                )
            else:
                log.check(
                    "cancel_req_state_release_documented",
                    rounds["scheduler_has_req"] is False,
                    "无法从宿主侧观察协议状态（已明确记录）："
                    f"{state_a2.get('reason')}；以 scheduler.requests/未完成计数释放作为替代证据",
                )
            workers = rounds["worker_req_ids"]
            result["observations"]["worker_req_ids_last_step"] = workers
            log.check(
                "worker_req_ids_release_evidence_recorded",
                True,
                "worker 侧 `execute_model_state.input_batch.req_ids` 是**上一次 forward 的快照**"
                f"（清理轮无调度步即不更新）：last={workers}；真正的 worker 侧释放以随后新请求的"
                "批次内容为准（见 worker_req_ids_after_new_request）",
            )

            # 4) 新带载荷请求 B：初始为 global；跑完 prefill + 首个 decode 步（真实标记）后自然结束。
            #    每个请求**结束再提交下一个**：`max_num_seqs=1` 下并发提交会让后一个请求饿死。
            probes_b: dict = {}

            def probe_b(step_index: int) -> None:
                probes_b[str(step_index)] = {
                    "state": protocol_state(llm, internal("new")),
                    "mode": protocol_mode(llm, internal("new")),  # 真实协议模式（逐 step）
                    "worker_req_ids": worker_req_ids(llm),
                }

            submit("new", params(2))
            drove_b = drive_until_finished(engine, ids["new"], budget_steps=4, on_step=probe_b,
                                       mode=output_mode)
            result["observations"]["new_req"] = drove_b
            result["observations"]["new_req_probes"] = probes_b
            log.check(
                "new_req_produced_token",
                len(drove_b["tokens"]) >= 2 and drove_b["finished"],
                f"B 产出 token={drove_b['tokens'][:4]}（步数 {drove_b['steps']}，finished={drove_b['finished']}）",
            )
            probed = [entry["state"] for entry in probes_b.values()]
            observable = [state for state in probed if state.get("observable")]
            if observable:
                log.check(
                    "new_req_payload_state_created",
                    any(state.get("in_configs") for state in observable),
                    f"B 在运行中确有请求级协议状态：{[ {k: s.get(k) for k in ('in_registry', 'in_configs', 'in_detok')} for s in observable ]}",
                )
                cfg_flags = [state.get("config_enforce_global") for state in observable]
                payload_flag = cfg_flags[0] if cfg_flags else None
                # 只有"协议模式**非 global** 且执行被 enforce_global 覆写"的步才留下 mark。
                marks = [mark for state in observable for mark in state.get("enforce_global_steps", ())
                         if str(mark.get("req_id")) == internal("new")]
                modes = {step: entry["mode"] for step, entry in probes_b.items()}
                observed_modes = [entry["mode"] for entry in modes.values() if entry.get("observable")]
                non_global_steps = [step for step, entry in modes.items()
                                    if entry.get("observable") and str(entry.get("mode")).lower() != "global"]
                result["observations"]["new_req_modes"] = modes
                for cname, cok, cmsg in cleanup_enforce_checks(
                        enforce_global=enforce_global, config_enforce_global=payload_flag, marks=marks,
                        non_global_steps=non_global_steps, observed_modes=observed_modes):
                    log.check(cname, cok, cmsg)
            else:
                log.check(
                    "new_req_state_unobservable_documented",
                    probed and all(state.get("observable") is False for state in probed),
                    "B 的协议状态不可从宿主侧观察（已明确记录）："
                    f"{probed[0].get('reason') if probed else '无探针'}",
                )
            batches = [entry["worker_req_ids"] for entry in probes_b.values()]
            if batches and any(batch is not None for batch in batches):
                observed_batches = [batch for batch in batches if batch is not None]
                log.check(
                    "worker_req_ids_after_new_request",
                    all(batch == [internal("new")] for batch in observed_batches),
                    f"B 的 worker 批次 req_ids={observed_batches}（期望 [{internal('new')}]：已取消的 A 不得再出现）",
                )
            else:
                log.check(
                    "worker_req_ids_after_new_request",
                    True,
                    "worker 侧批次内容不可从宿主观察（已记录），以 engine 侧状态作为替代证据",
                )
            log.check(
                "new_req_released_after_finish",
                _scheduler_has(engine, internal("new")) is False,
                f"B 正常结束后 scheduler.requests 含 B：{_scheduler_has(engine, internal('new'))}",
            )

            # 5) 正常结束路径：同样用带载荷请求，结束后再接一个新请求
            submit("normal", params(4))
            drove_c = drive_until_finished(engine, ids["normal"], budget_steps=6, mode=output_mode)
            result["observations"]["normal_req"] = drove_c
            log.check(
                "normal_req_produced_token_and_finished",
                bool(drove_c["tokens"]) and drove_c["finished"],
                f"C 产出 token={drove_c['tokens'][:4]} finished={drove_c['finished']}",
            )
            log.check(
                "normal_req_released_after_finish",
                _scheduler_has(engine, internal("normal")) is False,
                f"正常结束后 scheduler.requests 含 C：{_scheduler_has(engine, internal('normal'))}",
            )
            submit("follow", params(2))
            drove_d = drive_until_finished(engine, ids["follow"], budget_steps=4, mode=output_mode)
            result["observations"]["follow_req"] = drove_d
            log.check(
                "follow_req_produced_token",
                len(drove_d["tokens"]) >= 2 and drove_d["finished"],
                f"D 产出 token={drove_d['tokens'][:4]}（步数 {drove_d['steps']}，finished={drove_d['finished']}）",
            )
            log.check("cleanup_flow_completed", True, "清理段全部步骤执行完毕（无异常中断）")
        except Exception as exc:  # 引擎 API 异常/替身不实现：判失败，不吞成备注，也不让驱动崩溃
            log.check("cleanup_flow_completed", False, f"{type(exc).__name__}: {exc}")
            result["error"] = f"{type(exc).__name__}: {exc}"

    result["ok"] = bool(log.checks) and not log.failed
    result["failed_checks"] = log.failed
    return result


# --------------------------------------------------------------------------- #
# 产物校验（强制日志 / trace / logits / 轨迹）
# --------------------------------------------------------------------------- #


def capture_accounting(capture: LayerCapture, tokens: list[int], *, consumption_steps: int | None,
                       hook_steps: int | None = None, hook_source: str | None = None,
                       require_canonical_view: bool = True) -> dict:
    """按**真实信号**核对捕获步数、消费 token 与执行视图（四臂同一口径）。

    - **消费计数 = 宿主观测**：`engine.step()` 每轮里目标请求新增 token ≥ 1 的轮数
      （`drive_main_request` 的 `consuming_steps`；宿主输出 `RequestOutput.request_id` 是**外部 id**）。
      为什么不用 `force_step`：该计数只在适配层**实际替换 token 后**递增，`original` 臂未部署补丁、
      根本不调用钩子，拿它当通用计数会误判；
    - **forced 臂额外交叉核对**：钩子侧计数（`force_step` 终值；无强制轨迹时用 logits 钩子计数）
      必须 == 宿主观测，否则说明钩子与宿主观测分叉 ⇒ 失败；
    - **相位**：prefill/decode 由真实绝对位置判定，并与 `attn_metadata` 真实标记互核（见 `_end_step`）；
    - **执行视图**：本阶段三臂都必须是 canonical/global ⇒ FA metadata 的 `seq_lens[0]` 必须等于
      "本步最后位置 + 1"（受限读视图会缩短 `seqused_k`，adapter 覆写正是替换 `group_seq_lens`）；
    - 失败时把 `prefill 标记 / 消费步数(宿主) / decode 步号列表 / tokens 长度 / 钩子计数 / 视图证据`
      全写进 manifest。
    """
    steps = sorted(capture.records)
    phases = {step: capture.positions[step]["phase"] for step in steps if step in capture.positions}
    prefill_steps = [step for step, phase in phases.items() if phase == "prefill"]
    decode_indices = capture.decode_steps()
    account: dict = {
        "consumption_steps_host": consumption_steps,
        "hook_steps": hook_steps,
        "hook_source": hook_source,
        "tokens": len(tokens),
        "capture_steps": steps,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_indices,
        "decode_forwards": capture.decode_forwards(),
        "phases": {str(step): phases.get(step) for step in steps},
        "phase_evidence": {
            str(step): {
                "positions": capture.positions[step]["positions"][:4],
                "q_len": capture.positions[step]["q_len"],
                "positions_source": capture.positions[step]["positions_source"],
                "mrope_axes_shape": capture.positions[step]["mrope_axes_shape"],
                "phase": capture.positions[step]["phase"],
                "phase_source": capture.positions[step]["phase_source"],
                "phase_markers": capture.positions[step]["phase_markers"],
            }
            for step in steps
            if step in capture.positions
        },
        "view_evidence": {
            str(step): {
                "expected_seq_len": capture.positions[step]["positions"][-1] + 1,
                **capture.positions[step]["view"],
            }
            for step in steps
            if step in capture.positions
        },
    }
    problems: list[str] = []
    if len(prefill_steps) != 1:
        problems.append(f"prefill 步应恰好 1 个（真实位置判定），实际 {prefill_steps}")
    if consumption_steps is None:
        problems.append("宿主消费计数不可观察 —— 不得静默当成功")
    else:
        if consumption_steps != len(tokens):
            problems.append(f"宿主消费步数 {consumption_steps} != 消费 token 数 {len(tokens)}")
        if len(decode_indices) != consumption_steps - 1:
            problems.append(
                f"decode 捕获步数 {len(decode_indices)}（{decode_indices}）!= 宿主消费步数 "
                f"{consumption_steps} - 1（prefill 那一消费步不算 decode）"
            )
        if steps != list(range(1, consumption_steps + 1)):
            problems.append(f"捕获步 {steps} 与宿主消费步数 {consumption_steps} 不连续一致")
    if hook_steps is not None:
        if hook_steps != consumption_steps:
            problems.append(
                f"钩子侧计数（{hook_source or '未标注'}）{hook_steps} != 宿主消费步数 {consumption_steps}"
                "—— 钩子与宿主观测分叉"
            )
    if require_canonical_view:
        for step in steps:
            meta = capture.positions.get(step)
            if meta is None:
                continue
            view = meta["view"]
            expected = meta["positions"][-1] + 1
            if not view.get("observable"):
                problems.append(f"第 {step} 步无法取得 FA metadata 读长度证据：{view.get('reason')}")
                continue
            if view.get("seq_lens_first") != expected:
                problems.append(
                    f"第 {step} 步 FA 组读长度 seq_lens[0]={view.get('seq_lens_first')} != 本步最后位置+1"
                    f"（={expected}）：执行视图不是 canonical/global（受限读视图会缩短 seqused_k）"
                )
    account["problems"] = problems
    return account


def verify_force_log(path: Path, *, req_id: str, trajectory_tokens: list[int]) -> dict:
    """主请求的强制日志：step 从 1 连续、原始采样与强制值分列、**没有** warmup/其它 req_id 的记录。"""
    path = Path(path)
    if not path.exists():
        return {"exists": False, "records": 0, "steps": [], "problems": ["force.jsonl 不存在：强制钩子没有生效"]}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    problems: list[str] = []
    steps = [int(r.get("step", -1)) for r in records]
    if steps != list(range(1, len(records) + 1)):
        problems.append(f"step 不是从 1 连续：{steps[:12]}{'…' if len(steps) > 12 else ''}")
    if len(records) != len(trajectory_tokens):
        problems.append(f"强制消费步数 {len(records)} != 轨迹步数 {len(trajectory_tokens)}")
    for record in records:
        step = int(record.get("step", -1))
        if str(record.get("req_id")) != str(req_id):
            problems.append(f"step {step} 的 req_id={record.get('req_id')!r} 不是主请求 {req_id!r}")
        recorded_ids = [str(x) for x in (record.get("req_ids") or [])]
        if recorded_ids != [str(req_id)]:
            problems.append(f"step {step} 的 req_ids 混入其它请求：{recorded_ids}")
        raw = record.get("raw_sampled")
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
            problems.append(f"step {step} 的 raw_sampled 不是 tokens[step][row] 形状：{raw!r}")
        if not isinstance(record.get("forced"), list):
            problems.append(f"step {step} 的 forced 缺失或不是列表：{record.get('forced')!r}")
        if 1 <= step <= len(trajectory_tokens):
            expected = [int(trajectory_tokens[step - 1])]
            if [int(t) for t in (record.get("forced") or [])] != expected:
                problems.append(f"step {step} 的 forced={record.get('forced')} != 轨迹 {expected}")
    return {
        "exists": True,
        "records": len(records),
        "steps": steps,
        "consumption_steps": len(records),
        "problems": problems,
    }


def verify_override_trace(path: Path, *, req_id: str, expect_override_records: bool) -> dict:
    """受限读视图的覆写 trace（`steps.jsonl`）核对。

    本阶段三条臂的**执行视图都是 global**（`original` 无补丁；`patched-disabled` 请求无载荷 ⇒
    engine 侧不出计划；`patched-global` 载荷显式 `enforce_global=true` ⇒ engine 侧同样不出计划，见
    `attnview_engine.py:541-555`），因此 **override 记录本就不应存在**；它只可能由未来的 masked 臂产生。

    - `expect_override_records=False`（当前三臂）：文件缺失或为空都算通过；**出现任何记录即失败**
      （说明执行路径被改成了受限视图）；
    - `expect_override_records=True`（留给 masked 臂）：必须有记录、且都属于目标请求、且 override 非 null。

    绝不为"凑出 trace"而人为产出受限计划。
    """
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "records": 0,
            "problems": [] if not expect_override_records else ["steps.jsonl 不存在：目标请求没有覆写记录"],
        }
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    problems: list[str] = []
    for record in records:
        recorded_ids = [str(x) for x in (record.get("req_ids") or [])]
        if recorded_ids != [str(req_id)]:
            problems.append(f"trace 记录混入其它请求：{recorded_ids}（step={record.get('step')}）")
    if expect_override_records:
        if not records:
            problems.append("steps.jsonl 为空：目标请求没有覆写记录")
        if any(record.get("override") is None for record in records):
            problems.append("存在 override=None 的 trace 记录：该步并未真正覆写")
    elif records:
        problems.append(
            f"执行视图为 global 的臂不应有受限读视图覆写记录，却出现 {len(records)} 条"
            "（不得为造 trace 改执行路径）"
        )
    return {"exists": True, "records": len(records), "problems": problems}


def verify_logits_records(records: list, *, req_id: str, expected_steps: int, expect_argmax: bool) -> dict:
    """完整 logits：步号连续、只属主请求、形状为完整词表、且与主请求消费 token 对齐。"""
    problems: list[str] = []
    steps: list[int] = []
    argmax_mismatch: list[dict] = []
    for index, record in enumerate(records):
        steps.append(int(record.get("step", -1)))
        if str(record.get("req_id")) != str(req_id):
            problems.append(f"第 {index + 1} 条记录的 req_id={record.get('req_id')!r} 不是主请求")
        recorded_ids = [str(x) for x in (record.get("req_ids") or [])]
        if recorded_ids != [str(req_id)]:
            problems.append(f"第 {index + 1} 条记录的 req_ids 混入其它请求：{recorded_ids}")
        tensor = record.get("logits")
        if tensor is None or int(tensor.shape[0]) != 1:
            problems.append(f"第 {index + 1} 条记录不是单行完整 logits：{getattr(tensor, 'shape', None)}")
        elif expect_argmax:
            argmax = int(torch.argmax(tensor[0]).item())
            argmax_mismatch.append({"step": int(record.get("step", -1)), "argmax": argmax})
    if steps != list(range(1, len(records) + 1)):
        problems.append(f"logits 步号不是从 1 连续：{steps[:12]}")
    if len(records) != int(expected_steps):
        problems.append(f"logits 记录数 {len(records)} != 主请求步数 {expected_steps}")
    return {
        "records": len(records),
        "steps": steps,
        "argmax": argmax_mismatch,
        "problems": problems,
    }


def trajectory_token_sequence(trajectory: dict) -> list[int]:
    """把 `tokens[step][row]` 摊成本驱动比较用的**消费 token 序列**（单活跃请求：每步 1 行 × 1 token）。"""
    sequence: list[int] = []
    for index, rows in enumerate(trajectory["tokens"]):
        if len(rows) != 1 or len(rows[0]) != 1:
            raise RuntimeError(
                f"校准: 轨迹第 {index + 1} 步不是「1 行 × 1 token」（{rows!r}）—— 本驱动只支持单活跃请求"
            )
        sequence.append(int(rows[0][0]))
    return sequence


def compare_tokens(produced: list[int], expected: list[int]) -> dict:
    """逐 token 比对（每步消费 prefix 完全相同才算通过）。"""
    limit = min(len(produced), len(expected))
    first_divergence = None
    for index in range(limit):
        if int(produced[index]) != int(expected[index]):
            first_divergence = index
            break
    if first_divergence is None and len(produced) != len(expected):
        first_divergence = limit
    return {
        "produced": [int(t) for t in produced],
        "expected": [int(t) for t in expected],
        "compared_tokens": limit,
        "first_divergence": first_divergence,
        "identical": first_divergence is None,
    }


def load_trajectory(path: Path, *, prompt_ids: list[int]) -> dict:
    """读强制轨迹并**核对它属于同一个 prompt**（不同 prompt 的轨迹拒绝回放）。"""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "tokens" not in data:
        raise RuntimeError(f"校准: 轨迹文件 {path} 缺 'tokens' 字段")
    tokens = data["tokens"]
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"校准: 轨迹 {path} 的 tokens 为空")
    for index, rows in enumerate(tokens):
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise RuntimeError(
                f"校准: 轨迹 {path} 第 {index + 1} 步格式不对 —— 需要 tokens[step][row] = [token_ids...]"
            )
    recorded_len = data.get("prompt_len")
    recorded_hash = data.get("prompt_token_ids_sha256")
    if recorded_len is None or recorded_hash is None:
        raise RuntimeError(f"校准: 轨迹 {path} 缺 prompt_len/prompt_token_ids_sha256，拒绝回放")
    if int(recorded_len) != len(prompt_ids) or str(recorded_hash) != sha256_token_ids(prompt_ids):
        raise RuntimeError(
            f"校准: 轨迹 {path} 属于另一条 prompt（轨迹 len={recorded_len} hash={recorded_hash}，"
            f"本次 len={len(prompt_ids)} hash={sha256_token_ids(prompt_ids)}）—— 拒绝回放"
        )
    return {"path": str(path), "tokens": [[[int(t) for t in row] for row in rows] for rows in tokens],
            "source_arm": data.get("arm"), "created_cst": data.get("created_cst")}


def write_trajectory(path: Path, *, tokens: list[int], prompt_ids: list[int], arm: str,
                     request_id: str, internal_request_id: str | None = None, source: str,
                     logits_sha256: str | None) -> dict:
    """写强制轨迹（`tokens[step][row]`；单请求每步 1 个 token）。

    `request_id` 记**外部 id**（宿主侧可读），`internal_request_id` 记**内部 id**（钩子绑定所用）；
    轨迹回放的校验只看 prompt 长度与哈希，与 id 无关。
    """
    payload = {
        "schema": "attnview.p2-calib-trajectory/v1",
        "tokens": [[[int(t)]] for t in tokens],
        "arm": arm,
        "request_id": request_id,
        "internal_request_id": internal_request_id,
        "prompt_len": len(prompt_ids),
        "prompt_token_ids_sha256": sha256_token_ids(prompt_ids),
        "model_revision": SNAPSHOT.name,
        "trajectory_source": source,
        "logits_sha256": logits_sha256,
        "created_cst": now_cst(),
        "max_tokens": len(tokens),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SUP-004 模型校准驱动（三身份：original/patched-disabled/patched-global）")
    ap.add_argument("--arm", choices=ARMS, required=True,
                    help="original = 未打补丁的原版；patched-disabled = 打补丁但请求无载荷（插入层关闭）；"
                         "patched-global = 真实带载荷且 enforce_global=True；patched-masked = 真实带载荷且 enforce_global=False(诊断)")
    ap.add_argument("--out", type=Path, required=True, help="本次运行的**新**输出目录（已存在且非空即拒绝）")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--doc-fixture", type=Path,
                    help="含 document/question 的夹具(evidence/p1-cpu/demo-fixtures.json)。仅 patched-masked 臂可用。")
    ap.add_argument("--expect-fixture", type=Path,
                    help="已验收产物(filler_units/fine_units/context_sha256/prompt_len/spans/token 哈希)。")
    ap.add_argument("--timeline-config", type=Path,
                    help="提供 filler_unit/fine_char 的配置(configs/p2-masked-prep/crossblock.json)。")
    ap.add_argument("--emit-trajectory", type=Path, help="把本次主请求的 token 序列写成强制轨迹")
    ap.add_argument("--force-trajectory", type=Path, help="按给定轨迹强制（同轨迹回放；original 臂禁止）")
    ap.add_argument("--compare-to", type=Path, help="与给定轨迹**逐 token** 比对（original×2 机械断言）")
    ap.add_argument("--record-layers", action=argparse.BooleanOptionalAction, default=True,
                    help="层观测（默认开）")
    ap.add_argument("--capture-host-logits", action=argparse.BooleanOptionalAction, default=None,
                    help="宿主级完整 logits 捕获；默认：original 开（唯一来源）、patched 关（用 worker 钩子）")
    ap.add_argument("--cleanup-check", action="store_true",
                    help="额外做生命周期验收（真实提交 → 取消 → 清理轮 → 新带载荷请求）")
    return ap.parse_args(argv)


def prepare_output_dir(out: Path) -> None:
    """输出目录只写新 run：已存在且非空即拒绝（不覆盖旧目录/旧证据）。"""
    if out.exists():
        existing = sorted(p.name for p in out.iterdir())
        if existing:
            raise RuntimeError(f"校准: 输出目录 {out} 已存在且非空（{existing[:5]}）—— 拒绝覆盖旧 run")
    out.mkdir(parents=True, exist_ok=True)


def _run(args: argparse.Namespace, manifest: dict, manifest_path: Path, prompt, payload: dict,
         prompt_ids: list[int], extra_args: dict | None, arm_path: Path) -> int:
    out = args.out
    patched = args.arm != "original"
    log = CheckLog()

    trajectory = load_trajectory(args.force_trajectory, prompt_ids=prompt_ids) if args.force_trajectory else None
    compare_to = load_trajectory(args.compare_to, prompt_ids=prompt_ids) if args.compare_to else None
    if trajectory is not None and not patched:
        raise RuntimeError("校准: original 臂没有强制钩子（补丁未部署）—— 轨迹由它产生，不由它回放")
    trajectory_tokens = trajectory_token_sequence(trajectory) if trajectory is not None else []

    from vllm import LLM

    llm_kwargs = dict(
        model=str(SNAPSHOT),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        # 7834-token prompt 必须**单次 prefill**;仅 masked 臂固定该项,旧三臂 kwargs 不变。
        **({"max_num_batched_tokens": 8192} if args.arm == "patched-masked" else {}),
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
    manifest["config"]["llm_kwargs"] = {k: str(v) for k, v in llm_kwargs.items()}
    manifest["config"]["max_num_batched_tokens_requested"] = llm_kwargs.get("max_num_batched_tokens")
    try:   # 记录**真实生效值**(不是只看传入 kwargs)
        _sc = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        _sc = getattr(_sc, "scheduler_config", None)
        manifest["config"]["max_num_batched_tokens_effective"] = (
            None if _sc is None else getattr(_sc, "max_num_batched_tokens", None))
    except Exception as exc:  # noqa: BLE001
        manifest["config"]["max_num_batched_tokens_effective"] = f"unavailable: {type(exc).__name__}"

    with Watchdog(STARTUP_BUDGET_S, "startup", out, manifest_path) as startup_watch:
        llm = LLM(**llm_kwargs)
    startup_s = startup_watch.elapsed
    if startup_s > STARTUP_BUDGET_S:  # 双保险（看门狗正常路径下已退出）
        raise BudgetExceeded(f"启动耗时 {startup_s:.1f}s 超过预算 {STARTUP_BUDGET_S}s")
    manifest["startup_s"] = startup_s

    engine = llm.llm_engine
    manifest["engine"] = engine_observability(llm)
    manifest["geometry_rpc"] = (
        llm.llm_engine.model_executor.collective_rpc("attnview_geometry", single_value=True)
        if patched
        else {"skipped": "original 臂没有该 RPC（补丁未部署）"}
    )

    capture = install_capture(llm) if args.record_layers else None
    capture_host = args.capture_host_logits
    if capture_host is None:
        capture_host = not patched
    manifest["host_logits"] = {"enabled": bool(capture_host), "reason": (
        "original 臂唯一的完整 logits 来源" if not patched else "worker 钩子已提供逐步完整 logits；宿主捕获仅作交叉核对"
    )}
    if capture is not None:
        manifest["capture"] = capture.summary()

    run_stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    external_main_id = f"calib-main-{args.arm}-{run_stamp}"
    params = sampling_params(args.max_tokens, extra_args=extra_args)

    # ④ 直接提交阶段 03 的最终 token_ids（不重新 tokenize）；用返回的**内部 id** 做绑定与查询。
    #    控制文件必须在**首次 step() 之前**写入（首个消费步就要绑定到正确 id）。
    ledger = submit_request(engine, external_main_id, prompt_ids, params)
    internal_main_id = ledger["internal_id"]
    manifest["ids"] = ledger
    trajectory_steps = trajectory["tokens"] if trajectory is not None else []
    arm_content = {
        "target_req_id": internal_main_id,
        "tokens": trajectory_steps,
        "force_log": str(out / "force.jsonl") if trajectory_steps else None,
        "logits_path": str(out / "logits.pt") if patched else None,
        "trace_path": str(out / "steps.jsonl") if patched else None,
        "note": (
            "original 臂无补丁钩子：logits 由宿主级捕获写入 logits.pt（本文件不被读取）"
            if not patched
            else "tokens 为同轨迹强制序列（tokens[step][row]）；文件在 add_request 之后、首次 step() 之前写入"
        ),
        "external_req_id": external_main_id,
    }
    arm_path.write_text(json.dumps(arm_content, ensure_ascii=False, indent=2))
    manifest["arm_file"] = {
        "path": str(arm_path),
        "exists": True,
        "written_cst": now_cst(),
        "content": arm_content,
    }
    save_manifest(manifest_path, manifest)

    probe: dict = {}

    def on_first_step() -> None:
        requests = scheduler_requests(engine) or {}
        request = requests.get(internal_main_id)  # 内部 id 才是调度器账本的键
        probe["engine_prompt_ids"] = [int(t) for t in (getattr(request, "prompt_token_ids", ()) or ())]
        probe["scheduler_has_internal"] = _scheduler_has(engine, internal_main_id)
        probe["scheduler_has_external"] = _scheduler_has(engine, external_main_id)
        probe["protocol_state_first_step"] = protocol_state(llm, internal_main_id)
        probe["canonical_blocks"] = canonical_blocks(engine, internal_main_id)

    host_logits: list = []
    restore_host = wrap_host_logits(llm, host_logits, req_id=internal_main_id) if capture_host else None
    if capture is not None:
        capture.arm(internal_main_id, out / "capture", prompt_len=len(prompt_ids))
    try:
        with Watchdog(REQUEST_BUDGET_S, "request", out, manifest_path) as req_watch:
            # ⑤ engine.step() 驱动到完成（看门狗 + 步数上界）；宿主侧输出按**外部 id** 归属
            drove = drive_main_request(engine, external_main_id, max_tokens=args.max_tokens,
                                       on_first_step=on_first_step, mode=output_mode_of(params))
        request_s = req_watch.elapsed
        if request_s > REQUEST_BUDGET_S:  # 双保险
            raise BudgetExceeded(f"主请求耗时 {request_s:.1f}s 超过预算 {REQUEST_BUDGET_S}s")
    finally:
        if capture is not None:
            capture.disarm()
        if restore_host is not None:
            restore_host()

    tokens = drove["tokens"]
    steps = drove["steps"]
    log.check(
        "engine_prompt_ids_match_render",
        probe.get("engine_prompt_ids") == prompt_ids,
        f"引擎实收 prompt ids 与 render_arm 一致：{probe.get('engine_prompt_ids') == prompt_ids}"
        f"（引擎 {len(probe.get('engine_prompt_ids') or [])} 个 / 渲染 {len(prompt_ids)} 个）",
    )
    log.check(
        "scheduler_keyed_by_internal_id",
        probe.get("scheduler_has_internal") is True and probe.get("scheduler_has_external") is False,
        f"scheduler.requests 命中内部 id={probe.get('scheduler_has_internal')}、"
        f"外部 id={probe.get('scheduler_has_external')}（pin 随机化后外部 id 不应出现在调度器账本）",
    )
    log.check(
        "main_request_produced_tokens",
        len(tokens) == args.max_tokens,
        f"主请求消费 token={tokens}（期望 {args.max_tokens} 个）",
    )
    log.check("only_target_request_outputs", not drove["other_request_outputs"],
              f"驱动期间出现的其它请求输出：{drove['other_request_outputs'][:4] or '无'}")
    if capture is not None:
        observed = sorted(capture.records)
        missing_layers = {
            str(step): [capture.layer_names[i] for i in sorted(capture.layer_names)
                        if i not in (capture.records.get(step) or {})]
            for step in observed
        }
        missing_layers = {k: v for k, v in missing_layers.items() if v}
        log.check(
            "capture_steps_match_drive",
            observed == list(range(1, steps + 1)),
            f"捕获步 {observed} vs 驱动步数 {steps}",
        )
        log.check(
            "capture_all_fa_layers_present",
            not missing_layers,
            f"缺层的步：{missing_layers or '无'}（预期全注意力层 {len(capture.layer_names)} 个）",
        )
        log.check(
            "capture_positions_slice_consistent",
            all(meta["positions"] == list(range(meta["positions"][0], meta["positions"][0] + meta["q_len"]))
                for meta in capture.positions.values())
            and all(meta["q_len"] == len(meta["positions"]) for meta in capture.positions.values()),
            "逻辑位置必须连续且来自真实一维缓冲；mRoPE 三轴只作记录："
            f"{ {s: {'source': m['positions_source'], 'mrope_axes_shape': m['mrope_axes_shape'], 'q_len': m['q_len']} for s, m in sorted(capture.positions.items())} }",
        )
        manifest["capture"] = capture.summary()

    # worker 侧 FA metadata 块表 vs engine 侧 canonical 块表（执行视图旁证，记录用）
    if capture is not None:
        manifest["block_table_evidence"] = block_table_evidence(
            probe.get("canonical_blocks") or {"observable": False, "reason": "首步探针未取到"}, capture
        )

    # 强制日志 / trace
    host_consuming_steps = int(drove["consuming_steps"])
    hook_steps: int | None = None
    hook_source: str | None = None
    if trajectory is not None:
        force = verify_force_log(out / "force.jsonl", req_id=internal_main_id, trajectory_tokens=trajectory_tokens)
        force["ok"] = force.get("exists") and not force["problems"]
        log.check("force_log_is_target_only", bool(force["ok"]),
                  f"records={force['records']} steps={force.get('steps', [])[:6]}… problems={force['problems']}")
        manifest["force_log"] = force
        if force.get("exists"):
            hook_steps, hook_source = int(force["consumption_steps"]), "force.jsonl（force_step 终值）"
        log.check(
            "force_step_matches_host_consumption",
            force.get("exists") and int(force["consumption_steps"]) == host_consuming_steps,
            f"钩子侧 force_step={force.get('consumption_steps')} == 宿主消费计数 {host_consuming_steps}",
        )
    elif out.joinpath("force.jsonl").exists():
        log.check("force_log_absent_without_trajectory", False, "未给轨迹却出现 force.jsonl：钩子绑定越界")
    if patched:
        trace = verify_override_trace(out / "steps.jsonl", req_id=internal_main_id, expect_override_records=False)
        trace["ok"] = not trace["problems"]
        log.check("worker_trace_is_target_only", bool(trace["ok"]),
                  f"exists={trace['exists']} records={trace['records']} problems={trace['problems']}")
        manifest["worker_trace"] = trace

    # logits
    logits_path = out / "logits.pt"
    if patched:
        records = torch.load(logits_path, weights_only=False) if logits_path.exists() else []
        report = verify_logits_records(records, req_id=internal_main_id, expected_steps=steps, expect_argmax=False)
        log.check(
            "worker_logits_target_only",
            logits_path.exists() and not report["problems"],
            f"exists={logits_path.exists()} records={report['records']} == 主请求步数 {steps}，"
            f"problems={report['problems']}",
        )
        manifest["logits"] = {"path": str(logits_path), "source": "worker hook（arm.logits_path）", **report}
        if hook_steps is None and logits_path.exists():
            hook_steps, hook_source = int(report["records"]), "worker logits 钩子记录数（logits_step 终值）"
    if host_logits:
        report = verify_logits_records(host_logits, req_id=internal_main_id, expected_steps=steps,
                                      expect_argmax=not patched)
        argmax_ok = None
        if not patched:
            mismatches = [
                item for item in report["argmax"] if int(item["argmax"]) != int(tokens[item["step"] - 1])
            ] if len(tokens) >= len(report["argmax"]) else [{"note": "token 数不足"}]
            argmax_ok = not mismatches
            log.check(
                "host_logits_argmax_matches_sampled",
                bool(argmax_ok),
                f"逐步 argmax 与自然 greedy 采样一致：{argmax_ok}（不一致 {mismatches[:3]}）",
            )
        host_path = logits_path if not patched else out / "logits-host.pt"
        save_host_logits(host_logits, host_path)
        manifest["host_logits"].update({"path": str(host_path), "records": report["records"],
                                        "argmax_matches_sampled": argmax_ok})
        if not patched:
            log.check(
                "host_logits_target_only",
                logits_path.exists() and not report["problems"],
                f"records={report['records']} == 主请求步数 {steps}，problems={report['problems']}",
            )
            log.check(
                "host_logits_steps_match_host_consumption",
                int(report["records"]) == host_consuming_steps,
                f"宿主 logits 记录数 {report['records']} == 宿主消费计数 {host_consuming_steps}",
            )

    # 步数/消费 token 账本：消费计数取**宿主观测**（四臂同一口径）；forced 臂再与钩子侧计数交叉核对；
    # prefill/decode 由真实位置与 attn_metadata 真实标记判定，不用算术恒等式。
    if capture is not None:
        accounting = capture_accounting(
            capture,
            tokens,
            consumption_steps=host_consuming_steps,
            hook_steps=hook_steps,
            hook_source=hook_source,
        )
        log.check(
            "capture_accounting",
            not accounting["problems"],
            f"宿主消费步数={accounting['consumption_steps_host']} 钩子计数={accounting['hook_steps']}"
            f"（{accounting['hook_source']}）tokens={accounting['tokens']} "
            f"捕获步={accounting['capture_steps']} prefill={accounting['prefill_steps']} "
            f"decode={accounting['decode_steps']} problems={accounting['problems']}",
        )
        manifest["capture_accounting"] = accounting

    # 轨迹比对（original×2 的机械断言；patched 臂同样必须与轨迹逐 token 相同）
    if compare_to is not None:
        comparison = compare_tokens(tokens, trajectory_token_sequence(compare_to))
        comparison["reference"] = compare_to["path"]
        comparison["resolution"] = (
            "逐 token 相同：自然 greedy 在原版上可重复（patched 臂亦为同轨迹重放）"
            if comparison["identical"]
            else "不一致 ⇒ 本项失败；后续臂必须**改用同轨迹重放**（--force-trajectory 用同一份轨迹）"
                 "，不得把不同 prefix 的 logits 当数值误差比较"
        )
        manifest["trajectory_comparison"] = comparison
        log.check("trajectory_identical_to_reference", bool(comparison["identical"]),
                  f"compared={comparison['compared_tokens']} first_divergence={comparison['first_divergence']} "
                  f"({comparison['resolution']})")

    # 轨迹输出（original 臂的自然 greedy 序列是后续臂的回放来源）
    if args.emit_trajectory is not None:
        emitted = write_trajectory(
            args.emit_trajectory,
            tokens=tokens,
            prompt_ids=prompt_ids,
            arm=args.arm,
            request_id=external_main_id,
            internal_request_id=internal_main_id,
            source="natural-greedy" if not patched else "forced-replay",
            logits_sha256=sha256_file(logits_path) if logits_path.exists() else None,
        )
        manifest["trajectory_emitted"] = {"path": str(args.emit_trajectory), **emitted}

    # cleanup / 生命周期验收（主请求产物已先落盘）
    if args.cleanup_check:
        cleanup = run_cleanup_check(
            llm, prompt_ids, payload=dict(payload), patched=patched, out_dir=out,
            enforce_global=(args.arm != "patched-masked")
        )
        manifest["cleanup"] = cleanup
        log.check("cleanup_check_ok", bool(cleanup["ok"]),
                  f"清理验收 ok={cleanup['ok']} 失败项={cleanup.get('failed_checks')}")
        (out / "cleanup.json").write_text(json.dumps(cleanup, ensure_ascii=False, indent=2))

    # 主请求结束后的引擎侧状态（含 protocol 释放）
    post_state = protocol_state(llm, internal_main_id)
    if post_state.get("observable"):
        leftovers = [k for k in ("in_registry", "in_configs", "in_detok", "in_pending") if post_state.get(k)]
        log.check(
            "main_req_protocol_released_after_finish",
            not leftovers,
            f"主请求正常结束后仍存在的协议状态字段：{leftovers or '无'}"
            f"（registry={post_state.get('registry_ids')}）",
        )
    else:
        log.check(
            "main_req_protocol_release_documented",
            _scheduler_has(engine, internal_main_id) is False,
            "无法从宿主侧观察协议状态（已明确记录）："
            f"{post_state.get('reason')}；以 scheduler.requests 释放作为替代证据",
        )
    manifest["main_request"] = {
        "request_id": external_main_id,
        "internal_request_id": internal_main_id,
        "prompt_ids_sha256": sha256_token_ids(prompt_ids),
        "steps": steps,
        "tokens": tokens,
        "elapsed_s": request_s,
        "capture_steps": sorted(capture.records) if capture is not None else [],
        "first_step_probe": probe,
        "scheduler_has_req_after": _scheduler_has(engine, internal_main_id),
        "protocol_state_after": post_state,
    }
    if capture is not None and capture.records:
        dump_capture(capture, out / "capture" / "layers.npz")
        manifest["capture"]["npz"] = str(out / "capture" / "layers.npz")
        manifest["capture"]["npz_sha256"] = sha256_file(out / "capture" / "layers.npz")
    manifest["engine_traces"] = dump_engine_traces(llm, out / "engine-traces.json")

    manifest["checks"] = log.checks
    manifest["failures"] = log.failed
    save_manifest(manifest_path, manifest)
    print(json.dumps({
        "arm": args.arm,
        "startup_s": round(startup_s, 1),
        "request_s": round(request_s, 2),
        "steps": steps,
        "tokens": tokens,
        "failures": log.failed,
    }, ensure_ascii=False))
    return 1 if log.failed else 0


def freeze_run_source(out: Path) -> dict:
    """运行源冻结证据：脚本自身 sha256 + 源码快照（复制到本次 run 目录）。

    流程约束（R2）：**先提交再运行**；起时落指纹、运行期不覆写脚本；结束后才修，并**新目录**起新 run。
    这里在起时把当前脚本复制到 `<out>/source/` 并记录双方 sha256，供事后核对"跑的到底是哪份代码"。
    """
    out = Path(out)
    script = Path(__file__).resolve()
    digest = sha256_file(script)
    snapshot_dir = out / "source"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / script.name
    shutil.copyfile(script, snapshot)
    snapshot_digest = sha256_file(snapshot)
    if snapshot_digest != digest:
        raise RuntimeError(f"校准: 源码快照与运行脚本不一致（{snapshot_digest} != {digest}）")
    return {
        "script_path": str(script),
        "script_sha256": digest,
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": snapshot_digest,
        "note": "运行期不得覆写本脚本；结束后如需修改，请用新输出目录起新 run",
    }


def assert_run_source_unchanged(snapshot: dict) -> str | None:
    """结束前复核脚本未被覆写；不一致返回差异说明（写进 manifest 并由调用方判失败）。"""
    script = Path(snapshot.get("script_path", ""))
    if not script.exists():
        return f"运行脚本 {script} 已不存在"
    digest = sha256_file(script)
    if digest != snapshot.get("script_sha256"):
        return f"运行期脚本被修改：{digest} != {snapshot.get('script_sha256')}（本 run 结论不可信）"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out
    prepare_output_dir(out)
    manifest_path = out / "manifest.json"

    # ① 只设置控制文件路径，**不创建文件**（warmup 期间钩子必须未 armed）
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    arm_path = out / "arm.json"
    os.environ["ATTNVIEW_CALIB_ARM"] = str(arm_path)
    if arm_path.exists():
        raise RuntimeError(f"校准: {arm_path} 已存在 —— 本次运行必须从「文件不存在」开始")

    if args.doc_fixture is not None or args.expect_fixture is not None or args.timeline_config is not None:
        if args.arm != "patched-masked":
            raise RuntimeError(f"校准: fixture 参数仅 patched-masked 臂使用(当前 {args.arm})——不改变旧三臂输入")
        if not (args.doc_fixture and args.expect_fixture and args.timeline_config):
            raise RuntimeError("校准: 需同时给出 --doc-fixture / --expect-fixture / --timeline-config")
        prompt, payload, fixture_evidence = build_prompt_from_fixture(
            args.doc_fixture, args.expect_fixture, args.timeline_config)
    else:
        prompt, payload = build_prompt()
        fixture_evidence = None
    # 载荷语义(原契约不变):original 无补丁;patched-disabled 无载荷;patched-global 载荷+True;
    # 新增 patched-masked = 真实载荷 + enforce_global=False(诊断,不经 enforce_global 走 global)。
    payload["enforce_global"] = args.arm == "patched-global"
    prompt_ids = [int(t) for t in prompt.token_ids]
    payload_arms = ("patched-global", "patched-masked")
    extra_args = {"attnview": payload} if args.arm in payload_arms else None

    frozen = freeze_run_source(out)
    manifest: dict = {
        "schema": SCHEMA,
        "arm": args.arm,
        "started_cst": now_cst(),
        "source": frozen,
        "head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain").splitlines(),
        "pin_commit": _git("rev-parse", "HEAD", cwd=REPO / "vllm"),
        #: 部署态前置校验结果：**起时 manifest 先落盘**，校验在 try 内执行（失败也有 manifest/error.txt 落点）
        "deployment": None,
        "model": {"snapshot": str(SNAPSHOT), "revision": SNAPSHOT.name},
        "prompt": {
            "prompt_len": len(prompt_ids),
            "token_ids": prompt_ids,
            "token_ids_sha256": sha256_token_ids(prompt_ids),
            "rendered_sha256": sha256_text(prompt.rendered),
            "segment_spans": [list(s) for s in prompt.segment_spans],
        },
        "payload": payload,
        "fixture_evidence": fixture_evidence,
        "extra_args": extra_args,
        "config": {
            "llm_kwargs": None,
            "env": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "ATTNVIEW_CALIB_ARM": str(arm_path)},
        },
        "budgets": {"startup_s": STARTUP_BUDGET_S, "request_s": REQUEST_BUDGET_S, "cleanup_s": CLEANUP_BUDGET_S},
        "arm_file": {"path": str(arm_path), "exists": False, "content": None},
        "sources": {
            "emit_trajectory": str(args.emit_trajectory) if args.emit_trajectory else None,
            "force_trajectory": str(args.force_trajectory) if args.force_trajectory else None,
            "compare_to": str(args.compare_to) if args.compare_to else None,
        },
        "ended_cst": None,
        "exit_code": None,
    }
    save_manifest(manifest_path, manifest)

    exit_code = 1
    try:
        # 部署态前置校验（起时 manifest 之后的第一个阶段）：任何失败都走统一异常路径
        # ⇒ 写 manifest（failures/ended_cst/exit_code）+ 写 error.txt（完整 traceback）+ 非零退出。
        manifest["deployment"] = deployment_fingerprint(args.arm)
        save_manifest(manifest_path, manifest)
        exit_code = _run(args, manifest, manifest_path, prompt, payload, prompt_ids, extra_args, arm_path)
    except BudgetExceeded as exc:
        manifest.setdefault("notes", []).append(str(exc))
        exit_code = 3
    except Exception as exc:
        manifest.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        manifest.setdefault("failures", []).append(f"exception:{type(exc).__name__}")
        (out / "error.txt").write_text(traceback.format_exc())
        exit_code = 1
    finally:
        drift = assert_run_source_unchanged(manifest.get("source") or {})
        manifest.setdefault("source", {})["unchanged_at_end"] = drift is None
        if drift:
            manifest.setdefault("failures", []).append("script_modified_during_run")
            manifest.setdefault("errors", []).append(drift)
            if int(exit_code) == 0:
                exit_code = 1  # 运行期脚本被改 ⇒ 本 run 结论不可信，不得以 0 退出
        manifest["ended_cst"] = now_cst()
        manifest["exit_code"] = int(exit_code)
        save_manifest(manifest_path, manifest)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
