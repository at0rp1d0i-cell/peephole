#!/usr/bin/env python3
"""阶段 05 SUP-004 单请求模型校准：独立 FP32 dense 全局注意力参考（oracle）。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-calib-oracle.py --capture <capture.npz> --out <report.json>
    # 选定点模式（只算少量点；默认全量行为不变）：
    "$ATTNVIEW_PYTHON" tools/p2-calib-oracle.py --capture <capture.npz> --out <report.json> \
        --points <points.json>
    其中 points.json 形如
        {"prefill": [{"layer": 14, "position": 123}, …],
         "decode":  [{"layer": 14, "step": 1}, …]}
    选定点模式下每个点在报告的 `comparisons` 数组里（scope=prefill|decode，字段与全量模式同键同值），
    并额外给出尺度指标：`l2_diff`、`rel_l2_out`、`rel_l2_ref`、`|ref|` 的 p50/p99、
    `abs_ref_near_zero_fraction`（|ref| ≤ 1e-3 占比）、`elements`。
    注意 `rel_err`（= max_abs_err / L2(out)）是既有口径，**不是** torch.allclose 的 rtol
    （报告里以 `rel_err_is_allclose_rtol=false` 标注）。本工具不给任何通过/不通过阈值。

退出码：0=全部比较有限且报告已写出；2=参数/格式/必需元数据缺失；3=存在非有限值（报告仍写出）。

观测点与数值口径（README）
--------------------------
* 观测点：`out_step{i}_L{L}` 是该步该层**注意力子层（FA）的输出**，即 softmax(Q·Kᵀ·scale)·V；
  不含 o_proj、残差与后续 MLP。`q_step{i}_L{L}` 是该步该层的 query，`k_prefill_L{L}`/`v_prefill_L{L}`
  是 canonical 顺序（绝对位置 0..prompt_len-1）的 prefill K/V。
* Q 的 norm/RoPE 与 K 的 norm/RoPE 已包含在捕获张量里，`scale` 由捕获文件给出——本 oracle
  **不重复施加 RoPE，也不重算 scale**；`scale` 缺失直接报错，不用 head_dim^-0.5 之类兜底。
  若捕获自带 `scale_source` 声明且其值为 derived（非运行时真实值），本工具不拒绝，
  而是把它单列为已知近似来源（报告 `numerics.scale_is_approximate` 与 `warnings`）。
* 全部计算在 FP32：捕获到的 BF16 张量先 `.float()`（被升位的数组在报告 `numerics.upcast_to_float32`
  里逐个列出，`dtype_name` 记录模型侧精度）；softmax 数值稳定（每行减最大值再归一化）。
* 因果上界：绝对位置 p 的 query 只使用 key 下标 0..p（含自身）。GQA 按**连续分组**展开：
  query head h 使用 kv head h // (num_heads // num_kv_heads)。
* decode 步链：canonical KV = `k_prefill_L{L}` + `k_current_step1_L{L}` … `k_current_step{i}_L{L}`
  （每步当前写入 token，位置来自各自的 `decode_pos_step{i}`）。**步号必须从 1 连续、累计位置必须与
  prefill 段拼成连续区间（无空洞）**，该步才给完整参考；任何缺口（缺中间步的 current、位置空洞、
  形状不符、层覆盖不全）一律退出码 2，并指明缺哪一段/哪一步，不猜测补齐。

已知近似来源（都会进入误差，不能全部算作候选实现的问题）
--------------------------------------------------------
1. BF16 捕获输入的舍入，以及该舍入经 softmax/加权和放大后的影响；
2. FA kernel 与 oracle 的实现差异：在线 softmax、累加顺序与归约、累加器精度等；
3. 若 `out` 的实际观测点不在 FA 之后（例如在 o_proj 之后），本 oracle 不复现该线性层——
   观测点以捕获钩子说明为准，本报告只按"FA 输出"记账。

本 oracle **不能**声称覆盖的东西
--------------------------------
1. decode 步历史 KV 的物理布局：块表/块清单是否筛对、物理块映射是否正确——本 oracle 只按 canonical
   顺序与因果上界取 key，不读任何块表、块清单或候选侧集合；
2. 非因果结构：局部窗口、被裁掉的位置、chunked 掩码等——本 oracle 只实现"全局因果"，
   非全局参考必须由别的判据回答；
3. prefill 段的边界：`positions_step{i}` 中绝对位置 >= prompt_len 的 query 不在 prefill 段内
   （其可见 key 需要段外历史），本 oracle 拒绝该步（退出码 2），不产出不完整参考；
4. 不完整的 decode 链：步号不连续（缺中间步的 current）或累计位置有空洞时，该步及其后的可见历史
   都缺 KV → 拒绝（退出码 2，指明缺哪一段/哪一步），不做部分参考；
5. 通过/不通过结论：只给误差量级与逐位置/逐层/逐步分布，不给阈值判据（阈值尚未冻结）。

独立性（结构约束，见 tests/test_p2_calib_oracle.py 的守卫用例）
--------------------------------------------------------------
本文件是**独立判据**：不 import 项目内任何包模块，不调用候选的视图构造/块表转换代码，不读取候选
生成的集合；输入只有捕获 npz（张量 + canonical 顺序 + 元数据）与命令行参数。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from _lib import now_cst, sha256_file

SCHEMA = "attnview.p2-calib-oracle/v1"
COMPUTE_DTYPE = "float32"
REL_ERR_EPS = 1e-12
# 选定点模式：|ref| ≤ 该阈值视为“近零”，用于报告参考幅度分布（不是通过/不通过判据）。
NEAR_ZERO_THRESHOLD = 1e-3
# 选定点模式额外给出的尺度指标（非有限值时为 null）。
EXTENDED_NUMERIC_KEYS = (
    "l2_diff",
    "rel_l2_out",
    "rel_l2_ref",
    "abs_ref_p50",
    "abs_ref_p99",
    "abs_ref_near_zero_fraction",
)
REQUIRED_SCALARS = ("prompt_len", "scale", "num_heads", "num_kv_heads", "head_dim")

_LAYER_K = re.compile(r"^k_prefill_L(\d+)$")
_LAYER_V = re.compile(r"^v_prefill_L(\d+)$")
_LAYER_NAME = re.compile(r"^layer_name_L(\d+)$")
_STEP_Q = re.compile(r"^q_step(\d+)_L(\d+)$")
_STEP_OUT = re.compile(r"^out_step(\d+)_L(\d+)$")
_STEP_POS = re.compile(r"^positions_step(\d+)$")
_DECODE_Q = re.compile(r"^decode_q_step(\d+)_L(\d+)$")
_DECODE_OUT = re.compile(r"^decode_out_step(\d+)_L(\d+)$")
_DECODE_POS = re.compile(r"^decode_pos_step(\d+)$")
_DECODE_K = re.compile(r"^k_current_step(\d+)_L(\d+)$")
_DECODE_V = re.compile(r"^v_current_step(\d+)_L(\d+)$")


class CaptureError(Exception):
    """捕获文件缺件/不合规（对应退出码 2）。"""


# ---------------------------------------------------------------- 捕获加载


def _scalar(raw: dict, key: str):
    arr = np.asarray(raw[key])
    if arr.size != 1:
        raise CaptureError(f"元数据 {key} 不是标量：shape={tuple(arr.shape)}")
    return arr.reshape(-1)[0]


def _as_int(raw: dict, key: str) -> int:
    value = _scalar(raw, key)
    if isinstance(value, (np.floating, float)) and not float(value).is_integer():
        raise CaptureError(f"元数据 {key} 不是整数：{value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"元数据 {key} 不是整数：{value!r}") from exc


def _as_float(raw: dict, key: str) -> float:
    try:
        value = float(_scalar(raw, key))
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"元数据 {key} 不是浮点数：{raw[key]!r}") from exc
    if not math.isfinite(value):
        raise CaptureError(f"元数据 {key} 非有限：{value!r}")
    return value


def _optional_meta(raw: dict, key: str):
    """可选元数据：0 维/单元素给标量，多元素给列表；缺失给 None。"""
    if key not in raw:
        return None
    arr = np.asarray(raw[key])
    items = arr.reshape(-1).tolist()
    if not items:
        return []
    return items[0] if len(items) == 1 else items


def _as_tensor(raw: dict, name: str) -> torch.Tensor:
    arr = np.asarray(raw[name])
    if arr.dtype.kind != "f":
        raise CaptureError(f"{name} 的存储 dtype 不是浮点：{arr.dtype}")
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)


def _stored_dtype(raw: dict, name: str) -> str:
    return str(np.asarray(raw[name]).dtype)


def _is_tensor_key(name: str) -> bool:
    """判断 npz 键是否为参与比较的张量（排除元数据/positions）。"""
    return any(
        pattern.match(name)
        for pattern in (_LAYER_K, _LAYER_V, _STEP_Q, _STEP_OUT, _DECODE_Q, _DECODE_OUT, _DECODE_K, _DECODE_V)
    )


def load_capture(path: Path) -> dict:
    """读取并校验捕获 npz；任何缺件/形状/位置问题都以 CaptureError 抛出（退出码 2）。"""
    if not path.is_file():
        raise CaptureError(f"捕获文件不存在：{path}")
    with np.load(path) as npz:
        raw = {name: npz[name] for name in npz.files}
    if not raw:
        raise CaptureError(f"捕获文件为空：{path}")

    missing = [key for key in REQUIRED_SCALARS if key not in raw]
    if missing:
        raise CaptureError("必需元数据缺失：" + ", ".join(missing))

    prompt_len = _as_int(raw, "prompt_len")
    scale = _as_float(raw, "scale")
    num_heads = _as_int(raw, "num_heads")
    num_kv_heads = _as_int(raw, "num_kv_heads")
    head_dim = _as_int(raw, "head_dim")
    for name, value in (
        ("prompt_len", prompt_len),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if value <= 0:
            raise CaptureError(f"元数据 {name} 必须为正：{value}")
    if scale <= 0:
        raise CaptureError(f"元数据 scale 必须为正：{scale}")
    if num_heads % num_kv_heads:
        raise CaptureError(
            f"num_heads={num_heads} 不是 num_kv_heads={num_kv_heads} 的整数倍，GQA 分组无定义"
        )

    meta = {
        "prompt_len": prompt_len,
        "scale": scale,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "layer_index": _optional_meta(raw, "layer_index"),
        "dtype_name": _optional_meta(raw, "dtype_name"),
        # 捕获可声明 scale 的来源（如 impl.scale / derived_head_dim**-0.5）；
        # 未声明时记 "capture"（值确实取自捕获文件，但来源未进一步声明）。
        "scale_source": _optional_meta(raw, "scale_source") or "capture",
    }

    # --- 每层 K/V：canonical 顺序，形状 [num_kv_heads, prompt_len, head_dim] ---
    layer_ids: set[int] = set()
    for name in raw:
        for pattern in (_LAYER_K, _LAYER_V):
            match = pattern.match(name)
            if match:
                layer_ids.add(int(match.group(1)))
    if not layer_ids:
        raise CaptureError("捕获中没有 prefill K/V（k_prefill_L{L} / v_prefill_L{L}）")

    layers: dict[int, dict] = {}
    problems: list[str] = []
    for layer in sorted(layer_ids):
        k_name, v_name = f"k_prefill_L{layer}", f"v_prefill_L{layer}"
        absent = [name for name in (k_name, v_name) if name not in raw]
        if absent:
            problems.append(f"L{layer} 缺 {' / '.join(absent)}")
            continue
        k = _as_tensor(raw, k_name)
        v = _as_tensor(raw, v_name)
        for name, tensor in ((k_name, k), (v_name, v)):
            if tuple(tensor.shape) != (num_kv_heads, prompt_len, head_dim):
                problems.append(
                    f"{name} 形状 {tuple(tensor.shape)} != "
                    f"(num_kv_heads={num_kv_heads}, prompt_len={prompt_len}, head_dim={head_dim})"
                )
        layers[layer] = {
            "k": k,
            "v": v,
            "dtypes": {k_name: _stored_dtype(raw, k_name), v_name: _stored_dtype(raw, v_name)},
        }

    # --- 每个记录步：positions + 逐层 q/out；形状 [num_heads, q_len, head_dim] ---
    step_ids: set[int] = set()
    q_out_layers: set[tuple[int, int]] = set()
    for name in raw:
        for pattern in (_STEP_Q, _STEP_OUT):
            match = pattern.match(name)
            if match:
                step_ids.add(int(match.group(1)))
                q_out_layers.add((int(match.group(1)), int(match.group(2))))
        match = _STEP_POS.match(name)
        if match:
            step_ids.add(int(match.group(1)))
    steps: dict[int, dict] = {}
    for step in sorted(step_ids):
        pos_name = f"positions_step{step}"
        if pos_name not in raw:
            problems.append(f"step{step} 缺 {pos_name}")
            continue
        pos_arr = np.asarray(raw[pos_name])
        if pos_arr.dtype.kind not in "iu":
            problems.append(f"{pos_name} 的 dtype={pos_arr.dtype} 不是整数")
            continue
        positions = tuple(int(x) for x in pos_arr.reshape(-1))
        if not positions:
            problems.append(f"{pos_name} 为空")
            continue
        record = {"positions": positions, "q": {}, "out": {}}
        for layer in sorted(layers):
            for name, target in (
                (f"q_step{step}_L{layer}", record["q"]),
                (f"out_step{step}_L{layer}", record["out"]),
            ):
                if name not in raw:
                    problems.append(f"step{step} / L{layer} 缺 {name}")
                    continue
                tensor = _as_tensor(raw, name)
                if tuple(tensor.shape) != (num_heads, len(positions), head_dim):
                    problems.append(
                        f"{name} 形状 {tuple(tensor.shape)} != "
                        f"(num_heads={num_heads}, q_len={len(positions)}, head_dim={head_dim})"
                    )
                    continue
                target[layer] = tensor
        for index, position in enumerate(positions):
            if not 0 <= position < prompt_len:
                problems.append(
                    f"step{step} 第 {index} 个 query 位置 {position} 不在捕获段 "
                    f"[0, prompt_len={prompt_len}) 内：该步可见 key 会超出捕获的 prefill 段"
                    f"（含当前 token 自身），本 oracle 拒绝产出不完整参考"
                )
        steps[step] = record

    for step, layer in sorted(q_out_layers):
        if layer not in layers:
            problems.append(
                f"step{step} 有 L{layer} 的 q/out 记录，但捕获中没有该层的 prefill K/V："
                f"不静默跳过，需捕获钩子补齐该层 K/V 或去掉该记录"
            )

    # --- decode 步：prefill 段 + 当前 token 的 canonical KV ---
    # 只有与 prefill 段连续的首个 decode 步（位置 = prompt_len..）能给出**完整**参考：
    # 后续 decode 步的可见历史还包含中间生成 token 的 KV，本捕获不含 → 拒绝（见下）。
    decode_step_ids: set[int] = set()
    decode_pairs: set[tuple[int, int]] = set()
    for name in raw:
        for pattern in (_DECODE_Q, _DECODE_OUT, _DECODE_K, _DECODE_V):
            match = pattern.match(name)
            if match:
                decode_step_ids.add(int(match.group(1)))
                decode_pairs.add((int(match.group(1)), int(match.group(2))))
        match = _DECODE_POS.match(name)
        if match:
            decode_step_ids.add(int(match.group(1)))

    decode_steps: dict[int, dict] = {}
    for step in sorted(decode_step_ids):
        pos_name = f"decode_pos_step{step}"
        if pos_name not in raw:
            problems.append(f"decode step{step} 缺 {pos_name}（拒绝自行推断位置）")
            continue
        pos_arr = np.asarray(raw[pos_name])
        if pos_arr.dtype.kind not in "iu":
            problems.append(f"{pos_name} 的 dtype={pos_arr.dtype} 不是整数")
            continue
        positions = tuple(int(x) for x in pos_arr.reshape(-1))
        if not positions:
            problems.append(f"{pos_name} 为空")
            continue
        record = {"positions": positions, "q": {}, "out": {}, "k_current": {}, "v_current": {}}
        for layer in sorted(layers):
            for name, target, expected in (
                (f"decode_q_step{step}_L{layer}", record["q"], (num_heads, len(positions), head_dim)),
                (f"decode_out_step{step}_L{layer}", record["out"], (num_heads, len(positions), head_dim)),
                (f"k_current_step{step}_L{layer}", record["k_current"], (num_kv_heads, len(positions), head_dim)),
                (f"v_current_step{step}_L{layer}", record["v_current"], (num_kv_heads, len(positions), head_dim)),
            ):
                if name not in raw:
                    problems.append(f"decode step{step} / L{layer} 缺 {name}")
                    continue
                tensor = _as_tensor(raw, name)
                if tuple(tensor.shape) != expected:
                    problems.append(f"{name} 形状 {tuple(tensor.shape)} != {expected}")
                    continue
                target[layer] = tensor
        decode_steps[step] = record

    # 链校验：步号必须从 1 连续，且累计位置与 prefill 段拼成连续区间（无空洞）。
    # 否则该步（及其后各步）的可见历史缺 KV，一律拒绝并指明缺哪一段/哪一步。
    chain_positions: list[int] = []
    expected_step = 1
    for step in sorted(decode_steps):
        record = decode_steps[step]
        if step != expected_step:
            absent_steps = list(range(expected_step, step))
            if not absent_steps:  # 例：步号从 0 开始或出现重复编号
                problems.append(
                    f"decode 链步号异常：出现 step{step}，而链要求从 step1 起连续编号 → 拒绝该步及其后的步"
                )
            else:
                problems.append(
                    f"decode 链不完整：step{step} 的参考需要 "
                    f"{'、'.join(f'step{s}' for s in absent_steps)} 的 current K/V，但捕获里没有 "
                    f"→ 拒绝 step{step} 及其后的步（不猜测补齐）"
                )
            break
        chain_positions.extend(record["positions"])
        expected_step += 1
        if chain_positions != list(range(prompt_len, prompt_len + len(chain_positions))):
            captured = set(range(prompt_len)) | set(chain_positions)
            # 只报第一段缺口；上界截断，避免异常大的位置导致扫描爆炸。
            high = min(max(chain_positions), prompt_len + len(chain_positions))
            missing = [p for p in range(prompt_len, high + 1) if p not in captured]
            span = f"{missing[0]}..{missing[-1]}" if missing else "（位置重复或未按升序）"
            problems.append(
                f"decode 链位置有空洞：截至 step{step} 的累计位置 {chain_positions} 与 prefill 段"
                f"（0..{prompt_len - 1}）不能拼成连续 canonical KV：缺少位置 {span} 的 KV"
                f"（中间生成 token 未捕获）→ 拒绝该步及其后的步"
            )
            break

    for step, layer in sorted(decode_pairs):
        if layer not in layers:
            problems.append(
                f"decode step{step} 有 L{layer} 的记录，但没有该层 prefill K/V：不静默跳过，"
                f"需捕获钩子补齐该层 K/V 或去掉该记录"
            )

    # 层覆盖：被 layer_name_L{L} / layer_index 声明的层必须真的在捕获里（缺层即报错，不静默少覆盖）。
    declared_layers: set[int] = set()
    for name in raw:
        match = _LAYER_NAME.match(name)
        if match:
            declared_layers.add(int(match.group(1)))
    if isinstance(meta["layer_index"], int):
        declared_layers.add(meta["layer_index"])
    elif isinstance(meta["layer_index"], list):
        declared_layers.update(value for value in meta["layer_index"] if isinstance(value, int))
    for layer in sorted(declared_layers):
        if layer not in layers:
            problems.append(
                f"层 L{layer} 被 layer_name_L{layer} / layer_index 声明，却没有 prefill K/V（层覆盖不全）"
            )

    if not step_ids and not decode_step_ids:
        raise CaptureError(
            "捕获中既没有 prefill 记录步（q_step{i}_L{L} 等）也没有 decode 记录步"
            "（decode_q_step{i}_L{L} 等）：没有可比对对象"
        )

    if problems:
        raise CaptureError("捕获格式问题：\n  - " + "\n  - ".join(problems))

    warnings: list[str] = []

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "meta": meta,
        "layers": layers,
        "steps": steps,
        "decode_steps": decode_steps,
        "stored_dtypes": {name: _stored_dtype(raw, name) for name in sorted(raw)},
        "warnings": warnings,
    }


# ---------------------------------------------------------------- 参考计算


def dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """FP32 dense 全局因果参考。

    `q` 形状 [num_heads, q_len, head_dim]；`k`/`v` 形状 [num_kv_heads, prompt_len, head_dim]，
    canonical 顺序；`positions[j]` 是第 j 个 query 的绝对位置 p，其可见 key 为下标 0..p。
    返回 [num_heads, q_len, head_dim] 的 FP32 张量。
    """
    if num_heads % num_kv_heads:
        raise CaptureError(f"num_heads={num_heads} 不是 num_kv_heads={num_kv_heads} 的整数倍")
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    q_len = len(positions)
    if tuple(q.shape) != (num_heads, q_len, int(k.shape[-1])):
        raise CaptureError(
            f"q 形状 {tuple(q.shape)} 与 (num_heads={num_heads}, q_len={q_len}, head_dim={int(k.shape[-1])}) 不符"
        )
    if tuple(k.shape) != tuple(v.shape) or k.dim() != 3 or k.shape[0] != num_kv_heads:
        raise CaptureError(
            f"k/v 形状必须一致且为 [num_kv_heads={num_kv_heads}, prompt_len, head_dim]："
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    repeat = num_heads // num_kv_heads
    out = torch.empty((num_heads, q_len, int(k.shape[-1])), dtype=torch.float32)
    for index, position in enumerate(positions):
        keys_used = int(position) + 1
        if not 1 <= keys_used <= int(k.shape[1]):
            raise CaptureError(
                f"位置 {int(position)} 需要的 key 数 {keys_used} 超出捕获段长度 {int(k.shape[1])}"
            )
        # 只切出因果上界内的 key；段外（未来）位置不参与任何计算。
        k_used = k[:, :keys_used, :].repeat_interleave(repeat, dim=0)  # [H, keys_used, D]
        v_used = v[:, :keys_used, :].repeat_interleave(repeat, dim=0)
        q_row = q[:, index, :]  # [H, D]
        scores = torch.matmul(k_used, q_row.unsqueeze(-1)).squeeze(-1) * scale  # [H, keys_used]
        row_max = scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores - row_max)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out[:, index, :] = torch.matmul(weights.unsqueeze(1), v_used).squeeze(1)
    return out


def _nonfinite_count(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum())


def _metrics(ref: torch.Tensor, observed: torch.Tensor) -> dict:
    diff = (ref.to(torch.float32) - observed.to(torch.float32)).to(torch.float32)
    max_abs_err = float(diff.abs().max())
    rms_err = float(torch.sqrt(torch.mean(diff * diff)))
    out_norm = float(torch.linalg.vector_norm(observed.to(torch.float32)))
    ref_norm = float(torch.linalg.vector_norm(ref.to(torch.float32)))
    return {
        "max_abs_err": max_abs_err,
        "rms_err": rms_err,
        "out_norm": out_norm,
        "ref_norm": ref_norm,
        "rel_err": max_abs_err / max(out_norm, REL_ERR_EPS),
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {"min": ordered[0], "median": float(statistics.median(ordered)), "max": ordered[-1]}


def _group_stats(comparisons: list[dict], key_of) -> dict:
    groups: dict[object, list[dict]] = {}
    for item in comparisons:
        groups.setdefault(key_of(item), []).append(item)
    grouped: dict[str, dict] = {}
    for name in sorted(groups):
        items = groups[name]
        finite = [item for item in items if item["finite"]]
        grouped[str(name)] = {
            "comparisons": len(items),
            "finite": len(finite),
            "non_finite": len(items) - len(finite),
            **{
                metric: _stats([item[metric] for item in finite])
                for metric in ("max_abs_err", "rms_err", "rel_err", "out_norm", "ref_norm")
            },
        }
    return grouped


# ---------------------------------------------------------------- 报告

METRIC_KEYS = ("max_abs_err", "rms_err", "out_norm", "ref_norm", "rel_err")


def _scale_metrics(ref_row: torch.Tensor, out_row: torch.Tensor) -> dict:
    """选定点模式的尺度指标：L2 相对误差与参考幅度分布（不含任何通过/不通过判定）。"""
    ref = ref_row.to(torch.float32)
    out = out_row.to(torch.float32)
    diff = ref - out
    l2_diff = float(torch.linalg.vector_norm(diff))
    l2_out = float(torch.linalg.vector_norm(out))
    l2_ref = float(torch.linalg.vector_norm(ref))
    magnitude = ref.abs().flatten()
    return {
        "elements": int(ref.numel()),
        "l2_diff": l2_diff,
        "rel_l2_out": l2_diff / max(l2_out, REL_ERR_EPS),
        "rel_l2_ref": l2_diff / max(l2_ref, REL_ERR_EPS),
        "abs_ref_p50": float(torch.quantile(magnitude, 0.5)),
        "abs_ref_p99": float(torch.quantile(magnitude, 0.99)),
        "abs_ref_near_zero_fraction": float(
            (magnitude <= NEAR_ZERO_THRESHOLD).to(torch.float32).mean()
        ),
        # 明确标注：既有 rel_err 不是 allclose 的 rtol。
        "rel_err_is_allclose_rtol": False,
    }


def _comparison_item(
    *,
    scope: str,
    layer: int,
    step: int,
    position: int,
    keys_used: int,
    q_row: torch.Tensor,
    out_row: torch.Tensor,
    ref_row: torch.Tensor,
    non_finite: list[dict],
    extended: bool = False,
) -> dict:
    """单个 (scope, layer, step, position) 的比较项；非有限值单列进 non_finite，指标置 null。

    `extended=True`（选定点模式）时额外给出尺度指标（L2 相对误差、|ref| 分位数与近零占比）。
    """
    item = {
        "scope": scope,
        "layer": layer,
        "step": step,
        "position": int(position),
        "keys_used": int(keys_used),
        "finite": True,
    }
    broken: dict[str, int] = {}
    for kind, tensor in (("q", q_row), ("out", out_row), ("ref", ref_row)):
        count = _nonfinite_count(tensor)
        if count:
            broken[kind] = count
    if broken:
        item["finite"] = False
        for kind, count in broken.items():
            non_finite.append(
                {
                    "level": "comparison",
                    "scope": scope,
                    "layer": layer,
                    "step": step,
                    "position": int(position),
                    "tensor": kind,
                    "non_finite": count,
                    "elements": int(out_row.numel()),
                }
            )
        item.update(dict.fromkeys(METRIC_KEYS))
        if extended:
            item.update(dict.fromkeys(EXTENDED_NUMERIC_KEYS))
            item.update({"elements": int(out_row.numel()), "rel_err_is_allclose_rtol": False})
    else:
        item.update(_metrics(ref_row, out_row))
        if extended:
            item.update(_scale_metrics(ref_row, out_row))
    return item


# ---------------------------------------------------------------- 选定点模式


def load_points(path: Path) -> dict:
    """读取选定点 JSON：{"prefill": [{"layer","position"}], "decode": [{"layer","step"}]}。

    只负责格式校验；点是否存在于捕获里由 build_report 在选定点模式下核查（缺即报错，不产报告）。
    """
    if not path.is_file():
        raise CaptureError(f"选定点文件不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"选定点文件不是合法 JSON：{path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise CaptureError('选定点文件必须是 JSON 对象：{"prefill": [...], "decode": [...]}')
    unknown = sorted(set(payload) - {"prefill", "decode"})
    if unknown:
        raise CaptureError(f"选定点文件出现未知键：{', '.join(unknown)}（只接受 prefill/decode）")

    points: dict[str, list[dict]] = {"prefill": [], "decode": []}
    for scope, keys in (("prefill", ("layer", "position")), ("decode", ("layer", "step"))):
        raw = payload.get(scope, [])
        if not isinstance(raw, list):
            raise CaptureError(f"选定点 {scope} 必须是列表")
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise CaptureError(f"选定点 {scope}[{index}] 必须是对象：{item!r}")
            missing = sorted(set(keys) - set(item))
            extra = sorted(set(item) - set(keys))
            if missing or extra:
                raise CaptureError(
                    f"选定点 {scope}[{index}] 键不符：需要 {list(keys)}；"
                    f"缺 {missing or '无'}；多 {extra or '无'}"
                )
            for key in keys:
                if isinstance(item[key], bool) or not isinstance(item[key], int):
                    raise CaptureError(f"选定点 {scope}[{index}].{key} 必须是整数：{item[key]!r}")
            points[scope].append(item)
    if not points["prefill"] and not points["decode"]:
        raise CaptureError("选定点文件没有任何点（prefill 与 decode 都为空）")
    return points


def _prefill_position_index(steps: dict) -> dict[int, list[tuple[int, int]]]:
    index_of: dict[int, list[tuple[int, int]]] = {}
    for step in sorted(steps):
        for index, position in enumerate(steps[step]["positions"]):
            index_of.setdefault(int(position), []).append((step, index))
    return index_of


def _prefill_hint(steps: dict) -> str:
    if not steps:
        return "（捕获里没有 prefill 记录步）"
    parts = [
        f"step{step}: {len(record['positions'])} 个位置（{record['positions'][0]}..{record['positions'][-1]}）"
        for step, record in sorted(steps.items())
    ]
    return "；".join(parts)


def _selected_comparisons(capture: dict, points: dict, non_finite: list[dict]) -> list[dict]:
    """只算选定点：每点走与全量模式相同的参考实现，因此同点指标同值（额外给尺度指标）。"""
    meta = capture["meta"]
    layers = capture["layers"]
    steps = capture["steps"]
    decode_steps = capture["decode_steps"]
    num_heads, num_kv_heads = meta["num_heads"], meta["num_kv_heads"]
    index_of = _prefill_position_index(steps)
    comparisons: list[dict] = []

    for point in points["prefill"]:
        layer, position = int(point["layer"]), int(point["position"])
        if layer not in layers:
            raise CaptureError(f"选定 prefill 点 L{layer} 不在捕获层集合 {sorted(layers)} 中")
        matches = index_of.get(position, [])
        if not matches:
            raise CaptureError(
                f"选定 prefill 点 L{layer}/位置 {position} 不在捕获的记录步里：{_prefill_hint(steps)}"
            )
        if len(matches) > 1:
            raise CaptureError(
                f"选定 prefill 点 L{layer}/位置 {position} 在多个记录步里出现 {matches}："
                f"请改用能唯一定位的点"
            )
        step, index = matches[0]
        record = steps[step]
        q_row = record["q"][layer][:, index : index + 1, :]
        ref = dense_reference(
            q_row, layers[layer]["k"], layers[layer]["v"], [position], meta["scale"], num_heads, num_kv_heads
        )
        comparisons.append(
            _comparison_item(
                scope="prefill",
                layer=layer,
                step=step,
                position=position,
                keys_used=position + 1,
                q_row=q_row[:, 0, :],
                out_row=record["out"][layer][:, index, :],
                ref_row=ref[:, 0, :],
                non_finite=non_finite,
                extended=True,
            )
        )

    for point in points["decode"]:
        layer, step = int(point["layer"]), int(point["step"])
        if layer not in layers:
            raise CaptureError(f"选定 decode 点 L{layer} 不在捕获层集合 {sorted(layers)} 中")
        if step not in decode_steps:
            raise CaptureError(
                f"选定 decode 点 step{step} 不在捕获里：可用 {sorted(decode_steps)}"
            )
        record = decode_steps[step]
        k_chain = layers[layer]["k"]
        v_chain = layers[layer]["v"]
        for earlier in sorted(decode_steps):
            if earlier > step:
                break
            k_chain = torch.cat((k_chain, decode_steps[earlier]["k_current"][layer]), dim=1)
            v_chain = torch.cat((v_chain, decode_steps[earlier]["v_current"][layer]), dim=1)
        for index, position in enumerate(record["positions"]):
            q_row = record["q"][layer][:, index : index + 1, :]
            ref = dense_reference(
                q_row, k_chain, v_chain, [position], meta["scale"], num_heads, num_kv_heads
            )
            comparisons.append(
                _comparison_item(
                    scope="decode",
                    layer=layer,
                    step=step,
                    position=int(position),
                    keys_used=int(position) + 1,
                    q_row=q_row[:, 0, :],
                    out_row=record["out"][layer][:, index, :],
                    ref_row=ref[:, 0, :],
                    non_finite=non_finite,
                    extended=True,
                )
            )
    return comparisons


def _all_comparisons(capture: dict, non_finite: list[dict]) -> list[dict]:
    """全量模式：所有 (scope, layer, step, position) 都出参考与指标。"""
    meta = capture["meta"]
    layers = capture["layers"]
    steps = capture["steps"]
    decode_steps = capture["decode_steps"]

    comparisons: list[dict] = []
    for layer in sorted(layers):
        for step in sorted(steps):
            record = steps[step]
            q_tensor = record["q"].get(layer)
            out_tensor = record["out"].get(layer)
            if q_tensor is None or out_tensor is None:  # load_capture 已拦，防御性跳过
                continue
            ref = dense_reference(
                q_tensor,
                layers[layer]["k"],
                layers[layer]["v"],
                record["positions"],
                meta["scale"],
                meta["num_heads"],
                meta["num_kv_heads"],
            )
            for index, position in enumerate(record["positions"]):
                comparisons.append(
                    _comparison_item(
                        scope="prefill",
                        layer=layer,
                        step=step,
                        position=position,
                        keys_used=int(position) + 1,
                        q_row=q_tensor[:, index, :],
                        out_row=out_tensor[:, index, :],
                        ref_row=ref[:, index, :],
                        non_finite=non_finite,
                    )
                )

    # decode 路径：canonical KV = prefill 段 + step1..step i 的 current token（链完整性由 load_capture 校验）。
    for layer in sorted(layers):
        k_chain = layers[layer]["k"]
        v_chain = layers[layer]["v"]
        for step in sorted(decode_steps):
            record = decode_steps[step]
            q_tensor = record["q"].get(layer)
            out_tensor = record["out"].get(layer)
            k_current = record["k_current"].get(layer)
            v_current = record["v_current"].get(layer)
            if None in (q_tensor, out_tensor, k_current, v_current):  # load_capture 已拦
                continue
            k_chain = torch.cat((k_chain, k_current), dim=1)
            v_chain = torch.cat((v_chain, v_current), dim=1)
            ref = dense_reference(
                q_tensor,
                k_chain,
                v_chain,
                record["positions"],
                meta["scale"],
                meta["num_heads"],
                meta["num_kv_heads"],
            )
            for index, position in enumerate(record["positions"]):
                comparisons.append(
                    _comparison_item(
                        scope="decode",
                        layer=layer,
                        step=step,
                        position=position,
                        keys_used=int(position) + 1,
                        q_row=q_tensor[:, index, :],
                        out_row=out_tensor[:, index, :],
                        ref_row=ref[:, index, :],
                        non_finite=non_finite,
                    )
                )
    return comparisons


def build_report(capture: dict, points: dict | None = None) -> dict:
    meta = capture["meta"]
    layers = capture["layers"]
    steps = capture["steps"]
    decode_steps = capture["decode_steps"]

    non_finite: list[dict] = []
    warnings = list(capture["warnings"])
    # scale 来源声明为 derived 时（非运行时真实值），它进入全部误差 → 单列为已知近似来源。
    scale_source = meta["scale_source"]
    scale_derived = isinstance(scale_source, str) and "derived" in scale_source.lower()
    if scale_derived:
        warnings.append(
            f"scale 来源声明为 {scale_source}（非运行时真实值）：该假设进入本报告全部误差，属已知近似来源"
        )
    # 捕获级扫描：prefill K/V 与 decode 当前 token 的 K/V 全数组（含未被任何 query 用到的尾部）。
    for layer in sorted(layers):
        for kind in ("k", "v"):
            tensor = layers[layer][kind]
            count = _nonfinite_count(tensor)
            if count:
                non_finite.append(
                    {
                        "level": "capture",
                        "layer": layer,
                        "step": None,
                        "tensor": f"{kind}_prefill_L{layer}",
                        "non_finite": count,
                        "elements": int(tensor.numel()),
                    }
                )
        for step in sorted(decode_steps):
            for kind in ("k_current", "v_current"):
                tensor = decode_steps[step][kind].get(layer)
                if tensor is None:  # load_capture 已拦
                    continue
                count = _nonfinite_count(tensor)
                if count:
                    non_finite.append(
                        {
                            "level": "capture",
                            "layer": layer,
                            "step": step,
                            "tensor": f"{kind}_step{step}_L{layer}",
                            "non_finite": count,
                            "elements": int(tensor.numel()),
                        }
                    )

    comparisons: list[dict] = (
        _selected_comparisons(capture, points, non_finite)
        if points is not None
        else _all_comparisons(capture, non_finite)
    )

    per_layer = _group_stats(comparisons, lambda item: item["layer"])
    per_scope = _group_stats(comparisons, lambda item: item["scope"])
    per_step = _group_stats(comparisons, lambda item: f"{item['scope']}:{item['step']}")
    for step in sorted(steps):
        entry = per_step.get(f"prefill:{step}")
        if entry is not None:
            entry.update(
                {
                    "q_len": len(steps[step]["positions"]),
                    "positions": [int(p) for p in steps[step]["positions"]],
                }
            )
    for step in sorted(decode_steps):
        entry = per_step.get(f"decode:{step}")
        if entry is not None:
            entry.update(
                {
                    "q_len": len(decode_steps[step]["positions"]),
                    "positions": [int(p) for p in decode_steps[step]["positions"]],
                    # 该步参考实际用到的 current K/V 链（load_capture 保证为 1..step）。
                    "chain_steps": list(range(1, step + 1)),
                }
            )

    finite = [item for item in comparisons if item["finite"]]
    compared_layers = sorted({item["layer"] for item in comparisons})
    compared_prefill_steps = sorted(
        {item["step"] for item in comparisons if item["scope"] == "prefill"}
    )
    compared_decode_steps = sorted(
        {item["step"] for item in comparisons if item["scope"] == "decode"}
    )
    summary = {
        "layers": compared_layers,
        "prefill_steps": compared_prefill_steps,
        "decode_steps": compared_decode_steps,
        "decode_steps_referenced": compared_decode_steps,
        "comparisons": len(comparisons),
        "finite_comparisons": len(finite),
        "comparisons_by_scope": {
            scope: sum(1 for item in comparisons if item["scope"] == scope)
            for scope in ("prefill", "decode")
            if any(item["scope"] == scope for item in comparisons)
        },
        "max_abs_err_max": max((item["max_abs_err"] for item in finite), default=None),
        "non_finite_count": len(non_finite),
        "status": "non_finite" if non_finite else "ok",
        "rel_err_eps": REL_ERR_EPS,
    }

    report = {
        "schema": SCHEMA,
        "generated_at_cst": now_cst(),
        "capture": {"path": capture["path"], "sha256": capture["sha256"]},
        "metadata": meta,
        "numerics": {
            "compute_dtype": COMPUTE_DTYPE,
            "softmax": "数值稳定：scores 每行减最大值后 exp，再按行归一化",
            "rms_err_definition": "sqrt(mean((ref - out)^2))，在该位置的 head × head_dim 元素上求",
            "rel_err_definition": f"max_abs_err / max(out_norm, {REL_ERR_EPS:g})",
            "out_norm_definition": "该位置观测输出 out 的 L2 范数（head × head_dim 元素）",
            "ref_norm_definition": "该位置 FP32 参考输出的 L2 范数（head × head_dim 元素）",
            "gqa_grouping": "连续分组：query head h → kv head h // (num_heads // num_kv_heads)",
            "causal_rule": "绝对位置 p 的 query 只用 key 下标 0..p（含自身），key 取自 canonical prefill 段",
            "decode_reference": "canonical KV = prefill 段 + step1..step i 的 current token K/V（步号从 1 连续、位置无空洞才参考；缺链拒绝并指明缺哪一段/哪一步）",
            "rope_scale_reapplied": False,
            "dtype_name": meta["dtype_name"],
            "scale_source": scale_source,
            "scale_is_approximate": scale_derived,
            "upcast_to_float32": sorted(
                name
                for name, dtype in capture["stored_dtypes"].items()
                if dtype != COMPUTE_DTYPE and _is_tensor_key(name)
            ),
            "upcast_note": "按 npz 实际存储 dtype 判定；模型侧原始精度见 dtype_name",
        },
        "comparisons": comparisons,
        "per_layer": per_layer,
        "per_scope": per_scope,
        "per_step": per_step,
        "non_finite": non_finite,
        "summary": summary,
        "warnings": warnings,
    }

    if points is not None:
        # 选定点模式：只加自己的字段，默认（全量）报告不变。
        report["mode"] = "points"
        report["points_requested"] = {
            "prefill": len(points["prefill"]),
            "decode": len(points["decode"]),
            "total": len(points["prefill"]) + len(points["decode"]),
        }
        report["definitions"] = {
            "point_records": "每个选定点的记录在 comparisons 数组里（scope=prefill|decode，字段与全量模式同键同值）",
            "elements": "该点元素数 = 该 query 的 head 数 × head_dim",
            "l2_diff": "‖out - ref‖₂（同一批元素）",
            "rel_l2_out": f"‖out - ref‖₂ / max(‖out‖₂, {REL_ERR_EPS:g})",
            "rel_l2_ref": f"‖out - ref‖₂ / max(‖ref‖₂, {REL_ERR_EPS:g})",
            "abs_ref_p50": "|ref| 的 50% 分位数（torch.quantile，linear 插值）",
            "abs_ref_p99": "|ref| 的 99% 分位数（同上）",
            "abs_ref_near_zero_fraction": f"|ref| ≤ {NEAR_ZERO_THRESHOLD:g} 的元素占比",
            "rel_err": "max_abs_err / max(‖out‖₂, 1e-12)：既有口径；这是绝对误差相对 L2 尺度的比值，"
            "不是 torch.allclose 的 rtol（每条记录另有 rel_err_is_allclose_rtol=false 标注）",
            "no_thresholds": "本报告不含任何通过/不通过判定（阈值未冻结，由用户决定）",
        }
        report["summary"]["mode"] = "points"
        report["summary"]["points_requested"] = report["points_requested"]["total"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SUP-004 单请求模型校准：独立 FP32 dense 全局因果注意力参考（只读捕获 npz）"
    )
    parser.add_argument("--capture", required=True, help="捕获 npz 路径（prefill K/V + 逐步 q/out + 元数据）")
    parser.add_argument("--out", required=True, help="JSON 报告输出路径")
    parser.add_argument(
        "--points",
        default=None,
        help='可选：选定点 JSON，形如 {"prefill":[{"layer":14,"position":123}],'
        '"decode":[{"layer":14,"step":1}]}；只算这些点并额外给尺度指标（不给则保持全量行为）',
    )
    args = parser.parse_args(argv)

    try:
        capture = load_capture(Path(args.capture))
        points = load_points(Path(args.points)) if args.points else None
        report = build_report(capture, points)
    except CaptureError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    try:
        if out_path.parent != Path(""):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"错误：无法写出报告 {out_path}：{exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    layer_maxima = [
        stats["max_abs_err"]["max"]
        for stats in report["per_layer"].values()
        if stats["max_abs_err"]["max"] is not None
    ]
    layer_max = max(layer_maxima) if layer_maxima else None
    shown = "none" if layer_max is None else f"{layer_max:.6e}"
    points_suffix = f" points={summary['points_requested']}" if "points_requested" in summary else ""
    print(
        f"p2-calib-oracle: comparisons={summary['comparisons']}"
        f"{points_suffix} "
        f"finite={summary['finite_comparisons']} layers={len(summary['layers'])} "
        f"prefill_steps={len(summary['prefill_steps'])} decode_steps={len(summary['decode_steps'])} "
        f"max_abs_err_max={shown} "
        f"non_finite={summary['non_finite_count']} out={out_path}"
    )
    return 3 if summary["non_finite_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
