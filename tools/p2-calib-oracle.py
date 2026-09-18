#!/usr/bin/env python3
"""阶段 05 SUP-004 单请求模型校准：独立 FP32 dense 全局注意力参考（oracle）。

用法：
    source /root/attnview/env.sh
    "$ATTNVIEW_PYTHON" tools/p2-calib-oracle.py --capture <capture.npz> --out <report.json>

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
3. 捕获段之外的位置：若某步 query 的绝对位置 >= prompt_len，其可见 key 必然落在捕获的 prefill 段
   之外（含当前 token 自身），本 oracle 拒绝该步（退出码 2），不产出不完整参考；
4. 通过/不通过结论：只给误差量级与逐位置/逐层/逐步分布，不给阈值判据（阈值尚未冻结）。

独立性（结构约束，见 tests/test_p2_calib_oracle.py 的守卫用例）
--------------------------------------------------------------
本文件是**独立判据**：不 import 项目内任何包模块，不调用候选的视图构造/块表转换代码，不读取候选
生成的集合；输入只有捕获 npz（张量 + canonical 顺序 + 元数据）与命令行参数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

SCHEMA = "attnview.p2-calib-oracle/v1"
COMPUTE_DTYPE = "float32"
REL_ERR_EPS = 1e-12
REQUIRED_SCALARS = ("prompt_len", "scale", "num_heads", "num_kv_heads", "head_dim")

_LAYER_K = re.compile(r"^k_prefill_L(\d+)$")
_LAYER_V = re.compile(r"^v_prefill_L(\d+)$")
_STEP_Q = re.compile(r"^q_step(\d+)_L(\d+)$")
_STEP_OUT = re.compile(r"^out_step(\d+)_L(\d+)$")
_STEP_POS = re.compile(r"^positions_step(\d+)$")


class CaptureError(Exception):
    """捕获文件缺件/不合规（对应退出码 2）。"""


def now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return any(pattern.match(name) for pattern in (_LAYER_K, _LAYER_V, _STEP_Q, _STEP_OUT))


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
        "scale_source": "capture",
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
    if not step_ids:
        raise CaptureError("捕获中没有 q/out 记录步（q_step{i}_L{L} / out_step{i}_L{L} / positions_step{i}）")

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

    if problems:
        raise CaptureError("捕获格式问题：\n  - " + "\n  - ".join(problems))

    warnings: list[str] = []
    layer_index = meta["layer_index"]
    if isinstance(layer_index, int) and layer_index not in layers:
        warnings.append(
            f"元数据 layer_index={layer_index} 不在捕获到的层集合 {sorted(layers)} 中"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "meta": meta,
        "layers": layers,
        "steps": steps,
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


def _group_stats(comparisons: list[dict], key: str) -> dict:
    groups: dict[int, list[dict]] = {}
    for item in comparisons:
        groups.setdefault(item[key], []).append(item)
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


def build_report(capture: dict) -> dict:
    meta = capture["meta"]
    layers = capture["layers"]
    steps = capture["steps"]

    non_finite: list[dict] = []
    warnings = list(capture["warnings"])
    # scale 来源声明为 derived 时（非运行时真实值），它进入全部误差 → 单列为已知近似来源。
    scale_source = meta["scale_source"]
    scale_derived = isinstance(scale_source, str) and "derived" in scale_source.lower()
    if scale_derived:
        warnings.append(
            f"scale 来源声明为 {scale_source}（非运行时真实值）：该假设进入本报告全部误差，属已知近似来源"
        )
    # 捕获级扫描：K/V 全数组（含未被任何 query 用到的尾部）。
    for layer in sorted(layers):
        for kind in ("k", "v"):
            tensor = layers[layer][kind]
            count = _nonfinite_count(tensor)
            if count:
                non_finite.append(
                    {
                        "scope": "capture",
                        "layer": layer,
                        "tensor": f"{kind}_prefill_L{layer}",
                        "non_finite": count,
                        "elements": int(tensor.numel()),
                    }
                )

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
                out_row = out_tensor[:, index, :]
                ref_row = ref[:, index, :]
                broken: dict[str, int] = {}
                for kind, tensor in (
                    ("q", q_tensor[:, index, :]),
                    ("out", out_row),
                    ("ref", ref_row),
                ):
                    count = _nonfinite_count(tensor)
                    if count:
                        broken[kind] = count
                item = {
                    "layer": layer,
                    "step": step,
                    "position": int(position),
                    "keys_used": int(position) + 1,
                    "finite": not broken,
                }
                if broken:
                    for kind, count in broken.items():
                        non_finite.append(
                            {
                                "scope": "comparison",
                                "layer": layer,
                                "step": step,
                                "position": int(position),
                                "tensor": kind,
                                "non_finite": count,
                                "elements": int(out_row.numel()),
                            }
                        )
                    item.update(
                        {key: None for key in ("max_abs_err", "rms_err", "out_norm", "ref_norm", "rel_err")}
                    )
                else:
                    item.update(_metrics(ref_row, out_row))
                comparisons.append(item)

    per_layer = _group_stats(comparisons, "layer")
    per_step = _group_stats(comparisons, "step")
    for step in sorted(steps):
        per_step[str(step)].update(
            {
                "q_len": len(steps[step]["positions"]),
                "positions": [int(p) for p in steps[step]["positions"]],
            }
        )

    finite = [item for item in comparisons if item["finite"]]
    summary = {
        "layers": sorted(layers),
        "steps": sorted(steps),
        "comparisons": len(comparisons),
        "finite_comparisons": len(finite),
        "max_abs_err_max": max((item["max_abs_err"] for item in finite), default=None),
        "non_finite_count": len(non_finite),
        "status": "non_finite" if non_finite else "ok",
        "rel_err_eps": REL_ERR_EPS,
    }

    return {
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
        "per_step": per_step,
        "non_finite": non_finite,
        "summary": summary,
        "warnings": warnings,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SUP-004 单请求模型校准：独立 FP32 dense 全局因果注意力参考（只读捕获 npz）"
    )
    parser.add_argument("--capture", required=True, help="捕获 npz 路径（prefill K/V + 逐步 q/out + 元数据）")
    parser.add_argument("--out", required=True, help="JSON 报告输出路径")
    args = parser.parse_args(argv)

    try:
        capture = load_capture(Path(args.capture))
        report = build_report(capture)
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
    print(
        f"p2-calib-oracle: comparisons={summary['comparisons']} "
        f"finite={summary['finite_comparisons']} layers={len(summary['layers'])} "
        f"steps={len(summary['steps'])} max_abs_err_max={shown} "
        f"non_finite={summary['non_finite_count']} out={out_path}"
    )
    return 3 if summary["non_finite_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
