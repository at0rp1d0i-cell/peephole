#!/usr/bin/env python3
"""阶段 05 检查点 1 的 CPU 通道探针：协议参数经既有 `SamplingParams.extra_args` 的序列化往返。

只读、纯 CPU（`CUDA_VISIBLE_DEVICES=''`），不加载模型、不启动引擎、不联网。
目的：回答 SUP-004 §2.1「有源码字段不代表序列化可用」——给出实际往返证据或失败证据。

用法：
    source /root/attnview/env.sh
    CUDA_VISIBLE_DEVICES='' "$ATTNVIEW_PYTHON" tools/p2-probe-params-channel.py \
        --evidence evidence/p2-single
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REQUIRED_MODULES = ("vllm", "msgspec")

# 与设计一致的最小协议载荷：模式 + 片段引用 + 该步可见长度 + 可见逻辑块（升序、无重叠）。
PAYLOAD = {
    "attnview": {
        "mode": "focus",
        "refs": [3, 1],
        "visible_logical_blocks": [0, 5, 6, 7],
        "visible_kv_len": 6272,
        "effect_step": 41,
        "protocol": "v1.0",
    }
}


def _now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +0800")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="evidence/p2-single")
    args = ap.parse_args()

    out_dir = Path(args.evidence)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"date_cst": _now_cst(), "argv": sys.argv, "payload": PAYLOAD, "checks": []}
    ok_all = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    import vllm  # noqa: E402
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402

    report["version"] = {
        "vllm": getattr(vllm, "__version__", None),
        "vllm_file": vllm.__file__,
        "serial_utils": __import__("vllm.v1.serial_utils", fromlist=["x"]).__file__,
        "sampling_params": __import__("vllm.sampling_params", fromlist=["x"]).__file__,
    }

    sp = SamplingParams(temperature=0.0, max_tokens=64, extra_args=json.loads(json.dumps(PAYLOAD)))
    check("构造后 extra_args 未被 __post_init__ 改写", sp.extra_args == PAYLOAD,
          f"extra_args={sp.extra_args!r}")

    enc, dec_sp = MsgpackEncoder(), MsgpackDecoder(SamplingParams)
    rt_sp = dec_sp.decode(enc.encode(sp))
    check("SamplingParams 往返类型", isinstance(rt_sp, SamplingParams), type(rt_sp).__name__)
    check("SamplingParams 往返后 extra_args 逐字段相等", rt_sp.extra_args == PAYLOAD,
          f"往返={rt_sp.extra_args!r}")
    check("嵌套值类型保真", rt_sp.extra_args["attnview"]["refs"] == [3, 1]
          and isinstance(rt_sp.extra_args["attnview"]["effect_step"], int),
          f"refs={rt_sp.extra_args['attnview']['refs']!r} "
          f"effect_step={type(rt_sp.extra_args['attnview']['effect_step']).__name__}")
    check("其它采样字段同时保真",
          (rt_sp.temperature, rt_sp.max_tokens) == (sp.temperature, sp.max_tokens),
          f"temperature={rt_sp.temperature} max_tokens={rt_sp.max_tokens}")

    # API server -> EngineCore 的真实一跳（跨进程时经 msgspec/msgpack）
    from vllm.v1.engine import EngineCoreRequest  # noqa: E402
    req = EngineCoreRequest(
        request_id="probe-req-0",
        prompt_token_ids=list(range(16)),
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    enc2, dec2 = MsgpackEncoder(), MsgpackDecoder(EngineCoreRequest)
    rt_req = dec2.decode(enc2.encode(req))
    check("EngineCoreRequest 往返类型", isinstance(rt_req, EngineCoreRequest), type(rt_req).__name__)
    check("EngineCoreRequest 往返后 extra_args 保持",
          rt_req.sampling_params is not None and rt_req.sampling_params.extra_args == PAYLOAD,
          f"往返={rt_req.sampling_params.extra_args if rt_req.sampling_params else None!r}")

    # core -> worker 的调度结构：TP=1 走 UniProcExecutor 为同进程直调（无序列化），
    # 但仍验证该结构本身可被 msgspec 编码，以免未来换多进程执行器时才暴露问题。
    from vllm.v1.core.sched.output import NewRequestData  # noqa: E402
    nrd = NewRequestData(
        req_id="probe-req-0",
        prompt_token_ids=list(range(16)),
        mm_features=[],
        sampling_params=sp,
        pooling_params=None,
        block_ids=([0, 1, 2, 3],),
        num_computed_tokens=0,
        lora_request=None,
    )
    # 注意：`MsgpackDecoder(NewRequestData)` 在本环境会因该模块把 `torch.Tensor | None`
    # 写成字符串前向引用、而模块命名空间没有 `torch` 而抛 NameError（msgspec 求值注解）。
    # TP=1 的 `UniProcExecutor` 是**同进程直调**（`serial_utils.run_method`），不经过该解码器，
    # 因此本项只验证「编码可用 + 解码为原生 dict 时载荷保真」，并把注解 quirk 如实记录。
    dec3 = MsgpackDecoder()
    try:
        MsgpackDecoder(NewRequestData)
        decoder_typing = "ok"
    except Exception as exc:  # noqa: BLE001
        decoder_typing = f"{type(exc).__name__}: {exc}"
    report["new_request_data_decoder_typing"] = decoder_typing
    enc3 = MsgpackEncoder()
    rt_nrd = dec3.decode(enc3.encode(nrd))
    nrd_extra = rt_nrd["sampling_params"]["extra_args"]
    check("NewRequestData 编码后可解码且 extra_args/block_ids 保真",
          nrd_extra == PAYLOAD and rt_nrd["block_ids"] == [[0, 1, 2, 3]],
          f"extra_args={nrd_extra!r} block_ids={rt_nrd['block_ids']!r} "
          f"| NewRequestData 类型化解码: {decoder_typing}")

    report["ok"] = ok_all
    (out_dir / "params-channel-probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str)
    )
    print(f"\n{'PASS' if ok_all else 'FAIL'} -> {out_dir / 'params-channel-probe.json'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
