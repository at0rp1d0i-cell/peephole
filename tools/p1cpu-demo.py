#!/usr/bin/env python3
"""阶段 03 演示：合成资料 → 真实 tokenizer 分段/三臂 prompt → 固定声明轨迹 → 读取视图 → 公开提取。

只用 CPU：显式 `CUDA_VISIBLE_DEVICES=''`，tokenizer 从已落盘快照以 `local_files_only=True` 加载，
不 import/初始化 vLLM 引擎，不加载权重。

产出（全部写入 `evidence/p1-cpu/`）：

- `demo-input.json`：原文、question、segment 列表（字符偏移/token 数/哈希）
- `prompt-facts.json`：三臂渲染后的 prompt 长度、segment/scaffold token 区间、sink 前 16 token、模板/分词器哈希
- `prompt-da.txt` / `prompt-vanilla.txt`：实际渲染文本（可复跑核对）
- `fixed-trace.json` / `fixed-trace.md`：逐步轨迹（可读表 + 机器可读）
- `extraction.json`：公开提取结果与记账
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from attnview.extract import extract_public_output, public_stream_accounting  # noqa: E402
from attnview.parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL  # noqa: E402
from attnview.prompt import render_arm, sha256_file, sha256_text  # noqa: E402
from attnview.readview import TokenLayout  # noqa: E402
from attnview.reference import declared_positions, reference_visible_positions  # noqa: E402
from attnview.segmenter import build_offsets_index, join_segments, segment_context  # noqa: E402
from attnview.state import RequestProtocolState  # noqa: E402
from attnview.trace import build_step_trace, independence_summary  # noqa: E402

TOKENIZER_DIR = (
    REPO
    / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)

QUESTION = "What was the approval year of the Falcon program, and what was its budget?"

#: 固定声明轨迹（生成流）：覆盖 global → focus(1) → local → global → focus(2,3) → local → answer。
GENERATION_SCRIPT = (
    "<global>The question needs two facts: the approval year and the budget. "
    "Magic Chunk 1 should hold the programme history.</global>"
    '<focus magic_chunks="1">Approved in 2019.</focus>'
    "<global>The budget is still missing; Magic Chunk 2 covers funding lines.</global>"
    '<focus magic_chunks="2,3">Budget of 41 million.</focus>'
    "<local>Approval 2019; budget 41 million.</local>"
    "<answer>Approved in 2019, with a budget of 41 million.</answer>"
)


def build_chapters(tokenizer, target_tokens: int = 1900, seed: int = 20260918) -> list[str]:
    """按目标 token 数构造三段资料（每段都在上限内，整篇超过上限）。

    问题需要其中两段：Chapter A 给批准年份，Chapter B 给预算；Chapter C 是无关后勤内容。
    """
    rng = random.Random(seed)
    subjects = [
        "the platform review", "the supplier audit", "the field trial", "the funding line",
        "the steering committee", "the deployment plan", "the safety review", "the cost model",
        "the staffing plan", "the procurement round",
    ]
    verbs = [
        "was recorded as complete", "remained under review", "was scheduled for week",
        "was revised during quarter", "was flagged as pending", "reported a delta of",
    ]
    chapters: list[tuple[str, list[str], str]] = [
        (
            "Chapter A: programme history",
            [
                "The Falcon program was approved in 2019 after a two-year review cycle.",
                "Its first prototype was assembled at the northern site.",
                "The approving minute lists the programme office as the owner of record.",
            ],
            "Falcon",
        ),
        (
            "Chapter B: funding lines",
            [
                "The Falcon program carries a budget of 41 million for its first phase.",
                "Later phases are not yet allocated in the current plan.",
                "The funding line is reviewed each quarter against the cost model.",
            ],
            "budget",
        ),
        (
            "Chapter C: unrelated logistics",
            [
                "Warehouse throughput improved after the conveyor upgrade.",
                "The logistics annex describes pallet routing and dock scheduling.",
                "Nothing in this chapter concerns the Falcon program.",
            ],
            "logistics",
        ),
    ]

    out: list[str] = []
    for title, anchors, topic in chapters:
        lines = [title, ""] + list(anchors) + [""]
        count = len(tokenizer("\n".join(lines), add_special_tokens=False)["input_ids"])
        i = 0
        while count < target_tokens:
            subject = subjects[i % len(subjects)]
            verb = verbs[(i * 7 + rng.randint(0, 5)) % len(verbs)]
            line = (
                f"Note {i:04d}: {subject} of the {topic} dossier {verb} {100 + (i * 13) % 900} "
                f"in record {i * 3 + 1}; the reviewer initialled item {i * 5 + 2}."
            )
            lines.append(line)
            count += len(tokenizer(line, add_special_tokens=False)["input_ids"]) + 1
            i += 1
            if i % 40 == 0:
                lines.append("")
        out.append("\n".join(lines).strip() + "\n")
    return out


def make_segments(tokenizer, document: str):
    offsets = tokenizer(document, add_special_tokens=False, return_offsets_mapping=True)
    index = build_offsets_index(offsets["offset_mapping"])
    return segment_context(document, index), index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "evidence/p1-cpu"))
    parser.add_argument("--kernel-block-size", type=int, default=784)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    tokenizer_hash = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    template_hash = sha256_file(TOKENIZER_DIR / "chat_template.jinja")

    chapters = build_chapters(tokenizer, target_tokens=1900)
    document = "\n\n".join(c.rstrip("\n") for c in chapters) + "\n"
    segments, _ = make_segments(tokenizer, document)

    facts: dict = {
        "tokenizer": str(TOKENIZER_DIR),
        "tokenizer_sha256": tokenizer_hash,
        "chat_template_sha256": template_hash,
        "question": QUESTION,
        "document_sha256": sha256_text(document),
        "document_chars": len(document),
        "document_tokens": len(tokenizer(document, add_special_tokens=False)["input_ids"]),
        "segment_count": len(segments),
        "segments": [
            {
                "index": s.index,
                "char_span": [s.char_start, s.char_end],
                "tokens": s.token_count,
                "sha256": sha256_text(s.text),
                "head": s.text[:80],
            }
            for s in segments
        ],
        "lossless_join_ok": join_segments(segments) == document,
        "single_segment_rule_ok": (
            len(segments) > 1
        ) == (len(tokenizer(document, add_special_tokens=False)["input_ids"]) > 2560),
    }

    arms = {}
    for arm in ("da", "da_no_mask", "vanilla"):
        rendered = render_arm(
            arm,
            segments,
            QUESTION,
            document,
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            template_hash=template_hash,
            enable_thinking=False,
        )
        arms[arm] = rendered
        (out_dir / f"prompt-{arm}.txt").write_text(rendered.rendered, encoding="utf-8")

    da = arms["da"]
    facts["arms"] = {
        arm: {
            "prompt_len": r.prompt_len,
            "segment_spans": [list(s) for s in r.segment_spans],
            "question_span": list(r.question_span),
            "local_window_span": list(r.scaffold.local_window_span),
            "system_content_span": list(r.scaffold.system_content_span),
            "sink_span": list(r.scaffold.sink_span),
            "sink_message_role": r.scaffold.sink_message_role,
            "sink_decoded": list(r.scaffold.sink_decoded),
            "sink_inside_system_content": (
                r.scaffold.sink_span[0] >= r.scaffold.system_content_span[0]
                and r.scaffold.sink_span[1] <= r.scaffold.system_content_span[1]
            ),
            "sink_overlaps_context": any(
                min(r.scaffold.sink_span[1], seg_end) > max(r.scaffold.sink_span[0], seg_start)
                for seg_start, seg_end in (r.segment_spans or ())
            ),
            "prompt_sha256": sha256_text(r.rendered),
            "enable_thinking": r.enable_thinking,
            "notes": list(r.notes),
            "tools_declared": bool(r.tools),
        }
        for arm, r in arms.items()
    }
    facts["da_no_mask_same_prompt_as_da"] = (
        arms["da"].rendered == arms["da_no_mask"].rendered
        and arms["da"].tools == arms["da_no_mask"].tools
    )

    # 模板传参方式核对（C2.5 thinking 关闭）：直接 kwargs vs chat_template_kwargs vs 默认
    probe_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    variants = {
        "direct_kwarg": tokenizer.apply_chat_template(
            probe_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ),
        "chat_template_kwargs": tokenizer.apply_chat_template(
            probe_msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        ),
        "default": tokenizer.apply_chat_template(
            probe_msgs, tokenize=False, add_generation_prompt=True
        ),
    }
    (out_dir / "template-kwargs-check.txt").write_text(
        "\n".join(f"--- {k} ---\n{v!r}" for k, v in variants.items()) + "\n",
        encoding="utf-8",
    )
    facts["template_kwargs_check"] = {
        "direct_kwarg_disables_thinking": "<think>\n\n</think>" in variants["direct_kwarg"],
        "chat_template_kwargs_disables_thinking": "<think>\n\n</think>"
        in variants["chat_template_kwargs"],
        "default_disables_thinking": "<think>\n\n</think>" in variants["default"],
        "direct_equals_default": variants["direct_kwarg"] == variants["default"],
    }

    # --- 固定声明轨迹 ---------------------------------------------------
    layout = TokenLayout(
        prompt_len=da.prompt_len,
        segment_spans=da.segment_spans,
        local_window_span=da.scaffold.local_window_span,
        sink_span=da.scaffold.sink_span,
    )
    script_ids = tokenizer(GENERATION_SCRIPT, add_special_tokens=False)["input_ids"]
    state = RequestProtocolState(
        "demo-request-1",
        arm="da",
        num_segments=len(segments),
        prompt_len=da.prompt_len,
    )
    fed: list[str] = []
    for token_index, token_id in enumerate(script_ids):
        text = tokenizer.decode([token_id])
        fed.append(text)
        state.feed_generated_token(token_index, token_id, text)
    state.finish()

    if "".join(fed) != GENERATION_SCRIPT:
        raise SystemExit("生成流逐 token decode 拼接与脚本不一致；轨迹不可信")

    kernel_block_size = args.kernel_block_size
    kv_len_max = da.prompt_len + len(script_ids)
    total_blocks = (kv_len_max + kernel_block_size - 1) // kernel_block_size
    physical = list(range(101, 101 + total_blocks))
    random.Random(7).shuffle(physical)
    canonical_blocks = tuple(physical)
    max_width = len(canonical_blocks)

    rows = build_step_trace(
        state,
        layout,
        canonical_blocks=canonical_blocks,
        kernel_block_size=kernel_block_size,
        max_width=max_width,
    )

    # 参考对照：每个 generation step 的视图与独立参考逐位置一致
    mismatches = []
    for row in rows:
        if row.decode_step == 0:
            continue
        expected = reference_visible_positions(
            mode=row.mode,
            refs=row.refs,
            layout=layout,
            kv_len=row.kv_len,
            kernel_block_size=kernel_block_size,
        )
        if expected != row.view.positions():
            mismatches.append(row.decode_step)

    # 至少一步：784 对齐外扩之后仍有**已写入**的全注意力块被排除
    excluded_evidence = []
    for row in rows:
        written_blocks = set(range((row.kv_len + kernel_block_size - 1) // kernel_block_size))
        excluded = sorted(written_blocks - set(row.view.visible_blocks))
        if row.decode_step >= 1 and excluded:
            excluded_evidence.append(
                {
                    "decode_step": row.decode_step,
                    "mode": row.mode,
                    "refs": list(row.refs),
                    "written_blocks": len(written_blocks),
                    "excluded_blocks": excluded,
                    "visible_blocks": list(row.view.visible_blocks),
                    "visible_tokens": len(row.view.positions()),
                    "kv_len": row.kv_len,
                    "declared_positions_all_visible": all(
                        pos in row.view.positions()
                        for pos in declared_positions(layout, row.refs, row.kv_len)
                    ),
                }
            )

    trace = {
        "request_id": state.request_id,
        "arm": "da",
        "kernel_block_size": kernel_block_size,
        "manager_block_size": kernel_block_size,
        "prompt_len": da.prompt_len,
        "generation_tokens": len(script_ids),
        "kv_len_max": kv_len_max,
        "max_width": max_width,
        "canonical_blocks": list(canonical_blocks),
        "total_sequence_budget": 8192,
        "budget_ok": kv_len_max <= 8192,
        "generation_budget_headroom": 8192 - kv_len_max,
        "script": GENERATION_SCRIPT,
        "reference_mismatch_steps": mismatches,
        "excluded_block_steps": len(excluded_evidence),
        "excluded_block_examples": excluded_evidence[:6],
        "independence": independence_summary(rows),
        "protocol_trace": state.trace(),
        "rows": [row.as_dict() for row in rows],
    }

    extraction = extract_public_output(GENERATION_SCRIPT)
    (out_dir / "demo-input.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "prompt-facts.json").write_text(
        json.dumps(facts["arms"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "fixed-trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "extraction.json").write_text(
        json.dumps(
            {
                "raw_stream": GENERATION_SCRIPT,
                "result": extraction.as_dict(),
                "accounting": public_stream_accounting(GENERATION_SCRIPT),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "fixed-trace.md").write_text(_render_trace_markdown(trace), encoding="utf-8")

    print(json.dumps(
        {
            "segments": facts["segment_count"],
            "segment_tokens": [s["tokens"] for s in facts["segments"]],
            "lossless_join_ok": facts["lossless_join_ok"],
            "prompt_len_da": da.prompt_len,
            "prompt_len_vanilla": arms["vanilla"].prompt_len,
            "kv_len_max": kv_len_max,
            "budget_ok": trace["budget_ok"],
            "reference_mismatch_steps": mismatches,
            "excluded_block_steps": len(excluded_evidence),
            "read_list_changes": trace["independence"]["read_list_changes"],
            "extraction_ok": extraction.ok,
            "answer": extraction.answer,
            "sink_role_da": da.scaffold.sink_message_role,
            "sink_overlaps_context": any(a["sink_overlaps_context"] for a in facts["arms"].values()),
            "template_kwargs_check": facts["template_kwargs_check"],
            "segment_token_range": [min(x["tokens"] for x in facts["segments"]), max(x["tokens"] for x in facts["segments"])],
            "sink_role_vanilla": arms["vanilla"].scaffold.sink_message_role,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def _render_trace_markdown(trace: dict) -> str:
    lines = [
        "# 固定声明轨迹（da 臂，kernel/manager 块 = %d）" % trace["kernel_block_size"],
        "",
        f"- request_id: `{trace['request_id']}`；prompt_len = {trace['prompt_len']}；"
        f"生成 {trace['generation_tokens']} token；最大 kv_len = {trace['kv_len_max']}"
        f"（8192 预算内余量 {trace['generation_budget_headroom']}）",
        f"- 物理块映射（非连续，逻辑块 0..{trace['max_width'] - 1}）：`{trace['canonical_blocks']}`",
        f"- 独立参考逐位置对照：不一致 step = {trace['reference_mismatch_steps'] or '无'}",
        f"- 有已写入块被排除的 step 数 = {trace['excluded_block_steps']}",
        f"- 读清单变化次数 = {trace['independence']['read_list_changes']}；"
        f"写位置第 1 步 = {trace['independence']['write_position_step1']}、"
        f"末步 = {trace['independence']['write_position_last']}、"
        f"严格 +1 = {trace['independence']['write_monotonic_plus_one']}",
        "",
        "| step | 输入 token (idx/text) | 本步闭合事件 → effect_step | 下步模式 | 写位置 | kv_len | 可见原始区间 | 外扩区间 | 逻辑块 | 物理块 | 尾块有效 | 可见 token |",
        "| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in trace["rows"]:
        view = row["view"]
        events = "; ".join(
            f"{e['kind']}:{e['name']}"
            + (f"({','.join(map(str, e['refs']))})" if e["refs"] else "")
            + f"→{e['effect_step']}"
            + (f" [{e['reason']}]" if e["reason"] else "")
            for e in row["events"]
        ) or "—"
        token_text = (row["input_token_text"] or "").replace("|", "\\|").replace("\n", "\\n")
        lines.append(
            "| {step} | {tok} | {events} | {mode} | {wpos} | {kv} | {raw} | {vis} | {lb} | {pb} | {tail} | {ntok} |".format(
                step=row["decode_step"],
                tok=f"{row['input_token_index']}:`{token_text}`" if row["input_token_index"] is not None else "prefill",
                events=events,
                mode=row["mode"] + (f"[{','.join(map(str, row['refs']))}]" if row["refs"] else ""),
                wpos=row["write_position"] if row["write_position"] is not None else "—",
                kv=row["kv_len"],
                raw=" ".join(f"[{a},{b})" for a, b in view["raw_spans"]) or "—",
                vis=" ".join(f"[{a},{b})" for a, b in view["visible_spans"]) or "—",
                lb=",".join(map(str, view["visible_blocks"])) or "—",
                pb=",".join(map(str, view["physical_block_ids"])) or "—",
                tail=view["tail_block_valid_len"],
                ntok=view["visible_tokens"],
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
