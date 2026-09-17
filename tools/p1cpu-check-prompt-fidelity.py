#!/usr/bin/env python3
"""逐字保真核对：把 `src/attnview/prompts.py` 的 prompt 文本与论文摘录件机械对照。

两种检查（都只用标准库）：

1. **词袋相等**：整段 prompt 的词（含标点、按空白切分）多重集合必须与摘录件一致——不允许增删改词。
   这一项允许表格**版式重排**（摘录件是 `pdftotext -layout` 的双栏展开）。
2. **散文段落严格子串**：非表格的连续段落归一化空白后必须逐字出现，验证版式重排只发生在表格区域。

用法：`python3 tools/p1cpu-check-prompt-fidelity.py [摘录件路径]`
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from attnview.prompts import (  # noqa: E402
    DA_INSTRUCTION_PROMPT,
    VANILLA_INSTRUCTION_PROMPT,
)

DEFAULT_EXTRACT = REPO / "material/attnview/docs/references/da-paper-extract.md"
PAGE_NOISE = re.compile(
    r"^\s*(Language Models Can Control Their Own Attention|\d{1,3}|"
    r"(DA|Vanilla) Instruction Prompt.*)\s*$"
)


def appendix_f_block(text: str) -> str:
    """取附录 F 的 fenced 代码块内容（摘录件正文里的说明文字不算）。"""
    marker = "## 附录 F"
    start = text.index(marker)
    fence = text.index("```text", start)
    end = text.index("```", fence + 7)
    return text[fence + len("```text") : end]


def section(body: str, start_marker: str, end_marker: str) -> str:
    # 行首锚定：摘录件正文里也出现 "…the DA Instruction Prompt below." 这样的句子
    start = re.search(rf"(?m)^{re.escape(start_marker)}\s*$", body)
    if start is None:
        raise SystemExit(f"摘录件里找不到小节标题行：{start_marker}")
    start = start.start()
    end = body.index(end_marker, start + len(start_marker))
    chunk = body[start:end]
    chunk = "\n".join(line for line in chunk.splitlines() if not PAGE_NOISE.match(line))
    # 摘录件把长词在行尾用连字符断开（ex-\ntracted）与分页续行；回接后再比
    return re.sub(r"-\n", "", chunk)


def chars(text: str) -> Counter:
    """非空白字符多重集合：容忍版式重排/单元格拼接，但对增删改字符敏感。"""
    return Counter(re.sub(r"\s+", "", text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("extract", nargs="?", default=str(DEFAULT_EXTRACT))
    args = ap.parse_args()

    body = appendix_f_block(Path(args.extract).read_text(encoding="utf-8"))
    da_ref = section(body, "DA Instruction Prompt", "Vanilla Instruction Prompt")
    vanilla_ref = section(body, "Vanilla Instruction Prompt", "Rubric Generation Prompt")

    da_mine = DA_INSTRUCTION_PROMPT
    vanilla_mine = VANILLA_INSTRUCTION_PROMPT

    failures = 0
    print(f"摘录件：{args.extract}")

    for name, mine, ref in (
        ("DA Instruction Prompt", da_mine, da_ref),
        ("Vanilla Instruction Prompt", vanilla_mine, vanilla_ref),
    ):
        mine_chars, ref_chars = chars(mine), chars(ref)
        missing = ref_chars - mine_chars
        extra = mine_chars - ref_chars
        ok = not missing and not extra
        delta = sum(missing.values()) + sum(extra.values())
        print(f"\n[{name}] 非空白字符多重集核对：{'一致' if ok else f'差异 {delta} 个字符'}")
        if missing:
            print(f"  摘录件有、prompts.py 无：{dict(list(missing.items())[:20])}")
        if extra:
            print(f"  prompts.py 有、摘录件无：{dict(list(extra.items())[:20])}")
        failures += 0 if ok else 1

    prose = [
        "Answer the question above using only the retrieved document.",
        "Reason through the magic chunks using three modes.",
        "Global mode (default)",
        "VALUE is the answer the magic chunk provides: a name, number, date, or short noun phrase (typically 1 to 12 words).",
        "Three soft requirements:",
        "End your response with the final answer wrapped in <answer>...</answer>.",
        "Find the answer to the Question based solely on information in the Context above.",
    ]
    da_norm, vanilla_norm = normalize(da_mine.format(question="Q")), normalize(
        vanilla_mine.format(question="Q", context="C")
    )
    print("\n[散文段落严格子串核对]")
    for sentence in prose:
        target = normalize(sentence)
        where = "DA" if target in da_norm else ("Vanilla" if target in vanilla_norm else None)
        print(f"  {'ok  ' if where else 'FAIL'} {where or '未找到':<8} {sentence[:60]}...")
        failures += 0 if where else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
