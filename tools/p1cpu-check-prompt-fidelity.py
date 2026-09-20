#!/usr/bin/env python3
"""逐字保真核对：`src/attnview/prompts.py` 的 prompt 文本 vs 论文摘录件（附录 F）。

**能证明什么 / 不能证明什么**（结论写在输出里，避免把弱检查说成逐字通过）：

- 能证明：① 非表格正文按**原文顺序**逐字出现（归一化空白后严格子串 + 匹配位置递增）；
  ② 表格按**行列单元格**逐字一致：用表头行的列起始偏移把摘录件的换行碎片归位到列、再拼回单元格，
  与本地重建的单元格做**逐格**比较（含行列对应关系）；
  ③ 全文词多重集合相等（无增删改词）。
- 不能证明：论文 PDF 的**版面**（列宽、换行位置、单元格内换行）与本地重建一致——
  `pdftotext -layout` 已把双栏版面压平成文本，版面信息不可恢复；本地重建只是"把碎片按列拼回单元格"。
- 剔除的页面噪声：页眉、纯页码、`… Instruction Prompt (continued)` 整行。
- PDF 行尾连字符（摘录件里的 `ex-` + 换行 + `tracted`）按"去版式连字符"拼回 `extracted`；
  每次拼接都计入输出里的 de-hyphenation 计数（这是去版式，不是改词）。
- 列归位用表头行的碎片起始偏移（≥2 空格切分）作为列锚点；列 0 上以**小写字母开头**的碎片判为上一行同列的续行
  （本摘录件里唯一出现这种情况的是 `<focus` / `magic_chunks="K">` 这一处），该规则在输出中声明。

用法：`python3 tools/p1cpu-check-prompt-fidelity.py [摘录件路径]`

默认摘录件为仓内 `docs/references/da-paper-extract.md`（相对仓库根）；仓内不存在时回退到
`$ATTNVIEW_MATERIAL/docs/references/da-paper-extract.md`（私有素材仓快照，不随本仓发行）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from attnview.prompts import DA_INSTRUCTION_PROMPT, VANILLA_INSTRUCTION_PROMPT  # noqa: E402

EXTRACT_REL = "docs/references/da-paper-extract.md"
DEFAULT_EXTRACT = REPO / EXTRACT_REL  # 仓内副本（相对仓库根解析）


def resolve_extract() -> Path:
    """默认摘录件：优先仓内副本，其次私有素材仓快照（$ATTNVIEW_MATERIAL）。"""
    material = os.environ.get("ATTNVIEW_MATERIAL")
    candidates = [DEFAULT_EXTRACT]
    if material:
        candidates.append(Path(material) / EXTRACT_REL)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if material:
        fallback = f"{material}/{EXTRACT_REL}"
    else:
        fallback = f"$ATTNVIEW_MATERIAL/{EXTRACT_REL}（未设置或为空）"
    raise SystemExit(
        "找不到论文摘录件 da-paper-extract.md（附录 F）。已查找：\n"
        f"  1) 仓内副本：{DEFAULT_EXTRACT}\n"
        f"  2) 素材仓快照：{fallback}\n"
        "公开 checkout 不发行 material/；请显式传入摘录件路径，或设置 ATTNVIEW_MATERIAL。"
    )


PAGE_NOISE = re.compile(
    r"^\s*(Language Models Can Control Their Own Attention|\d{1,3}|"
    r"(DA|Vanilla) Instruction Prompt.*)\s*$"
)
FRAG_RE = re.compile(r"\S.*?(?=\s{2,}|$)")
CONTINUATION_COL0 = re.compile(r"^[a-z]")
BIG_INDENT = re.compile(r"^ {20,}\S")


def appendix_f_block(text: str) -> str:
    start = text.index("## 附录 F")
    fence = text.index("```text", start)
    end = text.index("```", fence + 7)
    return text[fence + len("```text") : end]


def section(body: str, start_marker: str, end_marker: str) -> str:
    start = re.search(rf"(?m)^{re.escape(start_marker)}\s*$", body)
    if start is None:
        raise SystemExit(f"摘录件里找不到小节标题行：{start_marker}")
    end = body.index(end_marker, start.start() + len(start_marker))
    return "\n".join(
        line for line in body[start.start() : end].splitlines() if not PAGE_NOISE.match(line)
    )


def fragments(line: str) -> list[tuple[int, str]]:
    """按"2 个及以上空格"切出非空白片段及其起始偏移（忽略行首缩进）。"""
    return [(m.start(), m.group(0).strip()) for m in FRAG_RE.finditer(line)]


def split_prose_and_tables(text: str) -> tuple[list[str], list[list[str]]]:
    """切分为 (prose 行, 表格块列表)。

    表格识别：某行有 ≥3 个片段即开启/延续一张表；其后若一行的**所有**片段起点都落在该表的列锚点（±2）上，
    则算作该表的续行。同一小节里出现两张列宽不同的表（本任务的 DA prompt 就是如此）时，
    第一/第二列锚点明显不同则开启新表块。
    """
    prose: list[str] = []
    blocks: list[list[str]] = []
    anchors: list[int] | None = None
    for line in text.splitlines():
        parts = fragments(line)
        starts = [s for s, _ in parts]
        if len(parts) >= 3:
            same_table = (
                anchors is not None
                and abs(starts[0] - anchors[0]) <= 2
                and abs(starts[1] - anchors[1]) <= 8
            )
            if not same_table:
                blocks.append([])
                anchors = starts
            blocks[-1].append(line)
            continue
        if (
            anchors is not None
            and parts
            and all(any(abs(s - a) <= 2 for a in anchors) for s in starts)
        ):
            blocks[-1].append(line)
            continue
        anchors = None
        prose.append(line)
    return prose, [b for b in blocks if b]


def cells_by_columns(
    lines: list[str], hyphenations: list[int] | None = None
) -> tuple[list[list[str]], int]:
    """按列锚点把碎片归位并拼回单元格 → 二维单元格表（行 × 列）。"""
    if hyphenations is None:
        hyphenations = [0]
    anchors: list[int] = []
    for line in lines:
        starts = [s for s, _ in fragments(line)]
        if len(starts) >= 2:
            anchors = starts
            break

    rows: list[list[list[str]]] = []
    unassigned: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = fragments(line)
        if not parts:
            continue

        def column_of(offset: int) -> int:
            best = 0
            for idx, anchor in enumerate(anchors):
                if offset >= anchor - 2:
                    best = idx
            return best

        col0 = [t for s, t in parts if column_of(s) == 0]
        starts_row = bool(col0) and not CONTINUATION_COL0.match(col0[0])
        if starts_row or not rows:
            rows.append([[] for _ in anchors])
        for s, text in parts:
            col = column_of(s)
            if col >= len(anchors):
                unassigned.append(text)
                continue
            rows[-1][col].append(text)
    joined: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for parts_in_cell in row:
            text = ""
            for fragment in parts_in_cell:
                if text.endswith("-"):
                    # PDF 行尾连字符：去连字符拼回原词（记为一次 de-hyphenation，不是改词）
                    text = text[:-1] + fragment
                    hyphenations[0] += 1
                else:
                    text = f"{text} {fragment}".strip()
            cells.append(text)
        joined.append(cells)
    return joined, len(unassigned)


def prose_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_prompt(name: str, mine: str, ref: str) -> list[str]:
    failures: list[str] = []
    mine_prose, mine_tables = split_prose_and_tables(mine)
    ref_prose, ref_tables = split_prose_and_tables(ref)
    if len(mine_tables) != len(ref_tables):
        failures.append(
            f"[{name}] 表格块数量不一致：本地 {len(mine_tables)} / 摘录件 {len(ref_tables)}"
        )

    ref_prose_text = normalize(" ".join(ref_prose))
    cursor = 0
    checked = 0
    for block in prose_blocks(mine_prose):
        target = normalize(block)
        if len(target) < 12:
            continue
        at = ref_prose_text.find(target, cursor)
        if at < 0:
            where = ref_prose_text.find(target)
            if where < 0:
                failures.append(f"[{name}] 正文段落未逐字出现：{target[:70]}…")
            else:
                failures.append(f"[{name}] 正文段落顺序不符：{target[:70]}…")
            continue
        cursor = at + len(target)
        checked += 1
    print(f"[{name}] ① 有序正文逐字：{checked} 段通过")

    mine_rows: list[list[str]] = []
    ref_rows: list[list[str]] = []
    unassigned = 0
    hyphenations = [0]
    # 表格块数不一致时上面已记入 failures：这里逐块比较公共前缀，不重复抛错（strict=False 是有意截断）。
    for index, (mine_block, ref_block) in enumerate(zip(mine_tables, ref_tables, strict=False)):
        rows_a, _ = cells_by_columns(mine_block, hyphenations)
        rows_b, unassigned_b = cells_by_columns(ref_block, hyphenations)
        if len(rows_a) != len(rows_b):
            failures.append(
                f"[{name}] 表 {index + 1} 行数不一致：本地 {len(rows_a)} / 摘录件 {len(rows_b)}"
            )
        mine_rows.extend(rows_a)
        ref_rows.extend(rows_b)
        unassigned += unassigned_b
    mine_cells = [normalize(c) for row in mine_rows for c in row]
    ref_cells = [normalize(c) for row in ref_rows for c in row]
    if mine_cells == ref_cells:
        print(
            f"[{name}] ② 表格单元格逐格一致：{len(mine_tables)} 张表 / {len(mine_rows)} 行 / "
            f"{len(mine_cells)} 格"
        )
    else:
        print(f"[{name}] ② 表格单元格不一致：本地 {len(mine_cells)} 格 / 摘录件 {len(ref_cells)} 格")
        for i in range(max(len(mine_cells), len(ref_cells))):
            a = mine_cells[i] if i < len(mine_cells) else "<缺失>"
            b = ref_cells[i] if i < len(ref_cells) else "<缺失>"
            if normalize(a) != normalize(b):
                print(f"    格 #{i}\n      本地   : {a}\n      摘录件 : {b}")
        failures.append(f"[{name}] 表格单元格序列不一致")
    if unassigned:
        print(f"    （摘录件有 {unassigned} 个碎片未能归位到列，已计入差异）")
        failures.append(f"[{name}] 有 {len(unassigned)} 个表格碎片未归位")

    def dehyphenate(text: str) -> str:
        return re.sub(r"-\n\s*", "", text)

    # 词级比较建立在**已拼回的文本**上（散文 + 单元格），否则同一处版式断词会在两侧形成不同的词：
    # 摘录件把 "extracted" 断成单元格内的 "ex-" + 续行 "tracted"，而本地重建是一个词。
    mine_words = Counter(re.findall(r"\S+", dehyphenate(" ".join(mine_prose)) + " " + " ".join(mine_cells)))
    ref_words = Counter(re.findall(r"\S+", dehyphenate(" ".join(ref_prose)) + " " + " ".join(ref_cells)))
    word_diff = (ref_words - mine_words) or (mine_words - ref_words)
    if word_diff:
        print(f"[{name}] ③ 词多重集合：不一致 {dict(list(word_diff.items())[:10])}")
        failures.append(f"[{name}] 词多重集合不一致")
    else:
        print(f"[{name}] ③ 词多重集合：{'一致' if not word_diff else '不一致'}（de-hyphenation {hyphenations[0]} 处）")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "extract",
        nargs="?",
        default=None,
        help="摘录件路径（默认：仓内 docs/references/da-paper-extract.md，"
        "回退 $ATTNVIEW_MATERIAL/docs/references/da-paper-extract.md）",
    )
    args = ap.parse_args()
    extract = Path(args.extract) if args.extract is not None else resolve_extract()
    body = appendix_f_block(extract.read_text(encoding="utf-8"))
    print(f"摘录件：{extract}")
    print("剔除：页眉 / 纯页码 / `… Instruction Prompt (continued)` 整行")
    print("列归位：以表头行碎片起始偏移为列锚点；列 0 上小写字母开头的碎片判为续行\n")
    failures: list[str] = []
    failures += check_prompt(
        "DA Instruction Prompt",
        DA_INSTRUCTION_PROMPT,
        section(body, "DA Instruction Prompt", "Vanilla Instruction Prompt"),
    )
    print()
    failures += check_prompt(
        "Vanilla Instruction Prompt",
        VANILLA_INSTRUCTION_PROMPT,
        section(body, "Vanilla Instruction Prompt", "Rubric Generation Prompt"),
    )
    print("\n结论：" + ("①②③ 全部通过" if not failures else "存在未通过项（见下）"))
    for f in failures:
        print("  FAIL " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
