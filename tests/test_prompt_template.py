"""三臂 prompt 与真实 tokenizer 验收（工作单 §4「分段与模板」面）。

只用 CPU：`local_files_only=True` 从已落盘快照加载 tokenizer；不加载权重、不初始化引擎。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attnview import prompts  # noqa: E402
from attnview.parser import MODE_FOCUS, MODE_GLOBAL, MODE_LOCAL, TagParser  # noqa: E402
from attnview.prompt import render_arm, sha256_file  # noqa: E402
from attnview.segmenter import Segment, build_offsets_index, join_segments, segment_context  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = (
    REPO
    / "models/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
QUESTION = "Which two magic chunks hold the founding year and the budget?"
SCRIPT = (
    "<global>chunk 1 first</global>"
    '<focus magic_chunks="1">founded 2003</focus>'
    "<local>year only</local>"
    '<focus magic_chunks="2,3">budget 41</focus>'
    "<answer>2003 and 41</answer>"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class PromptTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
        cls.tokenizer_hash = sha256_file(TOKENIZER_DIR / "tokenizer.json")
        cls.template_hash = sha256_file(TOKENIZER_DIR / "chat_template.jinja")
        cls.segments = [
            Segment(index=1, text="Alpha block: the company was founded in 2003.\n", char_start=0, char_end=0, token_count=0),
            Segment(index=2, text="Beta block: the first budget was 41 million.\n", char_start=0, char_end=0, token_count=0),
            Segment(index=3, text="Gamma block: unrelated logistics notes.\n", char_start=0, char_end=0, token_count=0),
        ]
        cls.context = "\n".join(s.text for s in cls.segments)

    def render(self, arm: str):
        return render_arm(
            arm,
            self.segments,
            QUESTION,
            self.context,
            self.tokenizer,
            tokenizer_hash=self.tokenizer_hash,
            template_hash=self.template_hash,
            enable_thinking=False,
        )

    def test_arms_share_task_and_prompt_provenance(self) -> None:
        da = self.render("da")
        no_mask = self.render("da_no_mask")
        vanilla = self.render("vanilla")
        self.assertEqual(da.rendered, no_mask.rendered, "DA 与 DA-no-mask 必须用同一份 prompt")
        self.assertEqual(da.tools, no_mask.tools)
        self.assertIsNone(vanilla.tools)
        self.assertIn("Magic Chunk 1", da.rendered)
        self.assertIn("Magic Chunk 3", da.rendered)
        self.assertNotIn("Magic Chunk", vanilla.rendered)
        self.assertIn("Context", vanilla.rendered)

    def test_da_instruction_prompt_is_carried_verbatim(self) -> None:
        da = self.render("da")
        expected = prompts.DA_INSTRUCTION_PROMPT.format(question=QUESTION)
        self.assertIn(normalize(expected), normalize(da.rendered))
        vanilla = self.render("vanilla")
        self.assertIn(
            normalize(prompts.VANILLA_INSTRUCTION_PROMPT.format(context=self.context, question=QUESTION)),
            normalize(vanilla.rendered),
        )

    def test_segment_spans_cover_segment_text(self) -> None:
        da = self.render("da")
        for seg, span in zip(self.segments, da.segment_spans):
            decoded = self.tokenizer.decode(da.token_ids[span[0] : span[1]])
            self.assertIn(seg.text.strip(), decoded)
        self.assertEqual(len(da.segment_spans), 3)
        self.assertLess(da.segment_spans[0][1], da.segment_spans[1][0])
        self.assertLess(da.segment_spans[1][1], da.segment_spans[2][0])

    def test_sink_never_covers_context(self) -> None:
        for arm in ("da", "vanilla"):
            rendered = self.render(arm)
            sink = set(range(*rendered.scaffold.sink_span))
            for seg_start, seg_end in rendered.segment_spans or ():
                self.assertEqual(sink & set(range(seg_start, seg_end)), set(), arm)
            self.assertEqual(len(rendered.scaffold.sink_decoded), 16)

    def test_sink_deviation_from_literal_contract(self) -> None:
        """合同 C4.2 要求前 16 token 落在固定的无内容 system **指令**上；实测：

        - vanilla 臂：sink 覆盖 system 指令正文（含角色包装）→ 满足字面要求；
        - da 臂：模型的**原生 tool 声明格式**把 tools 块排在 system 正文之前，
          sink 落在固定的 tool 声明脚手架上（无内容，但**不是** system 指令正文）。
        这里断言的是实测事实，供报告列为待决项；不据此改 prompt 或补字。
        """
        da = self.render("da")
        vanilla = self.render("vanilla")
        # 实测：两个臂的 16-token sink 都**不完全**落在 system 指令正文内
        # （该模板的 system 指令只有 6 个 token，sink 必然延伸到角色包装/后续脚手架）
        self.assertFalse(da.scaffold.sink_inside_system_content)
        self.assertFalse(vanilla.scaffold.sink_inside_system_content)
        # 两臂 sink 都落在 system 消息块内；vanilla 含指令正文，da 是 tool 声明脚手架
        self.assertEqual(da.scaffold.sink_message_role, "system")
        self.assertEqual(vanilla.scaffold.sink_message_role, "system")
        self.assertIn("# Tools", "".join(da.scaffold.sink_decoded))
        self.assertIn(
            prompts.SYSTEM_INSTRUCTION.split()[0],
            " ".join("".join(vanilla.scaffold.sink_decoded).split()),
        )

    def test_real_tokenization_splits_tags_and_effects_land_on_gt_token(self) -> None:
        ids = self.tokenizer(SCRIPT, add_special_tokens=False)["input_ids"]
        pieces = [self.tokenizer.decode([i]) for i in ids]
        self.assertGreater(len(ids), 5, "脚本必须被切成多个 token")
        spans_multiple = any(p.count("<") + p.count(">") < 2 for p in pieces if "<" in p or ">" in p)
        self.assertTrue(spans_multiple, "至少应有一个标签跨多个 token（否则轨迹不具代表性）")

        parser = TagParser(num_segments=3)
        events = []
        for index, piece in enumerate(pieces):
            events.extend(parser.feed(piece, index))
        transitions = [
            (e.mode_before, e.mode_after, e.refs)
            for e in events
            if e.kind == "transition"
        ]
        self.assertEqual(
            transitions,
            [
                (MODE_GLOBAL, MODE_FOCUS, (1,)),
                (MODE_FOCUS, MODE_GLOBAL, ()),
                (MODE_GLOBAL, MODE_LOCAL, ()),
                (MODE_LOCAL, MODE_GLOBAL, ()),
                (MODE_GLOBAL, MODE_FOCUS, (2, 3)),
                (MODE_FOCUS, MODE_GLOBAL, ()),
            ],
            "transition 顺序应严格按脚本",
        )
        self.assertEqual(
            [e.name for e in events if e.kind == "marker"], ["answer", "answer"],
            "开闭 answer 标签都是标记，不产生模式转移",
        )
        # effect_step 必须等于包含 `>` 的那个 token 的序号 + 1
        for event in events:
            if event.kind != "transition":
                continue
            piece = pieces[event.closed_at_token_index]
            self.assertIn(">", piece)
            self.assertEqual(event.effect_step, event.closed_at_token_index + 1)
        self.assertEqual(parser.mode, MODE_GLOBAL)

    def test_real_segmenter_produces_multiple_addressable_fragments(self) -> None:
        chapter = "Sentence about topic {i}. " * 1
        doc = "\n\n".join(
            ("Chapter " + name + "\n" + "".join(chapter.format(i=i) for i in range(170)))
            for name in ("A", "B", "C")
        )
        offsets = self.tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
        index = build_offsets_index(offsets["offset_mapping"])
        segs = segment_context(doc, index)
        total = index.count(0, len(doc))
        self.assertGreater(len(segs), 1, f"整篇 {total} token 必须被切成多个可寻址片段")
        self.assertEqual(join_segments(segs), doc)
        for seg in segs:
            self.assertLessEqual(seg.token_count, 2560)
        rendered = render_arm(
            "da", segs, QUESTION, doc, self.tokenizer, enable_thinking=False
        )
        self.assertEqual(len(rendered.segment_spans), len(segs))
        self.assertLess(rendered.prompt_len, 8192)


if __name__ == "__main__":
    unittest.main()
