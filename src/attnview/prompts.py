"""C2 prompt 文本（依据 `docs/references/da-paper-extract.md` 附录 F，逐字）。

证据级别与保真说明（写入 `evidence/p1-cpu/prompt-fidelity.md`，由
`tools/p1cpu-check-prompt-fidelity.py` 与提取件做归一化逐字对照）：

- `DA_INSTRUCTION_PROMPT` / `VANILLA_INSTRUCTION_PROMPT`：**论文原文逐字**（附录 F）。
  提取件来自 `pdftotext -layout`，双栏页面的表格在纯文本里按列展开；本文件把表格单元格的
  文字**逐字**重排为固定宽度表（不增删改词），并把被页面断行切开的长句接回。重排是确定性的，
  对照脚本会打印差异；这是**版式重建**而非文字改动，列为待复核的保真项。
- `BOOTSTRAP_USER_TURN` / `GET_MAGIC_CHUNK_TOOL`：论文只**描述**其语义（"A bootstrap user turn asks
  the assistant to retrieve the document one magic chunk at a time"、"declared as retrieving arbitrary
  units that do not align with the document's own sections or chapters"），附录 F 未给出逐字文本。
  本文件按描述构造，标记 `[项目决定]`；论文若发布原文，应替换而非近似保留。
- `SYSTEM_INSTRUCTION`、`MAGIC_CHUNK_HEADER`：附录 F/B 逐字（"You are a helpful assistant."、"Magic Chunk N"）。
"""

from __future__ import annotations

SYSTEM_INSTRUCTION = "You are a helpful assistant."

#: [项目决定] 依附录 F 对 bootstrap turn 的描述构造
BOOTSTRAP_USER_TURN = (
    "Retrieve the document one magic chunk at a time using the get_magic_chunk tool."
)

#: [项目决定] 依附录 F 对 tool 声明的描述构造；用模型原生 tool 声明格式（apply_chat_template(tools=...)）渲染
GET_MAGIC_CHUNK_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_magic_chunk",
        "description": (
            "Retrieve one magic chunk of the document. Magic chunks are arbitrary retrieval "
            "units that do not align with the document's own sections or chapters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk": {
                    "type": "integer",
                    "description": "1-based magic chunk index.",
                }
            },
            "required": ["chunk"],
        },
    },
}

MAGIC_CHUNK_HEADER = "Magic Chunk {n}"

# --- 论文附录 F 逐字：DA Instruction Prompt ---------------------------------
# 页 37-39；页间断行已接回（"DA Instruction Prompt (continued)"）。

DA_INSTRUCTION_PROMPT = """Question
{question}

Instructions
Answer the question above using only the retrieved document. Each get_magic_chunk response above
is one magic chunk of the document, in order from magic chunk 1. Magic chunks are arbitrary retrieval
splits and do not align with the document\u2019s own sections or chapters. The document is now fully
retrieved. Do not call get_magic_chunk again.

Reason through the magic chunks using three modes. Use the <answer> tag to output your final answer,
e.g., <answer>...</answer>.


  Mode                           What you can see              Use it when
  <global> (default)             all magic chunks              . . . you\u2019re identifying which magic chunk
                                                               to focus on next.
  <focus magic_chunks="K">       only magic chunk K (plus      . . . you\u2019re pulling a verbatim value out of
                                 values you\u2019ve already       magic chunk K.
                                 extracted)
  <local>                        only values you\u2019ve already  . . . you\u2019re planning over the question, or
                                 extracted (no magic chunks)   synthesizing values you\u2019ve already extracted.


Global mode (default)
Use global mode to identify the next magic chunk to examine. Briefly explain why the magic chunk is
relevant. Do not reason about the answer itself in global mode.

Focus mode
Switch to focus mode to examine a specific magic chunk and extract facts.

Syntax: <focus magic_chunks="K">VALUE</focus> (or <focus magic_chunks="K,M">, <focus
magic_chunks="K,M,N"> for multiple magic chunks).

VALUE is the answer the magic chunk provides: a name, number, date, or short noun phrase (typically
1 to 12 words).

When the document was delivered without any magic chunk retrievals (short document, inline only),
this tag is unavailable.

Local mode
In local mode you cannot see the document magic chunks, and you cannot re-read them. Local
mode is for reasoning that builds on facts you have already extracted via earlier <focus> blocks, the
question, and your prior reasoning.

Syntax: <local>your reasoning here</local>


Refer to verbatim values you extracted in earlier focus blocks by name. You do not need to re-state
them. If you find you need a value you have not yet extracted, close the <local> block, open <global>,
and identify the magic chunk to focus on next.

Strategy
Use <global>, <focus ...>, and <local> in any order. Repeat any of them as needed, then write
<answer>.

Three soft requirements:
1. Use at least one <focus> block. The point of focus mode is to pull the exact value(s) into your
   attention before answering. Skipping focus and guessing from memory is the most common failure
   mode.
2. End with a <local> block that names the single value, phrase, or summary you will put inside
   <answer>. This is the commitment step. Without it, the model often retrieves multiple values and
   forgets which one the question actually asks for.
3. Don\u2019t try to recall magic chunk content in <local>. Once you open <local>, the magic chunks
   are no longer visible to you. Anything you \u201cremember\u201d about a magic chunk you didn\u2019t focus on is a
   guess. If you need a value you haven\u2019t extracted, close </local> and go back to <global> to pick
   the magic chunk to focus on next.

Examples of valid orders:
\u2022 <local> \u2192 <global> \u2192 <focus> \u2192 <local> \u2192 <answer>. Typical shape for most questions: plan,
  locate, extract, conclude.
\u2022 <global> \u2192 <focus> \u2192 <local> \u2192 <answer>. Short factual / needle-in-haystack: the target magic
  chunk is obvious from the question.
\u2022 <global> \u2192 <focus> \u2192 <global> \u2192 <focus> \u2192 <local> \u2192 <answer>. Multi-fact, separate magic
  chunks.
\u2022 <global> \u2192 <focus> \u2192 <local> \u2192 <global> \u2192 <focus> \u2192 <local> \u2192 <answer>. Extract, sanity-
  check, fetch what\u2019s still missing, commit.

Answer
End your response with the final answer wrapped in <answer>...</answer>. The content depends on the
question type:


  Question type      Answer format        Example
  Multiple choice    the letter only      <answer>D</answer>
  Cloze (<mask-N>)   the missing word(s)  <answer>Britney</answer>
  Short factual      the value            <answer>March 14, 2024</answer>
  Summary            2 to 3 sentences     <answer>Congress passed the BUILD Act in 2018, creating
                                          the IDFC.</answer>


After </answer>, the response is complete."""

# --- 论文附录 F 逐字：Vanilla Instruction Prompt -----------------------------

VANILLA_INSTRUCTION_PROMPT = """Context
{context}

Question
{question}

INSTRUCTIONS:
Find the answer to the Question based solely on information in the Context above. Reason through the
context step by step.

Answer
End your response with the final answer wrapped in <answer>...</answer>. The content depends on the
question type:


  Question type      Answer format        Example
  Multiple choice    the letter only      <answer>D</answer>
  Cloze (<mask-N>)   the missing word(s)  <answer>Britney</answer>
  Short factual      the value            <answer>March 14, 2024</answer>
  Summary            2\u20133 sentences       <answer>Congress passed the BUILD Act in 2018, creating
                                          the IDFC.</answer>


After </answer>, the response is complete."""
