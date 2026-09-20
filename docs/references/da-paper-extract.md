# 论文原文摘录：附录 B（vLLM 集成）与附录 F（prompt 构造、评测 prompt）

来源：*Language Models Can Control Their Own Attention*（Declarative Attention），arXiv:2609.02737v1，
Namgyu Ho, Huzama Ahmad, Woosung Koh, Se-Young Yun, Tal Schuster, Cicero Nogueira dos Santos（KAIST AI / Google DeepMind）。
论文许可：CC BY 4.0（https://creativecommons.org/licenses/by/4.0/，2026-09-17 读取 arXiv abs 页确认）——本文件按署名要求标注来源后引用。

提取信息：2026-09-17，`curl -sSL -o da-paper.pdf https://arxiv.org/pdf/2609.02737v1` 后
`pdftotext -layout da-paper.pdf da-paper.txt`，本文件为下列行段的逐字复制（去掉分页符，保留页眉与页码）：

- 附录 B（DA vLLM integration）：`da-paper.txt` 第 1514–1555 行
- 附录 F（Prompt construction，含 DA / Vanilla Instruction Prompt、rubric 生成 prompt、judge prompt）：第 2257–2534 行

说明与限制：

- arXiv 的 HTML 版本（`https://arxiv.org/html/2609.02737v1`，同日读取）在附录 F 处渲染失败：正文止于
  "The final user turn is the DA Instruction Prompt below."，随后是 LaTeXML 的报错提示，**实际 prompt 文本缺失**。
  因此以 PDF 提取为准；本文件是项目实现协议 prompt 的唯一文本依据。
- `-layout` 模式保留双栏页面布局：表格在纯文本中按列展开，页眉（"Language Models Can Control Their Own Attention"）与
  页码（37–42）夹在正文之间；引用具体条目时以小节标题定位，不要以行号引用。
- 本文件是**输入资料**，不是项目自身的合同或结论；协议合同见 `docs/protocol-contract.md`。

---

## 附录 B · DA vLLM integration（原文）

```text
B. DA vLLM integration
DA’s state machine extends vLLM through hooks on the attention metadata builder, requiring no modifications
to vLLM’s kernels or scheduler. This section details the regions the mask keeps attended in every mode, the
per-kernel patches for different attention backends, the layers the mask applies to, its block alignment, and its
effect on decode latency.


Always-attended regions. Three regions stay attended in every mode, realizing the scaffold-and-response
guarantee of Section 2.3. The first is a fixed attention sink of the prompt’s first 16 tokens, in the style of
StreamingLLM (Xiao et al., 2024b). We place a short, fixed system instruction at the start of the prompt so these
16 tokens always land on content-free scaffolding, never on context. The second is a local window covering the
question and the DA instruction through the end of the prompt, so the model never loses sight of what it is
answering. The third is the entire response generated so far. <focus> keeps these three regions plus the named
segment spans, and <local> keeps only the three.


Per-kernel patches. vLLM routes different models to different attention kernels based on their architecture.
Qwen-3.6’s global attention layers route to FlashAttention (Dao et al., 2022), while Gemma-4’s hybrid attention
design routes to a Triton-based paged-attention kernel. For each kernel, the DA state machine writes the mask
into the request’s KV-cache block table at every decode step, so that only the kept blocks remain visible to the
kernel. The block-table modification is implemented as a hook on the attention metadata builder, analogous
across both backends and requiring no changes to the kernel internals.


Scope: global attention layers only. DA’s mask construction targets only the global attention layers of each
model. For Qwen-3.5/6, the remaining layers use Gated DeltaNet (GDN) (Yang et al., 2025b), a linear-attention
variant whose per-step state is bounded by a recurrent hidden representation rather than a context-length-
dependent KV cache, so per-token masking does not apply. For Gemma-4, the remaining layers use sliding
window attention (SWA) (Beltagy et al., 2020) with a 1,024-token window, which already bounds per-step KV
reads independently of context length. DA leaves both kinds of layers unmodified.


Block alignment. The per-token mask is rounded outward to vLLM’s KV-cache block boundaries. This adds
at most 𝑏 − 1 extra attended tokens at each edge of a kept span, where 𝑏 is the block size (typically 16 to 32
tokens), negligible against the 2048-token segments. Block alignment is what allows existing kernels to run
unchanged on the masked KV cache.


Effect on decode latency. By reducing the number of KV blocks loaded per step, the integration cuts the
per-step attention read to the kept blocks. The wall-clock effect is largest in the regime where attention
dominates total decode time, which we estimate with the roofline analysis of Section 5.4.

```

---

## 附录 F · Prompt construction（原文，含评测用 rubric / judge prompt）

```text
F. Prompt construction
Context segmentation. We split the context into segments with a tokenizer-aware semantic segmenter. The
segmenter targets a 2048-token segment size with a 2560-token hard cap, and descends a delimiter hierarchy
from coarse to fine: blank-line paragraph breaks, then single newlines, then sentence ends, then clause ends,
and finally whitespace between words. It splits only a unit that exceeds the cap, and cuts at the coarsest
boundary that unit contains, so a segment edge lands on the most semantically meaningful split the size budget
allows. A whitespace-free run, such as a long base64 blob, is atomic: it is never split, and instead becomes its
own over-cap segment rather than being cut mid-word.
   The segmenter works entirely in character-offset space. It tokenizes the context once with offset mapping,
measures candidate spans by token count through those offsets, and emits each segment as a real substring of
the context, never by decoding token-id slices. A segment therefore can never be corrupted by a multibyte
character split across a slice boundary, and the segments form an exact, lossless partition of the context:
concatenating them reproduces the input. A context that fits under the cap is a single segment, and an empty
or whitespace-only context is rendered as an <empty_context> placeholder so the protocol always has at least
one addressable segment. Segments are numbered 1 to 𝑁 .


Magic-chunk delivery. Each segment is delivered to the model as a magic chunk through a simulated tool-use
conversation. The prompt opens with a short fixed system instruction, “You are a helpful assistant.”, whose role
is to occupy the attention sink so context never enters it (Appendix B). A bootstrap user turn asks the assistant
to retrieve the document one magic chunk at a time, and for each segment the conversation carries an assistant
get_magic_chunk tool call followed by a tool response headed Magic Chunk N that contains the segment text.
The get_magic_chunk tool is declared as retrieving arbitrary units that do not align with the document’s own
sections or chapters, reinforcing the magic-chunk framing. The final user turn is the DA Instruction Prompt
below.




                                                                                                              37
                             Language Models Can Control Their Own Attention



DA Instruction Prompt

Question
{question}

Instructions
Answer the question above using only the retrieved document. Each get_magic_chunk response above
is one magic chunk of the document, in order from magic chunk 1. Magic chunks are arbitrary retrieval
splits and do not align with the document’s own sections or chapters. The document is now fully
retrieved. Do not call get_magic_chunk again.

Reason through the magic chunks using three modes. Use the <answer> tag to output your final answer,
e.g., <answer>...</answer>.


  Mode                         What you can see                       Use it when
  <global> (default)           all magic chunks                       . . . you’re identifying which magic
                                                                      chunk to focus on next.
  <focus                       only magic chunk K (plus values        . . . you’re pulling a verbatim value
  magic_chunks="K">            you’ve already extracted)              out of magic chunk K.
  <local>                      only values you’ve already ex-         . . . you’re planning over the
                               tracted (no magic chunks)              question, or synthesizing values
                                                                      you’ve already extracted.


Global mode (default)
Use global mode to identify the next magic chunk to examine. Briefly explain why the magic chunk is
relevant. Do not reason about the answer itself in global mode.

Focus mode
Switch to focus mode to examine a specific magic chunk and extract facts.

Syntax: <focus magic_chunks="K">VALUE</focus> (or <focus                  magic_chunks="K,M">, <focus
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




                                                                                                              38
                              Language Models Can Control Their Own Attention



DA Instruction Prompt (continued)

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
3. Don’t try to recall magic chunk content in <local>. Once you open <local>, the magic chunks
   are no longer visible to you. Anything you “remember” about a magic chunk you didn’t focus on is a
   guess. If you need a value you haven’t extracted, close </local> and go back to <global> to pick the
   magic chunk to focus on next.

Examples of valid orders:
• <local> → <global> → <focus> → <local> → <answer>. Typical shape for most questions: plan,
  locate, extract, conclude.
• <global> → <focus> → <local> → <answer>. Short factual / needle-in-haystack: the target magic
  chunk is obvious from the question.
• <global> → <focus> → <global> → <focus> → <local> → <answer>. Multi-fact, separate magic
  chunks.
• <global> → <focus> → <local> → <global> → <focus> → <local> → <answer>. Extract, sanity-
  check, fetch what’s still missing, commit.

Answer
End your response with the final answer wrapped in <answer>...</answer>. The content depends on
the question type:


  Question type          Answer format                      Example
  Multiple choice        the letter only                    <answer>D</answer>
  Cloze (<mask-N>)       the missing word(s)                <answer>Britney</answer>
  Short factual          the value                          <answer>March 14, 2024</answer>
  Summary                2 to 3 sentences                   <answer>Congress passed the BUILD Act
                                                            in 2018, creating the IDFC.</answer>


After </answer>, the response is complete.




                                                                                                          39
                             Language Models Can Control Their Own Attention



Vanilla Instruction Prompt

Context
{context}

Question
{question}

INSTRUCTIONS:
Find the answer to the Question based solely on information in the Context above. Reason through the
context step by step.

Answer
End your response with the final answer wrapped in <answer>...</answer>. The content depends on
the question type:


  Question type          Answer format                     Example
  Multiple choice        the letter only                   <answer>D</answer>
  Cloze (<mask-N>)       the missing word(s)               <answer>Britney</answer>
  Short factual          the value                         <answer>March 14, 2024</answer>
  Summary                2–3 sentences                     <answer>Congress passed the BUILD Act
                                                           in 2018, creating the IDFC.</answer>


After </answer>, the response is complete.




                                                                                                       40
                               Language Models Can Control Their Own Attention



Rubric Generation Prompt

You are an AI assistant designing rigorous rubrics for long-context language model evaluation.

INSTRUCTIONS:
You are given a long context, a question about that context, and the reference answer. Your task is
to design a rubric that lets an external evaluator judge whether a model’s response to the question is
correct, WITHOUT giving the evaluator access to the context.

Step 1. Design a rubric to evaluate whether a given response is correct.
1. Assume the evaluator does NOT have access to the context.
2. The first bullet of the rubric MUST state the exact correct answer (paraphrasing the reference answer
   is fine), so the evaluator can judge correctness without the context.
3. Allow paraphrases or alternate phrasings of the same specific fact (e.g. “Joe Biden” vs “President
   Biden”). Reject answers that state a different fact, an incomplete fact, or a related-but-wrong entity.
4. Identify any distractor information from the context that seems plausible but is NOT the correct
   answer.
5. The rubric only differentiates CORRECT from WRONG — no partial credit, no intermediate scores.

Notes on Rubric Formatting:
1. Refer to the given response as “the response” in the rubric.
2. The rubric is a bulleted list with no more than 5 items.
3. Use “CORRECT” and “WRONG” in capital letters.
4. Do not use the word “incorrect”.
5. Do not use double negatives.

Step 2. Format Output.
Format your final output strictly as a single valid JSON object. Do not include markdown code blocks,
backticks (“‘), “json” labels, or any preamble/postscript text. The output must be a raw string that can
be directly passed into json.loads() in Python. The object must strictly contain this key:
1. "rubric": str

User message (sent alongside the system prompt above):

Context
{context}

Question
{question}

Reference answer
{reference_answer}




                                                                                                             41
                               Language Models Can Control Their Own Attention



Judge Prompt: for LLM-as-a-judge evaluation

You are an AI assistant specializing in language model evaluation.
You are the judge model, and your goal is to evaluate the response of a target model under evaluation.

The target model is expected to answer the question based on some given context. You will be given a
question, the target model’s response, and a binary evaluation rubric. Your goal is to evaluate the target
model’s response based on the rubric.

Question
{question}

Response from target model
<model_response>
{response}
</model_response>

Evaluation rubric
{rubric}

Instructions
Evaluate whether the target model’s response enclosed in the <model_response> tags is correct, based
solely on the given evaluation rubric. Do not rely on your knowledge of the context or the question.

When applying the rubric, ignore the following surface-level differences (they do not by themselves
make a response wrong):
1. Case (“sigmoid” vs “Sigmoid”, “methicillin” vs “Methicillin”) — treat string matches as case-insensitive
   unless the rubric explicitly demands a specific case.
2. Minor morphological variation (singular vs plural, gerund vs noun, e.g. “sigmoid activation function”
   vs “sigmoid activation functions”) — accept unless the rubric calls out the form explicitly.
3. Numeric formatting equivalent values (“3” vs “Three”, “900,000” vs “nine hundred thousand”,
   “$17B” vs “$17 billion”) — accept as equivalent unless the rubric specifies a particular format.
4. Whitespace, punctuation, and trivial typography (LaTeX vs unicode, hyphens, surrounding
   quotes).
5. Wrapper text or paraphrasing that preserves the substantive content the rubric requires (e.g. “The
   answer is X.” or “The UI designer; locating the remote when it is lost.” both satisfy a rubric that
   accepts X / “The UI designer; finding the remote.”).

Do not ignore differences that the rubric explicitly flags as wrong, or that change the meaning of the
answer (e.g. swapping a different value, naming a different entity, hedging that contradicts the source
— “over X” when the rubric demands the value “stated as X” remains a judgment call: read the rubric
carefully).

Output your assessment in the following JSON format. Do not include any other text in your response.

{ "correct": boolean }

```
