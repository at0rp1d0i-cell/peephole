# Qwen3.8-27B 配置预检（阶段 02 E3 的只读前置）

日期：2026-09-17。执行者：主代理（远端 `lab` 只读核对，**未启动任何服务、未渲染 prompt、未占用 GPU**）。
性质：权重下载进行中（约 28%）时的辅助工作产物。**它是预读，不替代工作单 E3/E4 的实测输出**；其中每条都标了来源与证据级别。

对象：`Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，快照位于 `$HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/<rev>/`。

## 1. 方案 §3.2 的数字首次在"下载到本机的文件"上核对通过

| 项 | 值 | 来源 |
| --- | ---: | --- |
| `model.safetensors.index.json` 的 `metadata.total_size` | **55,562,855,904 B** | 已落盘的 index 文件 |
| 方案 §3.2 引用值 | 55,562,855,904 B | 此前仅来自"读官方权重索引" |
| 仓库文件总字节 | 55,586,114,863 B（+23.26 MB） | Hub API |
| 分片数 / 张量数 | 18 / 1199 | index `weight_map` |

**结论：方案 §3.2 的 checkpoint 张量总量逐字节吻合**（此前只有资料推导的依据）。两数不可混用：`total_size` 是张量总量，仓库总字节还含 config/tokenizer/vocab/merges/README/`crc32.txt` 等。

## 2. 架构与 KV 维度（与方案 §3.1/§3.2 对照，全部一致）

| 项 | 实测（config.json） | 方案记载 |
| --- | --- | --- |
| `architectures` / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5`（文本子配置 `qwen3_5_text`） | 一致 |
| 层结构 `layer_types` | **48 × `linear_attention` + 16 × `full_attention`**（`full_attention_interval: 4`） | "64 层，16 全注意力 + 48 GDN"，一致 |
| 全注意力 KV | `num_key_value_heads: 4`、`head_dim: 256` ⇒ **2×16×4×256×2 = 65,536 B/token = 64.0 KiB/token** | 一致（逐位吻合） |
| `num_attention_heads` | 24 | — |
| 上下文 | `max_position_embeddings: 262,144` | 一致 |
| dtype / 量化 | `bfloat16` / `quantization_config: null` | 一致（BF16、无量化） |
| 词表 | `vocab_size: 248,320`、`tie_word_embeddings: false` | 新增事实 |

**新增事实（方案未记录，影响 R03 的位置语义）**：

- `partial_rotary_factor: 0.25` —— **RoPE 只作用在 `head_dim` 的 1/4（64 维），不是全部 256 维**；
- `rope_parameters`: `mrope_interleaved: true`、`mrope_section [11,11,10]`、`rope_theta 1e7`、`rope_type default` —— 三段 mrope（多模态位置编码）；
- `attn_output_gate: true`、`output_gate_type: swish` —— 注意力输出带门控；
- GDN 侧：`linear_num_key_heads 16`、`linear_key_head_dim 128`、`linear_num_value_heads 48`、`linear_value_head_dim 128`、`linear_conv_kernel_dim 4`、`mamba_ssm_dtype float32`；
- **`mtp_num_hidden_layers: 1`** —— 存在一层 MTP；方案 §3.3 要求关闭 MTP/投机解码。

来源：本地快照的 `config.json`（读值，未在 GPU 上验证生效方式）。

## 3. 这是多模态 checkpoint，但视觉塔很小

config 顶层含 `vision_config`（depth 27、hidden 1152）、`image_token_id 248056`、`video_token_id 248057`，仓库含 `preprocessor_config.json` 与 `video_preprocessor_config.json`，且 `language_model_only: false`。

张量统计：

| 前缀 | 张量数（index） | 已读分片实测体积 |
| --- | ---: | ---: |
| `model.language_model.*` | 850 | 23.08 GiB（96.4%） |
| `model.visual.*` | 333 | **0.86 GiB（3.6%）** |
| `mtp.*` | 15 | 未落在已读的 8 个分片中 |
| `lm_head` | 1 | — |

实测方法：对已落盘的分片读 safetensors header（前 8 字节 → header 长度 → JSON），用 index 的 `weight_map` 交叉确认分片身份，按 `shape × dtype` 求和；覆盖 8/18 个分片（共 23.94 GiB），其余分片尚未下载到可读状态。

**结论**：视觉塔**只有约 0.86 GiB**（不是数 GiB 量级）。因此 `--language-model-only` 的价值主要是"不走多模态路径"，而不是省大量显存。`[INFERENCE]` 15 个 MTP 张量约相当于 1 层主体，量级应在 0.4–0.5 GiB。

## 4. vLLM 0.29.0 侧预检

| 检查 | 结果 | 出处 |
| --- | --- | --- |
| 架构是否被 pin 支持 | **支持**：`registry.py:575` 注册 `Qwen3_5ForConditionalGeneration ("qwen3_5")`；`models/qwen3_5.py`、`qwen3_5_mtp.py` 存在 | 安装轮子 + 源码 checkout |
| 跳过视觉塔的开关 | **存在**：`--language-model-only`（`arg_utils.py:1368`）；唯一已知冲突是 `cudagraph_mm_encoder=True`（本项目不使用） | 源码 |
| MTP 是否会被加载 | **不会**：`qwen3_5.py` 的权重映射 `orig_to_new_prefix={"mtp.": None}`（"mtp." 映射为 None = 丢弃） | 源码 |
| 是否会自动启用投机解码 | **未确认**，需在 E4 启动日志里核对（grep `speculative` / `mtp`） | 待实测 |

## 5. tokenizer 与 chat template

- `tokenizer_class: Qwen2Tokenizer`，`model_max_length: 262144`，`eos_token: <|im_end|>`，`bos_token: null`，`added_tokens_decoder` **33 条**（后续 token-span 映射要用到）；
- `chat_template.jinja` = 8,952 字符，**含 `enable_thinking`** → 方案 §3.3 要求的"关闭 thinking"在模板层面有开关可用；
- `generation_config.json` 仅 202 字节（`do_sample: true`、`temperature`、`eos/pad` 等）。

**未做**：渲染后的 prompt 全文与 token 数（E3 的实测项）；模板实际分支选择。

## 6. 对 E4 的具体建议（据此微调工作单执行）

1. 启动加 `--language-model-only`，并在日志中确认视觉塔未加载；
2. 显式确认没有自动启用 MTP / 投机解码（日志 grep）；
3. 首启用小 `--max-model-len`（如 8192），把"能否加载 + SM120 后端是否接受"与"容量探索"分开；
4. **记录加载后实际显存**，与 §3.2 的 51.75 GiB（含视觉 0.86 GiB、MTP）对照——这将是"文本路径实际权重占用"的第一次实测，直接喂 R04 的容量核算；
5. RoPE 维数只有 64（不是 256）这一事实，写进 R03 的对照实验设计输入。

## 7. 未验证项（仍在 E3/E4 范围内）

渲染后 prompt 与 token 数；实际 attention backend / `block_size` / `kernel_block_size` / 图模式 / 生效 dtype；加载后实际显存；MTP 是否被自动启用；KV cache 预算与可容纳 token 数；任何长上下文、性能与质量结论。本文件不据此得出这些结论。
