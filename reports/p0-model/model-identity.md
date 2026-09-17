# 模型 identity（阶段 02 交付）

仓库：`Qwen/Qwen3.8-27B`　revision：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

- 文件数：32（Hub 元数据）／本地 32，missing=0，extra=[]
- 总字节：55586114863（Hub）／55586114863（本地，逐文件求和）——一致
- 大小核对：32/32 通过；LFS sha256 核对：19 通过 / 0 失败
- license：apache-2.0；gated=False；private=False
- Hub API endpoint：`https://hf-mirror.com`；huggingface_hub 1.32.0
- 非 LFS 文件另有 git blob id 校验（13/13 通过，见 evidence-index）

## 文件清单（大小 + sha256）

| 文件 | 字节 | sha256（本地实算，LFS 文件与 Hub 元数据一致） |
| --- | ---: | --- |
| `.gitattributes` | 1,570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` ✓ |
| `LICENSE` | 11,544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` ✓ |
| `README.md` | 65,012 | `57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641` ✓ |
| `chat_template.jinja` | 8,952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` ✓ |
| `config.json` | 4,312 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` ✓ |
| `crc32.txt` | 238 | `b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83` ✓ |
| `generation_config.json` | 202 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` ✓ |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` ✓ |
| `model-00001-of-00018.safetensors` | 3,966,730,552 | `ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c` ✓ |
| `model-00002-of-00018.safetensors` | 3,043,080,328 | `06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb` ✓ |
| `model-00003-of-00018.safetensors` | 2,542,796,952 | `2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968` ✓ |
| `model-00004-of-00018.safetensors` | 3,988,973,152 | `511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a` ✓ |
| `model-00005-of-00018.safetensors` | 2,099,339,864 | `635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701` ✓ |
| `model-00006-of-00018.safetensors` | 3,979,553,696 | `0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0` ✓ |
| `model-00007-of-00018.safetensors` | 2,108,759,344 | `80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892` ✓ |
| `model-00008-of-00018.safetensors` | 3,979,553,696 | `7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79` ✓ |
| `model-00009-of-00018.safetensors` | 2,108,759,344 | `af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8` ✓ |
| `model-00010-of-00018.safetensors` | 3,979,553,696 | `163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9` ✓ |
| `model-00011-of-00018.safetensors` | 2,108,759,344 | `5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8` ✓ |
| `model-00012-of-00018.safetensors` | 3,979,553,696 | `a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55` ✓ |
| `model-00013-of-00018.safetensors` | 2,108,759,344 | `06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a` ✓ |
| `model-00014-of-00018.safetensors` | 3,979,553,696 | `4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b` ✓ |
| `model-00015-of-00018.safetensors` | 2,108,759,344 | `69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22` ✓ |
| `model-00016-of-00018.safetensors` | 3,979,564,040 | `73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe` ✓ |
| `model-00017-of-00018.safetensors` | 2,108,759,344 | `beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8` ✓ |
| `model-00018-of-00018.safetensors` | 3,392,197,344 | `1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06` ✓ |
| `model.safetensors.index.json` | 112,216 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` ✓ |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` ✓ |
| `tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` ✓ |
| `tokenizer_config.json` | 17,928 | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` ✓ |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` ✓ |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` ✓ |

## tokenizer 与模板

- `tokenizer.json` sha256：`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- `chat_template.jinja` sha256：`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`（与 `tokenizer_config.json` 内嵌副本逐字节相同：True）
- `tokenizer_config.json` sha256：`b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`
- `config.json` sha256：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
- tokenizer_class：`Qwen2Tokenizer`；model_max_length=262144
- eos_token=`<|im_end|>`；bos_token=None；pad_token=`<|endoftext|>`；unk_token=None
- 模板开关：enable_thinking ✓、preserve_thinking ✓、reasoning_effort ✓（默认 `xhigh`）

### 特殊 token 表

| id | 内容 | special |
| ---: | --- | --- |
| 248044 | `<|endoftext|>` | True |
| 248045 | `<|im_start|>` | True |
| 248046 | `<|im_end|>` | True |
| 248047 | `<|object_ref_start|>` | True |
| 248048 | `<|object_ref_end|>` | True |
| 248049 | `<|box_start|>` | True |
| 248050 | `<|box_end|>` | True |
| 248051 | `<|quad_start|>` | True |
| 248052 | `<|quad_end|>` | True |
| 248053 | `<|vision_start|>` | True |
| 248054 | `<|vision_end|>` | True |
| 248055 | `<|vision_pad|>` | True |
| 248056 | `<|image_pad|>` | True |
| 248057 | `<|video_pad|>` | True |
| 248058 | `<tool_call>` | False |
| 248059 | `</tool_call>` | False |
| 248060 | `<|fim_prefix|>` | False |
| 248061 | `<|fim_middle|>` | False |
| 248062 | `<|fim_suffix|>` | False |
| 248063 | `<|fim_pad|>` | False |
| 248064 | `<|repo_name|>` | False |
| 248065 | `<|file_sep|>` | False |
| 248066 | `<tool_response>` | False |
| 248067 | `</tool_response>` | False |
| 248068 | `<think>` | False |
| 248069 | `</think>` | False |
| 248070 | `<|audio_start|>` | True |
| 248071 | `<|audio_end|>` | True |
| 248072 | `<tts_pad>` | True |
| 248073 | `<tts_text_bos>` | True |
| 248074 | `<tts_text_eod>` | True |
| 248075 | `<tts_text_bos_single>` | True |
| 248076 | `<|audio_pad|>` | True |

## 渲染后的完整 prompt 样例

样例消息：system="You are a precise assistant. Answer with a single short sentence."，user="Name the capital of France."

### thinking_disabled（kwargs={'enable_thinking': False}，36 token）

```text
<|im_start|>system
You are a precise assistant. Answer with a single short sentence.<|im_end|>
<|im_start|>user
Name the capital of France.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

### thinking_default（kwargs={}，72 token）

```text
<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.

You are a precise assistant. Answer with a single short sentence.<|im_end|>
<|im_start|>user
Name the capital of France.<|im_end|>
<|im_start|>assistant
<think>

```

服务端渲染交叉验证（`/tokenize` + `/detokenize`，`tools/e5-prompt-check.py`）：

- `enable_thinking=false`：36 token，token id 序列与本地渲染**完全一致**（`server_matches_local_off_ids=true`）
- `enable_thinking=true`：72 token（注入 reasoning 指令 + 未闭合 `<think>`）

## 协议标签切分（下一阶段 token-span 映射的输入）

8 个控制标签是否在词表中：{"<global>": false, "</global>": false, "<focus>": false, "</focus>": false, "<local>": false, "</local>": false, "<answer>": false, "</answer>": false}

| 探测串 | token 数 | 切分 |
| --- | ---: | --- |
| `<global>` | 3 | ['<', 'global', '>'] |
| `</global>` | 3 | ['</', 'global', '>'] |
| `<focus>` | 3 | ['<', 'focus', '>'] |
| `<focus magic_chunks="8">` | 7 | ['<', 'focus', 'Ġmagic', '_chunks', '="', '8', '">'] |
| `</focus>` | 3 | ['</', 'focus', '>'] |
| `<local>` | 3 | ['<', 'local', '>'] |
| `</local>` | 3 | ['</', 'local', '>'] |
| `<answer>` | 3 | ['<', 'answer', '>'] |
| `</answer>` | 3 | ['</', 'answer', '>'] |
| `<answer>42</answer>` | 8 | ['<', 'answer', '>', '4', '2', '</', 'answer', '>'] |

