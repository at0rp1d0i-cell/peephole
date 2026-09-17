# 阶段 03 证据索引

生成时间：2026-09-18（远端 CPU 会话，R1 返工 + advisor 两条意见后）

| 文件 | 大小 (B) | sha256（前 16） | 说明 |
| --- | ---: | --- | --- |
| `demo-fixtures.json` | 40435 | `97dd1a4834dac8c1` | 完整原文、各段文本与字符区间、生成脚本、tokenizer/模板哈希（E2 保留原文与无损分段） |
| `demo-input.json` | 8340 | `8f476f905efb7034` | `8f476f905efb7034` |
| `extraction.json` | 933 | `47d85d6f1d4a09e7` | `47d85d6f1d4a09e7` |
| `fixed-trace.json` | 225153 | `2aab9723de3dc030` | `2aab9723de3dc030` |
| `fixed-trace.md` | 18529 | `aa097476ff12b363` | `aa097476ff12b363` |
| `prompt-da.txt` | 25943 | `1f5e2361f32622b4` | `1f5e2361f32622b4` |
| `prompt-da_no_mask.txt` | 25943 | `1f5e2361f32622b4` | `1f5e2361f32622b4` |
| `prompt-facts.json` | 6556 | `44af7fa453e51732` | `44af7fa453e51732` |
| `prompt-fidelity.txt` | 771 | `1654068d1f3c83dc` | `1654068d1f3c83dc` |
| `prompt-ids-offsets-da.json` | 148550 | `3dd4feecc9c9738c` | DA 臂全量 token_ids + offsets + segment/scaffold span |
| `prompt-ids-offsets-da_no_mask.json` | 148558 | `fe235d66882168e6` | DA-no-mask 臂全量 token_ids + offsets（应与 DA 相同） |
| `prompt-ids-offsets-vanilla.json` | 119347 | `5292ab4329448f4d` | Vanilla 臂全量 token_ids + offsets |
| `prompt-vanilla.txt` | 20157 | `34c33bf6c38ce8d0` | `34c33bf6c38ce8d0` |
| `run.log` | 1445 | `6bb079a06b710e42` | `f962dbcdef10885a` |
| `template-kwargs-check.txt` | 883 | `b610e633bec6564e` | `b610e633bec6564e` |
