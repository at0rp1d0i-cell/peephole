# 阶段 03 证据索引（E0–E4）

生成时间：2026-09-18（远端 CPU 会话）

| 文件 | 大小 (B) | sha256（前 16） | 说明 |
| --- | ---: | --- | --- |
| `demo-input.json` | 5608 | `fdfc09f3e797102d` | 合成资料、真实 tokenizer 分段事实（3 段、无损拼接、token 数） |
| `extraction.json` | 899 | `204f6094590aae48` | 公开提取结果与内部/公开记账 |
| `fixed-trace.json` | 203345 | `78f1563326861a27` | 115 个生成步 + prefill 的完整轨迹（机器可读，含独立参考对照结果） |
| `fixed-trace.md` | 19033 | `f7ae46dbde0d1c01` | 同一轨迹的可读表（step/输入 token/事件→effect_step/模式/写位置/kv_len/区间/块/物理块/尾块） |
| `prompt-da.txt` | 25943 | `1f5e2361f32622b4` | DA 臂实际渲染文本（含 tool 声明与 DA Instruction Prompt 全文） |
| `prompt-da_no_mask.txt` | 25943 | `1f5e2361f32622b4` | DA-no-mask 臂（应与 da 逐字相同） |
| `prompt-facts.json` | 3470 | `b00bf8511d99e8be` | 三臂渲染后的 prompt 长度、segment/scaffold token 区间、sink 前 16 token、模板/分词器哈希 |
| `prompt-fidelity.txt` | 736 | `47e80aea8c60ae8a` | prompt 文本与论文摘录件的机械对照结果 |
| `prompt-vanilla.txt` | 20157 | `34c33bf6c38ce8d0` | Vanilla 臂实际渲染文本（内联 context） |
| `run.log` | 922 | `6ba1e683869a657f` | 本阶段测试与演示的原始输出（含 61 项单测与演示统计） |
| `template-kwargs-check.txt` | 883 | `b610e633bec6564e` | transformers 5.17 模板传参方式实测（chat_template_kwargs 被静默忽略） |
