# 阶段 03 证据索引

生成时间：2026-09-18（远端 CPU 会话，R1 返工后）

| 文件 | 大小 (B) | sha256（前 16） | 说明 |
| --- | ---: | --- | --- |
| `demo-input.json` | 8340 | `8f476f905efb7034` | 合成资料与真实 tokenizer 分段事实（3 段、无损拼接、token 数、哈希） |
| `extraction.json` | 933 | `47d85d6f1d4a09e7` | 公开提取结果与内部/公开记账 |
| `fixed-trace.json` | 225153 | `2aab9723de3dc030` | 114 次 forward + prefill 的完整轨迹（含 focus 统计、stop token、独立参考对照） |
| `fixed-trace.md` | 18529 | `aa097476ff12b363` | 同一轨迹的可读表（四列时序字段 + 区间/块/物理块/尾块） |
| `prompt-da.txt` | 25943 | `1f5e2361f32622b4` | DA 臂实际渲染文本 |
| `prompt-da_no_mask.txt` | 25943 | `1f5e2361f32622b4` | DA-no-mask 臂（应与 da 逐字相同） |
| `prompt-facts.json` | 6556 | `44af7fa453e51732` | 三臂渲染事实 + sink 完整区间/多输入固定前缀 + 中英混排逐 token 解码与词表扫描（953/248077） |
| `prompt-fidelity.txt` | 771 | `1654068d1f3c83dc` | prompt 与论文摘录件的①有序正文②逐格单元格③词多重集合核对结果 |
| `prompt-vanilla.txt` | 20157 | `34c33bf6c38ce8d0` | Vanilla 臂实际渲染文本 |
| `run.log` | 1943 | `76c70479a26b42f0` | 测试（79 项）+ 保真核对 + 演示的原始输出 |
| `template-kwargs-check.txt` | 883 | `b610e633bec6564e` | transformers 5.17 模板传参实测（chat_template_kwargs 被静默忽略） |
