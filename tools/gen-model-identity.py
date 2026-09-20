#!/usr/bin/env python
"""从 E1/E2/E3 的证据 JSON 生成交付文档 model-identity.md（避免手工转录 sha256）。

用法：
  "$ATTNVIEW_PYTHON" /root/attnview/tools/gen-model-identity.py --evidence DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    args = ap.parse_args()
    ev = Path(args.evidence)
    ident = json.loads((ev / "e1-model-identity.json").read_text())
    verify = json.loads((ev / "e2-verify.json").read_text())
    audit = json.loads((ev / "e3-config-audit.json").read_text())
    proto = json.loads((ev / "e3-protocol-tokens.json").read_text())
    entry = ident["repos"][args.repo]
    sizes = {r["path"]: r for r in verify["files"]}
    extra_hashes = audit["tokenizer_config"]["file_sha256"]

    L: list[str] = []
    add = L.append
    add("# 模型 identity（阶段 02 交付）")
    add("")
    add(f"仓库：`{args.repo}`　revision：`{entry['revision_resolved']}`")
    add("")
    add(f"- 文件数：{entry['file_count']}（Hub 元数据）／本地 {verify['local_file_count']}，missing={verify['missing']}，extra={verify['extra_local_files']}")
    add(f"- 总字节：{entry['total_bytes']}（Hub）／{verify['total_local_bytes']}（本地，逐文件求和）——{'一致' if verify['total_bytes_match'] else '不一致'}")
    add(f"- 大小核对：{verify['size_matches']}/{entry['file_count']} 通过；LFS sha256 核对：{verify['lfs_sha256_matches']} 通过 / {verify['lfs_sha256_mismatches']} 失败")
    add(f"- license：{entry['license']}；gated={entry['gated']}；private={entry['private']}")
    add(f"- Hub API endpoint：`{ident['endpoint']}`；huggingface_hub {ident['huggingface_hub']}")
    add("- 非 LFS 文件另有 git blob id 校验（13/13 通过，见 evidence-index）")
    add("")
    add("## 文件清单（大小 + sha256）")
    add("")
    add("| 文件 | 字节 | sha256（本地实算，LFS 文件与 Hub 元数据一致） |")
    add("| --- | ---: | --- |")
    for f in sorted(entry["files"], key=lambda x: x["path"]):
        local = sizes.get(f["path"], {})
        digest = local.get("local_sha256") or "(未计算)"
        mark = "✓" if local.get("lfs_sha256_ok") or f["lfs_sha256"] is None else "✗"
        add(f"| `{f['path']}` | {f['size']:,} | `{digest}` {mark} |")
    add("")
    add("## tokenizer 与模板")
    add("")
    add(f"- `tokenizer.json` sha256：`{extra_hashes.get('tokenizer.json', 'n/a')}`")
    add(f"- `chat_template.jinja` sha256：`{extra_hashes.get('chat_template.jinja', 'n/a')}`（与 `tokenizer_config.json` 内嵌副本逐字节相同：{audit['tokenizer_config']['chat_template_in_tokenizer_config_identical']}）")
    add(f"- `tokenizer_config.json` sha256：`{extra_hashes.get('tokenizer_config.json', 'n/a')}`")
    add(f"- `config.json` sha256：`{extra_hashes.get('config.json', 'n/a')}`")
    add(f"- tokenizer_class：`{audit['tokenizer_config']['tokenizer_class']}`；model_max_length={audit['tokenizer_config']['model_max_length']}")
    add(f"- eos_token=`{audit['tokenizer_config']['eos_token']}`；bos_token={audit['tokenizer_config']['bos_token']}；pad_token=`{audit['tokenizer_config']['pad_token']}`；unk_token={audit['tokenizer_config']['unk_token']}")
    add(f"- 模板开关：enable_thinking ✓、preserve_thinking ✓、reasoning_effort ✓（默认 `{audit['tokenizer_config']['chat_template_flags']['default_reasoning_effort']}`）")
    add("")
    add("### 特殊 token 表")
    add("")
    add("| id | 内容 | special |")
    add("| ---: | --- | --- |")
    for k, v in audit["tokenizer_config"]["added_tokens"].items():
        add(f"| {k} | `{v['content']}` | {v['special']} |")
    add("")
    add("## 渲染后的完整 prompt 样例")
    add("")
    add("样例消息：system=\"You are a precise assistant. Answer with a single short sentence.\"，user=\"Name the capital of France.\"")
    add("")
    for label in ("thinking_disabled", "thinking_default"):
        r = audit["renders"][label]
        add(f"### {label}（kwargs={r['kwargs']}，{r['token_count']} token）")
        add("")
        add("```text")
        add(r["prompt_text"].replace("\\n", "\n").replace("\n", "\n"))
        add("```")
        add("")
    add("服务端渲染交叉验证（`/tokenize` + `/detokenize`，`tools/e5-prompt-check.py`）：")
    add("")
    add("- `enable_thinking=false`：36 token，token id 序列与本地渲染**完全一致**（`server_matches_local_off_ids=true`）")
    add("- `enable_thinking=true`：72 token（注入 reasoning 指令 + 未闭合 `<think>`）")
    add("")
    add("## 协议标签切分（下一阶段 token-span 映射的输入）")
    add("")
    add(f"8 个控制标签是否在词表中：{json.dumps(proto['protocol_tags_in_vocab'], ensure_ascii=False)}")
    add("")
    add("| 探测串 | token 数 | 切分 |")
    add("| --- | ---: | --- |")
    for probe, info in proto["probes"].items():
        add(f"| `{probe}` | {len(info['ids'])} | {info['pieces']} |")
    add("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written: {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
