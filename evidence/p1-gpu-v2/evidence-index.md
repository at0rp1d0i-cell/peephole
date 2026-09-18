# 阶段 04 返工证据索引（v2）

命令：`source env.sh && python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check-v2.json`（exit 0，852 条判据 / 0 失败）；
负对照：`python3 tools/p1gpu-overread-control.py`（exit 0）。

旧夹具证据保留在 `evidence/p1-gpu/`（结论已撤回，仅供追溯）。

| 文件 | 大小 (B) | sha256 |
| --- | ---: | --- |
| `negative-control.json` | 3438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` |
| `negative-control.log` | 405 | `4eb57fa301a36dd44ec261d1727ac665898b9fec1db2474ebefb8f4a8edd72a0` |
| `v2-cases-seed0.json` | 53906 | `0820dbf7a1340134241eee033f2fe69b94c1382728b9510802c0ca43a20212e3` |
| `v2-cases-seed1.json` | 53915 | `36701d2a3b0a7a18f42c7dad1d9f67d575d6604be9ac3c287b9d75acc2c3ea51` |
| `v2-cases-seed2.json` | 53898 | `0f789e267dda7e15bf6f9c3987614acf2683471a640cc33decebf2c5100b7a9d` |
| `v2-run.log` | 7539 | `71f548748d665ccb0c10ea58ac9060574ba57dcb0d1bf32b0412923300d2e3f6` |
| `v2-summary.json` | 175989 | `596d3abf98378d8e70125ddcdb3861af3e7ed8a15b786368dc6edd583f269f7f` |
