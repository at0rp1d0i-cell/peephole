# 阶段 04 交付运行证据索引（v6，R2 提交时序合规）

**本目录为交付依据**：运行清单由实验自身在起止时刻写入。

- HEAD：`38982cf773790ab936db120e0dd0a0c52990b870`（运行开始时 `git rev-parse HEAD`），`code_clean = True`（排除运行自身产物后的工作区状态）
- 配置：`/root/autodl-tmp/attnview/configs/p1-gpu/read-view-check-v3.json`，sha256 `3797326784d336030c8c5c59817b20d0010618a2904407668045dd36adbf96d1`
- 起止：`2026-09-18 10:09:22 +0800` → `2026-09-18 10:09:34 +0800`，exit `0`，判据 972 / 失败 0
- 负对照：`control-manifest.json`（同 HEAD、同配置、exit 0）

历史（不作为交付依据）：`evidence/p1-gpu/`（v1，夹具错位，已撤回）、`evidence/p1-gpu-v2/`（预期数组有误）、`evidence/p1-gpu-v3/`（配置提交晚于运行，时序不合规）、`evidence/p1-gpu-v4/`、`evidence/p1-gpu-v5/`（清单口径/命名未定稿）。

| 文件 | 大小 (B) | sha256 |
| --- | ---: | --- |
| `cases-seed0.json` | 69001 | `aa7520c26456e741f867191dc57b128956f3f2c7b201045e9f98b7af2211962a` |
| `cases-seed1.json` | 69007 | `9209cb789526e1f424874d8b2f4d2776cff2034c6b3059f8753f21567c731bbc` |
| `cases-seed2.json` | 68991 | `4d94754caea9226f10cd06abb2691b2f89ae75349fd4ffa14c317aaaa76648e8` |
| `control-manifest.json` | 1028 | `86a4369820b4e2fbfb9911fbf34477097d90826d3901b47d41b3ae0978db2bbc` |
| `negative-control.json` | 3438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` |
| `negative-control.log` | 551 | `147d7d9dcd0dba62ca6e7988f80176d1470b06cff3efc2d397c459640bdb2d10` |
| `run-manifest.json` | 1030 | `422d5e673d737bacc9604bc5b8389647eeadb20e47a36d94350375e2d8acdc83` |
| `run.log` | 8300 | `5a1146ddebca69ca99da0759722e26b7de4b17d9ff128b82467ccda491344bfb` |
| `summary.json` | 238574 | `7953a1eb80474fd0ebcdca3fbc7619f834824c5b8a7486fd9d2cdbc7df614bbb` |
