# 阶段 04 证据索引（GPU 读取视图数值验证）

生成时间：2026-09-18（远端 GPU 会话）。命令：`source env.sh && python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check.json`（exit 0）；
负对照：`python3 tools/p1gpu-overread-control.py`（exit 0）。

| 文件 | 大小 (B) | sha256 |
| --- | ---: | --- |
| `cases-seed0.json` | 16773 | `d825996514160ad9561351ce3df059db44532e1f79f15902592c65a077327a49` |
| `cases-seed1.json` | 16781 | `379916273f30b7815e5fc47365aeab814b6385c2347d8c9324cb58aada699860` |
| `cases-seed2.json` | 16784 | `d9b721a1a216718530b6eeb68ebadc763fa03fbd5372b507874ce620ab6dad78` |
| `negative-control.json` | 2056 | `d2ff889980024218fc862a8f10309554348ca78d48441347b6fe8834bd94d471` |
| `negative-control.log` | 402 | `865da10c9a4003a66316a8c444ad6d530c6cff1c0506c6adb5bc61548d2b1566` |
| `run.log` | 3260 | `9dd7c7482f1b573d024fdb177899c7935f7539478ee9c75b03ed4c01c88d70ba` |
| `summary.json` | 61124 | `fad0472659e155fb34e5bf7745299f07aeb0b937f515adc6430a70ed6e513191` |
