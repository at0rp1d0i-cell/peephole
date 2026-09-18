# 阶段 04 返工证据索引（v3，SUP-003-R2 收尾）

命令：`source env.sh && python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check-v3.json --evidence evidence/p1-gpu-v3`（exit 0，972 条判据 / 0 失败）；
独立负对照：`python3 tools/p1gpu-overread-control.py --config configs/p1-gpu/read-view-check-v3.json --evidence evidence/p1-gpu-v3`（exit 0）。

`summary.json` 含主记录与 `control_records`（越读负对照 + 边界拒绝检查）。
历史证据保留：`evidence/p1-gpu/`（v1，结论已撤回）、`evidence/p1-gpu-v2/`（首次正确夹具，expect_blocks 数组有误）。

| 文件 | 大小 (B) | sha256 |
| --- | ---: | --- |
| `cases-seed0.json` | 69001 | `12e20a676c1a067d299c50d9249d4d0ce3791bb05b90a9359918084a5939ebe0` |
| `cases-seed1.json` | 69007 | `a585b22b8dd3dec7dd72c07c5d83c15f9d1d3563ed67f48e5d0ef49b14c553ef` |
| `cases-seed2.json` | 68991 | `266d3c4f5dc8ec7229a187da9d11f74f4ca73cf23b74afd9fadedc4eb428ce42` |
| `negative-control.json` | 3438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` |
| `negative-control.log` | 405 | `1139e95ee8343872357489b609e83516fef69ce7b3b631b24992c228d73fe247` |
| `summary.json` | 238574 | `aa01627e091b42eb7e6eaf95e9e6d918a2e34c2deae4a393131915befaa554fe` |
| `v3-run.log` | 8085 | `659869990e3dec3ba83d78b7820f1108a335371a8ff8e4eea180949540734a4b` |
