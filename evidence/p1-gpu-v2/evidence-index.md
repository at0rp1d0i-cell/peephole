# 阶段 04 返工证据索引（v2）

命令：`source env.sh && python3 tools/p1gpu-read-view-check.py --config configs/p1-gpu/read-view-check-v2.json`（exit 0，738 条判据 / 0 失败）；
负对照：`python3 tools/p1gpu-overread-control.py`（exit 0）。

旧夹具证据保留在 `evidence/p1-gpu/`（结论已撤回，仅供追溯）。

| 文件 | 大小 (B) | sha256 |
| --- | ---: | --- |
| `negative-control.json` | 3438 | `3f2261c6fca5731bff8bd6e3a399805b2f529ed27ffd91392879b6995306c1b3` |
| `v2-cases-seed0.json` | 53401 | `05f7ac725b22dc7ba87113d4799591ad1740f7375b08f53f46b795f99db4e22b` |
| `v2-cases-seed1.json` | 53410 | `2ea9f33751af177bb77d9b75f3bfeb07d074005847b3365a4cca0eebac48c523` |
| `v2-cases-seed2.json` | 53393 | `371f21ea17ed011a732ccac07388f477e969f10f6451332912144010357b024b` |
| `v2-run.log` | 7539 | `02315c4a83ae59570c878cc378fe6113ad8be0f16b44c3a3798451520eccbad1` |
| `v2-summary.json` | 174360 | `4beae951fbc27d9d173102b7f6143c09211d75d287940a123621dd4ac6d9f577` |
