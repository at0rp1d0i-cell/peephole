# 阶段 05 证据索引（`evidence/p3-calib/` 及同批复核目录）

范围：`evidence/p3-calib/` 下的 11 个 run 目录、`masked-prep/`、metadata 探针族、`gate-record.json`、`oracle-original-3a.json`（聚合段入库，全量件树外登记）、`traj*.json`，以及 8 个同批独立证据目录（含 7 个 `*local-review*`）。原始大件（张量 dump）留在远端数据盘；**字节不随仓发行**。本轮另将 7 个 `*local-review*` 目录（`independent-review`）与 `oracle-original-3a.json` 全量件（`detail-only`）移出仓内发行：原始字节归档在数据盘，path + size + sha256 见 `reports/evidence-index.md` 的树外登记表。

## 0. 边界声明（先读）

- 本索引**只登记可机械核对的事实**：文件存在性、字节数、sha256、manifest 自述字段。**不做任何结论标签**（不写「通过」「完成」「达标」「已验收」）。发现不一致只报告，不改数据。
- **stage-05 尚未完成验收**；`smoke_structure`（7 项结构用例）是**草稿而非门禁**，其失败证据按原样保留、含义不因本索引而改变。本索引不得被引用为 `smoke_structure` 已通过的依据。
- **历史已重写**（2026-09-21 两次 `filter-branch`：AGENTS.md 全历史移除；树内容脱敏）。本索引抄录的 `head` / `pin_commit` 字段是**重写前**的 commit SHA，已被重写作废；old→new 对照见 `reports/git-history-map-20260921.txt`（142 行）。本索引出现（含 §3.4 补记的 2 个复核产物）的**全部 9 个不同 `head` 值均可在该表 old 侧命中**（§3.4）。
- 大件登记口径：`capture/*.npz` 与 `logits.pt` 均命中 `.gitignore` 的 `*.npz` / `*.pt` 规则，**不入库**；其 文件名 / 字节 / sha256 登记于 §2，并注明哈希来源。
- **本轮发布范围收缩**：7 个 `*local-review*` 复核目录（`independent-review`）与 `oracle-original-3a.json` 全量件（`detail-only`）已移出仓内发行，规则见 `reports/evidence-registry.json` 的 R1–R9；原件归档在远端数据盘，path + size + sha256 以 `reports/evidence-index.md` 的树外登记表为准。本索引对这些件仍按登记值抄录其数值，**其字节不在仓内**。
- 路径与文件名按原始字节抄录，未做改写；除既定的脱敏项（§6 注）外不改任何数值与结论。
- 时间字段一律为该 run 自身 manifest 记录的 CST（`started_cst` / `ended_cst` / `created_cst`）。

### 计数摘要

| 项 | 值 |
| --- | --- |
| 已登记 run | **11**（§1）；另有 1 个空目录 `run-original-1/` 不登记，见 §1.12 |
| 其中失败尝试（保留原始日志） | 3（`run-disabled-3`、`run-original-r1-1`、`run-original-r1-2`，§4） |
| 排除大件（不入仓） | **61** 个文件 / **16,707,551,279** B（15.56 GiB）（§2） |
| 交叉核对不一致（声明但缺失） | **7** 条声明点 / 6 个不同路径（§3.1） |
| 交叉核对：记录哈希 vs 实际 | 逐字一致，0 条不符（§3.2） |
| 未能判定（manifest 不声明完整清单） | 4 类（§3.3） |
| 文本层合计 | 11,648,169 B = run 目录内 1,774,647 + `p3-calib` 顶层非 run 9,126,439 + 8 个复核目录 747,083（后者中 7 个 `*local-review*` 目录已移出仓内发行，见 §6） |
| 抽样重算大件 | 2 个（2.77 GB）与 manifest 记录对比：1 致 1 无记录（§7） |

## 1. run 登记（11 个有产物的 run）

每条给出：arm（`arm.json`，缺失则注明来源）、`started_cst`、manifest 的 `head`（附 `reports/git-history-map-20260921.txt` 的 new SHA）、以及该 run 目录内全部文件与字节。`capture/*.npz` 与 `logits.pt` 只列字节，sha256 见 §2。

| # | run | arm | started_cst | head（重写前 → 重写后） | 文件数 | 目录总字节 |
| ---: | --- | --- | --- | --- | ---: | ---: |
| 1 | `run-vanilla-1` | vanilla | manifest 未记录 | `eceb695dedb4…` → `50fac5602f65…` | 2 | 9,940,886 |
| 2 | `run-disabled-3` | patched-disabled（据 `error.txt` 命令行；无 arm.json/manifest） | manifest 未记录 | —（无 manifest） | 2 | 123,914 |
| 3 | `run-disabled-4` | patched-disabled | 2026-09-18 15:31:29 +0800 | `c65d8e52907f…` → `c76ef5cfe62b…` | 16 | 2,783,198,639 |
| 4 | `run-disabled-5` | patched-disabled | 2026-09-18 15:38:13 +0800 | `757edddb3d26…` → `a9fff6ab6cb1…` | 16 | 2,783,203,229 |
| 5 | `run-global-4` | patched-global | 2026-09-18 15:32:40 +0800 | `c65d8e52907f…` → `c76ef5cfe62b…` | 16 | 2,783,204,119 |
| 6 | `run-global-5` | patched-global | 2026-09-18 15:39:18 +0800 | `757edddb3d26…` → `a9fff6ab6cb1…` | 16 | 2,783,208,708 |
| 7 | `run-original-2a` | original | 2026-09-18 15:16:16 +0800 | `cb57a5d71063…` → `429f4a2ef0da…` | 1 | 24,815 |
| 8 | `run-original-3a` | original | 2026-09-18 15:26:54 +0800 | `07de025b662b…` → `0668e288046e…` | 14 | 2,783,178,016 |
| 9 | `run-original-3b` | original | 2026-09-18 15:28:02 +0800 | `07de025b662b…` → `0668e288046e…` | 14 | 2,783,177,745 |
| 10 | `run-original-r1-1` | original | 2026-09-18 14:55:21 +0800 | `3d3f994c1bce…` → `56cdd9012439…` | 3 | 32,514 |
| 11 | `run-original-r1-2` | original | 2026-09-18 15:04:55 +0800 | `291dcbbd1b41…` → `b6d8756ca2fa…` | 3 | 33,341 |

### 1.1 `evidence/p3-calib/run-vanilla-1/`

manifest **无 `schema` 字段**（`arm`/`head`/`pin_commit`/`model_revision`/`payload`/`result`，无 `capture`/`logits` 段）；startup_s=94.78；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `logits.pt` | 9,938,421 | 见 §2 |
| `manifest.json` | 2,465 | `a07654f463684663…` |

### 1.2 `evidence/p3-calib/run-disabled-3/`

**无 `manifest.json`**（异常发生在 manifest 构建之前，见 §4）。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `error.txt` | 1,294 | `479194f61d0f8f0f…` |
| `source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6…` |

### 1.3 `evidence/p3-calib/run-disabled-4/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:32:30 +0800；startup_s=50.18；exit_code=1；failures=["cleanup_check_ok"]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 813 | `771161e7a4eba15e…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 6,693 | `56c698f762b41300…` |
| `engine-traces.json` | 7,600 | `3b7f1492be9e22ac…` |
| `force.jsonl` | 2,608 | `0ea885d19aac2d1f…` |
| `logits.pt` | 7,951,935 | 见 §2 |
| `manifest.json` | 120,313 | `d16403764805cd31…` |
| `source/p2-calib-run.py` | 124,517 | `0f2040a04770d841…` |

### 1.4 `evidence/p3-calib/run-disabled-5/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:39:13 +0800；startup_s=49.20；exit_code=0；failures=[]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 813 | `08fcbc6d9e890d81…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 7,240 | `aa9ae9e37b627a33…` |
| `engine-traces.json` | 7,600 | `7eb4aafc5346a3ac…` |
| `force.jsonl` | 2,608 | `aba861b766577c0e…` |
| `logits.pt` | 7,951,935 | 见 §2 |
| `manifest.json` | 120,965 | `5cf9576262d83522…` |
| `source/p2-calib-run.py` | 127,908 | `e2d5d0440bb8aaba…` |

### 1.5 `evidence/p3-calib/run-global-4/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:33:39 +0800；startup_s=48.22；exit_code=1；failures=["cleanup_check_ok"]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 803 | `274d44795116f421…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 6,697 | `9f8888854e01710a…` |
| `engine-traces.json` | 12,764 | `bada76543c35c0f4…` |
| `force.jsonl` | 2,576 | `3bcabe65d6114523…` |
| `logits.pt` | 7,951,935 | 见 §2 |
| `manifest.json` | 120,667 | `8ce6b52dbfb5ca84…` |
| `source/p2-calib-run.py` | 124,517 | `0f2040a04770d841…` |

### 1.6 `evidence/p3-calib/run-global-5/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:40:18 +0800；startup_s=49.51；exit_code=0；failures=[]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 803 | `0f8e9aed468a3081…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 7,244 | `c5b00f7fb720a2dc…` |
| `engine-traces.json` | 12,764 | `b8caf2627143970d…` |
| `force.jsonl` | 2,576 | `6864b9fcff18878b…` |
| `logits.pt` | 7,951,935 | 见 §2 |
| `manifest.json` | 121,318 | `ba12082e1051376f…` |
| `source/p2-calib-run.py` | 127,908 | `e2d5d0440bb8aaba…` |

### 1.7 `evidence/p3-calib/run-original-2a/`

manifest `schema=attnview.p2-calib-run/v2`；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `manifest.json` | 24,815 | `2e648615e8dae167…` |

### 1.8 `evidence/p3-calib/run-original-3a/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:27:52 +0800；startup_s=48.21；exit_code=0；failures=[]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 323 | `7e001ca614aa3182…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 5,457 | `df5126624862fd52…` |
| `logits.pt` | 7,950,079 | 见 §2 |
| `manifest.json` | 115,377 | `6362678768d876a3…` |
| `source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6…` |

### 1.9 `evidence/p3-calib/run-original-3b/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:29:00 +0800；startup_s=48.45；exit_code=0；failures=[]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 323 | `6c699c676abd4f3e…` |
| `capture/forward1.npz` | 1,380,888,464 | 见 §2 |
| `capture/forward2.npz` | 951,250 | 见 §2 |
| `capture/forward3.npz` | 951,250 | 见 §2 |
| `capture/forward4.npz` | 951,250 | 见 §2 |
| `capture/forward5.npz` | 951,250 | 见 §2 |
| `capture/forward6.npz` | 951,250 | 见 §2 |
| `capture/forward7.npz` | 951,250 | 见 §2 |
| `capture/forward8.npz` | 951,250 | 见 §2 |
| `capture/layers.npz` | 1,387,436,946 | 见 §2 |
| `cleanup.json` | 5,457 | `46c64f4a2d0cae77…` |
| `logits.pt` | 7,950,079 | 见 §2 |
| `manifest.json` | 115,106 | `30724ec1df368bcd…` |
| `source/p2-calib-run.py` | 122,620 | `8758db24e91e6ce6…` |

### 1.10 `evidence/p3-calib/run-original-r1-1/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 14:56:10 +0800；startup_s=46.82；exit_code=1；errors=["RuntimeError: 校准: 本步 positions 形状 (3, 1505) 不是一维绝对位置，无法作为独立参考"]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 254 | `224bae8178aa3bf0…` |
| `error.txt` | 4,013 | `f614804991a7fbbb…` |
| `manifest.json` | 28,247 | `d277243669328083…` |

### 1.11 `evidence/p3-calib/run-original-r1-2/`

manifest `schema=attnview.p2-calib-run/v2`；ended_cst=2026-09-18 15:05:48 +0800；startup_s=49.34；exit_code=1；errors=["RuntimeError: 校准: 第 1 步的相位判定不一致：按真实位置是 prefill，按 attn_metadata 真实标记是 decode（{'num_prefill_reqs': 0, 'num_decode_reqs': 0, 'num_prefill_tokens': 0, 'num_decode_tokens': 0, 'max_query_len': 1505}）"]；pin_commit=`98dff2a81d747d1dba01a47f939f48c3526d4206`。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `arm.json` | 323 | `0e18b782ba461a50…` |
| `error.txt` | 4,345 | `9e9c07d85e644937…` |
| `manifest.json` | 28,673 | `89001215d1876664…` |

### 1.12 未登记：`run-original-1/`

本次登记开始时（2026-09-21）`evidence/p3-calib/run-original-1/` 为**空目录**（0 个文件、无 `manifest.json`、无 `capture/`），登记过程中该空目录已从工作区消失（git 不跟踪空目录）。因此**不登记为 run**，也不计入 §1 的 11 个。该目录是否曾产出过工件、其产物是否已并入 `run-original-2a/`：**未能判定**（目录内无任何文件可查）。

## 2. 排除大件登记表（`capture/*.npz` 与 `logits.pt`）

**这些字节不随仓发行**：`*.npz` 与 `*.pt` 命中仓库根 `.gitignore` 第 17 行（`*.pt`）、第 22 行（`*.npz`），`git ls-files` 不含其中任何一个；原始大件留在远端数据盘。

表的 **sha256 来源**：`manifest.json` 的 `capture.npz_sha256` 只覆盖 **`capture/layers.npz`** 一个文件（该值在 6 个 run 上相同；交叉验证见 §7）。
`capture/forward1..8.npz` 与全部 `logits.pt`（共 55 个文件）**在 manifest 中没有任何文件级 sha256 记录** → 这些行的 sha256 列标 `未记录`，**只登记 size**；不要把这些行的 sha256 当作已核对。

合计：**61** 个文件 / **16,707,551,279** B（15.56 GiB）= `.npz` 54 个 / `.pt` 7 个。

| run | 文件 | 字节 | sha256 | 来源 |
| --- | --- | ---: | --- | --- |
| `run-vanilla-1` | `logits.pt` | 9,938,421 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-4` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-disabled-4` | `logits.pt` | 7,951,935 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-disabled-5` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-disabled-5` | `logits.pt` | 7,951,935 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-4` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-global-4` | `logits.pt` | 7,951,935 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-global-5` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-global-5` | `logits.pt` | 7,951,935 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3a` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-original-3a` | `logits.pt` | 7,950,079 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward1.npz` | 1,380,888,464 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward2.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward3.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward4.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward5.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward6.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward7.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `forward8.npz` | 951,250 | 未记录 | manifest 无该文件哈希（只登记 size） |
| `run-original-3b` | `layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | manifest.json:capture.npz_sha256 |
| `run-original-3b` | `logits.pt` | 7,950,079 | 未记录 | manifest 无该文件哈希（只登记 size） |

### 2.1 每个 run 的大小构成（同构关系）

6 个带 `capture/` 的 run（`run-disabled-4/5`、`run-global-4/5`、`run-original-3a/3b`）的 9 个 `.npz` **逐文件字节完全相同**：

| 文件 | 字节 | 每 run 出现次数 |
| --- | ---: | ---: |
| `capture/forward1.npz` | 1,380,888,464 | 6 |
| `capture/forward2..8.npz` | 951,250（每个） | 6 × 7 = 42 |
| `capture/layers.npz` | 1,387,436,946 | 6 |
| `logits.pt`（disabled/global 臂） | 7,951,935 | 4 |
| `logits.pt`（original 臂） | 7,950,079 | 2 |
| `logits.pt`（`run-vanilla-1`） | 9,938,421 | 1 |

`run-vanilla-1/logits.pt`（9,938,421 B）与其余 6 个不同：其 `manifest.json` **没有任何 `logits` / `host_logits` 字段**（该 manifest 属另一套 schema：`arm`/`head`/`pin_commit`/`model_revision`/`payload`/`result`，无 `capture` 段），该文件与本次运行的因果关系**未能判定**；其 sha256（`6bda922d43bceba70590c72b21ca18d92e51589f3ebee79b19b32145a339949d`）在本仓库 `evidence/`、`reports/` 与 `git grep` 全量检索中**未被任何文件记录**。

## 3. 交叉核对：`manifest.json` 的自述清单 vs 实际目录

方法：把每个 `manifest.json` 里所有字符串叶子取出，筛出形如 `evidence/…` 或带工件后缀（`.json/.jsonl/.pt/.npz/.txt/.py`）的路径，逐条对照磁盘（存在性、字节数）。运行时 `git status --porcelain` 快照（`?? path/`）不是文件声明，已排除（§3.6）。

### 3.1 不一致：**声明但缺失**（7 条声明点 / 6 个不同路径）

| # | run | manifest 字段（JSON 路径） | 声明的路径 | 磁盘 | 说明 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `run-disabled-4` | `arm_file.content.trace_path` | `evidence/p3-calib/run-disabled-4/steps.jsonl` | **缺失** | `arm.json` 的 `trace_path`；该 run 无 `steps.jsonl` |
| 2 | `run-disabled-5` | `arm_file.content.trace_path` | `evidence/p3-calib/run-disabled-5/steps.jsonl` | **缺失** | 同构 |
| 3 | `run-global-4` | `arm_file.content.trace_path` | `evidence/p3-calib/run-global-4/steps.jsonl` | **缺失** | 同构 |
| 4 | `run-global-5` | `arm_file.content.trace_path` | `evidence/p3-calib/run-global-5/steps.jsonl` | **缺失** | 同构 |
| 5 | `run-original-2a` | `arm_file.path` | `evidence/p3-calib/run-original-2a/arm.json` | **缺失** | manifest 自身已记 `arm_file.exists=false`、`arm_file.content=null`——与磁盘一致，但仍是「声明了却不存在的文件」 |
| 6 | `run-original-r1-1` | `sources.emit_trajectory` | `evidence/p3-calib/traj-r1.json` | **缺失** | 失败尝试（§4）未写出轨迹；该路径在全仓库不存在 |
| 7 | `run-original-r1-2` | `sources.emit_trajectory` | `evidence/p3-calib/traj-r1.json` | **缺失** | 同构 |

**大小不符：0 条**——manifest 不为任何单个声明文件记录字节数（`capture.npz_sha256` 是哈希不是大小，`logits.records` / `force_log.records` 是记录条数不是字节），因此「大小不符」这一维度**未能判定**，不作 0 的结论。

### 3.2 记录哈希 vs 实际内容：逐字一致（0 条不符）

| 断言方 | 断言的哈希 | 对象 | 核对结果 |
| --- | --- | --- | --- |
| `run-*/manifest.json` → `capture.npz_sha256`（6 个 run） | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | `capture/layers.npz` | 重算一致（§7） |
| `manifest.json` → `source.script_sha256` 与 `source.source_snapshot_sha256`（6 个 run） | 各 run 自述 | `tools/p2-calib-run.py` 快照 | **6/6 一致**；`unchanged_at_end=true` 6/6 |
| `arm_file.content`（6 个 run） | —（内容比对） | `arm.json` | **6/6 与磁盘 `arm.json` 内容逐字相同** |
| `force_log.records`（4 个 run） | —（条数比对） | `force.jsonl` | **4/4**：lines=8 == records=8 |
| `engine_traces.count`（4 个 run） | —（条数比对） | `engine-traces.json` | **4/4**：数组长度 12/12/21/21 == count |
| `masked-prep` 复核 manifest（树外登记 `independent-review`）→ `output_sha256` | `ca842c58579d63cb40e623e7cf0628b986b7c21f669afb17653c30d3c1c6e921` | `evidence/p3-calib/masked-prep/crossblock-note.json` 及复核目录同名 `crossblock.json`（后者已移出仓内发行，哈希见树外登记表） | **双方一致**（两文件同哈希） |
| `masked-prep` 复核 manifest（树外登记 `independent-review`）→ `input_sha256`（3 项） | 自述 | `configs/p2-masked-prep/crossblock.json`、`evidence/p1-cpu/demo-fixtures.json`、`tools/p2-masked-prep-crossblock.py` | 前 **2/3** 与当前工作区一致；第 3 个与**当前**工作区不同 → 见下行 |
| 同上，改按**记录 head 提交**核对 | `691b04ad…b35c8e58` | `tools/p2-masked-prep-crossblock.py` @ e5ca9aeb | **一致**：`git show 205fa013…:tools/p2-masked-prep-crossblock.py \| sha256sum` = 691b04ad…（脚本在**其后**的提交里改成了 `a06846faff0dcc71…`） |
| `gate-record.json` → `source_hashes`（4 项，sha256 前 16 hex） | 自述 | 4 个源文件 | 与**当前工作区**：1/4 一致；与**记录 head 提交 eceb695d**：**4/4 一致** |
| `p3-masked-smoke/traj-7834-generated.json` → `context_sha256` | `8dfc679f9cf687019f69750191e79be27bc2b062fae15fdb0709abe4ff3e9cf7` | `masked-prep/crossblock-note.json` 的 `context_sha256` | **一致** |
| `final-v5-artifacts.json`（树外登记 `independent-review`）→ `manifest_sha256`（`run-disabled-5`、`run-global-5`） | `5cf95762…`、`ba12082e…` | 两个 `manifest.json` | **2/2 一致** |
| `p2-single/calibration-report.md` → oracle SHA256 前 8 | `3615eaa6…110225` | 全量件 `evidence/p3-calib/oracle-original-3a.json`（**已移出仓内发行**，`detail-only`）；本行 size + sha256 由入库的聚合段 `evidence/p3-calib/oracle-original-3a.aggregates.json` 自带 | **一致**（8,268,191 B / `3615eaa619134745879b98038a5f8783f925520e2d0ba73e1263cd4ecc110225`） |

`run-original-3a` / `run-original-3b` 的 `host_logits.argmax_matches_sampled` / `trajectory_comparison.identical` 等**语义断言**不在本索引的核对范围内（需要重跑或读 `.pt`），本索引只核对文件层事实。

### 3.3 未能判定：磁盘有、manifest 未声明

`manifest.json` **不声明完整文件清单**（不含 `files[]` 之类的枚举），因此「多文件」这个维度只在 manifest 确有枚举时才有意义。实际存在的「磁盘有而未声明」项如下，逐项给出可判定程度：

| 项 | 出现范围 | 判定 |
| --- | --- | --- |
| `cleanup.json` | 10 个 run 目录 | `cleanup` 段只记 `ok` / `checks[]`，**不含路径**；全部 manifest 中 `cleanup.json` 字符串出现 0 次。文件由谁写出、是否为脚本预期产物：**未能判定** |
| `capture/forward1..8.npz` | 6 个 run | manifest 只声明 `capture.npz`（= `layers.npz`）与 `capture.request_steps`/`forward_steps_total`；**8 个 forward 分片**无逐文件声明、无哈希 → 存在性与步数枚举一致（§3.5），但逐文件哈希**未能判定** |
| `run-vanilla-1/logits.pt` | 1 个 run | 该 manifest 无 `logits` / `host_logits` 段，9,938,421 B 的来源**未能判定**（§2.1） |
| `error.txt` | 3 个 run（§4） | 由失败处置流程写入（`run-disabled-3/error.txt` 自述来源为后台作业原始 stdout/stderr），非 manifest 声明产物；不影响文件层一致性 |
| `source/p2-calib-run.py` | `run-disabled-3` | 该目录无 `manifest.json`，故无声明可对；快照哈希 `8758db24…` 与 `run-original-3a/3b` 的快照相同（同一时刻的脚本版本） |

### 3.4 `head` 字段 × 历史重写映射

11 个 run 的 manifest 与 `gate-record.json` 共用 **7 个**不同 `head`（下表），复核产物另有 **2 个**（下段；两个复核目录均已移出仓内发行，`independent-review`，`head` 按树外登记件自述抄录）——**共 9 个不同值，全部可在 `reports/git-history-map-20260921.txt` 的 old 侧命中**（该表 142 行，两侧重写后无一相同）。`pin_commit=98dff2a81d747d1dba01a47f939f48c3526d4206` 未出现在该表中（它是 **vLLM 上游 checkout** 的 SHA，不是本仓提交，不受本仓重写影响）。

| head（重写前，本索引抄录值） | 出现处 | 重写后 SHA |
| --- | --- | --- |
| `c65d8e52907f62596d43ba9b4fceb16196650a76` | `run-disabled-4`, `run-global-4` | `c76ef5cfe62bded7aac8f27a1502140727c761d3` |
| `757edddb3d26ab130634dc487b3ac494e8fa9d68` | `run-disabled-5`, `run-global-5` | `a9fff6ab6cb126716500604df7633f2cbd1508f5` |
| `cb57a5d710636e38189f66bb1f5cff0491eadbf5` | `run-original-2a` | `429f4a2ef0dafdd2c9ba3c8d82f016b0c32f2468` |
| `07de025b662b5ee3af4cd2d504562c6fab84ea76` | `run-original-3a`, `run-original-3b` | `0668e288046ea2d2bc22287657b4e309643b9d2b` |
| `3d3f994c1bce228e6885e04fc8d001db2b46ffb9` | `run-original-r1-1` | `56cdd90124390907f409816f8a34a160154b2fcd` |
| `291dcbbd1b419588064904120863095a0e003601` | `run-original-r1-2` | `b6d8756ca2fa11a9b27a6f1ea676c3985c066bdf` |
| `eceb695dedb4cb665d4603340736f8970d20f748` | `gate-record.json`, `run-vanilla-1` | `50fac5602f65274e509f865deff53ddb91847d9d` |

另有两个不出现在 run manifest 中、但出现在复核产物 `head` 字段的值，同样命中映射：`e5ca9aeba07038bc572f589f2effc4988487fd92` → `205fa013dbc4c5ef57ed7bdd09b5fbb9a6d525f6`（`p3-masked-prep-local-review-20260920`）、`9118dafc6832199e281b092b1a3af15f600c6010` → `61a905cb689e8580e6becc5c6291ec0c9fbf884d`（`p3-masked-ref-local-review-20260920`）；`cb57a5d710636e38189f66bb1f5cff0491eadbf5` → `429f4a2ef0da…`（`run-original-2a` 与两个复核产物共用）。上述两个复核目录本轮已移出仓内发行（`independent-review`），其 `head` 值按树外登记件自述抄录。

### 3.5 声明枚举 vs 实际：`capture` 段

| run | `capture.forward_steps_total` | `capture.request_steps` | 实际 `capture/` 文件 | 判定 |
| --- | ---: | --- | --- | --- |
| `run-vanilla-1` | —（无 `capture` 段） | — | 0 | 无枚举可比 |
| `run-disabled-4` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-disabled-5` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-global-4` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-global-5` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-original-2a` | —（无 `capture` 段） | — | 0 | 无枚举可比 |
| `run-original-3a` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-original-3b` | 8 | `[1, 2, 3, 4, 5, 6, 7, 8]` | forward=8，layers=1，共 9 | 一致 |
| `run-original-r1-1` | 0 | `[]` | forward=0，layers=0，共 0 | 一致 |
| `run-original-r1-2` | 0 | `[]` | forward=0，layers=0，共 0 | 一致 |

`run-original-r1-1` / `run-original-r1-2` 的 `capture/` 是**空目录**（0 文件），与 `forward_steps_total=0` / `request_steps=[]` 一致。

### 3.6 已排除的误报类

每个 manifest 的 `git_status_porcelain` 是**运行时 `git status --porcelain` 的快照**，条目形如 `?? evidence/…/`（目录标记）。按路径检查这些字符串会全部报「缺失」，但它们不是文件声明——这些路径在快照当时均为未跟踪；就本索引的登记口径：run 目录与 `traj-original-1.json` 现随仓发行，5 个 `*local-review*` 目录与 `evidence/p3-calib/oracle-original-3a.json` 现为树外登记、不随仓发行（§6、§3.2），`vllm-patch/deployed.json` 与 `vllm-patch/orig/` 不在本索引与 `reports/evidence-registry.json` 的登记范围内（此处不判定其跟踪状态）。本索引按此排除，不计入 §3.1 的不一致数。

## 4. 失败尝试（保留原始日志，不重写为成功）

| run | 保留的证据 | 退出码（manifest） | manifest `errors`/失败点（原文摘录） |
| --- | --- | ---: | --- |
| `run-disabled-3` | `error.txt` (1,294 B)、`source/p2-calib-run.py` (122,620 B)；**无 manifest** | —（manifest 未生成） | `RuntimeError: 校准: pin checkout 与 manifest 声明的 pre 状态不一致（v1/core/sched/output.py: pin=75451f40… manifest pre=41cf5780…）` |
| `run-original-r1-1` | `error.txt` (4,013 B)、`manifest.json` (28,247 B)、`arm.json` (254 B)、空 `capture/` | 1 | `RuntimeError: 校准: 本步 positions 形状 (3, 1505) 不是一维绝对位置，无法作为独立参考` |
| `run-original-r1-2` | `error.txt` (4,345 B)、`manifest.json` (28,673 B)、`arm.json` (323 B)、空 `capture/` | 1 | `RuntimeError: 校准: 第 1 步的相位判定不一致：按真实位置是 prefill，按 attn_metadata 真实标记是 decode（… 'max_query_len': 1505）` |

`run-disabled-3/error.txt` 首段自述其来源与处置口径（逐字）：

> `# 原始失败证据（NATIVE-009 要求：原样保留，不重写为成功）`
> `# 来源：后台作业 bg_5 的原始 stdout/stderr（命令见下方），异常发生在 manifest 构建/try 之外，`
> `#       故当时该目录只有 source 快照；本文件为该次运行的完整原始尾部输出与退出码记录。`

另有 2 个 run 的 manifest `exit_code=1`、`failures=["cleanup_check_ok"]`（**不是**本节的「失败尝试」：其文本层与大件均完整，`cleanup.json` 存在）：
`run-disabled-4`（`ok=false`）、`run-global-4`（`ok=false`）。二者其后分别由 `run-disabled-5`、`run-global-5`（`failures=[]`、`exit_code=0`）承接；是否构成替代关系属研究判断，**不在本索引结论范围内**。

## 5. `evidence/p3-calib/` 顶层非 run 产物

合计 **24** 个文件 / **9,126,439** B（**入库时登记值**；本轮收缩后仓内实际为 **927,463** B = 9,126,439 − 8,268,191 + 69,215，即 oracle 全量件换为聚合段）。其中 `masked-prep/` 的 9 个文件单列于 §5.1；本表只列顶层散件（15 行），15 + 9 = 24。

注（本轮发布范围收缩）：本表数值为入库时的实测登记值。`oracle-original-3a.json` 一行的全量件已移出仓内发行（`detail-only`），入库的对应件为聚合段 `oracle-original-3a.aggregates.json`，其 `full_file` 段自带该全量件的 size + sha256（§3.2 同源）。

| 文件 | 字节 | sha256 | 自述要点（只抄字段，不作判定） |
| --- | ---: | --- | --- |
| `gate-record.json` | 1,078 | `efd0e337034a55a35cee90af1af2c99be355720055919d2621ed521beb614c00` | `kind=metadata_gate_postcommit`；`head=eceb695d…`；`probe_exit_code=0`；`failed_checks=[]`；`checks=14`；`metering` 段 6 个计数；`cst=2026-09-18 13:51:43 +0800`；`source_hashes` 4 项（§3.2） |
| `metadata-probe-gate.json` | 3,478 | `93d902d3bd7ac1a2bae1fd507b8556cf227f3ff102926ee391bff94df56b1237` | 与 `metadata-probe.json` 同 schema（`device`/`capability`/`stream_before`/`stream_after`/`metering`/`checks`/`failed`/`notes`）；`checks=14`；`failed=[]` |
| `metadata-probe-gate2.chrome.json` | 32,930 | `332116a2e839abe85509bd554474984e0ed310aa97917fe945a1d31f8204ebe0` | chrome trace（`schemaVersion`/`deviceProperties`/`traceEvents`/…）；**脱敏改动过**（§6 注） |
| `metadata-probe-gate2.json` | 5,349 | `25dd36324c4563b714448d45763713092e172dfb49d605f72d08e497ffda796e` | 加 `runtime` 段；`checks=16`；`failed=["运行期未触发隐式同步告警（torch sync_debug warn）"]` |
| `metadata-probe-gate3.chrome.json` | 34,074 | `426c8d26c4590f057e489de3bcf44319b3bba375cc3c7e3a897de017e878f5fa` | chrome trace（`schemaVersion`/`deviceProperties`/`traceEvents`/…）；**脱敏改动过**（§6 注） |
| `metadata-probe-gate3.json` | 5,145 | `97b294c42a4c1b6cb0a6068a09a90c670aebcea494e88e21f2bdf90cd76bbec3` | 同 gate2 结构；`checks=16`；`failed=["运行期未触发隐式同步告警（torch sync_debug warn）"]` |
| `metadata-probe-gate4.chrome.json` | 34,060 | `d2be21748e325117a33c6bc9bcbd26736652a0ce322e5c4c5dbb0ff99be61065` | chrome trace（`schemaVersion`/`deviceProperties`/`traceEvents`/…）；**脱敏改动过**（§6 注） |
| `metadata-probe-gate4.events.json` | 5,435 | `7199a91e847bf56264aa37f1005d2a8199c9009d26e4c8e1e08ed608c5806dc6` | chrome-trace 事件数组，8 条，元素键 `name/attrs/args/device_type` |
| `metadata-probe-gate4.json` | 5,194 | `357b22ee1c703e21c976e3b3d1782be2f3f6ac7ab11c167eb6de943851a8978b` | 同 gate2 结构；`checks=16`；`failed=[]` |
| `metadata-probe-gate5.chrome.json` | 34,070 | `524545a739f399d3d1c100398bc5fe6f503b5bd8372d977f2b37d0300cac1b78` | chrome trace（`schemaVersion`/`deviceProperties`/`traceEvents`/…）；**脱敏改动过**（§6 注） |
| `metadata-probe-gate5.json` | 5,354 | `076fb27be3d6ce90f4fff36872e0208b1e004880d2475dc3d8666fce86ef702b` | 同 gate2 结构；`checks=18`；`failed=[]` |
| `metadata-probe.json` | 3,478 | `b248242410e5b8b1e15166d76086d4aae4c7b97b9d0f0121825e64711a460611` | 基线探针（`checks=14`、`failed=[]`，**无 `runtime` 段**）；其后 gate2..gate5 加入 `runtime` 段（profiler CUDA activity + sync-debug + chrome trace） |
| `oracle-original-3a.json` | 8,268,191 | `3615eaa619134745879b98038a5f8783f925520e2d0ba73e1263cd4ecc110225` | **聚合段 `oracle-original-3a.aggregates.json` 入库；全量件已移出仓内发行（`detail-only`），本行 size + sha256 由聚合段的 `full_file` 段自带**。全量件顶层键（剥离前实测）：`schema`/`generated_at_cst`/`capture`/`metadata`/`numerics`/`comparisons`/`per_layer`/`per_scope`/`per_step`/`non_finite`/`summary`/`warnings` |
| `traj-original-1.json` | 924 | `da01f10eeae306ddd5f599180a4b56192fe399dc0ab3cd86460098eee5e923af` | `attnview.p2-calib-trajectory/v1`；8 步 token；`arm=original`；`trajectory_source=natural-greedy`；`created_cst=2026-09-18 15:27:48 +0800`；`logits_sha256=24c6ad2f…` |
| `traj.json` | 88 | `019e8493a3681e6a76d0106b117e922f2933d6d5188d39e99aea76d2f8c7f8c4` | 仅 `tokens`（8 步，与 `traj-original-1.json` 同序同值）；无 schema/head/时间字段 |

### 5.1 `masked-prep/`（9 个文件 / 687,591 B）

由 `tools/p2-masked-prep-crossblock.py` 离线（`CUDA_VISIBLE_DEVICES=""`）产出，运行记录在 `evidence/p3-masked-prep-local-review-20260920/`（§6；该目录已移出仓内发行，`independent-review`）；两者的 `crossblock-note.json` / `crossblock.json` 逐字相同（同 sha256）。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `masked-prep/crossblock-events.json` | 156,414 | `b8f544fa3b04e2edb3d973f11495c04cd350e7817f9b207d318bc7f610df0b5a` |
| `masked-prep/crossblock-note.json` | 153,051 | `ca842c58579d63cb40e623e7cf0628b986b7c21f669afb17653c30d3c1c6e921` |
| `masked-prep/crossblock-r2.json` | 147,148 | `30a41c2889846175babb08976ed1e0f311cbd6c80bd3b6085e11b41087f4bc3a` |
| `masked-prep/crossblock.json` | 102,935 | `b8a109e75c903cb9a0fa463e0e15925725613a6045f304322cd0442d6a048423` |
| `masked-prep/points.json` | 3,915 | `d4790b03c504334aa6eab1a6902e28b24a9bcd3dfd32f62369a7f758817ce234` |
| `masked-prep/ref-cpu-checks.json` | 9,718 | `eda2d12a9a4f128e77970256b0e6982afbce96a4efc1ef3f56bb6db451593b8b` |
| `masked-prep/scale-metrics-dedup.json` | 1,371 | `04530cd02694bd50cec961e8b291338cfa63965fbe91cad474db51c2de0ad375` |
| `masked-prep/scale-metrics.json` | 98,318 | `850bda144bb5ba90ba24d5eea9b2b3fa69e443db8f862d9714e607e22f855f37` |
| `masked-prep/sets.json` | 14,721 | `2df7bb74a2e09f20180fbbe5e6811b9f11840a2b91f5cdb6405e870af69e8cfb` |

`traj-original-1.json` 的 `created_cst`（15:27:48）晚于 `run-original-2a` 的 `started_cst`（15:16:16）：`run-original-2a/` 目录内只有 `manifest.json`，其 `sources.emit_trajectory` 指向该文件；`run-original-3a` 的 `trajectory_emitted.path` 也指向它。该文件由哪一次运行写出：**未能判定**。

## 6. 同批独立证据目录（8 个）

这 8 个目录既不属于 `p3-calib` 的 run，也不是其 manifest 产出的；它们记录阶段 01/02/03 的**本地复核**与 `masked` 系列的准备/参考复核。合计 747,083 B。

**发布范围（本轮收缩）**：§6.1 `p3-masked-smoke/` 随仓发行；§6.2–§6.8 的 7 个 `*local-review*` 目录**已移出仓内发行**（`independent-review`），其 path / size / sha256 与原件现登记于 `reports/evidence-index.md` 的树外登记表（原件归档在远端数据盘）。各节的 bytes / sha256 仍按入库时登记值抄录，**不代表其字节在仓内**。

**脱敏注**：其中 5 个文件在 2026-09-21 的树内容脱敏中被改写（容器主机名/GPU UUID/内部路径），改写前后 sha256 见 `/root/autodl-tmp/pass2-hash-table.tsv`：
`p3-calib-local-review-20260918/metadata-retry.chrome.json`、`p3-calib/metadata-probe-gate2.chrome.json`、`…gate3.chrome.json`、`…gate4.chrome.json`、`…gate5.chrome.json`（只改这 5 个；`metadata-probe-gate4.events.json` 未改）。下表登记的是**改写后**的值。

### 6.1 `evidence/p3-masked-smoke/` — P3 masked 冒烟轨迹（第 29 步仅采样，终止）

1 个文件 / 1,812 B。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `traj-7834-generated.json` | 1,812 | `5b15d841fff0436a7714b52dbc36dd140270ded0eb3c9669d72f632cb6f549ea` |

### 6.2 `evidence/p3-masked-prep-local-review-20260920/` — masked 跨块准备复核（离线，含 run manifest 与 run.log）

3 个文件 / 154,966 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `crossblock.json` | 153,051 | `ca842c58579d63cb40e623e7cf0628b986b7c21f669afb17653c30d3c1c6e921` |
| `manifest.json` | 1,343 | `1fd86238502a3a44f1ce4d3a651080a10b5bc38da80bf3335d4d56331089dceb` |
| `run.log` | 572 | `7f8c64dbf7ea086b2761a87884a150c7bed69404fd8b5c977848b9aaa44b8faa` |

### 6.3 `evidence/p3-masked-ref-local-review-20260920/` — masked 参考实现集成前 findings

1 个文件 / 820 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `preintegration-findings.json` | 820 | `a84017e9ae4f54598457a72c8b276bbb3e98573a0df6b335bee2f19fa810d8ae` |

### 6.4 `evidence/p1-gpu-local-review-20260918-1011/` — 阶段 01 GPU 复核（3 个 seed 的 cases + summary + run-manifest）

5 个文件 / 446,675 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `cases-seed0.json` | 69,001 | `2bb0601b77c8bb0d3103b9594b1f98b2358483bf33814e7253d5441e045c72b8` |
| `cases-seed1.json` | 69,007 | `4168ff3553b77bd000b44fae182cbada9efb9f109a24f9746e2694c0a81b172f` |
| `cases-seed2.json` | 68,991 | `b07d7bb9020276495aa8916a6f8cdff27923ea37569c001502a471ba38e51da2` |
| `run-manifest.json` | 1,102 | `9f0264bdaf7124d74b76209ddfc0c6a9b2a0b80ad26b69db3c9c296712d35a7d` |
| `summary.json` | 238,574 | `cc30f3de73c698e536f7e9855c30a7af44e8d2058abcffe914c1e9c482ec6301` |

### 6.5 `evidence/p2-single-local-review-params-20260918/` — 阶段 02 params 通道探针

1 个文件 / 2,605 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `params-channel-probe.json` | 2,605 | `b0fe8b6fe2404189b8911288a20522f2fee48f213288e4f1762c836574a63dd7` |

### 6.6 `evidence/p2-single-local-review-r1-20260918/` — 阶段 02 R1 findings

1 个文件 / 1,001 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `findings.json` | 1,001 | `ef0cc46c3a5cfc2b775ca83ea147d01d60faf9a09361e1ca0eb88d7f11099ea7` |

### 6.7 `evidence/p2-single-local-review-r2-20260918/` — 阶段 02 R2 部署未知态

1 个文件 / 516 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `deployment-unknown-state.json` | 516 | `60e24446c69e9ad0502604a2f20c4ff7f8106158d463e47df6768aa1ac8b7784` |

### 6.8 `evidence/p3-calib-local-review-20260918/` — 阶段 03 校准本地复核（11 件）

11 个文件 / 138,688 B。**已移出仓内发行**（`independent-review`）：path / size / sha256 见 `reports/evidence-index.md` 的树外登记表。

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `capture-byte-identity.json` | 859 | `655b4a2843cdc5652b0be9fa1cd01f3a632e4bf2ac6a689d40d162c7d8ff93e9` |
| `driver-findings.json` | 264 | `bcce8eef53b699973dbb424e3db80cdc37781aafafb8279ee30c99a75d2c1a5c` |
| `engine-contract-findings.json` | 500 | `8c296b1fe6caf78b7a7ff4208c08e8a33f632ecbd9c6da277bedcf3455642a35` |
| `final-v5-artifacts.json` | 1,089 | `79e1a40f50bb4647fb17254b20c5805e1d8f75aa9d57c5eba5e8a90d6284c4e9` |
| `metadata-retry.chrome.json` | 34,090 | `4e610ad31539b932068feb9efd2c0fa220fad98adfdc5a8dd0d444c638bfe0af` |
| `metadata-retry.json` | 5,370 | `b1709a1a134229a286751efca5918b855b632e4e8d9cf46640b7c9826e559a25` |
| `oracle-worstcase-independent.json` | 1,324 | `b3ea79ddd71cfe90d1efa9ddc4bb0b42062a362b62f5fff262faeca1e582e50e` |
| `original-repeat-tensors.json` | 28,137 | `e3c7c2f5970f9c8d0041af5941c60f9b66b76673a83bf7e95b169bd5bd30dda9` |
| `output-mode-findings.json` | 1,532 | `6d5ab34f8053da2f1e4c75925ffeb4d43a4592c4e06562b823a2e07d9f070971` |
| `patched-v4-main-tensors.json` | 63,815 | `887bae9b0a139e6e8481f72b1f233f72759c0078f36217dc3a90e800a1b29756` |
| `r2-note-recheck.json` | 1,708 | `8195b79b611fcd2167eaefa1da0048f2b426c307f4b65421f6f61ad3c6f0cbe3` |

`p3-calib-local-review-20260918/` 的其余自述要点：`final-v5-artifacts.json` 记 `run-disabled-5`/`run-global-5` 的 `manifest_sha256`、`head`、`capture_sha256`、`logit_count=8`、`capture_equal_to_v4=true`、`logits_equal_to_original=true`；`capture-byte-identity.json` 记 4 个 run 的 `capture/layers.npz` 的 `sha256`+`bytes` 并断言 `byte_identical=true`；`oracle-worstcase-independent.json` 记 2 个最坏 case 的独立 NumPy float64 复算（`method` 字段声明为 2 例、非全量重跑）。

## 7. 大件抽样重算（2 个，共 2.77 GB）

数据盘顺序读约 2 GB/s；**未做全量重算**（16.7 GB 只有 6 个文件有 manifest 哈希，其余 55 个无记录可对）。
抽样选取：1 个**有** manifest 记录的文件 + 1 个**无**记录的文件，用于同时验证「记录可信」与「记录缺口确实存在」。

| 文件 | 字节 | 重算 sha256 | 用时 | 与 manifest 记录对比 |
| --- | ---: | --- | ---: | --- |
| `run-original-3a/capture/layers.npz` | 1,387,436,946 | `e60ec7d1be13895e2e157d281db98fce3e1cb822dccb27b3fea9f1f2e1437008` | 1.1 s | **一致**（`run-original-3a/manifest.json` → `capture.npz_sha256`） |
| `run-disabled-5/capture/forward1.npz` | 1,380,888,464 | `fdb9b06ff8e64b3b4d015ac71a33e68bccf6b572f35aa39861565370e1aff655` | 2.2 s | **无记录可比**（manifest 不含 forward 分片哈希） |

由此得到的两个可引用结论（仅限事实层面）：

1. `manifest.json` 的 `capture.npz_sha256` **确实覆盖 `capture/layers.npz` 且值可复现**（1/1 抽样一致）；该值在 6 个 run 上相同，且 `p3-calib-local-review-20260918/capture-byte-identity.json`（该件已移出仓内发行，`independent-review`；§6.8）对其中 4 个 run 独立记录了同一哈希。**`forward2..8.npz` 未被抽样**（其值未在任何文件中记录）。
2. **55/61 个大件没有可对照的哈希记录**，其中 `logits.pt` 全部 7 个无记录、`forward*.npz` 全部 48 个无记录（`layers.npz` 6 个有记录）。若要把这些大件纳入可核对范围，需要重新生成哈希登记——**本索引不做此事**。

## 8. 可执行核对命令

```bash
cd /root/autodl-tmp/attnview

# 8.1 大件清单（文件名 + 字节），不入库
find evidence/p3-calib -name '*.npz' -o -name '*.pt' | sort | xargs stat -c '%s	%n'
git check-ignore -v evidence/p3-calib/run-original-3a/capture/layers.npz evidence/p3-calib/run-original-3a/logits.pt

# 8.2 抽样重算大件哈希（约 2.8 GB 读取；不要对全部 16.7 GB 重算）
sha256sum evidence/p3-calib/run-original-3a/capture/layers.npz \
          evidence/p3-calib/run-disabled-5/capture/forward1.npz
jq -r '.capture.npz_sha256' evidence/p3-calib/run-original-3a/manifest.json   # → e60ec7d1…

# 8.3 §3.1 不一致逐条复现（应全部打印 MISS）
for p in evidence/p3-calib/run-disabled-4/steps.jsonl \
         evidence/p3-calib/run-disabled-5/steps.jsonl \
         evidence/p3-calib/run-global-4/steps.jsonl \
         evidence/p3-calib/run-global-5/steps.jsonl \
         evidence/p3-calib/run-original-2a/arm.json \
         evidence/p3-calib/traj-r1.json ; do
  [ -e "$p" ] && echo "PRESENT $p" || echo "MISS    $p"
done
jq -r '.arm_file.content.trace_path, .sources.emit_trajectory' evidence/p3-calib/run-global-4/manifest.json

# 8.4 manifest 声明的路径是否存在（通用式；含 §3.6 的 git status 噪音，需自行过滤）
for f in evidence/p3-calib/run-*/manifest.json; do
  jq -r '.. | strings | select(test("evidence/") and test("^/root/")|not)' "$f" \
    | sort -u | while read -r p; do [ -e "$p" ] || echo "MISS $f -> $p"; done
done

# 8.5 记录哈希 vs 实际（§3.2 逐条）
sha256sum evidence/p3-calib/run-original-3a/source/p2-calib-run.py   # 对上 manifest.source.source_snapshot_sha256
jq -r '.source | .script_sha256, .source_snapshot_sha256' evidence/p3-calib/run-original-3a/manifest.json
sha256sum tools/p2-masked-prep-crossblock.py                        # → a06846fa…（当前工作区）
# 按记录 head 提交核对（masked-prep manifest 写入时的版本）：
git show 205fa013dbc4c5ef57ed7bdd09b5fbb9a6d525f6:tools/p2-masked-prep-crossblock.py | sha256sum   # → 691b04ad…

# 8.6 head 字段 → 重写后 SHA（§3.4）
grep -P '^eceb695dedb4cb665d4603340736f8970d20f748\t' reports/git-history-map-20260921.txt

# 8.7 gate-record 的 source_hashes 在记录 head 提交上核对（4/4 一致）
NEW=$(grep -P '^eceb695dedb4cb665d4603340736f8970d20f748\t' reports/git-history-map-20260921.txt | cut -f2)
for f in vllm-patch/files/vllm/v1/worker/gpu/attnview_adapter.py vllm-patch/manifest.json \
         tools/p2-calib-metadata-probe.py src/attnview/step_plan.py; do
  printf '%s  %s\n' "$(git show "$NEW:$f" | sha256sum | cut -c1-16)" "$f"
done

# 8.8 登记项与仓库跟踪状态（大件不在其中）
# 注：下列 3 个 *local-review* 目录已移出仓内发行（independent-review），git ls-files 不再列出其文件；
#     仓内部分只剩 evidence/p3-calib 与 evidence/p3-masked-smoke。树外登记的 path / size / sha256 见 reports/evidence-index.md。
git ls-files evidence/p3-calib evidence/p3-masked-smoke \
  evidence/p3-masked-prep-local-review-20260920 evidence/p3-masked-ref-local-review-20260920 \
  evidence/p3-calib-local-review-20260918 | wc -l
```

## 9. 本索引不做的事

- 不判定 `smoke_structure`（草稿非门禁）与 stage-05 的验收状态；不给出任何结论标签。
- 不修改任何数据文件；不重算全部大件哈希；不为缺失文件补造替代物。
- 不判定 `run-disabled-4`/`run-global-4`（`cleanup_check_ok` 失败）与 `run-disabled-5`/`run-global-5` 的替代关系。
- 不判定 `run-vanilla-1/logits.pt` 与 `traj-original-1.json` 的产出归属（§2.1、§5.1：未能判定）。
- 不核对 `.pt` / `.npz` 内部的张量级断言（`argmax_matches_sampled`、`identical`、`logits_equal_to_original` 等）——那需要读取大件或在 GPU 上重跑，超出「证据索引」的工作范围。

