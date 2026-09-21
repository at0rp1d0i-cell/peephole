# harness 接入 runbook（远端模型侧 + Windows 验收侧，公网映射）

日期：2026-09-21。性质：可执行操作手册（草案，未冻结）。范围：只覆盖"模型侧在本机、Docker 验收侧在用户 Windows 机器、两侧经平台公网映射对接"这一条链路；不改协议、不改阈值、不改模型/后端。

来源与依据：`project-plan.md` §1/§4.1/§6.4、`reports/p2-single/api-contract-proposal.md`（本轮提案）、`docs/coding-eval-research-report.md`（SWE-bench 链路与环境口径）、本轮实测（端口映射、scaffold token 开销、HF 直连状态）。

## 0. 当前状态：哪些现在就能跑

| 环节 | 状态 | 依据 |
| --- | --- | --- |
| 平台公网映射 6006 | **已实测可用**（本地 200 / 公网 200） | 本轮探针 |
| Windows 侧容器验收链路 | **可执行**（需装 Docker Desktop） | SWE-bench 官方要求：x86_64、宿主 Docker daemon、≥120 GB 空闲、16 GB RAM |
| 远端原版引擎服务 | **可执行**（需 GPU 授权） | `tools/serve-vanilla.sh` 已存在并在阶段 02 实跑过 |
| 协议载荷经 HTTP 注入 | **未验证** | `vllm_xargs` → `SamplingParams.extra_args` 源码存在（`chat_completion/protocol.py:492,705,746`），跨进程序列化未实测 |
| 薄封装服务（协议渲染 + 标签过滤 + 记账） | **未实现** | `api-contract-proposal.md` 全文 |
| 超 8192 上下文 | **未放行** | `max_model_len=8192`；协议臂固定开销 1,434 token |

## 1. 拓扑与端口

```
Windows（Docker harness + agent）                远端 AutoDL（单卡 96 GB）
  swebench eval ──┐                                  vllm serve 127.0.0.1:8000
  agent scaffold ─┼─ HTTPS ─→ :8443 ──映射──→ 6006 ←┤   ▲
                  │                                  └───┘ 薄封装（待实现，0.0.0.0:6006）
                  └─→ 本地容器验收（不调模型）
```

- **只暴露 6006**（薄封装）。vLLM 本体保持 `127.0.0.1:8000`，不出网。
- 6008 的映射地址待确认（用户给出的地址含双 `u`，实测返回 404）；当前无服务监听 6008。
- 单卡装不下两个 27B（51.75 GiB × 2 > 96 GB）→ 候选/原版**分时**启动，不并行。

---

## 2. 远端（这台机器）执行

### R0 环境自检（无副作用）

```bash
cd /root/attnview && source ./env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
"$ATTNVIEW_PYTHON" -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
df -h /root/autodl-tmp
```

判据：GPU 无他人计算进程；`vllm 0.29.0`；数据盘空闲 ≥ 50 GB。

### R1 通道自检（不需要 GPU；已实测通过）

```bash
cd /root/autodl-tmp/attnview/tmp && python3 -m http.server 6006 --bind 0.0.0.0 &
curl -s -o /dev/null -w "local=%{http_code}\n" http://127.0.0.1:6006/
curl -s -o /dev/null -w "public=%{http_code}\n" https://u249165-182f-041a0abe.westd.seetacloud.com:8443/
kill %1
```

判据：`local=200`、`public=200`。**再由 Windows 侧访问同一公网 URL 复验一次**（容器内自证不等于外部可达）。

### R2 启动引擎（**需用户/主代理授权**，占 GPU）

```bash
cd /root/attnview && source ./env.sh
# 原版（对照臂）：
bash tools/serve-vanilla.sh --tag api-orig-01 --port 8000 -- --api-key "$ATTNVIEW_API_KEY"
# 候选（部署补丁后同一条命令即为 patched 引擎）：
"$ATTNVIEW_PYTHON" tools/p2-apply-patch.py apply && "$ATTNVIEW_PYTHON" tools/p2-apply-patch.py verify
bash tools/serve-vanilla.sh --tag api-cand-01 --port 8000 -- --api-key "$ATTNVIEW_API_KEY"
```

判据：`logs/serve-api-*.log` 出现 `Started server` / `Uvicorn running on http://127.0.0.1:8000`；`curl -H "Authorization: Bearer $ATTNVIEW_API_KEY" http://127.0.0.1:8000/v1/models` 返回 200 且 `id=qwen3.8-27b`。

注意：`tools/serve-vanilla.sh` 硬编码 `--host 127.0.0.1`（脚本内两处）；**不要**为暴露而改它——暴露交给薄封装（R3）。

### R3 暴露（薄封装实现前的最小手段）

- **目标形态（推荐）**：薄封装监听 `0.0.0.0:6006`，转发到 `127.0.0.1:8000`，并完成协议渲染、载荷注入、标签过滤、记账（见 `api-contract-proposal.md`）。**待实现**。
- **临时形态（仅用于通道/基线验证）**：把 vLLM 直接绑到 `0.0.0.0:6006`：在 `tools/serve-vanilla.sh` 给 `--host` 加一个可选参数（改动点：脚本内 `--host 127.0.0.1` 两处），然后
  ```bash
  bash tools/serve-vanilla.sh --tag api-exposed --port 6006 -- --host 0.0.0.0 --api-key "$ATTNVIEW_API_KEY"
  ```
  ⚠️ 这是**仅验证用**的临时形态：它把原版接口直接暴露在公网，且不含协议层——**不得**用它产出任何机制结论。

### R4 记账与日志

- 引擎日志：`$ATTNVIEW_LOGS_DIR/serve-<tag>.log`（脚本自动写）。
- 服务侧记账（目标形态）：每请求一行 `serving.jsonl`，字段见 `api-contract-proposal.md` §4；键 `(run_id, instance_id, attempt)`，与 Windows 侧 harness 结果对齐。
- 三类时间**分开记**：`prefill_ms` / `decode_ms`+`decode_tokens` / `tool_ms`（由 harness 侧回填）。

### R5 停止与回滚

```bash
# 停服务：Ctrl-C 或 kill 对应 vllm 进程；确认显存释放
nvidia-smi --query-gpu=memory.used --format=csv
# 撤销补丁（事务化，只动本事务写过的文件）
"$ATTNVIEW_PYTHON" tools/p2-apply-patch.py revert
```

判据：显存回落到启动前水平；`p2-apply-patch.py verify` 在回滚后应报"未部署"。

---

## 3. Windows 执行

### W0 自检

```powershell
docker version                                  # Docker Desktop（WSL2 后端）daemon 在跑
docker buildx version                           # harness 需要 buildx
Get-PSDrive C | Select-Object Free              # ≥120 GB 空闲
(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB
```

判据：Docker 可用、磁盘 ≥120 GB、内存 16 GB（**临界**）→ 全程 `-j 1`。
不通过时：装 Docker Desktop（WSL2）并启用文件共享；磁盘不足则先只跑 1 个实例。

### W1 安装 harness

```powershell
py -3.11 -m venv $env:USERPROFILE\swebench-env
& $env:USERPROFILE\swebench-env\Scripts\Activate.ps1
python -m pip install -U pip swebench
$env:HF_ENDPOINT = "https://hf-mirror.com"      # 国内拉数据集用镜像；HF 直连本轮已恢复 200，可对比
```

判据：`swebench --help` 正常输出。

### W2 gold 正控（**先证明容器链路，不涉及我们的模型**）

```powershell
swebench eval lite --gold -i <instance_id> --run-id gold-001
```

判据：产出 `report.json`，`resolved: true`（gold 必须通过）。**这一步不过，后面全部不成立**——先修 Docker/磁盘/镜像拉取。

### W3 公网通道复验（从 Windows 到本机服务）

```powershell
curl.exe -H "Authorization: Bearer <ATTNVIEW_API_KEY>" https://u249165-182f-041a0abe.westd.seetacloud.com:8443/v1/models
```

判据：200 + 模型 id。失败排查顺序：本机服务是否在跑 → 是否绑 `0.0.0.0` → 平台映射地址是否对应 6006 → key 是否正确。

### W4 agent 生成预测

- agent scaffold 在本机跑（只需 CPU + HTTP），上下文**必须限制在 8192 以内**（当前引擎上限；协议臂实际可用约 6,700 token）。
- 输出 `preds.jsonl`，每行：
  ```json
  {"instance_id": "...", "model_name_or_path": "qwen3.8-27b-<arm>", "model_patch": "<unified diff>"}
  ```
- **必须去重**：重复 `instance_id` 会被后者静默覆盖（本仓已核实）；生成侧做一次 `instance_id` 唯一性断言。
- 每个 arm 用**不同** `model_name_or_path` 与不同 `run_id`：`vanilla` / `da_no_mask` / `da`。

### W5 验收

```powershell
swebench eval lite -p preds.jsonl -i <id1,id2,...> -j 1 -t 1800 -r run-<arm>-01
```

判据：`results.json` 中逐实例 `resolved`；失败样本保留原始输出与测试日志，不筛选。
**不可复用 `run-id`**（同 id 会复用旧判定，不比 patch 内容）。

### W6 回传

回传：`preds.jsonl`、`results.json`、每实例 `report.json`、失败日志摘要（脱敏：不含私密文档）。
对应本机 `serving.jsonl`（同 `(run_id, instance_id, attempt)`）→ 三类时间分列成表。

---

## 4. 第一步先做什么（最小成本顺序）

1. **W0 → W2**（Windows，零成本，不占 GPU）：证明容器链路。
2. **R1 + W3**（远端不起引擎，零 GPU）：证明公网链路双向可达。
3. 以上都过，再申请 R2（占 GPU）+ 实现 R3 薄封装。
4. 最后才进 W4/W5 端到端单实例，再考虑小样本。

## 5. 已知限制与阻塞（不得绕过）

| 项 | 影响 |
| --- | --- |
| `max_model_len=8192`，协议臂固定开销 1,434 token | 只能跑短上下文任务；长上下文未放行前，不得用 SWE-bench/Live 的大多数实例 |
| 首版单活跃请求 | agent 并发调用需排队或限流；`-j 1` 同时限制容器并发 |
| 薄封装未实现 | R3 的临时暴露形态产出的一切结果**只能用于通道与 baseline 验证** |
| `vllm_xargs` 序列化未验证 | 协议载荷经 HTTP 是否保真未知；第一个冒烟必须逐字节比对 |
| SWE-bench Pro 公开镜像污染（issue #93，报告者称 100% 可复现 `git show` 取答案） | 不得用其 public 集出质量结论 |
| Live 榜单提交要求外发完整 rollout trajectory | 需用户明确同意；默认只本地跑、不提交 |
| Windows 16 GB RAM | 只能 `-j 1` 串行，长任务注意宿主内存 |

## 6. 失败处理

| 现象 | 处理 |
| --- | --- |
| W2 gold 正控失败 | 修 Docker/磁盘/镜像源；**不得**把 gold 失败当"环境无所谓"继续 |
| W3 curl 超时 | 依次查：服务进程、绑定地址、平台映射地址、key；不要反复重启引擎掩盖 |
| 引擎 OOM / 启动失败 | 保留完整日志后诊断；不反复加载掩盖同一故障（沿用 stage-05 预算与做法） |
| 上下文超限 4xx | 缩短任务上下文或改小样本；**不得**让引擎截断后当成功 |
| 载荷非法 / focus 全失败 | 按 C6.2 保持当前模式 + 计失败 + 单列；不得静默回退 global |
