# 内核 / 适配分界：初步计划清单（草案）

日期：2026-09-20。状态：**草案，待用户确认**；开发未完成，本清单只定「边界与顺序」，不冻结任何公开 API，
不改变协议语义、评测阈值、资源范围与发布形态。

归属：远端仓 `reports/`。经打包步骤进素材仓 `docs/`（建议文件名 `open-source-split-plan.md`）。
与命名决定（`peephole`，见台账 2026-09-20 决策记录）属同一「发布前布局冻结」窗口。

## 0. 目标与不变量

目标：开源时让读者一眼看清「哪部分是机制、哪部分是引擎胶水」，且内核可脱离任何推理框架独立测试。

不变量（本清单任何阶段都不得破坏）：

1. 读取视图语义与不变量 I1–I8 不变（`docs/read-view-spec.md`）。
2. 协议状态机与异常处置不变（`docs/protocol-contract.md`）。
3. 现有部署与证据链继续成立：`manifest.json`（逐文件 sha256）+ `p2-apply-patch.py` 的
   apply/verify/revert + `deployed.json` journal + 部署指纹。
4. 开发未完成，不得为「结构好看」重排生产路径；每个阶段都必须零行为变化，除显式标注者外。

## 1. 现状事实（2026-09-20 实测，作为分界依据）

| 事实 | 证据 |
| --- | --- |
| 部署集恰好是 **7 个**内核模块 | `tools/p2-gen-patch.py` `PACKAGE_FILES` = `__init__/decode/parser/readview/gpukv/state/step_plan`；`vllm-patch/manifest.json` `package_files` 逐文件 sha256 |
| 部署是**显式清单 + 哈希核对**，不是隐式环境依赖 | `tools/p2-apply-patch.py` `plan_targets()` 把 `package_files` 拷到 `venvs/.../site-packages/attnview/`，逐个核对 `post` sha256；journal 见 `vllm-patch/deployed.json` |
| 适配层 = 2 个新文件装进 vLLM 包内 | `new_files`：`v1/engine/attnview_engine.py`（727 行）、`v1/worker/gpu/attnview_adapter.py`（760 行） |
| 上游改写 8 处 | `manifest.edits`：`v1/engine/core.py`、`v1/worker/gpu/{attn_utils,model_runner,gpu_worker}.py`、`model_states/{default,interface,mamba_hybrid}.py`、`v1/core/sched/output.py` |
| 内核**对 vLLM 零 import** | `grep -rn vllm src/attnview/` 无命中；`src/attnview/` 共 3153 行 / 16 模块 |
| 但「无 vLLM import」≠ 语义框架无关 | `step_plan.check_supported_config`（`:195-225`）强制 `async_scheduling=False`、`cudagraph_mode=NONE`（`--enforce-eager`）、禁前缀缓存、禁投机/MTP |
| `gpukv` 是**生产路径**，不是探针 | `step_plan.build_step_plan`（`:333`）调用 `read_table_from_read_view` / `validate_mapping` 生成无 `-1` 物理读取表与 `seqused_k`/尾长/`next_write_position` |
| 无标准打包 | 仓内无 `pyproject.toml` / `setup.py`；测试靠 `sys.path.insert(0, REPO/"src")` |

工具输出说明：以上均为原始命令输出（非 RTK 摘要）。

**两处此前的误述，在此更正**：

- 不成立：「适配层裸 import、无安装声明，外部复跑必踩空」。实际存在显式部署链路（见上表第 2 行）。
  标准打包是分发改善项，**不作为分拆的理由**。
- 不成立：「按文件名前缀归类」。分界依据是**运行职责**，见 §2/§3。

## 2. 分界原则（按运行职责，不按前缀、不按依赖）

- **内核（kernel）**：纯函数、CPU 可跑、无 torch/vLLM 依赖，语义只由本仓不变量定义；
  是「模型声明 → 读取视图 → 物理读取表」这条链。
- **适配（adapter）**：框架特定的一切——几何从运行期 KV 配置读取、attention metadata 落位、
  块表格式与 kernel 契约、**部署门禁**、打点 hook 位置。
- **服务/任务准备（service）**：输入侧分段与 prompt 构造、公共输出提取；纯 CPU，但角色不是引擎内核。
- **工具/证据（tools）**：探针、oracle、参考实现；不进任何发行包。

## 3. 三层归属清单（推定，待用户确认）

### 3.1 内核

- 已部署 7 文件：`parser`（增量标签解析）、`state`（ProtocolRegistry）、`readview`（视图构造 + I1–I8）、
  `gpukv`（视图 → 无 `-1` 物理读取表与长度算术）、`step_plan`（每步计划装配、canonical 映射校验）、
  `decode`（增量 detokenize，供标签解析）、`__init__`。
- 待定：`extract`（公共输出提取/过滤，纯 CPU）——内核 or 服务层？
- 待定：`trace`（运行时观测）——当前未进部署集，随 API/指标设计再定归属。

### 3.2 服务 / 任务准备（纯 CPU，非引擎内核）

- `segmenter`（输入分段与 token-span 映射）、`prompt` / `prompts`（prompt 与协议文本构造）。

### 3.3 适配层（vLLM）

- `vllm/v1/engine/attnview_engine.py`、`vllm/v1/worker/gpu/attnview_adapter.py`、8 处上游改写。
- **需要从内核移出（阶段 P2）**：`check_supported_config` 中的 vLLM 部署门禁（异步调度 / CUDA Graph /
  前缀缓存 / 投机）。内核只保留与框架无关的输入合法性校验（模式、`effect_step`、
  `attention_kv_len`、`canonical_blocks` 前缀完整性等）。
- 线上通道（唯一跨语言/跨进程契约）：`extra_args["attnview"]`（8 处）与 `attnview_geometry`。

### 3.4 工具 / 证据（不发行）

- `gpuoracle`（唯一引 torch 的模块，独立 FP32 参考）、`reference`、`gpucheck`、`tools/p2-*`
  （calib 驱动、探针、gen/apply patch 工具）。
- 归此类的依据是**角色**（产生证据），不是依赖：`gpucheck`/`reference` 同样不引 torch。

## 4. 阶段清单（每阶段独立可停，按序）

| 阶段 | 内容 | 行为影响 | 完成判据 |
| --- | --- | --- | --- |
| **P0** 定边界 | 本清单经用户确认；命名（`peephole`）与路径/包名同步 | 无（文档） | 台账记录 + `docs/` 落定 |
| **P1** 内核自理 | 内核模块只依赖内核内部；`segmenter`/`prompt`/`extract`/`trace` 与内核的依赖方向单向化；加一条机器可检的「内核层不 import torch/vLLM」门禁 | 零行为变化 | 新增静态检查 + CPU 测试全过 |
| **P2** 门禁归位 | vLLM 部署门禁从 `step_plan` 移适配侧；内核仅保留框架无关校验 | 零行为变化（拒绝路径语义不变） | 三类不支持配置的拒绝用例仍成立（CPU + 真实入口） |
| **P3** 分发（可选） | 内核加标准打包（`pyproject` + wheel），与现有 manifest+哈希部署**共存并互相核对** | 无运行时变化 | 无 vLLM 环境可 `pip install` + 跑 CPU 测试 |
| **P4** 第二后端准备 | **仅在选定真实第二后端后启动**：先出「接入点对照表」（几何来源 / metadata 落位 / 块表格式 / 门禁 / hook），再决定是否需要最小 SPI | 无 | 对照表逐项有源码依据；**在此之前不引入注册表或基类** |

阶段顺序理由：P0–P2 是「把已存在的边界显式化」，风险最低、收益最高；P3 是分发改善；
P4 在缺第二个真实后端时做就是猜抽象。

## 5. 每阶段统一验收

1. `bash verify-runtime.sh` 全绿。
2. CPU 层测试全过。
3. `tools/p2-apply-patch.py` 的 apply / verify / revert 指纹一致，journal 归属正确。
4. 内核能在不 import vLLM、不 import torch 的解释器里跑通 CPU 层测试。
5. 证据与日志按既有约定**保留旧名不改写**（沿用 `da-runtime → attnview` 的处理方式）。

## 6. 明确不做

- 不做 `backends/` 注册表、`FrameworkAdapter` 基类、空实现的「其他框架」目录
  （只有一个真实后端时是 phantom abstraction）。
- 不把 `gpukv` 当探针移出内核（它是生产路径）。
- 不把 vLLM 专用门禁冻结成内核公共 API。
- 不合并/改变现有 manifest + 哈希部署链（它是证据与 custody 的骨架）。
- 不改协议、阈值、评测合同与发布形态（暂不上 PyPI）；这些属用户决定。

## 7. 待用户决定

1. **包形态**：先不打包（只做 P1/P2 边界自理）／内核单包／内核 + 适配两包。
   建议：**先不打包**，因为开发未完成，P1/P2 已能拿到全部可读性收益。
2. `extract` 与 `trace` 的归属（服务层 or 内核）。
3. SGLang 是否列为第二后端目标、P4 是否启动、何时启动。
4. 执行时机：建议与改名**同批**在阶段边界后执行（两者都触碰 `src/attnview/`、`ATTNVIEW_*`、
   `manifest.json` 哈希，分两次做等于把同一片区域改两遍）。

## 8. 未核验项

- 未逐项核查 GitHub 其余 168 条 `peephole` 同名命中；未查 npm/crates/HF 与商标库。
- P1 的「内核不 import torch/vLLM」静态门禁尚未实现，能否覆盖全部内核模块待实测。
- SGLang 侧接入点（几何来源 / 块表格式 / 门禁 / hook）**未做任何源码核对**，P4 前不得据此估算工作量。
- `trace` 的运行期归属与其是否应进部署集，未做依赖分析。
