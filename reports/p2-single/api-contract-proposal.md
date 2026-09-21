# attnview 服务端 API 契约提案（最小版，未冻结）

日期：2026-09-21。性质：设计提案，**不冻结**端点/字段/阈值，不改 `protocol-contract.md` v1.0，不替换 `project-plan.md` §4.1 的职责边界。触发：用户提出"本机无 Docker，但可以把 API 转发出去"，让容器验收侧与模型侧分离。

## 0. 为什么需要这一层（而不是直接暴露 `vllm serve`）

直接暴露原版/补丁版 `vllm serve` 的 `/v1/chat/completions` 只能得到原版模型输出，且调用方必须自己写协议 prompt、自己切 segment、自己解标签——等于把机制交给外部。`project-plan.md` §4.1 已定：调用方只接收最终结果，不接触内部注意力开关标签、答案包装与执行 trace。因此需要一个**薄封装**：外部只送任务输入，协议渲染与载荷注入在服务端完成。

## 1. 拓扑

```
外部客户端(Docker harness / 人工)
        │  HTTPS（平台端口映射）
        ▼
薄封装服务  0.0.0.0:6006        ← 本提案的唯一新增件
        │  HTTP 127.0.0.1:8000（含 vllm_xargs）
        ▼
vllm serve（版本锁定 patch 已部署）→ 引擎内部 attnview runtime
```

- 6006 = 候选（patched）。**6008 保留**给分时启用的 original 对照（单卡 96 GB 装不下两个 27B：51.75 GiB × 2 > 96 GB）。
- 薄封装只做四件事：协议 prompt 渲染、载荷注入、输出提取、记账。不做调度、不做流式。

## 2. 已核实的注入通道（源码级）

| 事实 | 位置 |
| --- | --- |
| OpenAI chat 协议含 `vllm_xargs` 字段 | `vllm/entrypoints/openai/chat_completion/protocol.py:492` |
| `vllm_xargs` → `SamplingParams.extra_args` | 同文件 `:705`、`:746` |
| `extra_args` 语义为"供自定义采样实现/插件使用" | `vllm/sampling_params.py:345-348` |
| 校准脚本已用同一通道注入 `{"attnview": payload}` | `tools/p2-calib-run.py`（`extra_args`） |

**实现约束（必须遵守）**：`vllm_xargs` 的类型是 `dict[str, str | int | float | list[str|int|float]]`（`:492-500`），**不能嵌套对象**。segment span、chunk 列表等结构必须序列化成 JSON 字符串（或 base64）再放入。

**未验证项（不得当作已可用）**：`vllm_xargs` 经 HTTP → `EngineCoreRequest` 跨进程 → worker 的**序列化是否保真**未实测（stage-05 §2 已记录该口径）。首个冒烟必须比对"请求里发的 JSON"与"worker 收到的 `extra_args`"逐字节相等。

## 3. 最小端点集

| 端点 | 作用 | 首版 |
| --- | --- | --- |
| `POST /v1/attnview/generate` | 任务输入 → 协议渲染 → 生成 → 提取答案 | ✅ |
| `GET /v1/models` | 健康检查 / 模型标识（含 revision） | ✅ |
| `POST /v1/attnview/detokenize` | 长度核验与调试用（可选） | 可选 |
| streaming / `/v1/chat/completions` 直通 | 首版不做 | ❌ |

请求体（字段最小集）：

```json
{
  "context": "<任务上下文原文>",
  "question": "<问题>",
  "arm": "da | da_no_mask | vanilla",
  "max_tokens": 512,
  "enable_thinking": false,
  "request_id": "外部可读 id"
}
```

- `arm` 是**受控枚举**，不是自由字符串；`vanilla` 用于对照臂。
- 服务端用阶段 03 入口 `render_arm(arm, segments, question, context, tokenizer, enable_thinking=False)` 渲染，segment 由 `segment_context` 切分——**不接受调用方自己的 segment 划分**。
- 响应：`{"content": "<提取后的答案>", "usage": {...}, "attnview": {...}}`；`content` 中不得出现 `<global>`/`<focus>`/`<local>`/`<answer>` 语法（按 §4.1 机械过滤完整标签，同名普通文本列为已知限制）。

## 4. 记账字段（与用户要求的三类时间对齐）

| 字段 | 含义 |
| --- | --- |
| `usage.prompt_tokens` / `usage.completion_tokens` | 引擎真实计量（**含**被过滤掉的协议标签 token） |
| `usage.attnview_protocol_tokens` | 协议包装本身的开销（实测基线：`da`/`da_no_mask` **1,434**、`vanilla` **210**，最小 context 13 token 时） |
| `timing.model_ms` | 本轮模型执行（prefill + decode，分开记 `prefill_ms` / `decode_ms` 与 `decode_tokens`） |
| `timing.tool_ms` | 由调用方填回（容器/测试/编译），服务端不测 |
| `attnview.modes_used` | 本请求是否真的用了 focus/local（分子/分母口径按 plan §5，无调用记 null 不记 100%） |

## 5. 错误与异常语义

| 情况 | 行为 |
| --- | --- |
| 载荷非法 / 无可用 segment | 按合同 C6.2 **保持当前模式**、记 trace、计失败；HTTP 仍 200（结果可用性由 `attnview` 字段暴露），不得静默回退 global |
| 请求超过 `max_model_len` | 显式 4xx + 明确字段（不允许引擎侧截断后当成功） |
| 上游 vLLM 异常 | 透传状态码，附 `request_id` |
| 未授权 | 401（Bearer key） |

## 6. 安全与边界

- 薄封装绑定 `0.0.0.0:6006` 仅为适配平台端口映射；**必须**启用 `Authorization: Bearer <key>`（vLLM 侧 `--api-key` 同步启用）。
- 只暴露生成能力；**不**暴露权重、原始 trace、内部模式明细（`attnview` 字段只给计数与模式名，不给 block 表）。
- 已知风险：公网地址无来源限制（平台映射），需配合限流与日志脱敏；请求内容可能被他方观测。

## 7. 明确不做（首版）

并发（`max-num-seqs=1` 语义）、prefix caching、streaming、多轮会话状态、抢占语义、`>8192` 的上下文（等长上下文放行）、把内部 trace 对外。
