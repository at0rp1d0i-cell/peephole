"""attnview 运行时实现的 CPU 侧协议层（阶段 03）。

模块职责：

- `segmenter`：C1 输入侧分段（字符偏移空间，精确无损划分）。
- `prompts` / `prompt`：C2 三臂 prompt 组装与最终 token-span 映射。
- `parser` / `state`：C3 增量状态机与请求级控制状态（含 C6 异常与 trace 字段）。
- `readview`：C4 读取视图构造（纯函数）与后端参数边界校验。
- `reference`：**独立**逐位置可见集合参考（不由 `readview` 生成期望值）。
- `extract`：方案 §4.1 的最终答案提取与规定标签机械过滤。
- `trace`：逐 step 轨迹（协议 + 视图 + 写入位置）。
"""

__all__ = ["segmenter", "prompts", "prompt", "parser", "state", "readview", "reference", "extract", "trace"]
