# 首次 masked GPU 诊断：聚焦启动审查

用户于 2026-09-21 直接将启动审查交给当前实现者，不再等待旧本地主代理 READY。旧 OMP 开发会话保持停点；只另开过一个只读 OMP 会话核对部署依赖。

## 首次启动决定与依据

首次实现提交为 `bfa728a`，五个任务文件显式暂存、提交，核对 `git show --stat` 和空的任务路径 diff 后才测试。启动入口补上实际输出 token 与固定轨迹比较后，运行提交为 `ba05322`。

- CPU 回归 85 项通过，原始命令、HEAD、退出码、日志在 `evidence/p3-masked-smoke/cpu-20260920T170715Z/`。真实 capture begin/end 流程覆盖 FA 组非零、非连续物理块、上一轮过期 execute state、后续缓冲覆写；15 种独立篡改均失败。实际 `_run` 在桩模型/引擎下正常返回 0，长度篡改返回 1，仍执行 cleanup。这些是 CPU 接线证据，不是 GPU 验收。
- 独立预期只取声明 token 分段与原始 renderer spans；现场 canonical 表仅提供物理分配。逐 forward、逐真实 FA 层保存位置、读取长度/块行、当前 forward context 写槽。请求、层、步骤、trace 缺失即失败；受限 view 覆写与 enforce_global 标记分别检查。
- 旧三臂保持输入与 canonical/global 断言。original 无补丁，disabled 无载荷，global 带载荷且 enforce_global=True；masked 带载荷且 False，单独执行精确结构门禁。
- 部署 manifest 的七个 attnview 包文件覆盖两个生产 adapter 及其传递依赖。宿主诊断从仓库 `src/` 加载，不安装 reference 替换 hook。生成器 `--check` 通过。初始无部署 journal；未部署时 verify 的退出 0 不当作已部署证明。事务执行 apply→verify→确认 deployed 状态→运行→finally/revert。
- 固定配置 `configs/p2-masked-smoke/diagnostic.json` 使用真实 7834-token prompt、29 采样/28 消费、decode 6/7/20/25 捕获点。输入哈希及轨迹逐 token 等于声明 script 加终止 token 的检查在 `launch-inputs.json`。
- vLLM pin 为 `98dff2a81d747d1dba01a47f939f48c3526d4206`，模型 revision 为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。BF16/TP1/显式 FA2、eager、同步、单活跃请求、8192 总长/批 token 预算、关闭 prefix caching、无投机。启动/主请求/cleanup 上限 900/180/120 秒不变。
- labops 快照 `20260921T005712+0800-07ebdb54ec27`：单张 RTX PRO 6000 Blackwell 96GB，驱动 580.95.05，CPU 配额 22 核，内存上限 110 GiB。启动前无 GPU 计算进程。空间预检 122.82 GiB 可用，大于预计输出 3 GiB 加 5 GiB 预留；无历史 labctl 基线，不作性能可比性声明。

## 失败与同故障窄修复

首次尝试 `diagnostic-20260921-first/` 部署/校验成功，模型实际使用 FA2、块大小 784，但在 decode 6 进入 FA forward 前失败：trace 访问 inference tensor 的 `_version`。用 CPU `torch.inference_mode()` 独立复现同一错误。原始 `error.txt`、forward 1–6 捕获全部保留。

首次 revert=0，27 个目标恢复，无 journal/残留进程，GPU 0 MiB；当次进程内生命周期 cleanup 未执行，不把退出清理等同于请求取消验收。

按工作单允许的同故障窄修复，`01ad171` 将不存在的版本计数记为 null/unobservable，不改候选读写算术；驱动补充请求异常时部分捕获/engine trace 落盘与有界取消。部署 manifest 仅生成时间与 adapter 哈希更新。

修复后 101 项受影响 CPU 检查通过：hooks、structure、adapter、patch_layout、committed_state。原始件在 `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/`，包括真实 trace 在 inference mode 下的回归与 `_run` 异常清理。其他 smoke suites 的首轮证据继续适用，不把两轮重叠测试数量相加。

修复后启动目录为 `diagnostic-20260921-retry1/`，模型、输入、协议、预算不变。只完成这次修复后诊断，无第三次启动。

## 可复跑入口与边界

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-diagnostic-run.py \
  --config configs/p2-masked-smoke/diagnostic.json \
  --out evidence/p3-masked-smoke/diagnostic-<new-id>
```

必须使用新目录。事务记录实际命令、阶段退出码、PID、HEAD、配置哈希；驱动保存全部 attnview 源码快照与输入哈希。prefill 只捕获一次 KV，每 decode 只捕获新增 KV，Q/out 只在四个代表点保留；合并 archive 会重复存储磁盘证据，但不重复从 GPU 复制历史 KV。

这是带同步的诊断，不用于性能结论，不设数值通过线。完整独立 masked attention/logits 参考、质量和净收益仍未验收。原始结果审计与 closeout 后停止扩展。
