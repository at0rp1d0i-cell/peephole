# 首次真实 masked GPU 诊断闭环

本检查点已完成：最小实现/配置先提交，受影响 CPU 回归、聚焦启动审查、真实 GPU 请求、失败审计及部署恢复均有原始证据。共两次尝试：首次在诊断 trace 中失败；同故障窄修复后，固定轨迹的完整诊断退出 0。保留两次目录，不将重试写成首次成功。

## 实现与稳定源码

- `bfa728a`：逐 forward/逐真实 FA 层结构快照、独立预期门禁、真实 `_run` 接线、CPU 反例、固定配置与部署事务入口。
- `ba05322`：启动审查及宿主实际 token 与固定轨迹的比较；首次失败尝试的运行源码。
- `01ad171`：inference tensor trace 窄修复、请求异常证据/取消路径、离线数值观察工具；成功 GPU 请求和数值审计使用此已提交源码。

用户并行进行 Git/发布整理，HEAD 在工作中正常前进。没有 reset、checkout 旧 SHA、改写历史、修改旧实验 manifest、删除失败件或模型资产；用户文档改动未夹带进任务实现提交。旧 SHA 查 `reports/git-history-map-20260921.txt`。

## 验证与原始件

| 内容 | 结果 | 仓库内路径 |
| --- | --- | --- |
| 首轮受影响 CPU 回归 | 85 项通过，exit 0 | `evidence/p3-masked-smoke/cpu-20260920T170715Z/` |
| 同故障修复后 CPU 回归 | 101 项通过，exit 0；与首轮部分重叠 | `evidence/p3-masked-smoke/cpu-retry1-20260920T171516Z/` |
| 收尾提交门禁 | pytest 340 项、152 个 subtests 通过，exit 0 | `evidence/p3-masked-smoke/closeout-gates-20260921/` |
| 首次 GPU 尝试 | exit 1，prefill + decode 1–5 已捕获 | `evidence/p3-masked-smoke/diagnostic-20260921-first/` |
| 修复后 GPU 诊断 | apply/verify/gpu/revert 全部 exit 0 | `evidence/p3-masked-smoke/diagnostic-20260921-retry1/` |
| 局部 FP32 数值观察 | 64 个层/代表点，exit 0，无数值通过线 | 同上 `numeric-audit.json`、`.log`、`.exit` |
| 部署恢复 | 27 目标恢复，原版逐文件指纹核对通过 | 同上 `restoration.json`、`revert.log` |

运行目录中的 `transaction.json` 保存命令、HEAD、PID、各阶段退出码；`run/manifest.json` 保存部署指纹、源码/输入哈希、请求身份、实际参数、结构判据、非有限值、cleanup 和失败项。`run/source/` 是当次源码快照。二进制捕获、logits 留本机，大小/SHA256 见 `artifact-index.json`，不入 Git。

CPU 反例包括长度、物理块、slot、位置、请求、层、步骤、删除/缺失/错位 trace、强制 global 标记、错误模式。真实 `_begin_step/_end_step` 使用非零 FA 组与非连续物理分配；缓冲复用后历史快照不变。真实 `_run` 桩引擎覆盖正常返回 0、结构失败返回 1 后 cleanup、请求异常时保存部分证据并取消。GPU 前沿用项目 unittest；用户后来明确要求 pytest 提交门禁，收尾时补齐固定开发依赖并用 pytest 收集原有用例。没有重跑旧四臂全量校准或 24192 项 oracle。

## 真实结构结果

真实模型为固定 Qwen3.8-27B revision、vLLM `98dff2a8`；BF16/TP1/显式 FA2、同步/eager、单请求、总长≤8192，prefix caching/投机/CUDA Graph 关闭。实际 FA cache group=3、块大小=784、真实 FA 层集合=16 层。

7834-token prompt 经重建及哈希核对。29 个采样 token 与固定轨迹逐项一致，其中前 28 个被后继 forward 消费；总计 prefill + 28 decode = 29 forward，每步全部 16 层均有结构快照。模式轨迹按预期为 global→local→global→focus(2)→global，解析位置 5/11/19/24 对应 decode 6/12/20/25 生效。

| decode | 模式 | 原逻辑位置 | 实际读取长度 | 实际物理块行 | canonical 写槽 |
| --- | --- | ---: | ---: | --- | ---: |
| 6 | local | 7839 | 2352 | 4,12,13 | 10975 |
| 7 | local | 7840 | 2353 | 4,12,13,14 | 10976 |
| 20 | focus(2) | 7853 | 4718 | 4,7,8,9,12,13,14 | 10989 |
| 25 | global | 7858 | 7859 | 4,5,6,7,8,9,10,11,12,13,14 | 10994 |

decode 7 的 KV 长度 7841，写入逻辑块 10 的第一个位置，读取尾块计数为 1。decode 25 恢复完整 canonical 行。每一实际层/步的读取长度、物理映射、原位置及写槽均与独立预期相等。全部 local/focus 覆写步骤有实际 trace；enforce_global 标记为空。结构门禁无失败项。

快照见 `run/capture/structure.json`；独立预期与逐项比较结果在 manifest 的 `masked_structure`。canonical 分配取自本步 runtime，独立读取集合取自声明配置、真实 token 分段及原始 spans，不从候选 view 反推。

## 数值观察与边界

所有 FA forward 的实际 Q/K/V/attention 输出非有限值计数为 0；29 份完整 logits 非有限值为 0，最大绝对值为 28。捕获保留全部 16 层的 prefill KV 与逐 decode 新增 KV，Q/out 仅保留 decode 6/7/20/25。合并 `layers.npz` 为 1,034,009,400 字节，数组总字节 1,033,701,784；无完整 prefill Q/out。

CPU 使用独立块外扩位置，从候选轨迹捕获的 Q/K/V 计算 FP32 attention，观察全部 16×4=64 个点。下面是各项在这些点上的最大值，不一定来自同一个层/步：

| 比较对象 | max_abs | RMS | relative L2 |
| --- | ---: | ---: | ---: |
| 实际输出 vs FP32 局部参考 | 0.1240158081 | 0.0084787980 | 0.0018260934 |
| 实际输出 vs 转为 BF16 的局部参考 | 0.125 | 0.0061230073 | 0.0014283564 |

这些是无通过线的局部误差观察。Q/K/V 来自候选运行，未执行独立模型分支，未直接逐槽验证 GPU KV 内容，未做独立 GDN 状态对照，也没有独立 masked logits 参考。生产 GDN 路径未修改；这不等于其独立数值验收。完整 attention/logits 参考、模型质量与净收益仍是后续必要工作；不宣布 P2 或 masked 数值通过。

成功尝试启动约 48.6 秒、主请求约 8.11 秒，均在原预算内，但包含捕获同步、JIT/预热影响，不能用于性能宣传。

## 失败与清理审计

首次失败的准确位置是首次 local 覆写记录、进入 decode 6 FA forward 前。`torch.inference_mode()` 创建的 tensor 没有版本计数，访问 `_version` 抛 `RuntimeError: Inference tensors do not track version counter.`。CPU 最小复现与原始 traceback 一致。修复只记 `version=null`、`version_observable=false`；不以空值宣称 canonical/GDN 输入不可变。读写值比较由逐步结构门禁承担。

首次目录保留 6×16 个层/forward 捕获，已观测张量非有限值为 0；缺少受限 forward，不能给出 masked 通过结论。首次未到进程内 lifecycle cleanup，实际依靠进程退出与事务回滚释放；此限制保留在报告。

成功尝试 lifecycle cleanup 全部通过，主请求 scheduler/protocol 状态已释放。两次 revert 都返回 0；成功尝试后原版指纹再次校验，无 `deployed.json`、无 `orig/`，无 GPU 计算进程，显存 0 MiB。日志有 NCCL 未显式 destroy 的退出警告；最终进程/GPU检查未见残留，不将警告隐藏为“无警告退出”。

## 可复跑与停点

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-diagnostic-run.py \
  --config configs/p2-masked-smoke/diagnostic.json \
  --out evidence/p3-masked-smoke/diagnostic-<new-id>
CUDA_VISIBLE_DEVICES= "$ATTNVIEW_PYTHON" tools/p2-masked-smoke-audit.py \
  --run evidence/p3-masked-smoke/diagnostic-<new-id>/run \
  --out evidence/p3-masked-smoke/diagnostic-<new-id>/numeric-audit.json
```

每次使用新目录；完整驱动参数由固定配置展开，并写入 transaction。不改变模型、协议、阈值、预算或资源租用。本次没有 push/tag/公开发布。检查点到此完成，停止扩展，等待用户决定下一步协作。

收尾门禁最初发现 pytest 缺失、文档中扫描命令自匹配；在用户授权继续收尾后，采用等价字符类写法消除自匹配，扫描规则覆盖不变。按 `requirements.dev.txt` 安装三个固定开发包，生产依赖未改变。最终 pytest、脱敏扫描、文件大小及密钥扫描门禁通过。只对用户指定的三条未推送提交重写消息；原始实验 SHA 保留，映射见 `reports/git-message-map-20260921.txt`，重写不改变对应提交树。
