# 独立 masked decode 参考：CPU 接线交付

本轮补齐已有参考组件到 `tools/p2-calib-run.py` 的请求生命周期接线。新增测试专用 `patched-reference` 臂；没有运行新的模型/GPU 实验，也没有设置数值通过线。此前文档的“CPU 实现完成”仅指组件检查，不能作为真实 harness 已接通的证据；本报告取代该完成声明，历史原始产物保持不变。

## 执行边界

| 臂 | attnview 载荷 | FA 执行 |
| --- | --- | --- |
| original | 无，未部署补丁 | 原实现 |
| patched-disabled | 无 | 原实现 |
| patched-global | `enforce_global=True` | canonical/global 原实现 |
| patched-masked | `enforce_global=False` | 候选读取视图 |
| patched-reference | 无 | global 原版 prefill；decode 独立 FP32 dense，cast 后原地写入 output |

旧三臂的默认输入、采样参数、canonical 校准断言保持。masked/reference 两臂强制使用已绑定哈希的 7834-token renderer 夹具、29 次采样/28 次消费、8192 上界、单请求、同步/eager、BF16/TP1 和显式 FA2。参考轨迹额外与真实 tokenizer 分段编码核对：配置声明给出前 28 个消费 token，再重复末 token 作为第 29 个仅采样 token。模式切换仍在 decode 6/12/20/25 生效。

候选与参考分别调用驱动，各自构造新引擎、提交新内部请求。参考臂提交前检查 scheduler 与 `runner.req_states.req_id_to_index` 均为空；不使用可能为空或属于前一步的 `execute_model_state` 判断活动请求。没有把候选 Q、隐藏状态、KV 或 GDN 状态送入参考臂。真实 GDN 初始化/传播仍须 GPU 验证；CPU 替身的状态隔离测试只验证驱动生命周期。

## 接线与独立性

`ReferenceRun` 只服务当前诊断，不进入生产 adapter。`prepare_inputs` 捕获当前 InputBatch，每次 `model.forward` 都要求新的 prepare 记录、唯一目标内部 ID、正确相位和真实逻辑位置。prefill 必须完整覆盖 prompt；decode 只消费一个 token。forward、decode、生成 token 下标分别为 `f`、`f-1`、`f-2`。

FA 集合来自真实 `FullAttentionSpec`。每层 forward 内记录独立快照，比较实际 metadata 与当前 cache group 的 canonical 有效前缀，同时核对 forward context 中该真实层名的写槽。padding 不进入参考读取；canonical 前缀不得在请求期间变化；本步 K/V 必须已经写入对应物理槽。Q 观测点仍是模型 norm/RoPE 之后、真实 FA impl 入口。

独立 mask 只取声明配置、真实 token 分段、renderer 原始 spans 和当前 KV 上界。语义位置按块外扩，再截到当前有效长度；不调用候选筛选/压缩 helper。原生 KV 保留 stride 视图，按可见位置批量 gather，不复制整份 cache。GQA 按连续 query-head 分组映射 KV head，使用真实 `impl.scale`，FP32 显式 QK/softmax/V 后 cast 到真实 BF16 output。

pin 的 `unified_attention_with_output` 忽略 impl 返回值，继续消费传入 output。hook 用 NaN 哨兵检出遗漏写回，dense 入口原地 `copy_`。日志输出真实 cast 误差、参考幅度和非有限计数，不用自比较宣称数值正确。CUDA 路径要求 TF32/autocast 关闭；非 BF16、ALiBi、sliding window、soft cap、KV sharing、sinks、DCP、非 FA2、cascade、特殊 mask 和未知参数均拒绝。pin 的 `supports_quant_query_input` 表示能力，不能单独作为已启用量化的判据；正常的 `output_scale=None, output_block_scale=None` 允许通过。

覆盖账本逐层、逐步、逐请求检查：prefill 为 passthrough，所有 decode 为 overrode。漏层、漏步、重复记录、错 ID 或错误 action 均失败。`_run` 把覆盖结论写入 manifest，失败返回非零；forward 异常保留原错误与部分证据，并执行有界取消。参考与捕获方法在 finally 恢复，恢复完成后才进行额外 cleanup 请求。源码冻结包含全部 `src/attnview/*.py` 与 driver 的公共工具依赖 `_lib.py`，结束时复核哈希。

## CPU 证据与复跑

交付证据：`evidence/p3-calib/reference-integration-20260921/summary.json`。提交前 CPU 验证绑定 HEAD `84116a2` 的 11 个源码/配置文件 SHA256；实现提交为 `ea7a2ad`，测试提交为 `1f9c1aa`，两者均通过同一版本化 pre-commit 门禁。验证结束时源码哈希未变。受影响 unittest **99 项通过**，提交后桥接检查 **32/32 通过**（`bridge-checks-postcommit.json`）。提交前工作区的七项门禁全部 exit 0；完整 pytest 收集 355 项，14 条既有 torch JIT 弃用警告。`precommit-gates.log` 明确属于提交前工作区验证，不冒充某个纯净历史树的运行。

开发期间日志已在独立协调目录保留，其中早期失败包括测试夹具的轨迹字段遗漏、28/29 边界错误、lazy import 的模块隔离问题，以及旧 mRoPE 测试重复包装同一模型造成的诊断覆盖。完整 pytest 的一次失败为并行暂存的两个生产文件与 manifest 尚未同步；同步后原断言通过。Watchdog 子进程补齐工具模块导入路径，原退出码 3 与结束时间断言保留。开发工作区测试与提交后验证分别记录。

```bash
source env.sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:tools:tests python -m unittest \
  tests.test_p2_reference_run tests.test_p2_calib_hooks \
  tests.test_p2_smoke_structure tests.test_p2_smoke_bounded_capture \
  tests.test_p2_smoke_llm_kwargs tests.test_p2_smoke_cleanup_checks \
  tests.test_p2_smoke_fixture -v
CUDA_VISIBLE_DEVICES= python tools/p2-ref-cpu-checks.py \
  --config configs/p2-ref-cpu/checks.json --out /tmp/attnview-reference-checks-new.json
bash tools/gates.sh
```

新测试通过实际 `_run → submit_request → drive_main_request → prepare_inputs → model.forward → FA wrapper` 控制流，边界引擎/模型为 CPU 替身。两个独立新请求各走 29 个 forward；首请求缓存和循环状态被污染后，不影响第二请求结果。故障注入在第三个 forward 第二层破坏读取长度，检查部分证据、取消与原方法恢复。另有小张量算术真值检查：用 NaN 填充排除块和未写尾槽，以零 query 的均匀 softmax 验证 GQA、块外扩、当前 token 与原地输出。

捕获仍有界：一次 prefill K/V、每 decode 增量 K/V、decode 6/7/20/25 的 Q/out，以及全部步骤结构元数据；不保存完整 prefill Q/out。

## 未验证与停点

没有新 GPU 运行，未验证真实 27B 参考 hook 的覆盖、原地替换后的 GDN 传播、独立 attention/logits 误差、缓存内容真值、质量或净收益。CPU `_run` 测试不能代替这些证据，也不是 masked 数值验收或 P2 通过声明。首次 masked smoke 的旧结论与原始 SHA 保持，历史重写通过既有映射追溯。

本交付只完成参考 CPU 接线。后续真实运行必须另做启动审查、apply/verify/run/revert 和独立数值验收；本轮不执行该实验、不推送、不打阶段 tag。

部署只读检查 `python tools/p2-apply-patch.py verify` exit 0：未部署，无事务 journal。本轮只有已退出的 CPU 验证子进程，没有创建模型/GPU 进程，也没有修改部署目标。
