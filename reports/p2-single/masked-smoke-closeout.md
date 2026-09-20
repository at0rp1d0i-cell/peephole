# Session closeout：首次真实 masked GPU 诊断

- 检查点完成并停止扩展。用户直接与当前实现者协作，旧 OMP 会话保持停点；不再等待旧本地主代理派单。
- 稳定实现：`bfa728a`、`ba05322`、`01ad171`；成功 GPU 与 CPU 数值审计源码为 `01ad171`。最终报告/文本证据另有收尾提交，不能将收尾 HEAD 冒充 GPU 运行版本。
- CPU：首轮 85 项通过，窄修复后受影响 101 项通过（有重叠）。真实 `_run` 正常/结构失败/请求异常分支已覆盖。
- 新规范收尾门禁：pytest 340 项、152 subtests 通过；14 条既有 torch JIT deprecation 警告保留。脱敏0命中、密钥扫描为空、最大文件低于10MB。仅新增固定的pytest/iniconfig/pluggy开发依赖。
- GPU：首次因 inference tensor trace 读取 `_version` 失败；原始目录保留。窄修复后完成 7834 prompt、29采样/28消费、29 forward×16 FA 层，结构门禁通过，非有限值为0，真实 cleanup 通过。
- 数值：64 个局部 FP32 观察点，max_abs 最大0.1240158081、relative L2最大0.0018260934；没有设置通过线。完整独立 masked attention/logits、GDN 状态/缓存内容验证、质量和净收益未验收，不宣布 P2 通过。
- 恢复：两次部署回滚均退出0，最终原版指纹通过，无journal/orig、无GPU进程、显存0 MiB。NCCL退出警告保留。
- 入口、精确结果、原始证据、未验证边界见 `masked-smoke-report.md`；复跑工具为 `tools/p2-masked-smoke-run.py`，离线观察工具为 `tools/p2-masked-smoke-audit.py`，均要求新输出路径。
- Git/发布整理由用户并行推进，不回滚、不夹带、不改写旧manifest或已推送历史。未push/tag。用户随后授权消息整改及门禁修复：三条指定提交仅重写消息，文档扫描示例的自匹配已用等价字符类消除；消息重写映射见 `reports/git-message-map-20260921.txt`。后续按新贡献规范用中文消息与报告、正文和四个trailer。
- 下一步仅由用户决定：是否推进完整独立参考及后续协作方案；本会话不自动开始下一阶段。
