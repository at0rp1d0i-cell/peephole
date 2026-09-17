# 依赖清单变更记录（阶段 02 会话内的 CUDA 修复）

本文件记录**相对阶段 01 冻结清单**的差异，符合"保留修复前记录"的要求。
冻结快照（均在远端 `evidence/p0-model/`，内容逐字节保留）：

| 快照 | 文件 | 含义 |
| --- | --- | --- |
| 修复前 | `cuda-upgrade-freeze-before.txt` | 用户授权修复前的原始状态（阶段 01 的结果） |
| 中间态 | `cuda-upgrade-freeze-after.txt` | 仅把 `nvidia-cuda-runtime` 升到 13.4.92 之后 |
| **最终态** | `cuda-upgrade-freeze-final.txt` | 全部已授权 CUDA 修复完成后的权威清单 |
| 交付副本 | `../p0-model/requirements.freeze.txt` | 最终态在素材仓中的副本（197 行） |

## 差异（修复前 → 最终态，逐行）

```
- nvidia-cuda-cupti==13.0.85        + nvidia-cuda-cupti==13.4.92
- nvidia-cuda-nvrtc==13.0.88        + nvidia-cuda-nvrtc==13.4.92
- nvidia-cuda-runtime==13.0.96      + nvidia-cuda-runtime==13.4.92
```

- 行数不变：before 197 行 / final 197 行；仅上述 3 行变化，**没有任何新增或删除的包**。
- `pip check`：`No broken requirements found.`（修复前与修复后均通过）。
- **未改** ：`nvidia-cuda-nvcc==13.4.92` 与 `nvidia-cuda-crt==13.4.92`（本来就是 13.4.92）。
- `nvidia-cuda-cccl==13.3.4.3.1` **保持不动**：上游 redist 的 cccl 最高只有 13.3.4.3（本机装的 13.3.4.3.1 是同一版本的构建后缀），不存在 13.4.x，无法"统一"到 13.4。
- torch / vllm / triton / transformers / flashinfer 等**全部未动**。

## 复现命令（项目 venv 内，仅动 CUDA 组件）

```bash
source /root/attnview/env.sh
"$ATTNVIEW_PYTHON" -m pip install --upgrade \
  "nvidia-cuda-runtime==13.4.92" "nvidia-cuda-nvrtc==13.4.92" "nvidia-cuda-cupti==13.4.92"
"$ATTNVIEW_PYTHON" -m pip check
bash /root/attnview/setup-local-cuda.sh    # 第 4 步会做版本一致性校验 + 补 dev 链接/stub + 链接自检
```

## 为什么必须成组对齐

`nvidia-cuda-nvcc` 的 `Requires-Dist: nvidia-cuda-runtime` **未锁版本**，所以 pip 允许出现
"nvcc 13.4 + cuda.h 13.0" 这种组合；而 flashinfer 内置 CCCL 会做编译期兼容性检查
（`cuda/std/__cccl/cuda_toolkit.h:41`，要求编译器 minor 与头文件 minor 相同），随后 JIT 链接
又需要轮子里缺失的 `libcudart.so` 与驱动 stub。三处（runtime 版本、dev 链接、stub）
已全部写进 `setup-local-cuda.sh` 第 4 步并在脚本里做硬校验，避免再次静默漂移。
