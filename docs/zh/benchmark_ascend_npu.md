# 昇腾 NPU 性能基准

本页持续跟踪 **TeleFuser** 与 **vLLM-Omni** 在华为昇腾 NPU 上的同底座端到端性能与数值精度对齐结果。
每一条记录均显式标明 **芯片具体型号**（区分 `Ascend 910B2 (64GB)` 与 `Ascend 910B4 (32GB)`）、**推理卡数**、
**并行拓扑**、**CPU 架构**与 **CANN 版本**。

容器构建与运行方式参见[《昇腾 910B 镜像》](ascend_910b.md)，
跨硬件算子分派边界参见[《硬件平台》](platforms.md)。

## 已验证同底座对标结果

热态延迟取同一容器底座与权重版本下「1 次冷启动预热 + 连续 3 次热请求」的中位数（括号内为 DiT 去噪阶段耗时）。

| 模型 / 任务 | 芯片规格与卡数 | 并行策略 | 架构 / CANN | 工作负载 | TeleFuser 热态 (DiT) | vLLM-Omni 热态 (DiT) | 相对 vLLM-Omni | 单卡 HBM 峰值 | 质量门禁 |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `aarch64` · CANN `9.1.0` | 864x480, 121f, 50 steps | **98.90 s** (90.4 s) | 106.95 s (96.0 s) `v0.23.0rc1` | **1.08x (-7.5%)** | 16,405 MB | PSNR 31.39 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **85.51 s** (78.0 s) | 101.57 s (93.0 s) `v0.34.0` | **1.19x (-15.8%)** | 17,805 MB (vs. 24,475 MB) | PSNR 26.89 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **2× Ascend 910B2 (64GB)** | CFG2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **46.67 s** (39.1 s) | 56.01 s (47.5 s) `v0.34.0` | **1.20x (-16.7%)** | 18,278 MB (vs. 24,906 MB) | PSNR 99.00 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **4× Ascend 910B2 (64GB)** | CFG2 x Ulysses2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **33.85 s** (26.3 s) | 38.66 s (29.8 s) `v0.34.0` | **1.14x (-12.4%)** | 19,039 MB (vs. 25,814 MB) | PSNR 21.65 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **8× Ascend 910B2 (64GB)** | CFG2 x Ulysses4 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **20.66 s** (13.1 s) | 25.84 s (16.5 s) `v0.34.0` | **1.25x (-20.0%)** | 18,153 MB (vs. 25,792 MB) | PSNR 21.95 dB, Finite |

## 待回填验收矩阵

| 模型 / 任务 | 芯片规格与卡数 | 并行策略 | 架构 / CANN / 底座镜像 | 标准工作负载 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MiniMax-H3` (T2VA) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **2× Ascend 910B2 (64GB)** | Resident TP2 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **4× Ascend 910B2 (64GB)** | Resident Ulysses4 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **8× Ascend 910B2 (64GB)** | Resident Ulysses8 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |

## 五阶可复现验收门禁

上表数据统一由 `benchmarks/ascend_npu/` 下的自动化验收脚手架产出，并通过
`python benchmarks/ascend_npu/render_board.py` 自动校验脱敏与硬件规格字段后生成，禁止手工填数：

1. **G0（设备与通信微基准）**：核验逻辑卡映射、NPU 融合算子、HCCL `all_reduce` / `all_to_all_single` 带宽与主存张量队列。
2. **G1（单卡功能与真值对齐）**：核验编码前张量无 `NaN/Inf`、`ffprobe` 音视频流契约（含音视频模型 32 kHz 立体声同步）及 `PSNR` / `MAE`。
3. **G2（多卡扩展与多实例隔离）**：核验多卡并行相对单卡同 seed 的输出一致性，以及多实例并发 HCCL 16 端口锁隔离。
4. **G3（热态延迟与显存剖析）**：在同一 `quay.io/ascend/vllm-omni` 容器底座上执行 `1 Cold + 3 Warm` 计时与各卡 HBM 峰值采样。
5. **G4（无状态盲跑门禁）**：通过 `env -i` 清空自定义环境变量、不挂载宿主机源码目录，独立执行生成的 `REPRODUCE.sh` 验证第三方一次跑通。
