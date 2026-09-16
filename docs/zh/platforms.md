# 硬件平台

TeleFuser 通过 `telefuser/platforms/` 中的平台层屏蔽底层硬件差异。每个进程在导入时解析出唯一的
`current_platform` 对象，[算子](ops.md)分发层根据它选择具体实现。Pipeline 与示例只需配置
`torch.device` 风格的设备，无需针对厂商编写分支。

## 平台选择

`telefuser/platforms/__init__.py` 中的 `_resolve_current_platform()` 按固定顺序探测环境，并实例化第一个
命中的平台：

1. **ROCm** — PyTorch `+rocm`（HIP）构建且至少有一块可见的 AMD GPU
2. **CUDA** — CUDA 构建且至少有一块可见的 NVIDIA GPU
3. **NPU** — 安装了 `torch_npu` 且有可见的昇腾设备
4. **CPU** — 未检测到加速器时的回退

HIP 构建同样暴露 `torch.cuda` API，因此 ROCm 平台复用它（`device_type` 仍为 `cuda`）；可通过
`torch.version.hip` 区分 ROCm 主机与 NVIDIA 主机。HIP 或 CUDA 构建在没有可见 GPU 时同样回退到 CPU。
两块 GPU 平台通过 `CUDA_VISIBLE_DEVICES` 控制设备可见性，NPU 使用 `ASCEND_RT_VISIBLE_DEVICES`。

## 平台矩阵

| 平台 | `device_type` | 分布式后端 | 算子分发 | `tf-kernel` | torch.compile |
|------|---------------|------------|----------|-------------|---------------|
| CUDA（NVIDIA） | `cuda` | NCCL | 优化的 `forward_cuda` 路径 | 支持 | 实验性（[详情](torch_compile_compatibility.md)） |
| ROCm（AMD） | `cuda` | RCCL（`nccl`） | Triton `forward_cuda` 路径与原生回退 | 不支持 | 未验证 |
| NPU（昇腾） | `npu` | HCCL | 原生回退 | 不支持 | 未验证 |
| CPU | `cpu` | Gloo | 原生回退 | 不支持 | 原生路径 |

## 各平台的注意力后端

注意力后端可用性在导入时解析（见[注意力机制](attention.md)）。依赖缺失的后端会带一次性警告回退到
`TORCH_SDPA`。

- **CUDA**：全部稠密后端 —— `TORCH_SDPA`、FlashAttention 2/3/4、经 `tf-kernel` 或 `sageattention` 提供的
  SageAttention，以及 cuDNN —— 取决于 GPU 架构；稀疏后端同理。
- **ROCm**：仅 `TORCH_SDPA`。`flash_attn` 没有面向消费级 RDNA GPU 的官方 ROCm 轮子，`tf-kernel`、
  SageAttention 与 SpargeAttn 仅支持 CUDA。AOTriton 支持的 SDPA 路径是 RDNA4 上的快速注意力内核。
- **NPU / CPU**：通过原生回退路径使用 `TORCH_SDPA`；未集成厂商注意力内核。

## CUDA

CUDA 是主要的已验证路径：Python 3.10–3.13、PyTorch 2.6 及以上、CUDA Toolkit 12.8 及以上，优化内核以
H100 为验证目标。可选的 `tf-kernel` 提供融合逐元素算子、量化 GEMM、SageAttention 与块稀疏注意力 ——
构建与制品兼容性见 [tf-kernel](tf_kernel.md)。多卡推理使用 NCCL，参见[并行推理](parallel.md)。

## ROCm

ROCm 支持面向使用 ROCm 7.x 与 PyTorch `+rocm` 构建的 AMD GPU；安装路径见[安装指南](installation.md)。

- 注意力使用 `TORCH_SDPA`；无需安装 `tf-kernel`、`flash_attn` 或 `sageattention`。
- 算子层在内核定义了 `forward_rocm` 时优先选择，否则复用 CUDA Triton 路径（Triton 支持 ROCm），并回退到
  PyTorch 原生实现。
- `tf-kernel` 导入被限定在 `CudaPlatform`，ROCm 主机不会加载任何 CUDA-only 扩展。
- 多卡推理使用 RCCL（AMD 与 NCCL 兼容的集合通信库）。PyTorch 的 ROCm 构建通过 `nccl` 后端字符串暴露它，
  平台层无需特殊配置。
- `torch.compile` 在 ROCm 上未验证；ROCm 示例以 eager 模式运行。
- 已验证入口为 `*_rocm.py` 示例，例如在 Radeon RX 9070（gfx1201，ROCm 7.2）上运行的
  [Wan2.1 1.3B 文生视频](https://github.com/Tele-AI/TeleFuser/tree/main/examples/wan_video)。多卡分支复用
  `_h100.py` 的并行配置，但尚未在 ROCm 上验证。

## NPU 与 CPU

NPU 平台通过 `torch_npu` 与 HCCL 分布式后端支持华为昇腾设备。平台层与算子分发层均已接入 NPU，但现有
示例在 CUDA 上验证、部分示例在 ROCm 上验证 —— 生产使用前请先在目标 NPU 上完成验证。CPU 平台是未检测到
加速器时的回退，面向测试以及显式请求 CPU 执行的 Pipeline。
