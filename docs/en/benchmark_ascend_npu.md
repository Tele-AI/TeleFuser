# Ascend NPU Benchmarks

This page tracks apples-to-apples performance and numerical parity between **TeleFuser** and **vLLM-Omni** on
Huawei Ascend NPUs. Every entry explicitly records the **accelerator SKU** (`Ascend 910B2 (64GB)` vs.
`Ascend 910B4 (32GB)`), **card count**, **parallel topology**, **CPU architecture**, and **CANN runtime**.

See [Run Wan2.2 on Ascend 910B](ascend_910b.md) for container setup and
[Hardware Platforms](platforms.md) for dispatch boundaries.

## Verified Head-to-Head Results

Warm latency reports the median of three consecutive warm requests after one cold warmup run on an identical
software stack and checkpoint revision (values in parentheses show the denoising loop duration).

| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN | Workload | TeleFuser Warm (DiT) | vLLM-Omni Warm (DiT) | Speedup vs. vLLM-Omni | Peak HBM / Card | Quality Gate |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `aarch64` · CANN `9.1.0` | 864x480, 121f, 50 steps | **98.90 s** (90.4 s) | 106.95 s (96.0 s) `v0.23.0rc1` | **1.08x (-7.5%)** | 16,405 MB | PSNR 31.39 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **85.51 s** (78.0 s) | 101.57 s (93.0 s) `v0.34.0` | **1.19x (-15.8%)** | 17,805 MB (vs. 24,475 MB) | PSNR 26.89 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **2× Ascend 910B2 (64GB)** | CFG2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **46.67 s** (39.1 s) | 56.01 s (47.5 s) `v0.34.0` | **1.20x (-16.7%)** | 18,278 MB (vs. 24,906 MB) | PSNR 99.00 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **4× Ascend 910B2 (64GB)** | CFG2 x Ulysses2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **33.85 s** (26.3 s) | 38.66 s (29.8 s) `v0.34.0` | **1.14x (-12.4%)** | 19,039 MB (vs. 25,814 MB) | PSNR 21.65 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **8× Ascend 910B2 (64GB)** | CFG2 x Ulysses4 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **20.66 s** (13.1 s) | 25.84 s (16.5 s) `v0.34.0` | **1.25x (-20.0%)** | 18,153 MB (vs. 25,792 MB) | PSNR 21.95 dB, Finite |

## Qualification Roadmap Matrix

| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN / Base Image | Target Workload | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MiniMax-H3` (T2VA) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **2× Ascend 910B2 (64GB)** | Resident TP2 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **4× Ascend 910B2 (64GB)** | Resident Ulysses4 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |
| `MiniMax-H3` (T2VA) | **8× Ascend 910B2 (64GB)** | Resident Ulysses8 | `x86_64` · CANN `9.3.0` (`vllm-omni:v0.34.0`) | 768p, 5s (24fps + 32kHz), 50 steps, BF16 | `PENDING` |

## Reproducible Five-Gate Harness

All rows above are produced by the automated harness in
`benchmarks/ascend_npu/` (`python benchmarks/ascend_npu/render_board.py`):

1. **Gate G0 (Device & HCCL Probe)**: Verifies clean device visibility, NPU fused ops, HCCL
   `all_reduce` / `all_to_all_single` bandwidth, and host-staged worker tensor queues.
2. **Gate G1 (Single-Card Parity)**: Verifies finite pre-encode outputs, `ffprobe` media stream contracts
   (including 32 kHz stereo audio sync for audio-video models), and `PSNR` / `MAE` parity.
3. **Gate G2 (Multi-Card Scaling & Isolation)**: Verifies multi-NPU equivalence against single-NPU
   outputs and concurrent multi-instance HCCL 16-port lease isolation.
4. **Gate G3 (Warm Latency & Memory Profile)**: Measures `1 Cold + 3 Warm` serial requests and per-card
   HBM peaks on the shared `quay.io/ascend/vllm-omni` container base.
5. **Gate G4 (Clean-Room Reproduction)**: Executes a standalone `REPRODUCE.sh` with a stripped
   environment (`env -i`) and zero source-tree bind mounts to guarantee third-party reproducibility.
