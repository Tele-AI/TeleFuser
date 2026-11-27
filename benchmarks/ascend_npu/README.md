# Ascend NPU Benchmark & Qualification Suite

Reproducible qualification gates (`G0`–`G4`) and sanitized head-to-head performance records comparing
**TeleFuser** with **vLLM-Omni** on Ascend 910B2 / 910B4 NPUs.

## Verified Results

| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN | Workload | TeleFuser Warm (DiT) | vLLM-Omni Warm (DiT) | Speedup vs. vLLM-Omni | Peak HBM / Card | Quality Gate |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `MiniMax-H3` (T2VA) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` | 1344x768, 124f, 50 steps | **783.39 s** (758.4 s) | 1615.90 s (1590.9 s) `v0.34.0` | **2.06x (-51.5%)** | 54,653 MB (vs. 23,152 MB) | PSNR 18.83 dB, AudioCos 0.9753, Finite |
| `MiniMax-H3` (T2VA) | **2× Ascend 910B2 (64GB)** | Resident TP2 | `x86_64` · CANN `9.3.0` | 1344x768, 124f, 50 steps | **498.34 s** (482.7 s) | 517.26 s (502.2 s) `v0.34.0` | **1.04x (-3.7%)** | 44,435 MB (vs. 25,942 MB) | PSNR 17.77 dB, AudioCos 0.9655, Finite |
| `MiniMax-H3` (T2VA) | **4× Ascend 910B2 (64GB)** | Resident Ulysses4 | `x86_64` · CANN `9.3.0` | 1344x768, 124f, 50 steps | **220.88 s** (211.2 s) | 246.38 s (236.5 s) `v0.34.0` | **1.12x (-10.3%)** | 63,514 MB (vs. 25,383 MB) | PSNR 25.07 dB, AudioCos 0.9950, Finite |
| `MiniMax-H3` (T2VA) | **8× Ascend 910B2 (64GB)** | Resident Ulysses8 | `x86_64` · CANN `9.3.0` | 1344x768, 124f, 50 steps | **109.53 s** (102.6 s) | 122.90 s (115.5 s) `v0.34.0` | **1.12x (-10.9%)** | 58,299 MB (vs. 25,379 MB) | PSNR 15.67 dB, AudioCos 0.7675, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `aarch64` · CANN `9.1.0` | 864x480, 121f, 50 steps | **98.90 s** (90.4 s) | 106.95 s (96.0 s) `v0.23.0rc1` | **1.08x (-7.5%)** | 16,405 MB | PSNR 31.39 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **1× Ascend 910B2 (64GB)** | Offload (Single-Card) | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **85.51 s** (78.0 s) | 101.57 s (93.0 s) `v0.34.0` | **1.19x (-15.8%)** | 17,805 MB (vs. 24,475 MB) | PSNR 26.89 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **2× Ascend 910B2 (64GB)** | CFG2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **46.67 s** (39.1 s) | 56.01 s (47.5 s) `v0.34.0` | **1.20x (-16.7%)** | 18,278 MB (vs. 24,906 MB) | PSNR 99.00 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **4× Ascend 910B2 (64GB)** | CFG2 x Ulysses2 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **33.85 s** (26.3 s) | 38.66 s (29.8 s) `v0.34.0` | **1.14x (-12.4%)** | 19,039 MB (vs. 25,814 MB) | PSNR 21.65 dB, Finite |
| `Wan2.2-TI2V-5B` (T2V) | **8× Ascend 910B2 (64GB)** | CFG2 x Ulysses4 | `x86_64` · CANN `9.3.0` | 864x480, 121f, 50 steps | **20.66 s** (13.1 s) | 25.84 s (16.5 s) `v0.34.0` | **1.25x (-20.0%)** | 18,153 MB (vs. 25,792 MB) | PSNR 21.95 dB, Finite |

## Updating the Board

1. Run the qualification harness (`harness/preflight_check.sh`, `harness/comm_probe.py`,
   `harness/bench_vs_vllm_omni.py`, and `harness/cleanroom_repro.sh`) on a clean container built from
   `docker/Dockerfile.ascend`.
2. Place the resulting sanitized JSON record into `benchmarks/ascend_npu/records/`.
3. Regenerate the repository README and bilingual documentation pages:

```bash
python benchmarks/ascend_npu/render_board.py
pytest tests/unit/benchmarks/test_ascend_npu_board.py
```

The renderer automatically rejects any record containing server hostnames, internal IP addresses,
usernames, or internal mount paths, and requires explicit `Ascend 910B2 (64GB)` / `Ascend 910B4 (32GB)`
SKU and card-count fields.
