# MiniMax-H3-Image validation

## Scope and provenance

This integration adds MiniMax-H3-Image only. MiniMax-H3 video/audio generation is an existing, separate integration;
its LightX2V video benchmark does not measure MiniMax-H3-Image.

The released-checkpoint results below were recorded on November 13, 2026, at TeleFuser commit
`e09df7a01875298248d343167ed3a9fee615dc1b`. They are historical validation evidence, not reruns of the split PR.
Every listed output was checked against its original SHA256 record during this split.

## Resident execution

The current example also supports `--gpu_num 4 --sp 1 --tp 4 --resident`: DiT TP4, independent Qwen3-VL
encoder TP4, resident mixed-precision VAE weights, round-robin VAE tiles and six-stream local decoder CUDA graphs.
VAE loading preserves FP32 weights before casting decoder GEMMs to BF16, retaining the existing FP32
encoder, normalization/scales and rotary-base safeguards. Existing
`--fp8 --sol-fp8` flags and hybrid DiT SP2×TP2 compose with this configuration. The default CPU-offload path remains
available. This removes the previous single-device restriction on image conditioning and VAE stages without
introducing a separate model implementation.

Native CUDA graph replay is compared against eager decoding, including input replacement and output ownership.
Two-GPU numerical tests compare tiled encode/decode against serial execution, including tile counts that do not
divide the world size and ranks assigned no tiles:

```bash
pytest tests/unit/models/test_minimax_h3_image_vae.py -k 'graph_replays or parallel_tiles'
```

Current performance comparisons use the upstream `benchmarks/diffusion/h3_image_bench.py` HTTP client for both
runtimes, with the same checkpoint, prompt, CPU seed, canvas, six PDD steps, warm-up policy and PNG encoding.
The measured charts and generated outputs are attached to the PR rather than committed as repository assets.
The older measurements below describe their original CPU-offload configurations; they do not describe the new
resident execution path.

## Hardware and runtime

- Host: 4 NVIDIA H100 80GB HBM3 GPUs with NVLink; released runs use **one or two** devices (GPU 2/3).
- CPU: dual Intel Xeon Platinum 8468, 192 logical CPUs; container memory limit 800 GiB.
- Driver 590.48.01; Python 3.11.13; PyTorch 2.11.0+cu130; Transformers 5.14.1; xFormers 0.0.35.
- Model CPU offload, `pin_cpu_memory=False`; separate model loading excluded from `pipeline_seconds`.
- Single-request wall time includes conditioning, denoising, decoding and runtime CPU offload/onload;
  no repeated-run throughput or pure-kernel speedup is inferred.
- FP8+Sol below is **FP8 linear GEMM with BF16 Sol attention**, not FP8 Q/K/V attention.
- Independent Diffusers reference revision: `843f1743e259e510f22065c0cb917d417b98d9f5`.

## External H200 reference benchmark

The official comparison reference is the [vLLM-Omni MiniMax-H3-Image H200 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Image-H200.md), not a MiniMax vendor speed claim. It reports NVIDIA H200 SXM/HGX at 141 GB physical memory (about 139.8 GiB usable per GPU), six steps, concurrency 1, one warm-up request followed by three requests, and the median client end-to-end latency including inference and image processing.

| vLLM-Omni profile | Memory / GPU | 1024² | 2048² | 4736×3552 | 2048² edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4× H200, Ulysses SP4, DiT resident | 81.3 GiB | 0.77 s | 0.88 s | 3.84 s | 1.35 s |
| 1× H200, DiT resident + text-encoder offload | 73.4 GiB | 3.08 s | 3.18 s | 12.39 s | 4.60 s |

The corresponding vLLM-Omni H100 recipe is explicitly **configuration-only** and says its numbers are reused from H200; it does not provide H100 measurements. The H100 recipe also notes that the 4× H200 Ulysses resident profile does not fit on H100.

## Released checkpoint runs

Checkpoint: `MiniMaxAI/MiniMax-H3-Image`, revision `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`.
Output: 2048×2048 RGB image, fixed six-step PDD schedule, seed 71.
Prompt: "A red fox standing in a snowy pine forest, detailed natural wildlife photograph, soft morning sunlight."
The edit run uses the BF16 text-to-image output as its reference and a **different** instruction:
"Change the snowy forest in <Picture 1> into an autumn forest with orange and golden leaves,
preserving the red fox and the photographic composition."
It is an editing demonstration, not a BF16-versus-FP8 quality comparison. The other six runs use the snow prompt above.

| Case | GPUs | SP×TP | Pipeline (s) | Output shape |
| --- | ---: | --- | ---: | --- |
| h3-image-bf16 | 1 | 1×1 | 113.012 | [2048, 2048, 3] |
| h3-image-edit-bf16 | 1 | 1×1 | 97.237 | [2048, 2048, 3] |
| h3-image-fp8-sol | 1 | 1×1 | 136.192 | [2048, 2048, 3] |
| h3-image-sp2 | 2 | 2×1 | 101.832 | [2048, 2048, 3] |
| h3-image-sp2-fp8-sol | 2 | 2×1 | 126.721 | [2048, 2048, 3] |
| h3-image-tp2 | 2 | 1×2 | 67.240 | [2048, 2048, 3] |
| h3-image-tp2-fp8-sol | 2 | 1×2 | 91.442 | [2048, 2048, 3] |

These are generation-completion checks, not a scored quality evaluation or a matched official speed benchmark.
Four-GPU hybrid SP2×TP2 is covered by small numerical fixtures; the full-checkpoint results above use at most two GPUs.

## Split-branch reproduction

On November 23, 2026, the standard example at split commit `ef5233d` completed full-checkpoint BF16 generation
on one H100 with all six PDD steps, the snow prompt above, seed 71 and 2048×2048 output.
Its PNG SHA256 matches the historical BF16 file exactly:
`7daec58c9326235ddce24b31f532b76237f2e9431015ba807070a40b551ed890`.
This confirms the standard example on the split branch; the timing table remains the historical measurements.

## TeleFuser H100 aligned benchmark status

The TeleFuser standard example was extended with `--dit-resident` to represent the one-GPU vLLM profile. On H100, the BF16 resident attempt fails before generation: the 79.18 GiB device has only about 79.1 GiB usable, while the resident DiT reaches 78.22 GiB allocated and the next 250 MiB allocation fails. This is a hardware-capacity result, not a quality result. Online `--fp8` GEMM was not enabled for this BF16 OOM attempt; a separate FP8 resident attempt is required before drawing a conclusion about FP8 memory fit.

The completed H100 comparison below uses the fitting model-CPU-offload path and the same request semantics. It is not directly comparable to the H200 resident profile because the execution path and hardware differ.

### Four-H100 hybrid SP2×TP2 result

The same 2048×2048 workload was also run with TeleFuser's hybrid Ulysses sequence parallel plus tensor parallel mesh (`SP2×TP2`).

| TeleFuser H100 profile | Timed runs (s) | Median | Peak / GPU |
| --- | --- | ---: | ---: |
| 4× H100, SP2×TP2, FP8 GEMM, Sol disabled | 61.385 / 63.064 / 63.025 | **63.025** | ~38 GiB |
| 4× H100, SP2×TP2, FP8 GEMM + FP8 Q/K/V Sol (Triton) | 63.119 / 61.964 / 61.943 | **61.964** | ~38 GiB |

Against the 4×H100 SP4 rows, TP2 reduces FP8-only latency from 82.858 s to 63.025 s (**1.315×**) and the full FP8+Sol latency from 82.493 s to 61.964 s (**1.331×**). This is the single-request tensor-parallel benefit that the SP4-only comparison would miss.

### Four-H100 TP4 result (SP1×TP4)

`TP4` means `SP1×TP4`: Ulysses sequence parallel is disabled, so this profile does not use sequence-parallel all-to-all communication.

| TeleFuser H100 profile | Timed runs (s) | Median | Peak / GPU |
| --- | --- | ---: | ---: |
| 4× H100, TP4 (SP1×TP4), FP8 GEMM, Sol disabled | 53.290 / 53.213 / 52.668 | **53.213** | ~21 GiB |
| 4× H100, TP4 (SP1×TP4), FP8 GEMM + FP8 Q/K/V Sol (Triton) | 63.207 / 62.626 / 62.251 | **62.626** | ~21 GiB |

TP4 is 1.185× faster than SP2×TP2 for FP8-only (53.213 s vs 63.025 s). For the full FP8+Sol workload, SP2×TP2 is 1.011× faster than TP4 (61.964 s vs 62.626 s).

### Four-H100 SP4 result

Full four-GPU TeleFuser runs used Ulysses SP4, 2048×2048, six steps, one warm-up and three timed requests at concurrency 1. The DiT was loaded/offloaded per request because this H100 host cannot keep the full BF16 DiT resident with the required runtime headroom.

| TeleFuser H100 profile | Timed runs (s) | Median | Peak / GPU |
| --- | --- | ---: | ---: |
| 4× H100, Ulysses SP4, BF16, Sol disabled | 135.867 / 108.862 / 108.325 | **108.862** | ~66.5 GiB |
| 4× H100, Ulysses SP4, FP8 GEMM, Sol disabled | 76.327 / 82.858 / 88.801 | **82.858** | ~67.4 GiB |
| 4× H100, Ulysses SP4, FP8 GEMM + FP8 Q/K/V Sol (Triton) | 80.480 / 82.493 / 83.119 | **82.493** | ~67.4 GiB |

The BF16 four-card median (108.862 s) is 1.023× the single-card BF16 median (106.415 s), so same-precision SP4 showed no speedup in this CPU-offload configuration. The FP8-only four-card median is 82.858 s. These historical measurements used a Triton FP8 Sol workaround for the NVVM `fmax` API mismatch. The native CuTe fix and new validation are recorded below; the global disable cache has been removed.

## Quantization status

The published `MiniMaxAI/MiniMax-H3-Image` checkpoint contains BF16 DiT weights, FP32 boundary weights and an FP32 image VAE; no official FP8 or FP4 H3-Image checkpoint was found in the model release or vLLM-Omni H200/H100 recipes. TeleFuser `--fp8` is online FP8 GEMM quantization of selected linear modules and must not be described as an official quantized weight release. vLLM-Omni’s H200 recipe marks quantization as not qualified.

## Numerical and sampling checks

The unit tests cover checkpoint conversion, conditioning precision, scheduling, output conversion and VAE behavior.
The distributed tools compare tiny models against the same dense model and exercise repeated offload/worker requests;
they do not measure full-model quality. Independent Diffusers comparisons require that pinned reference implementation.

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/validation/validate_h3_image_parallel.py \
  --sp 2 --tp 1 --fp8 --sol --offload --output work_dirs/h3_image-parallel.json

NCCL_IB_DISABLE=1 python tools/validation/validate_h3_image_sampling.py \
  --device-ids 0 1 --sp 2 --tp 1 --fp8 --sol \
  --output work_dirs/h3_image-sampling.json
```

See the [example](README.md) for released-checkpoint generation commands.

## Native CuTe Sol with hybrid parallelism

A November 25 investigation reproduced the hybrid Sol fallback in a single-GPU
kernel call too. It was not caused by SP/TP collectives: the vendored wrapper
selected the old three-positional-argument `nvvm.fmax` API from `CUDA_VERSION=12.9`,
while the loaded NVVM binding actually accepted two operands and inferred the result type.
The resulting TypeError caused CuTe-to-Triton fallback.

The wrapper now inspects the actual NVVM function signature, supporting both APIs.
The blanket process-wide CuTe-disable cache and forced BF16 Triton dispatch have been
removed. Native FP8 Sol regression tests cover both 56 heads and the 14 heads held
by a rank after TP2 and Ulysses SP2 split a 56-head attention tensor. They prohibit
Triton fallback and compare the exact-attention limit against SDPA on identical
quantized inputs.

The full-checkpoint SP2×TP2 validation forbids both Triton fallback and Sol-to-SDPA
fallback in every spawned worker, records successful native FP8 kernel execution,
and keeps the original six-step model, prompt, seed and output canvas.
Earlier Triton measurements remain historical measurements of a different backend.

Native CuTe full-checkpoint result after the API fix (November 25):

| Layout | Backend | Timed requests (s) | Median (s) |
| --- | --- | --- | ---: |
| 4×H100, SP2×TP2, FP8 GEMM + FP8 Q/K/V Sol | Native SM90 CuTe | 63.623 / 65.851 / 64.464 | **64.464** |

One warm-up and three serial requests; 2048×2048, six PDD steps, seed 42, prompt
"A red bicycle leaning against a white plaster wall, detailed natural photograph, soft morning sunlight."
Model CPU offload remains enabled. All four worker ranks reported native FP8
execution on `[1, 4114, 14, 128]` tensors. The test fails if Triton is invoked or
Sol falls back to SDPA. Dense warm-up/refiner attention still uses SDPA if FA4
is unavailable; this is distinct from native Sol execution on active sparse layers.
The result verifies native dispatch and generation, not a speedup over the earlier
Triton timing or a controlled H200 comparison.
