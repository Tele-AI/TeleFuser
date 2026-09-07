# FP8 Sol-Attn Sequence-Parallel Tuning

This microbenchmark isolates the per-rank FP8 Sol-Attn work after a Ulysses
all-to-all. `local_heads = global_heads / sp_degree`; communication itself is
covered by the distributed parity test rather than these kernel timings.

Environment: NVIDIA H100 80 GB HBM3, PyTorch 2.11.0+cu128, seed 0, BF16 random
Q/K/V, head dimension 128, warm CuTe specializations. Each reported value is the
mean of three or ten CUDA-event measurements. These synthetic results tune
execution parameters; they do not replace generated-video quality evaluation.

## KV Splits

`tau=1.0`, `threshold_type=diag`, 40 global heads:

| Tokens | SP degree | Local heads | Split 1 kernel | Split 2 kernel | Split 4 kernel | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 40 | 0.334 ms | 0.353 ms | 0.397 ms | 1 |
| 4,096 | 4 | 10 | 0.279 ms | 0.302 ms | 0.315 ms | 1 |
| 16,384 | 1 | 40 | 2.382 ms | 2.349 ms | 2.638 ms | 2 |
| 16,384 | 4 | 10 | 0.696 ms | 0.693 ms | 0.749 ms | 2 |
| 32,768 | 1 | 40 | 8.476 ms | 8.009 ms | 8.803 ms | 2 |
| 32,768 | 4 | 10 | 2.254 ms | 2.089 ms | 2.226 ms | 2 |
| 65,536 | 4 | 10 | - | 7.377 ms | 7.754 ms | 2 |

The measurements support the existing FP8 `auto` boundary: split 1 below
16,384 tokens and split 2 at or above it. Ulysses head sharding does not justify
raising the split count. Split 4 regresses even at 65,536 tokens because its
partial-output workspace and combine pass outweigh additional KV parallelism.

The target Wan2.1 1.3B production shape has 32,760 tokens and 12 global heads.
Ten-repeat measurements confirm the same split choice after head sharding:

| SP degree | Local heads | Split 1 kernel | Split 2 kernel | Split 4 kernel | Best |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 3.198 ms | 2.900 ms | 3.102 ms | 2 |
| 2 | 6 | 1.662 ms | 1.501 ms | 1.594 ms | 2 |
| 4 | 3 | 0.890 ms | 0.808 ms | 0.847 ms | 2 |

## Routing Parameters

At 16,384 tokens, SP=4, 10 local heads, split 2 (ten repetitions):

| Tau | Threshold | Kernel | Quantize + kernel |
|---:|---|---:|---:|
| 0.5 | diag | 0.913 ms | 1.004 ms |
| 0.5 | exact | 0.932 ms | 1.028 ms |
| 1.0 | diag | 0.638 ms | 0.733 ms |
| 1.0 | exact | 0.658 ms | 0.754 ms |
| 1.5 | diag | 0.489 ms | 0.584 ms |
| 1.5 | exact | 0.501 ms | 0.602 ms |

Use `tau=1.0`, `diag`, and `kv_splits=auto` as the conservative production
starting point. `tau=1.5` is an explicit aggressive profile: it was 23% faster
than `tau=1.0` in this kernel test but selects fewer exact KV blocks and therefore
requires model-level perceptual validation. `exact` was 2-3% slower here.

## Reproduce

```bash
PYTHONPATH=. python tools/validation/benchmark_fp8_sol_attention.py \
  --tokens 4096,16384,32768 \
  --global-heads 40 --sp-degrees 1,4 \
  --taus 1.0 --threshold-types diag \
  --kv-splits 1,2,4 --warmup 1 --repeats 3 \
  --output /tmp/fp8_sol_sp.json

PYTHONPATH=. python tools/validation/benchmark_fp8_sol_attention.py \
  --tokens 16384 --global-heads 40 --sp-degrees 4 \
  --taus 0.5,1.0,1.5 --threshold-types diag,exact \
  --kv-splits 2 --warmup 2 --repeats 10

PYTHONPATH=. python tools/validation/benchmark_fp8_sol_attention.py \
  --tokens 32760 --global-heads 12 --sp-degrees 1,2,4 \
  --taus 1.0 --threshold-types diag \
  --kv-splits 1,2,4 --warmup 2 --repeats 10
```

The corresponding NCCL correctness coverage is
`tests/integration/test_wan_fp8_sol_distributed.py` for two- and four-way
Ulysses execution.
