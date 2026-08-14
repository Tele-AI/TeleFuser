# ABot-World LightVAE LF=3 single-GPU microbatch capacity

> Scope: controlled native-model microbenchmark, not an end-to-end serving
> result. It supersedes the pre-LightVAE summaries that were previously kept in
> this branch. Raw JSON/CSV/log artifacts remain outside Git.

## Configuration

- GPU: one NVIDIA H100 80 GiB (CUDA device 5)
- Checkpoint: `ABot-World-0-5B-LF` with official `taew2_2` lightweight decoder
- `control_latent_frames=3`; each continuation chunk delivers 12 display frames per independent session
- 2 warmup chunks, then 5 synchronized steady-state samples per batch size
- Input: `examples/data/1.png`; fixed prompt and deterministic session seeds

## Results

| Concurrent sessions / DiT batch | Mean chunk time (s) | Aggregate FPS | FPS/session | Peak allocated GiB | 8 FPS deadline (1.5 s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.387 | 30.98 | 30.98 | 28.91 | pass |
| 2 | 0.703 | 34.13 | 17.07 | 34.56 | pass |
| 3 | 1.021 | 35.26 | 11.75 | 40.28 | pass |
| 4 | 1.297 | 37.01 | 9.25 | 45.95 | pass |
| 5 | 1.599 | 37.53 | 7.51 | 51.73 | fail |
| 6 | 1.961 | 36.71 | 6.12 | 57.36 | fail |

## Current conclusion

This measurement uses the official `taew2_2` LightVAE path and directly refutes
the old claim that one H100 is saturated at `B=1`: aggregate model throughput
increases from **30.98 FPS** at `B=1` to **37.01 FPS** at synchronized `B=4`
(+19.5%). The 8-FPS model-side limit is four synchronized sessions: `B=4` p95
chunk time is 1.371 s, while `B=5` mean latency already exceeds the 1.5 s
deadline. The approximately 37-FPS plateau is an 8-FPS SLO/model-execution
limit, not an HBM OOM limit; at `B=6`, mean DiT time is 1.402 s and LightVAE
decode is 0.056 s.

The benchmark deliberately synchronizes identical session histories before each
native call. It therefore measures the *available* model batch efficiency, not
whether a real serving trace forms `B=4`. In the public 16-user trace, the
observed DiT batch distribution and native LightVAE batch distribution must be
read from serving telemetry; session readiness, playout deadlines, and causal
decoder-state compatibility can keep those batches near `B=1` even though this
controlled microbenchmark can execute `B=4`.

Raw local artifacts (`results.json`, `results.csv`, `run.log`) are intentionally ignored by repository rules.
