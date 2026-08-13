# LingBot-VLA v2 Validation at `baf3d18`

This report records the native structured HTTP smoke and full-project CPU CI run performed on 2026-08-12 for
TeleFuser commit `baf3d18840a71363984edb46222ef86200efb689`.

The strict 38-item upstream parity calculation was not repeated. The model, loader, preprocessing, velocity sampling,
and action-generation path covered by the frozen parity baseline have not changed since `2d40ee2`; the changes up to
`baf3d18` affect the service and validation boundary. This report therefore adds service evidence without replacing
the frozen strict parity result.

## Native Structured HTTP Smoke

Environment:

- NVIDIA H100 80GB HBM3
- Python 3.10.12
- PyTorch 2.11.0+cu130 with CUDA 13.0
- One native TeleFuser service replica on `127.0.0.1:18080`
- One warmup request followed by 20 measured sequential requests
- Identical source image for the high, left-wrist, and right-wrist cameras
- Seed 7 and a zero-valued 14-dimensional RobotWin state

The service was started with:

```bash
TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action \
  --parallelism 1 \
  --host 127.0.0.1 \
  --port 18080
```

The measured workload was run with:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --warmup 1 \
  --requests 20 \
  --concurrency 1 \
  --service-pid <service-pid> \
  --gpu-indexes 0 \
  --output work_dirs/vla_service_validation/smoke_20_baf3d18.json
```

| Check | Result |
| --- | --- |
| Overall validation | Passed |
| Measured requests | 20 |
| Successful requests | 20 (100%) |
| Failed requests | 0 |
| Unique task IDs | 20 |
| Terminal status | 20 `completed` |
| Action contract | 20 finite `50 x 55` canonical normalized chunks |
| Policy status | 20 `unverified_official_6b_base` |
| Ready before and after | Passed |
| Warmup | Passed |
| Queue drained | Passed |
| Resource sampling | Passed, 26 CPU and GPU samples |

| Latency | Mean | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| End to end | 1.370 s | 1.372 s | 1.391 s | 1.391 s | 1.391 s |
| Accepted to terminal | 1.337 s | 1.339 s | 1.355 s | 1.356 s | 1.356 s |
| Target inference | 1.267 s | 1.252 s | 1.330 s | 1.330 s | 1.330 s |
| Submission | 0.033 s | 0.032 s | 0.035 s | 0.036 s | 0.036 s |

Sequential throughput was 0.729 requests/s. Sampled GPU process memory was constant at 13,302 MiB. Process-tree RSS
had a 3,617.9 MiB maximum and changed by -0.6 MiB between the first and last sample windows. The service was stopped
normally after the run, and no service process or GPU compute allocation remained.

The raw report remains under the Git-ignored `work_dirs/` tree. It contains bounded request statistics and action
fingerprints, not full actions or Base64 camera payloads.

## Full-Project CI

The repository CI entrypoint was run in the separate full-project `.venv`, with dependency installation skipped after
that environment was prepared. Cross-model dependencies such as PyAV, OpenCV, Diffusers, and ImageIO were installed
only in `.venv`; they were not added to `.venv-vla`.

```bash
PATH=/data/telefuser_vla_test/.venv/bin:$PATH \
  bash scripts/run_ci_tests.sh --skip-install
```

| Stage | Result |
| --- | --- |
| Ruff check | Passed |
| Ruff format check | Passed, 510 files checked |
| Ruff import check | Passed |
| CPU-only runtime assertion | Passed |
| Unit tests | 1,236 passed, 8 skipped, 114 deselected, 5 subtests passed |
| Server and OpenAI API tests | 62 passed |
| Overall CI result | Passed |

The unit skips were expected for CUDA-only operations, the optional local `tf-kernel` extension, and two LingBot-Video
refiner parity checks whose separate upstream checkout was unavailable. No VLA, structured-service, shared-service,
or cross-model test failed.

This smoke proves that the real 6B checkpoint crosses the native HTTP boundary, scheduler, pipeline service, result
serialization, and status polling while preserving the `50 x 55` action contract. It does not prove embodiment-specific
robot control semantics or change the base policy's `unverified_official_6b_base` status.
