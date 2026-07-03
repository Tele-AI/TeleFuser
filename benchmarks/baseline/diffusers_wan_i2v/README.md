# Diffusers Wan2.1 I2V Baseline

This directory contains a standalone benchmark baseline for the official
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` pipeline.

The goal is to compare:

- the same model
- the same I2V workload
- the same async `/v1/videos` HTTP semantics
- while changing only the inference framework

Current fixed workload:

- model: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- resolution: `480p`
- effective size: `832x480`
- frames: `81`
- denoising steps: `40`
- guidance scale: `5.0`
- output fps: `16`

## Layout

- `service.py`: standalone FastAPI server
- `configs/`: AIPerf configs for the baseline
- `scripts/run_service.sh`: launch helper
- `scripts/run_video_bench.sh`: benchmark helper

## Start The Service

```bash
python3 benchmarks/baseline/diffusers_wan_i2v/service.py
```

Or:

```bash
bash benchmarks/baseline/diffusers_wan_i2v/scripts/run_service.sh
```

Default address:

- `http://127.0.0.1:8010`

Health check:

```bash
curl http://127.0.0.1:8010/v1/service/health
```

## Run AIPerf

Install the vendored AIPerf first:

```bash
pip install -e ./benchmarks/aiperf
```

Then run:

```bash
bash benchmarks/baseline/diffusers_wan_i2v/scripts/run_video_bench.sh
```

This reuses the same prompt dataset as the TeleFuser benchmark:

- `benchmarks/telefuser_aiperf/data/video_prompts.jsonl`

## Notes

- The service supports both `multipart/form-data` and JSON requests.
- `reference_url` is treated as a local filesystem path in the current benchmark dataset.
- Remote HTTP image URLs are intentionally rejected in this first baseline to keep the benchmark focused on inference rather than network fetch behavior.
