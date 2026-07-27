# TeleFuser and AIPerf

TeleFuser exposes raw target-side facts; AIPerf owns workload execution, aggregation, resource collection, artifacts,
GreptimeDB history, and visualization. The checked-in integration currently covers batch video generation through
the OpenAI-compatible `/v1/videos` API.

The former LingBot streaming adapter and its SGLang comparison assets were removed with the legacy transport backend.
A LiveKit benchmark adapter has not been added to AIPerf yet, so this repository does not present an
unsupported stream benchmark as runnable. Target compute metrics emitted by streaming services remain available to
future LiveKit-aware benchmark clients.

## Repository boundary

```text
benchmarks/
├── telefuser_aiperf/   # Batch contracts, configs, data, and launcher
└── aiperf/             # Ignored external AIPerf checkout
```

The AIPerf implementation is not vendored. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the TeleFuser repository root:

```bash
bash scripts/setup_aiperf_repo.sh
```

The script clones AIPerf into `benchmarks/aiperf`, installs its non-development runtime, and creates `artifacts/`.
Pin a commit for reproducible runs:

```bash
AIPERF_REF=<commit> bash scripts/setup_aiperf_repo.sh
```

`AIPERF_REPO_URL`, `AIPERF_BRANCH`, and `AIPERF_REF` may select the source and revision, but never change the checkout
location.

## Batch video benchmark

Start the fixed Wan2.1 I2V target:

```bash
telefuser serve \
  examples/wan_video/wan21_14b_image_to_video_480p_service.py \
  --port 8000 \
  --task i2v
```

Run a smoke profile or fixed comparison workload:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml
```

The launcher checks `/v1/service/health` before profiling. Common overrides include `TELEFUSER_AIPERF_URL`,
`TELEFUSER_AIPERF_CONCURRENCY`, `TELEFUSER_AIPERF_REQUESTS`, `TELEFUSER_AIPERF_SIZE`, and
`TELEFUSER_AIPERF_SECONDS`.

| Config | Purpose |
|---|---|
| `video_generation_quick.yaml` | Connectivity and latency smoke test |
| `video_generation_e2e.yaml` | Warmup, trace, records, and server metrics |
| `video_generation_rate.yaml` | Poisson-arrival load |
| `video_generation_wan21_i2v_480p_compare.yaml` | Fixed Wan I2V comparison |

## Active resource history

Start persistent GreptimeDB storage:

```bash
docker volume create aiperf-greptime-data
docker run -d --name aiperf-greptime --restart unless-stopped \
  -p 127.0.0.1:4000:4000 \
  -v aiperf-greptime-data:/greptimedb_data \
  greptime/greptimedb:latest \
  standalone start \
  --http-addr 0.0.0.0:4000 \
  --data-home /greptimedb_data
```

Then start the bundled AIPerf history API and dashboard:

```bash
uv run --frozen --no-dev --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --host 127.0.0.1 \
  --port 8095
```

For active process-tree collection, export `AIPERF_HISTORY_URL` and `AIPERF_RESOURCE_TARGET_PID` before running the
batch launcher. GreptimeDB is required for History and active reporting; failures do not silently fall back to an
in-memory or file-only database.

## Reproducibility

Every result should retain the TeleFuser and AIPerf commits, model revision, accelerator model/count, driver, CUDA,
PyTorch, dtype, complete workload config, warmup rule, success/failure counts, and offload/cache/attention settings.
Dynamic results belong in GreptimeDB and replayable artifacts, not stable documentation.

The stable responsibilities and metric boundary are summarized above; dynamic results remain outside this guide.
