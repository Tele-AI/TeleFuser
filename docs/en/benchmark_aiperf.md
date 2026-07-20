# TeleFuser and AIPerf

TeleFuser exposes raw target-side facts; AIPerf owns workload execution, aggregation,
resource collection, artifacts, GreptimeDB history, and visualization. This separation
keeps the same benchmark and dashboard reusable across TeleFuser, SGLang-Diffusion, and
future targets.

The included assets cover:

- Wan2.1 image-to-video through the OpenAI-compatible `/v1/videos` API;
- LingBot-World-Fast sessions through WebRTC and DataChannel;
- a LingBot SGLang-Diffusion baseline through WebSocket and MessagePack.

## Repository layout

```text
benchmarks/
├── telefuser_aiperf/                 # TeleFuser contracts, configs, data, launchers
├── baseline/sglang_lingbot_stream/   # Stream baseline
└── aiperf/                            # Ignored external AIPerf checkout
```

The AIPerf implementation is not vendored into TeleFuser. The setup script always uses
`<TeleFuser>/benchmarks/aiperf`; neither the setup script nor the launchers accept a
checkout-path override. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run this once from
the TeleFuser repository root:

```bash
bash scripts/setup_aiperf_repo.sh
```

The script clones AIPerf, creates its isolated runtime environment with WebRTC support,
and creates `<TeleFuser>/artifacts` for benchmark output and History imports. The
dashboard is bundled, so runtime users do not need Node.js or a separate frontend
process. Pin a commit for reproducible runs:

```bash
AIPERF_REF=<commit> bash scripts/setup_aiperf_repo.sh
```

`AIPERF_REPO_URL`, `AIPERF_BRANCH`, and `AIPERF_REF` may select the source and revision,
but never change the checkout location.

## Batch video

Start the fixed Wan2.1 I2V target:

```bash
telefuser serve \
  examples/wan_video/wan21_14b_image_to_video_480p_service.py \
  --port 8000 \
  --task i2v
```

Run a smoke profile or the fixed comparison workload:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml
```

The launcher checks `/v1/service/health` before profiling. Common overrides include
`TELEFUSER_AIPERF_URL`, `TELEFUSER_AIPERF_CONCURRENCY`,
`TELEFUSER_AIPERF_REQUESTS`, `TELEFUSER_AIPERF_SIZE`, and
`TELEFUSER_AIPERF_SECONDS`.

## LingBot stream

Start TeleFuser:

```bash
telefuser stream-serve \
  examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  -p 8088 \
  --skip-validation
```

Then run:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_compare.json
```

The stream config enables `benchmark_metrics`. TeleFuser then reports synchronized raw
facts for runtime creation, actor-graph chunk compute, cache geometry, and environment
identity. Allocator peaks are omitted because generation runs in child actors and the
service-process allocator cannot represent the complete graph; active AIPerf resource
telemetry supplies process-tree GPU-memory curves instead. The native WebRTC path does
not report a separate payload encoding duration because encoding happens after the target
chunk fact. AIPerf computes warmup-aware summaries and keeps client delivery separate
from target compute.

The shared timed control trace is stored at
`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`.

## SGLang-Diffusion baseline

Use a compatible, version-pinned `sgl-project/sglang` environment. The TeleFuser tree
does not patch SGLang modules at import time.

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh
```

The baseline uses the same prompt, first frame, FPS target, session window, and control
trace. The adapter translates only transport semantics. For a performance comparison,
record the exact SGLang commit and model revision, use GPU-resident speed mode, and keep
offload and fallback settings identical. An OOM is a result for that configuration; do
not replace it with a mock or offloaded result under the same label.

Both documented launch commands default to one GPU. Override both targets explicitly
when comparing another accelerator count.

## Configs

| Config | Purpose |
|---|---|
| `video_generation_quick.yaml` | Batch connectivity and latency smoke test |
| `video_generation_e2e.yaml` | Batch warmup, trace, records, and server metrics |
| `video_generation_rate.yaml` | Poisson-arrival Batch load |
| `video_generation_wan21_i2v_480p_compare.yaml` | Fixed Wan I2V comparison |
| `stream_lingbot_world_fast_quick.json` | Bounded Stream smoke test |
| `stream_lingbot_world_fast_compare.json` | Fixed LingBot Stream comparison |

SGLang equivalents are under `benchmarks/baseline/sglang_lingbot_stream/configs`.

## Metric interpretation

The most important distinction is scope:

| Metric | Meaning |
|---|---|
| `stream_fps` | Frames received by the client divided by client session time |
| `chunk_compute_fps` | Frames divided by compute time for one target chunk |
| `chunk_compute_fps_weighted` | `sum(frames) / sum(compute_seconds)` after warmup exclusion |

AIPerf presents metrics under five stable dimensions: delivery, latency, throughput,
target execution, and resources. Implementation-specific fields remain raw evidence and
map into these canonical leaves; they do not become separate top-level metrics.

## Active resource history

Docker provides the shortest persistent GreptimeDB setup:

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

Pin the image tag or digest for production. The named volume keeps history across
container restarts. Then start the bundled AIPerf API and frontend from the TeleFuser
repository root; the `artifacts` root matches the benchmark launchers:

```bash
uv run --frozen --no-dev --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --host 127.0.0.1 \
  --port 8095
```

Verify the stack:

```bash
curl --fail http://127.0.0.1:8095/api/v1/history/health
curl --fail -X POST 'http://127.0.0.1:4000/v1/sql?db=public' \
  --data-urlencode 'sql=SELECT 1 AS ready'
```

Enable active collection on the target host:

```bash
export AIPERF_HISTORY_URL=http://<history-host>:8095
export AIPERF_RESOURCE_TARGET_PID=<service-pid>

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

The agent recursively observes the target process tree. It samples every second,
uploads every 15 seconds, timestamps samples at the source, and flushes at termination.
It reports process, container-when-detectable, and machine facts for CPU, memory, GPU,
VRAM, Ethernet, and RDMA. Capacity is kept separate from usage.

GreptimeDB is mandatory for History and active reporting. Startup, query, or final-flush
failure is surfaced; there is no SQLite, in-memory, or direct-file query fallback.

Open `http://127.0.0.1:8095/` for the Chinese desktop dashboard. It supports two run
groups, canonical metric-tree selection, aggregate curves, resource timelines, and
cross-run comparison.

For a remote benchmark host, keep the default loopback bind and forward it securely:

```bash
ssh -L 8095:127.0.0.1:8095 user@benchmark-host
```

## Artifacts and reproducibility

Batch and stream launchers write timestamped artifacts below `artifacts/`. Stream
artifacts include summaries, session and event JSONL, target metadata, normalized
metrics, and a standalone HTML report.

Every performance result should retain:

- TeleFuser or SGLang commit and model revision;
- accelerator model/count, driver, CUDA, PyTorch, and dtype;
- workload config and control trace;
- warmup policy and successful/failed session counts;
- offload, cache, attention, and fallback settings.

See the Chinese [benchmark design](/TeleFuser/zh/benchmark_aiperf_design/) for protocol and
ownership details.
