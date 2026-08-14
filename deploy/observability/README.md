# TeleFuser serving observability

This stack scrapes the LiveKit serving API, NVIDIA DCGM Exporter, and Node
Exporter into Prometheus, then provisions a Grafana dashboard automatically.
It is intended for the four-GPU ABot-World experiment, but has no model-specific
runtime dependency.

## Start the serving API first

Expose the TeleFuser metrics endpoint on the host port used below (default
`8088`). The endpoint includes low-cardinality scheduler/session/batch/pipeline
metrics in addition to generic TeleFuser metrics:

```bash
curl -fsS http://127.0.0.1:8088/metrics | grep '^telefuser_serving_'
```

The compose file maps `host.docker.internal` to Docker's host gateway. On an
older Docker engine that does not support `host-gateway`, replace
`host.docker.internal:8088` in `prometheus.yml` with the host's reachable IP.
The serving API must be reachable from the Prometheus container.

## Start the monitoring stack

```bash
cd deploy/observability
docker compose up -d
```

Open Grafana at `http://<host>:3000` (default credentials are `admin` / `admin`; set
`GRAFANA_ADMIN_PASSWORD` before starting in any shared environment). Prometheus
is available at port 9090.

DCGM Exporter needs the NVIDIA Container Toolkit. By default it observes
**physical GPUs 0--3**, through NVIDIA Container Toolkit's
`NVIDIA_VISIBLE_DEVICES` allowlist. Select a different physical GPU set without
editing Compose:

```bash
# The four GPUs used by the current ABot experiment (the default).
TELEFUSER_MONITOR_GPU_IDS=0,1,2,3 docker compose up -d

# Observe all eight physical GPUs from one DCGM Exporter.
TELEFUSER_MONITOR_GPU_IDS=0,1,2,3,4,5,6,7 docker compose up -d
```

These are host/physical GPU indices. `CUDA_VISIBLE_DEVICES` remaps only the
serving process to logical IDs; it does **not** remap an independent Docker
container. The custom `dcgm-metrics.csv` requests GPU/HBM, power/temperature,
PCIe, NVLink, DRAM, and tensor-core telemetry. Some profiling fields are
conditionally omitted when the driver, GPU, or DCGM version cannot expose them.

## Capture an experiment without Docker

When Docker, DCGM Exporter, or Grafana cannot run on the experiment host, save
the serving metrics as a durable artifact with the lightweight collector. It
queries the same `/metrics` and `/v1/service/metrics/json` APIs used by
Prometheus, requires only the Python standard library, and explicitly bypasses
all proxy environment variables (including for `127.0.0.1`).

```bash
python tools/validation/capture_abot_serving_metrics.py \
  --server-url http://127.0.0.1:8088 \
  --duration 420 \
  --interval 1 \
  --output-dir results/experiments/abot_livekit_realistic_peak16/metrics
```

For the physical GPUs used by this experiment, pair it with the direct NVML
fallback. It does not shell out to `nvidia-smi`:

```bash
python tools/validation/capture_gpu_nvml_metrics.py \
  --gpu-indices 0,1,2,3 \
  --duration 420 \
  --interval 1 \
  --output-dir results/experiments/abot_livekit_realistic_peak16/gpu_metrics
```

Run both collectors in parallel with the workload, then wait for them before
stopping the serving API. The service collector writes raw Prometheus snapshots,
an aggregate-only JSONL time series, and a manifest even if interrupted. The
NVML collector writes one JSONL row per sample with GPU utilization, framebuffer
memory, power, and temperature. It is a local fallback, not a replacement for
DCGM: PCIe/NVLink, DRAM, tensor-core profiling, host CPU/RAM, and network
counters still require the full DCGM/Node Exporter stack.


## Metric groups

- `telefuser_serving_worker_*`: worker/GPU retained sessions, capacity, busy
  ratio, latest model stage timing, and model-reported chunk latency.
- `telefuser_serving_sessions`, `*_queue_depth`, `*_session_status`: admission,
  active/idle/waiting session state without session-id labels.
- `telefuser_serving_batch_*`: observed coalesced batch distribution and mean.
- `telefuser_serving_pipeline_stage_latency_seconds`: cumulative stage
  histograms for VAE encode/decode, DiT, cache operations, and postprocessing.
- `telefuser_serving_action_to_first_frame_seconds`: validated action ingress
  to the first frame accepted by the LiveKit publisher. It is a server-side
  A2F measurement; network and browser decode are intentionally excluded.
- `telefuser_serving_published_fps`: trailing 30-second aggregate and
  per-active-session published FPS. This is not model compute FPS.
- `telefuser_serving_slo_*`: each chunk is judged against
  `frames / configured_fps`, using scheduler queue wait plus compute time.
- DCGM metrics cover GPU utilization, framebuffer/HBM use, power, temperature,
  PCIe, NVLink, DRAM, and tensor-core fields supported by the installed driver.
  Node Exporter provides CPU, RAM, disk, and host-network counters.

OpenTelemetry is deliberately optional: use it only when per-action trace IDs
are needed. Prometheus remains the experiment source of truth because it avoids
high-cardinality session/action labels.

## Useful PromQL

```promql
# P95 model chunk time
histogram_quantile(0.95, sum by (le) (rate(telefuser_serving_chunk_latency_seconds_bucket[1m])))

# Per-stage P95
histogram_quantile(0.95, sum by (stage, le) (rate(telefuser_serving_pipeline_stage_latency_seconds_bucket[1m])))

# SLO attainment over the last minute
sum(rate(telefuser_serving_slo_chunks_total{result="met"}[1m]))
/
sum(rate(telefuser_serving_slo_chunks_total[1m]))

# Published user-visible aggregate FPS
telefuser_serving_published_fps{scope="aggregate"}
```
