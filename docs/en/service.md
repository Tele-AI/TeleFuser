# TeleFuser Service Guide

This guide covers the TeleFuser API server, CLI usage, and HTTP API reference. For real-time streaming via WebRTC, see the [Stream Server Guide](stream_server.md).

## Table of Contents

- [Quick Start](#quick-start)
- [Serving Modes](#serving-modes)
- [CLI Usage](#cli-usage)
- [Supported Pipelines](#supported-pipelines)
- [Server Configuration](#server-configuration)
- [Pipeline Pool (Multi-Replica Serving)](#pipeline-pool-multi-replica-serving)
- [Service Metadata Guide](./service_metadata.md)
- [HTTP API Reference](#http-api-reference)
- [Client SDK](#client-sdk)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install TeleFuser

```bash
pip install telefuser
```

### 2. Start the Server

```bash
# For video generation
telefuser serve \
    ./examples/wan_video/wan21_14b_image_to_video_h100.py \
    --task i2v \
    --port 8000 \
    --parallelism 1

# For image generation
telefuser serve \
    /path/to/image_pipeline.py \
    --task t2i \
    --port 8000 \
    --parallelism 1

# For real-time world model streaming (WebRTC support is included in the default install)
# Set TF_MODEL_ZOO_PATH and PPL_CONFIG["parallelism"] in
# examples/lingbot/stream_lingbot_world_fast.py
telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py -p 8088 --skip-validation
```

### 3. Create a Task

```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2v",
        "prompt": "astronaut walking on the moon",
        "resolution": "720p",
        "aspect_ratio": "16:9"
    }'
```

---

## Serving Modes

TeleFuser provides two serving commands optimized for different workload types:

### `telefuser serve` — Batch Request-Response Mode

Use for batch text-to-video, image-to-video, image generation, and super-resolution.

- Task-based API under `/v1/tasks/*`
- OpenAI-compatible routes under `/v1/images` and `/v1/videos`
- Pipeline contracts for structured parameter exposure
- Optional async scheduling for pipelines that use the orchestrator

### `telefuser stream-serve` — Continuous Streaming Mode

Use for real-time world models, interactive generation, speech-driven animation, and streaming media.

- Server-push WebRTC for progressive video output
- Bidirectional WebRTC for interactive control loops (DataChannel + RTP)
- Stateful sessions with continuous chunk generation

See the [Stream Server Guide](stream_server.md) for full streaming documentation.

---

## Supported Pipelines

### World Model and Real-Time Oriented

| Pipeline | Task | Notes |
|----------|------|-------|
| `LingBot-World-Fast` | Bidirectional world-model streaming | Interactive WebRTC control loop — see [Stream Server Guide](stream_server.md) |
| `LiveAct` | S2V (speech-to-video) | Speech-driven talking head generation |
| `FlashVSR` | VSR | Streaming video super-resolution |
| `LongCat-Video` | T2V, I2V, VC | Long-form generation and continuation |

### Video Generation

| Pipeline | Task | Notes |
|----------|------|-------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | Main video generation family |
| `HunyuanVideo` | T2V, I2V | Supported via service examples |
| `LTX Video` | I2V + Audio | Unified audio-video generation |

### Image Generation

| Pipeline | Task | Notes |
|----------|------|-------|
| `Qwen-Image` | T2I, Edit | Image generation and editing |
| `Z-Image` | T2I | Image generation |
| `Flux2 Klein` | T2I | Image generation |

---

## CLI Usage

The TeleFuser CLI provides commands for starting API servers, validating pipelines, and scanning for security issues.

### Available Commands

| Command | Description |
|---------|-------------|
| `serve` | Start the TeleFuser API server |
| `validate` | Validate a pipeline configuration file |
| `scan` | Scan a directory for pipeline files |

### Serve Command

Start the TeleFuser API server.

```bash
telefuser serve /path/to/pipeline --task i2v [OPTIONS]
```

#### Parameters

| Parameter | Shortcut | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `pipe_path` | | string | **Required** | Positional path to the pipeline Python file |
| `--task` | `-t` | choice | `i2v` | Task type: t2v, i2v, fl2v, vc, t2i, i2i, s2v, vsr |
| `--port` | `-p` | int | `8000` | Server port |
| `--host` | | string | `127.0.0.1` | Server host address |
| `--cache-dir` | `-c` | string | `work_dirs/server_cache` | Cache directory |
| `--parallelism` | `-g` | int | `1` | Number of parallel workers |
| `--num-replicas` | `-n` | int | `1` | Number of independent pipeline replicas (Pipeline Pool) |
| `--security-level` | | choice | `strict` | Validation level: none/basic/strict/sandbox. `sandbox` is a best-effort restricted-load check, not runtime isolation. |
| `--skip-validation` | | flag | `False` | Skip security validation |
| `--validate-only` | | flag | `False` | Only validate without starting |

#### Examples

```bash
# Image-to-Video with full parameters
telefuser serve \
    ./examples/wan_video/wan21_14b_image_to_video_h100.py \
    --task i2v \
    --port 8080 \
    --host 0.0.0.0 \
    --parallelism 2

# Using short form
telefuser serve ./pipeline.py -t i2v -p 8080 -g 2

# Validate only
telefuser serve ./pipeline.py --validate-only

# Skip validation (not recommended for production)
telefuser serve ./pipeline.py --skip-validation
```

### Validate Command

Validate a pipeline file for security issues.

```bash
telefuser validate /path/to/pipeline.py [OPTIONS]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline_file` | **Required** | Path to pipeline Python file |
| `--level` | `strict` | Validation level: none/basic/strict/sandbox. `sandbox` is a best-effort restricted-load check, not runtime isolation. |
| `--json` | `False` | Output in JSON format |

```bash
# Default validation
telefuser validate ./pipeline.py

# Specific security level
telefuser validate ./pipeline.py --level basic

# JSON output
telefuser validate ./pipeline.py --json
```

### Scan Command

Scan a directory for pipeline files and validate them.

```bash
telefuser scan /path/to/directory [OPTIONS]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `directory` | **Required** | Directory to scan |
| `--level` | `strict` | Security validation level |
| `--recursive` / `--no-recursive` | `True` | Scan recursively |

```bash
# Scan recursively
telefuser scan ./examples

# Scan without recursion
telefuser scan ./examples --no-recursive
```

### Getting Help

```bash
# All commands
telefuser --help

# Specific command
telefuser serve --help
```

---

## Server Configuration

### Supported Task Types

| Task | Description |
|------|-------------|
| `t2v` | Text-to-Video: Generate video from text prompt |
| `i2v` | Image-to-Video: Generate video from input image |
| `fl2v` | First-Last to Video: Generate video from first and last frame images |
| `vc` | Video Continue: Continue an existing video |
| `t2i` | Text-to-Image: Generate image from text prompt |
| `i2i` | Image-to-Image: Generate image from input image and prompt |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEFUSER_SECURITY_LEVEL` | Security validation level | `STRICT` |
| `TELEFUSER_ALLOW_UNSAFE` | Allow unsafe pipelines | `false` |
| `TELEFUSER_MAX_PPL_SIZE` | Max pipeline file size (bytes) | `10485760` |
| `TELEFUSER_TASK_TIMEOUT` | Task timeout (seconds) | `3600` |
| `TELEFUSER_HOST` | Server host | `127.0.0.1` |
| `TELEFUSER_PORT` | Server port | `8000` |
| `TELEFUSER_RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
| `TELEFUSER_RATE_LIMIT_REQUESTS_PER_MINUTE` | Requests per minute limit | `60` |
| `TELEFUSER_TRUST_FORWARDED_FOR` | Trust `X-Forwarded-For` for rate-limit identity. Enable only behind trusted proxies. | `false` |
| `TELEFUSER_ARTIFACT_STORAGE_BACKEND` | Artifact backend. Only `local` is implemented currently. | `local` |
| `TELEFUSER_ARTIFACT_LOCAL_ROOT` | Local artifact root. Defaults to `TELEFUSER_CACHE_DIR`. | unset |
| `TELEFUSER_ARTIFACT_PERSISTENCE_MODE` | Local artifact retention mode: `persistent` or `ephemeral`. | `persistent` |
| `TELEFUSER_ARTIFACT_RETENTION_SECONDS` | Retention time for terminal task artifacts. `0` disables TTL cleanup. | `604800` |
| `TELEFUSER_ARTIFACT_TMP_RETENTION_SECONDS` | Retention time for temporary `.part` files. `0` disables tmp cleanup. | `3600` |
| `TELEFUSER_ARTIFACT_CLEANUP_INTERVAL_SECONDS` | Background artifact cleanup interval. | `3600` |
| `TELEFUSER_ARTIFACT_MAX_TOTAL_BYTES` | Local artifact cache size limit. `0` disables capacity cleanup. | `0` |
| `TELEFUSER_ARTIFACT_MAX_TASK_BYTES` | Per-task artifact size limit for terminal tasks. `0` disables per-task cleanup. | `0` |
| `TELEFUSER_ARTIFACT_PRESERVE_FAILED_OUTPUTS` | Preserve failed task directories from automatic artifact cleanup. | `false` |

### Configuration File Example

Create `.env` file:

```env
TELEFUSER_SECURITY_LEVEL=STRICT
TELEFUSER_PORT=8080
TELEFUSER_HOST=0.0.0.0
TELEFUSER_RATE_LIMIT_ENABLED=true
TELEFUSER_RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

### Runtime Boundaries

The request-response service is intentionally local and single-process by default:

- Task status, queue state, cancellation state, and rate-limit buckets are kept in process memory. They are not shared across multiple server instances and are not restored after restart.
- File inputs, generated outputs, and temporary `.part` files are managed by the local artifact store under the configured cache root.
- `artifact_storage_backend=s3` is declared as a future backend boundary, but it is not implemented yet. Setting it to `s3` fails at startup instead of silently falling back to local storage.
- `telefuser serve` exposes request-response routes only: `/v1/tasks/*`, `/v1/files/*`, `/v1/images/*`, `/v1/videos/*`, and `/v1/service/*`.
- `telefuser stream-serve` exposes stream routes only: `/v1/stream/*`, `/v1/stream/webrtc/*`, and `/v1/service/*`. It does not expose task, file-download, or OpenAI-compatible request-response routes.

### Artifact Storage and Cleanup

Artifacts are local files rooted at `TELEFUSER_ARTIFACT_LOCAL_ROOT` when set, otherwise `TELEFUSER_CACHE_DIR`.
Uploads and remote downloads are written through temporary `.part` files and atomically renamed after successful completion.

Outputs are task-scoped when a task id is available:

```text
<artifact_root>/tasks/<task_id>/inputs/
<artifact_root>/tasks/<task_id>/outputs/
<artifact_root>/tasks/<task_id>/tmp/
```

Only paths resolved inside the artifact root are accepted. Absolute output paths and `..` traversal are rejected.
Download routes accept validated output paths and local artifact ids. Local artifact ids use the
`local:tasks/<task_id>/outputs/<media_type>/<filename>` form and only refer to downloadable outputs in the local
artifact backend. They are not remote object-storage ids.

Cleanup is best-effort and never removes active tasks (`pending`, `processing`, or `streaming`). Terminal tasks
(`completed`, `failed`, or `cancelled`) may be removed by TTL cleanup, total-cache capacity cleanup, or per-task size cleanup.
Temporary `.part` files are cleaned independently by `TELEFUSER_ARTIFACT_TMP_RETENTION_SECONDS`.

`TELEFUSER_ARTIFACT_PERSISTENCE_MODE=persistent` keeps terminal task artifacts until TTL or capacity cleanup removes
them. `ephemeral` removes terminal task directories on the next cleanup pass. If
`TELEFUSER_ARTIFACT_PRESERVE_FAILED_OUTPUTS=true`, failed task directories are protected from automatic cleanup so
partial outputs can be inspected manually.

Recommended cleanup profiles:

```bash
# Development/debugging: keep outputs for one day and preserve failed task outputs.
export TELEFUSER_ARTIFACT_PERSISTENCE_MODE=persistent
export TELEFUSER_ARTIFACT_RETENTION_SECONDS=86400
export TELEFUSER_ARTIFACT_TMP_RETENTION_SECONDS=1800
export TELEFUSER_ARTIFACT_CLEANUP_INTERVAL_SECONDS=600
export TELEFUSER_ARTIFACT_PRESERVE_FAILED_OUTPUTS=true

# Temporary benchmarking: remove terminal task directories on the next cleanup pass.
export TELEFUSER_ARTIFACT_PERSISTENCE_MODE=ephemeral
export TELEFUSER_ARTIFACT_TMP_RETENTION_SECONDS=600
export TELEFUSER_ARTIFACT_CLEANUP_INTERVAL_SECONDS=300
export TELEFUSER_ARTIFACT_PRESERVE_FAILED_OUTPUTS=false

# Long-running service: keep artifacts for seven days with bounded local storage.
export TELEFUSER_ARTIFACT_PERSISTENCE_MODE=persistent
export TELEFUSER_ARTIFACT_RETENTION_SECONDS=604800
export TELEFUSER_ARTIFACT_TMP_RETENTION_SECONDS=3600
export TELEFUSER_ARTIFACT_CLEANUP_INTERVAL_SECONDS=3600
export TELEFUSER_ARTIFACT_MAX_TOTAL_BYTES=107374182400
export TELEFUSER_ARTIFACT_MAX_TASK_BYTES=10737418240
```

### Rate Limit and Proxy Headers

Rate limiting is in-memory and keyed by client identity. By default, the server uses the direct client address and does
not trust `X-Forwarded-For`, because arbitrary clients can spoof that header. Set
`TELEFUSER_TRUST_FORWARDED_FOR=true` only when TeleFuser is behind a trusted reverse proxy that sanitizes forwarded
headers.

The default limited paths cover expensive generation endpoints, artifact downloads, and stream negotiation:

```text
/v1/tasks/create
/v1/tasks/form
/v1/images
/v1/videos
/v1/files/download
/v1/stream
```

Liveness/readiness endpoints remain unthrottled for infrastructure probes.

### Pipeline Contract and Parameter Definitions

The service does not infer pipeline capabilities from Python signatures alone. Instead, it loads an explicit
pipeline contract from the example script when available.

#### Contract Entry Points

The server looks for one of the following in the pipeline file:

- `get_pipeline_contract()`
- `get_pipeline_manifest()`
- `PIPELINE_CONTRACT`
- `PIPELINE_MANIFEST`

If none of them exist, the server falls back to a legacy compatibility contract based on the CLI `--task`.

#### Contract Structure

At minimum, a pipeline contract should declare:

```python
PIPELINE_MANIFEST = {
  "contract_version": "v1",
  "pipeline_name": "wan22_A14B_i2v_h100_distill",
  "supported_tasks": ["i2v"],
  "supported_media_types": ["video"],
  "execution_mode": "serial_single_pipeline",
  "effective_max_concurrent_tasks": 1,
  "entrypoints": {
    "get_pipeline": "get_pipeline",
    "run_with_file": "run_with_file",
  },
  "task_contracts": {
    "i2v": {
      "media_type": "video",
      "required_inputs": ["first_image_path"],
      "optional_inputs": ["last_image_path"],
      "parameters": {
        "prompt": {
          "type": "string",
          "required": True,
          "default": "",
          "description": "Positive guidance text prompt.",
        },
        "resolution": {
          "type": "string",
          "required": False,
          "default": "720p",
          "enum": ["480p", "720p"],
          "description": "Output resolution exposed by this example.",
        },
      },
    },
  },
}
```

#### What Belongs in `task_contracts`

Each task contract is split into two parts:

- `required_inputs` and `optional_inputs`: file-like inputs that determine task eligibility and validation.
- `parameters`: user-facing runtime parameters that the server may default and validate.

Only parameters that users should know about belong in `parameters`. Internal implementation details such as
`num_inference_steps`, fixed distillation settings, or pipeline-private tuning knobs should remain in `PPL_CONFIG`
or inside `run()`/`run_with_file()`.

#### How the Server Uses the Contract

For `/v1/tasks/create`, `/v1/tasks/form`, and the OpenAI-compatible routes, the server processes requests in this order:

1. Resolve or validate the task name against `supported_tasks`.
2. Check file-style inputs against `required_inputs`.
3. Apply defaults from `task_contracts[task]["parameters"]` for user-facing parameters that the caller omitted.
4. Validate required user-facing parameters declared in the contract.
5. Fall back to built-in request-model defaults only for fields not defined by the contract.

This means contract defaults take precedence over generic API defaults when both exist.

#### Service Metadata

`GET /v1/service/metadata` exposes the active contract back to clients. This is the recommended way for a UI,
gateway, or automation layer to discover which tasks are currently supported and which user-facing parameters are
available for each task.

For client-side form generation, task routing, and gateway integration patterns, see
[Service Metadata Consumption Guide](./service_metadata.md).

---

## Pipeline Pool (Multi-Replica Serving)

Pipeline Pool enables concurrent inference by running multiple independent pipeline replicas, each in its own subprocess with exclusive GPU access. This provides true data-parallel serving without shared state between replicas.

### How It Works

When `--num-replicas` is set to N (with `--parallelism` providing the total GPU count), the server:

1. Splits the available GPUs evenly across N replicas (e.g., 4 GPUs with 2 replicas → 2 GPUs per replica)
2. Spawns each replica in a separate subprocess with isolated `CUDA_VISIBLE_DEVICES`
3. Routes incoming requests to idle replicas via round-robin dispatch
4. Rejects requests with HTTP 503 when the queue is full (configurable `max_queue_size`)

Each replica is a fully independent pipeline instance — there is no shared GPU memory or model state between replicas.

### CLI Usage

```bash
# 2 replicas on 2 GPUs (1 GPU each)
CUDA_VISIBLE_DEVICES=0,1 telefuser serve ./pipeline.py -g 2 -n 2

# 4 replicas on 8 GPUs (2 GPUs each, with tensor parallelism)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 telefuser serve ./pipeline.py -g 8 -n 4

# Single replica (default, same as before)
telefuser serve ./pipeline.py -g 1
```

The total GPU count (`-g`) must be evenly divisible by the number of replicas (`-n`).

### Monitoring

When Pipeline Pool is active, `GET /v1/service/status` returns additional pool information:

```json
{
  "service_status": "idle",
  "processing_count": 0,
  "max_concurrent_processing": 2,
  "execution_mode": "concurrent_pipeline_pool",
  "pool": [
    {"id": 0, "device_ids": ["0"], "status": "idle"},
    {"id": 1, "device_ids": ["1"], "status": "idle"}
  ]
}
```

`GET /v1/service/metadata` also includes pool configuration:

```json
{
  "pool": {
    "num_replicas": 2,
    "live_replicas": 2,
    "replica_device_ids": [["0"], ["1"]],
    "per_instance_execution_mode": "serial_single_pipeline"
  }
}
```

### Replica Eviction

If a replica subprocess crashes or becomes unresponsive, the pool automatically:

1. Evicts the dead replica from the rotation
2. Reduces the concurrent processing capacity
3. Continues serving with the remaining live replicas
4. Logs a critical error if all replicas are dead

### Best Practices

- **GPU sizing**: Each replica gets `parallelism / num_replicas` GPUs. Ensure each replica has enough VRAM for the model.
- **Queue size**: The default queue size is 10. For high-throughput scenarios, adjust via `TELEFUSER_MAX_QUEUE_SIZE`.
- **Rate limiting**: When stress-testing, increase the rate limit: `TELEFUSER_RATE_LIMIT_REQUESTS_PER_MINUTE=10000`.
- **Troubleshooting**: Check `GET /v1/service/status` → `pool` to verify all replicas are alive and GPU assignments are correct.

---

## HTTP API Reference

### Base URL

```
http://localhost:8000/v1
```

### Interactive Documentation

When the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tasks/create` | Create a new generation task |
| POST | `/tasks/form` | Create task with file upload |
| GET | `/tasks/{task_id}/status` | Get task status |
| DELETE | `/tasks/{task_id}` | Request cooperative task cancellation |
| GET | `/tasks/queue/status` | Get queue status |
| GET | `/files/download/{file_id}` | Download an output file by validated path or local artifact id |
| GET | `/service/health` | Health check |
| GET | `/service/status` | Service status |
| GET | `/service/metadata` | Service metadata |
| GET | `/service/metrics` | Prometheus metrics |
| GET | `/service/metrics/json` | Metrics in JSON format |

### Create Task

```
POST /v1/tasks/create
```

Create a new generation task.

**Request Body**: `TaskRequest`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | No | `t2v` | Task type: t2v, i2v, fl2v, vc, t2i, i2i |
| `prompt` | string | Yes | - | Text prompt for generation |
| `aspect_ratio` | string | No | `16:9` | Aspect ratio |
| `resolution` | string | No | `720p` | Resolution (720p, 1080p for video; 1024x1024 for image) |
| `seed` | int | No | `42` | Random seed |
| `negative_prompt` | string | No | `""` | Negative prompt |
| `first_image_path` | string | No | `""` | Input image path (for I2V, I2I) |
| `last_image_path` | string | No | `""` | Last frame image (for FL2V) |
| `ref_video_path` | string | No | `""` | Reference video (for VC) |
| `target_video_length` | int | No | `5` | Video length in seconds |
| `output_format` | string | No | `png` | Output format for images |
| `output_path` | string | No | Auto | Custom output path |

**Response**: `TaskResponse`

```json
{
  "task_id": "task_abc123xyz",
  "task_status": "pending",
  "output_path": "task_abc123xyz.mp4"
}
```

#### Examples

**Text-to-Video (T2V)**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2v",
        "prompt": "Astronaut walking on the moon",
        "resolution": "720p",
        "aspect_ratio": "16:9"
    }'
```

**Image-to-Video (I2V)**:
```bash
# Using base64 encoded image
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "i2v",
        "prompt": "Astronaut walking",
        "first_image_path": "data:image/jpeg;base64,/9j/4AAQ..."
    }'

# Or upload image with task creation via form endpoint
curl -X POST "http://127.0.0.1:8000/v1/tasks/form" \
    -F "first_image_file=@/path/to/input.jpg" \
    -F "prompt=Astronaut walking" \
    -F "task=i2v"
```

**Text-to-Image (T2I)**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2i",
        "prompt": "Beautiful landscape",
        "resolution": "1024x1024",
        "output_format": "png"
    }'
```

### Get Task Status

```
GET /v1/tasks/{task_id}/status
```

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/tasks/task_abc123xyz/status"
```

**Response**:

```json
{
  "task_id": "task_abc123xyz",
  "status": "completed",
  "start_time": "2024-01-15T08:30:00",
  "end_time": "2024-01-15T08:35:00",
  "error": null,
  "output_path": "task_abc123xyz.mp4"
}
```

### Cancel Task

```
DELETE /v1/tasks/{task_id}
```

**Example**:
```bash
curl -X DELETE "http://127.0.0.1:8000/v1/tasks/task_abc123xyz"
```

### Get Queue Status

```
GET /v1/tasks/queue/status
```

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/tasks/queue/status"
```

**Response**:

```json
{
  "is_processing": true,
  "current_task": "task-123",
  "pending_count": 3,
  "active_count": 1,
  "queue_size": 10
}
```

### Health Check

```
GET /v1/service/health
```

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/service/health"
```

**Response**:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T08:30:00Z",
  "version": "1.0.0",
  "pipeline_ready": true
}
```

### Service Status

```
GET /v1/service/status
```

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/service/status"
```

### Service Metadata

```
GET /v1/service/metadata
```

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/service/metadata"
```

**Response**:

```json
{
  "pipeline_file": "/path/to/pipeline.py",
  "parallelism": 1,
  "task": "i2v",
  "security_level": "STRICT",
  "supported_tasks": ["t2v", "i2v", "fl2v", "vc", "t2i", "i2i"],
  "supported_media_types": ["video", "image"]
}
```

### Get Prometheus Metrics

```
GET /v1/service/metrics
```

Get metrics in Prometheus-compatible text format for monitoring systems.

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/service/metrics"
```

**Response** (Prometheus format):
```
# HELP telefuser_tasks_created_total Total number of tasks created
# TYPE telefuser_tasks_created_total counter
telefuser_tasks_created_total 100

# HELP telefuser_task_duration_seconds Duration of task execution in seconds
# TYPE telefuser_task_duration_seconds histogram
telefuser_task_duration_seconds_bucket{le="0.005"} 10
telefuser_task_duration_seconds_bucket{le="0.01"} 25
...

# HELP telefuser_gpu_0_memory_used_bytes GPU 0 memory used in bytes
# TYPE telefuser_gpu_0_memory_used_bytes gauge
telefuser_gpu_0_memory_used_bytes 8589934592
```

**Available Metrics**:

| Metric | Type | Description |
|--------|------|-------------|
| `tasks_created_total` | Counter | Total tasks created |
| `tasks_completed_total` | Counter | Tasks completed successfully |
| `tasks_failed_total` | Counter | Tasks that failed |
| `tasks_cancelled_total` | Counter | Tasks cancelled |
| `task_duration_seconds` | Histogram | Task execution duration |
| `queue_size` | Gauge | Total queue size |
| `queue_pending` | Gauge | Pending tasks |
| `queue_processing` | Gauge | Processing tasks |
| `gpu_{id}_memory_used_bytes` | Gauge | GPU memory used |
| `gpu_{id}_utilization_ratio` | Gauge | GPU utilization (0-1) |

### Get Metrics (JSON)

```
GET /v1/service/metrics/json
```

Get metrics in JSON format for easy parsing and debugging.

**Example**:
```bash
curl "http://127.0.0.1:8000/v1/service/metrics/json"
```

**Response**:

```json
{
  "uptime_seconds": 3600,
  "tasks": {
    "created": 100,
    "completed": 95,
    "failed": 3,
    "cancelled": 2
  },
  "queue": {
    "size": 10,
    "pending": 5,
    "processing": 1
  },
  "metrics_count": 25,
  "registered_stages": ["stage_1", "stage_2"]
}
```

### Download Output File

```
GET /v1/files/download/{file_path}
```

```bash
curl "http://127.0.0.1:8000/v1/files/download/task_abc123xyz.mp4" \
    --output output.mp4
```

---

## OpenAI Compatible API

TeleFuser provides an **OpenAI-compatible** REST API that allows you to use standard OpenAI SDK clients and seamlessly migrate from OpenAI/Anthropic services.

### Overview

| API Type | Endpoint | Characteristics |
|---------|------|------|
| **OpenAI Compatible** | `/v1/images`, `/v1/videos` | Sync/Async, industry-standard format |
| **TeleFuser Native** | `/v1/tasks/*` | Async task management, more powerful |

### When to Use Which API?

| Scenario | Recommended API |
|------|----------|
| Quick prototyping / Simple needs | OpenAI API |
| Existing OpenAI project migration | OpenAI API |
| Production batch processing | TeleFuser Native API |
| Fine-grained control needed | TeleFuser Native API |
| Long-running task monitoring | TeleFuser Native API |

### Image Generation API

#### Endpoints

| Method | Endpoint | Description |
|------|------|------|
| POST | `/v1/images/generations` | Generate image from text (sync) |
| POST | `/v1/images/edits` | Image editing (I2I) |
| GET | `/v1/images/{id}/content` | Download generated image |

#### Request Parameters

```json
{
  "prompt": "a beautiful cat",          // Required: generation prompt
  "model": "qwen-image",                // Optional: model name
  "n": 1,                               // Optional: number of images (1-10)
  "quality": "auto",                    // Optional: quality (standard/hd/auto)
  "response_format": "url",             // Optional: url or b64_json
  "size": "1024x1024",                  // Optional: image size
  "style": "vivid",                     // Optional: style (vivid/natural)
  "user": "user_id",                    // Optional: user identifier
  "seed": 42,                           // Optional: random seed
  "negative_prompt": "blurry"           // Optional: negative prompt
}
```

#### Response Format

```json
{
  "created": 1699000000,
  "data": [
    {
      "url": "http://localhost:8000/v1/images/task_xxx/content",
      "revised_prompt": "a beautiful cat",
      "file_path": "/cache/tasks/task_xxx/outputs/images/result.png",
      "artifact_id": "local:tasks/task_xxx/outputs/images/result.png",
      "artifact_metadata": {
        "backend": "local",
        "relative_path": "tasks/task_xxx/outputs/images/result.png",
        "task_id": "task_xxx",
        "media_type": "image",
        "filename": "result.png",
        "size_bytes": 123456
      }
    }
  ],
  "peak_memory_mb": 4096.5,
  "inference_time_s": 2.5
}
```

#### Example

```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "a beautiful sunset",
        "size": "1024x1024",
        "response_format": "url"
    }'
```

**Python SDK:**
```python
from telefuser.client.openai import OpenAICompatibleClient

client = OpenAICompatibleClient("http://localhost:8000")

# Generate image
response = client.images.generate(
    prompt="a beautiful sunset",
    size="1024x1024",
    response_format="url"
)

# Save image
response.data[0].save("sunset.png")
```

### Video Generation API

#### Endpoints

| Method | Endpoint | Description |
|------|------|------|
| POST | `/v1/videos` | Create video generation task (async) |
| GET | `/v1/videos` | List video tasks |
| GET | `/v1/videos/{id}` | Get video status |
| DELETE | `/v1/videos/{id}` | Cancel/delete task |
| GET | `/v1/videos/{id}/content` | Download video |

#### Request Parameters

```json
{
  "prompt": "a cat playing piano",      // Required: generation prompt
  "input_reference": "/path/to/img",    // Optional: input image (I2V)
  "reference_url": "http://example.com",// Optional: input image URL
  "model": "wan-video",                 // Optional: model name
  "seconds": 5,                         // Optional: duration (1-60)
  "size": "1024x576",                   // Optional: video size
  "seed": 1024,                         // Optional: random seed
  "negative_prompt": "blurry",          // Optional: negative prompt
  "output_path": "custom-output.mp4"    // Optional: relative custom output path
}
```

#### Response Format

```json
{
  "id": "vid_xxx",
  "object": "video",
  "model": "wan-video",
  "status": "queued",
  "progress": 0,
  "created_at": 1699000000,
  "size": "1024x576",
  "seconds": "5",
  "url": null,
  "file_path": null,
  "artifact_id": null,
  "artifact_metadata": null
}
```

Completed video responses include a content URL and local artifact metadata:

```json
{
  "id": "vid_xxx",
  "status": "completed",
  "progress": 100,
  "url": "http://localhost:8000/v1/videos/vid_xxx/content",
  "file_path": "/cache/tasks/vid_xxx/outputs/videos/result.mp4",
  "artifact_id": "local:tasks/vid_xxx/outputs/videos/result.mp4",
  "artifact_metadata": {
    "backend": "local",
    "relative_path": "tasks/vid_xxx/outputs/videos/result.mp4",
    "task_id": "vid_xxx",
    "media_type": "video",
    "filename": "result.mp4",
    "size_bytes": 987654
  }
}
```

#### Example

```bash
# Create task
curl -X POST "http://localhost:8000/v1/videos" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "a cat playing piano",
        "seconds": 5,
        "size": "720p"
    }'

# Get status
curl "http://localhost:8000/v1/videos/{video_id}"

# Download video
curl "http://localhost:8000/v1/videos/{video_id}/content" \
    -o output.mp4
```

**Python SDK:**
```python
from telefuser.client.openai import OpenAICompatibleClient

client = OpenAICompatibleClient("http://localhost:8000")

# Create video task
video = client.videos.create(
    prompt="a cat playing piano",
    seconds=5,
    size="720p"
)

print(f"Task ID: {video.id}")
print(f"Status: {video.status}")

# Wait for completion
video.wait(timeout=300)

# Download video
video.download("output.mp4")
```

### Configuration

#### Enable/Disable OpenAI API

OpenAI compatible API is enabled by default. You can control it via:

```python
from telefuser.service.api.api_server import ApiServer

# Enable OpenAI API (default)
server = ApiServer(enable_openai_api=True)

# Disable OpenAI API
server = ApiServer(enable_openai_api=False)
```

Or via CLI:
```bash
# Default: OpenAI API enabled
telefuser serve ./pipeline.py --task t2v
```

---

## Client SDK

### Installation

```bash
pip install telefuser
```

### Basic Usage

```python
from telefuser.client import TFClient

client = TFClient(base_url="http://127.0.0.1:8000")

# Create a video task
task = client.create_t2v_task(
    prompt="Astronaut walking on the moon",
    resolution="720p",
    seed=42
)

print(f"Task created: {task['task_id']}")

# Wait for completion
if client.wait_for_completion(task['task_id']):
    client.download_result(task['task_id'], "./output.mp4")
```

### Task Methods

| Method | Description |
|--------|-------------|
| `create_t2v_task()` | Create text-to-video task |
| `create_i2v_task()` | Create image-to-video task |
| `create_fl2v_task()` | Create first-last to video task |
| `create_vc_task()` | Create video continue task |
| `create_t2i_task()` | Create text-to-image task |
| `create_i2i_task()` | Create image-to-image task |
| `get_task_status()` | Get task status |
| `wait_for_completion()` | Wait for task completion |
| `download_result()` | Download result file |

### Examples

**Image-to-Video**:
```python
task = client.create_i2v_task(
    prompt="Astronaut walking",
    first_image_path="input.jpg",
    resolution="720p"
)
```

**Text-to-Image**:
```python
task = client.create_t2i_task(
    prompt="Beautiful mountain landscape",
    resolution="1024x1024",
    output_format="png"
)
```

**Image-to-Image**:
```python
task = client.create_i2i_task(
    prompt="Transform into oil painting style",
    image_path="input.jpg",
    resolution="1024x1024",
    output_format="jpg"
)
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Task or file not found |
| 422 | Validation Error - Invalid parameters |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Server overloaded |

### Error Response Format

```json
{
  "detail": [
    {
      "loc": ["body", "task"],
      "msg": "Invalid task type",
      "type": "value_error"
    }
  ]
}
```

### Rate Limiting

- **Limit**: 60 requests per minute per IP
- **Burst**: 10 requests
- **Exempt paths**: `/v1/service/health`

---

## Best Practices

### Resource Management

- Specify appropriate `parallelism` based on GPU memory
- Use dedicated cache directory for production
- Clean up old cache files periodically

### Error Handling

- Wrap client calls in try-except blocks
- Implement exponential backoff for retries
- Check task status before assuming completion

### Security

- Use `strict` security level in production
- Validate all input files before uploading
- Run server behind a reverse proxy with SSL

### Performance

```python
# Batch processing
from telefuser.client import TFClient
import concurrent.futures

client = TFClient(base_url="http://127.0.0.1:8000")

prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

def generate(prompt):
    task = client.create_t2i_task(prompt=prompt, resolution="1024x1024")
    return client.wait_for_completion(task['task_id'])

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(generate, prompts))
```

---

## Troubleshooting

### Connection Refused

```bash
# Check if server is running
curl http://127.0.0.1:8000/v1/service/health

# Check firewall
sudo ufw allow 8000
```

### Out of Memory

- Reduce `--parallelism`
- Enable CPU offloading in pipeline config
- Use smaller batch sizes

### Task Timeout

```python
# Increase timeout
client.wait_for_completion(task_id, timeout=600)
```

### Pipeline Validation Failed

```bash
# Validate pipeline
telefuser validate /path/to/pipeline.py --level strict

# Skip validation (not recommended for production)
telefuser serve /path/to/pipeline.py --skip-validation
```

### Port Already in Use

```bash
telefuser serve /path/to/pipeline.py --port 8081
```

### Security Validation Failed

1. Check detailed report: `telefuser validate /path/to/pipeline.py`
2. Fix security issues in pipeline file
3. Or bypass validation: `telefuser serve /path/to/pipeline.py --skip-validation`

---

*For more information, visit the [TeleFuser documentation](https://github.com/telefuser/telefuser).*
