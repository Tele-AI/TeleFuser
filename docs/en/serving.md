# Serving and APIs

TeleFuser exposes two service modes with different lifecycle and transport contracts. Select the mode from the
workload rather than treating them as interchangeable server configurations.

## Choose a Serving Mode

| Workload | Command | Transport | Lifecycle |
| --- | --- | --- | --- |
| Image, video, audio-video, restoration, or VLA task | `telefuser serve` | HTTP request and polling | Finite task |
| Continuous generation or interactive world model | `telefuser stream-serve` | HTTP control plus LiveKit media/data | Stateful session |

Use the [Batch Service and API Reference](service.md) for finite generation tasks. It covers task APIs,
OpenAI-compatible image and video routes, the Python client, pipeline replicas, metrics, and error responses.

Use the [Stream Server Guide](stream_server.md) for server-push and bidirectional sessions. It covers LiveKit room
roles, worker admission, GPU placement, reconnect behavior, and production networking.

## Batch Service Contract

```bash
telefuser serve /path/to/pipeline.py --task t2v --port 8000
```

After startup, inspect the running contract instead of assuming every pipeline accepts the same inputs:

| Endpoint | Purpose |
| --- | --- |
| `/docs` | Interactive Swagger UI |
| `/openapi.json` | Machine-readable HTTP schema |
| `/v1/service/health` | Liveness check |
| `/v1/service/metadata` | Loaded pipeline and parameter contract |
| `/v1/tasks/create` | Submit a finite task |
| `/v1/tasks/{task_id}/status` | Poll task state |

## Streaming Contract

`telefuser stream-serve` manages retained model workers and session admission. LiveKit carries media and reliable
control messages; the HTTP API creates, inspects, and removes sessions. A batch task ID and a streaming session ID
have different ownership and cleanup semantics.

## Related Guides

- [LingBot-World v2 WebRTC Core Experience](streaming_quickstart.md) validates the interactive browser path.
- [Basic Inference Quickstart](quickstart.md) completes a standalone and batch-service request.
- [Service Metadata](service_metadata.md) defines introspection fields.
- [Stream Scheduler](stream_scheduler.md) explains actor ownership and bounded dataflow.
- [Operations and Reference](operations_reference.md) covers observability and diagnostics.
