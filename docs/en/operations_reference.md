# Operations and Reference

Use this section after a pipeline runs correctly in standalone mode. It collects service inspection, metrics,
logging, profiling, testing, and symptom-based diagnostics without mixing them into the first-run tutorial.

## Find the Right Reference

| Need | Reference |
| --- | --- |
| Inspect CLI commands and batch HTTP endpoints | [Batch Service and API Reference](service.md) |
| Inspect a loaded pipeline's accepted parameters | [Service Metadata](service_metadata.md) |
| Configure streaming workers and LiveKit sessions | [Stream Server](stream_server.md) |
| Interpret raw runtime measurements | [Metrics](metrics.md) |
| Configure process and request logging | [Logging](logging.md) |
| Capture stage timings and GPU traces | [Profiler](profiler.md) |
| Reproduce batch and streaming benchmarks | [TeleFuser and AIPerf](benchmark_aiperf.md) |
| Run CPU, GPU, distributed, or regression tests | [Testing](testing.md) |
| Diagnose common failures | [Troubleshooting](troubleshooting.md) |

## Runtime Inspection

For a batch server, start with these endpoints:

```bash
curl --fail http://127.0.0.1:8000/v1/service/health
curl --fail http://127.0.0.1:8000/v1/service/status
curl --fail http://127.0.0.1:8000/v1/service/metadata
curl --fail http://127.0.0.1:8000/v1/service/metrics
```

Record the TeleFuser commit, example file, checkpoint identifier, GPU model, CUDA and PyTorch versions, precision,
parallel configuration, and complete command when reporting a result or failure.

## Reference Policy

The running CLI `--help`, OpenAPI schema, Pydantic service schemas, configuration dataclasses, and tests are the API
source of truth. Narrative documentation should explain those contracts and must not introduce parallel options or
environment variables.
