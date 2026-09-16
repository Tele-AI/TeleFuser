# Troubleshooting

Start with the smallest failing layer: installation, model loading, standalone inference, batch service, distributed
execution, or LiveKit transport. Do not enable additional optimizations while isolating a failure.

## Capture the Environment

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
telefuser --help
```

Include the example path, full command, checkpoint identifier, GPU topology, and complete traceback in bug reports.

## Import or CUDA Is Unavailable

If `import telefuser` fails, confirm the active interpreter and reinstall the repository in that same environment:

```bash
python -m pip show telefuser
python -m pip install -e .
```

If PyTorch reports no CUDA device, resolve the driver and CUDA-enabled PyTorch installation first. The optional
`tf-kernel` package is not required for the native fallback paths. See [Installation](installation.md) and
[TF-Kernel](tf_kernel.md).

## A Checkpoint Cannot Be Loaded

- Confirm that the selected example supports a hub model ID before passing one directly.
- For local checkpoints, preserve the upstream repository layout and required auxiliary files.
- Compare the path with the model's [Cookbook guide](supported_models.md).
- Verify authentication and available disk space for gated or remote repositories.

Use the [Model Loading](model_loading.md) guide to distinguish Diffusers-format loading from ModuleManager file-list
loading.

## CUDA Out of Memory

First reproduce with the guide's validated resolution, frame count, precision, and GPU count. Then reduce one memory
dimension at a time: resolution, frame count, concurrent replicas, or retained streaming sessions. CPU offload and
quantization change performance and numerical behavior, so enable them only after the baseline runs.

See [CPU Offloading](offload.md), [Quantization](quantization.md), and [Parallel Inference](parallel.md).

## Attention or Kernel Failure

Return to the pipeline's documented dense or native attention backend. If that works, verify the optimized backend's
GPU architecture, PyTorch, CUDA, and package requirements in [Attention](attention.md). For `tf-kernel`, inspect its
recorded build information rather than bypassing compatibility checks.

## Distributed Startup Hangs

Check that every rank sees the intended GPUs and that no previous worker still owns the port or device. Record
`CUDA_VISIBLE_DEVICES`, world size, parallel degrees, and NCCL output. Reproduce on one GPU before restoring the exact
validated topology. See [Parallel Inference](parallel.md) and [Communication Architecture](communication.md).

## Batch Service Does Not Become Ready

Run the pipeline directly first. Then inspect the CLI and service endpoints:

```bash
telefuser validate /path/to/pipeline.py
telefuser serve /path/to/pipeline.py --task t2v --port 8000
curl --fail http://127.0.0.1:8000/v1/service/health
curl --fail http://127.0.0.1:8000/v1/service/status
```

Use [Batch Service and API Reference](service.md) for validation, port conflicts, task failures, and error responses.

## LiveKit Session Connects Without Media

Confirm the TeleFuser health endpoint, LiveKit server URL and credentials, room participants, worker admission, and
TURN reachability separately. Test on loopback before adding proxies or remote networking. Use the topology and
diagnostic sequence in [Stream Server](stream_server.md); use [Stream Scheduler](stream_scheduler.md) for worker and
session ownership.

## Documentation Build Fails

```bash
python -m unittest discover -s tests/docs -v
python scripts/docs/prepare_cookbook.py build
```

Do not edit `.build/` or `site/`. Fix the source document, `docs/cookbook.yml`, or the registered example README.
