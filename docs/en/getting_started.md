# Get Started

Choose the shortest path for the result you need. TeleFuser requires a CUDA-capable system for model execution;
documentation builds and CPU unit tests do not require a GPU.

## Core Path

1. [Install TeleFuser](installation.md) and verify the command-line interface.
2. [Verify the WebRTC path](streaming_quickstart.md) with the self-contained, model-free preflight.
3. Run LingBot-World v2 from the same guide to validate interactive world-model generation in the browser.

For a smaller single-GPU package and batch-API check, use [Basic Inference](quickstart.md).

## Choose a Workflow

| Goal | Start here |
| --- | --- |
| Experience interactive LingBot-World v2 in a browser | [Core WebRTC Experience](streaming_quickstart.md) |
| Run a model directly from Python | [Basic Inference](quickstart.md) |
| Expose batch image or video generation over HTTP | [Serving and APIs](serving.md) |
| Design or operate a streaming deployment | [Stream Server](stream_server.md) |
| Select a checkpoint and example | [Supported Models](supported_models.md) |
| Diagnose installation or runtime failures | [Troubleshooting](troubleshooting.md) |
| Integrate a new pipeline | [Developer Guide](adding_new_model.md) |

## Scope

TeleFuser covers offline generation, batch HTTP serving, and stateful streaming. The WebRTC preflight separates
transport failures from model and GPU failures, while the basic inference path isolates package and batch-service
setup. Choose the path that matches the capability you need to validate.
