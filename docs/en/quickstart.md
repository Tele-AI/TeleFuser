# Basic Inference Quickstart

This guide validates a maintained single-GPU text-to-video example and the batch HTTP service. For TeleFuser's
primary interactive workflow, start with the [LingBot-World v2 WebRTC Core Experience](streaming_quickstart.md).

## Prerequisites

- A Linux checkout of the TeleFuser repository
- Python 3.10 through 3.13 and a working CUDA-enabled PyTorch installation
- One CUDA GPU with enough memory for Wan2.1 T2V 1.3B at 480p
- An existing local Wan2.1 T2V 1.3B checkpoint; this workflow does not download models

Follow [Installation](installation.md) first. Confirm that `torch.cuda.is_available()` returns `True`.

## Run the Pipeline

From the repository root:

```bash
mkdir -p work_dirs
export WAN21_MODEL_SOURCE=/path/to/model_zoo/Wan2.1-T2V-1.3B
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/wan_video/wan21_1_3b_text_to_video_hf.py \
  --model_root "$WAN21_MODEL_SOURCE" \
  --resolution 480p \
  --prompt "A sailboat crosses a calm lake at sunrise"
```

A successful run ends with a `Video saved to:` message and writes:

```text
work_dirs/wan_video_wan21_1_3b_text_to_video_hf.mp4
```

The checkpoint is also published as
[Wan-AI/Wan2.1-T2V-1.3B on Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and
[Wan-AI/Wan2.1-T2V-1.3B on ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B).
Set `WAN21_MODEL_SOURCE` to the existing local repository directory without flattening its layout. The links above
identify the checkpoint; they are not download steps in this guide.

## Start the Batch Service

In the shell where `WAN21_MODEL_SOURCE` is set, keep this process running:

```bash
telefuser serve examples/wan_video/wan21_1_3b_text_to_video_hf.py \
  --task t2v \
  --port 8000
```

Wait for startup to complete, then check the service from another terminal:

```bash
curl --fail http://127.0.0.1:8000/v1/service/health
```

Create a task:

```bash
curl --fail --request POST http://127.0.0.1:8000/v1/tasks/create \
  --header "Content-Type: application/json" \
  --data '{
    "task": "t2v",
    "prompt": "A sailboat crosses a calm lake at sunrise",
    "resolution": "480p",
    "aspect_ratio": "16:9"
  }'
```

The response contains a `task_id`, `task_status`, and `output_path`. Poll the returned task ID until `status` is
`completed`:

```bash
curl --fail http://127.0.0.1:8000/v1/tasks/TASK_ID/status
```

The running server also publishes Swagger UI at `http://127.0.0.1:8000/docs` and its OpenAPI document at
`http://127.0.0.1:8000/openapi.json`.

## Next Steps

- [Supported Models](supported_models.md) lists model families and task-specific guides.
- [Serving and APIs](serving.md) explains batch and streaming modes.
- [Runtime and optimization](configuration.md) covers configuration, parallelism, attention, and caching.
- [Troubleshooting](troubleshooting.md) provides symptom-based diagnostics.

Continue with the [LingBot-World v2 WebRTC Core Experience](streaming_quickstart.md) to exercise the framework's
stateful, bidirectional streaming path.
