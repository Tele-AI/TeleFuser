# LingBot-Video Examples

These examples run the Dense 1.3B and MoE 30B LingBot-Video checkpoints for text-to-image, text-to-video, and
text-and-image-to-video generation. The MoE path can refine the base result in memory.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LingBot-Video Dense 1.3B | [robbyant/lingbot-video-dense-1.3b](https://huggingface.co/robbyant/lingbot-video-dense-1.3b) | [Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | Dense base generation |
| LingBot-Video MoE 30B A3B | [robbyant/lingbot-video-moe-30b-a3b](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b) | [Robbyant/lingbot-video-moe-30b-a3b](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | Mixture-of-experts base generation and refiner |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| T2I, T2V, and TI2V | Supported | Select with `--task`; I2V requires `--first_image_path` |
| Multi-GPU inference | Supported | One or four GPUs with FSDP plus CFG/Ulysses parallelism |
| MoE refinement | Supported | MoE video tasks support an in-memory high-resolution refiner |
| Quantization | Partial | MoE routed experts support an explicit memory-oriented FP8 backend |
| CPU offload | Supported | Used by single-GPU paths and the sequential refiner lifecycle |
| Feature cache | Unsupported | No feature cache is configured |
| Server API | Supported | Both files declare standard pipeline contracts |

## Requirements

- GPU: one CUDA GPU, or four GPUs for the validated distributed layouts
- Software: the standard TeleFuser installation with compatible Diffusers and Transformers versions
- Input assets: a structured JSON caption; TI2V also requires a readable first image

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/lingbot/
|-- lingbot-video-dense-1.3b/
|   |-- transformer/
|   |-- processor/
|   |-- text_encoder/
|   |-- vae/
|   \-- scheduler/
\-- lingbot-video-moe-30b-a3b/
    |-- transformer/
    |-- refiner/
    |-- processor/
    |-- text_encoder/
    |-- vae/
    \-- scheduler/
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

```bash
python examples/lingbot_video/lingbot_video_dense_1_3b.py \
  --model_root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-video-dense-1.3b" \
  --prompt "$(cat examples/lingbot_video/assets/t2v_5s.json.example)" \
  --task t2v --output_path work_dirs/lingbot-video-dense.mp4
```

The command writes a five-second, 832x480 landscape video to `work_dirs/lingbot-video-dense.mp4`.

## Examples

### Dense Generation

#### `lingbot_video_dense_1_3b.py`

```bash
python examples/lingbot_video/lingbot_video_dense_1_3b.py \
  --gpu_num 4 --cfg_parallel_degree 2 \
  --model_root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-video-dense-1.3b" \
  --prompt "$(cat examples/lingbot_video/assets/t2v_5s.json.example)" \
  --task t2v --output_path work_dirs/lingbot-video-dense-sp2.mp4
```

### MoE Generation And Refinement

#### `lingbot_video_moe_30b.py`

```bash
python examples/lingbot_video/lingbot_video_moe_30b.py \
  --gpu_num 4 --cfg_parallel_degree 2 \
  --refiner_gpu_num 4 --refiner_cfg_parallel_degree 2 \
  --refiner_co_resident --expert_backend fp8 --refine \
  --model_root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-video-moe-30b-a3b" \
  --prompt "$(cat examples/lingbot_video/assets/t2v_5s.json.example)" \
  --task t2v --output_path work_dirs/lingbot-video-moe-refined.mp4
```

CFG parallel and batch CFG are mutually exclusive. Use `--no-refiner_co_resident` when both DiTs do not fit, or
`--no-refine` for base-only MoE generation. The `fp8` expert backend is a memory-oriented option and must be
validated on the target GPU.

## Serving

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --port 8000
```

Serve the MoE model by replacing the script path. TeleFuser creates its workers; do not launch these services with
`torchrun`.
