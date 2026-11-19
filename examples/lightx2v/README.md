# LightX2V examples

TeleFuser runs LightX2V distilled Wan2.2 checkpoints through the existing Wan video pipeline. The four-step adapter is
loaded as the low/high noise DiT pair; VAE, text encoder, CPU offload, sequence parallelism and tensor parallelism
remain TeleFuser components.

## Quick start

```bash
python examples/wan_video/wan22_14b_image_to_video_distill_h100.py \
  --image_path inputs/first.png --prompt "A robot walks forward" --output_path work_dirs/lightx2v.mp4
```

Use the corresponding `wan22_14b_image_to_video_distill_fp8_h100.py` example for existing FP8 loading. The model paths
in those examples point to `Wan2.2-Distill-Models/*lightx2v*` checkpoints. Set `TF_MODEL_ZOO_PATH` or edit the paths in
the example configuration.

## TeleFuser features

- **Model abstraction:** `WanVideoPipeline` / `WanVideoStage` and `ModuleManager`.
- **LightX2V:** four-step distilled high/low-noise checkpoints, selected by the existing scheduler split.
- **SP + TP:** use the existing `ParallelConfig` and Wan Ulysses/TP stage options.
- **FP8:** use the existing FP8 example; FP8 support applies to the linear layers while the distilled LoRA/checkpoint
  remains in the model's expected precision.
- **CPU offload:** configure `WeightOffloadType.MODEL_CPU_OFFLOAD` when the full model does not fit on one GPU.

This example reuses the maintained Wan implementation instead of adding a parallel LightX2V framework API.
For benchmark collection, use the existing runtime metrics and save the raw JSON next to the generated MP4.
