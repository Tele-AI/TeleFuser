# LingBot-Video

Use a structured JSON caption produced by the LingBot rewriter. Dense T2V:

```bash
python examples/lingbot_video/lingbot_video_generate.py \
  --model-dir /path/to/lingbot-video-dense-1.3b \
  --caption-json /path/to/caption.json --output result.mp4
```

Pass `--image first_frame.png` for TI2V, or `--variant moe` for the MoE base.

For the MoE checkpoint refiner, use the in-memory base-to-refiner path:

```bash
python examples/lingbot_video/lingbot_video_generate.py \
  --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --refine \
  --caption-json /path/to/caption.json --output result.mp4
```

The command releases base-stage GPU weights before loading the separate refiner. Add `--image first_frame.png` to preserve the TI2V clean frame-zero condition. The default sorted eager MoE route path is source-equivalent with a diagnostic fallback, but it is not a grouped-GEMM or FP8 production-performance backend.

Serve structured-caption T2I/T2V/TI2V requests with:

```bash
LINGBOT_VIDEO_MODEL_ROOT=/path/to/lingbot-video-dense-1.3b \
  telefuser serve examples/lingbot_video/lingbot_video_service.py --port 8000
```

Set `LINGBOT_VIDEO_VARIANT=moe LINGBOT_VIDEO_ENABLE_REFINER=1` for the MoE base-plus-refiner service lifecycle.
