# LingBot-Video

Dense 1.3B and MoE 30B are separate examples. Each module exposes
`PPL_CONFIG`, `CONTRACT`, `get_pipeline`, `run`, and `run_with_file` for the
shared CLI runner and TeleFuser service.
Both examples default to the official five-second structured caption in `assets/t2v_5s.json.example` and the validated 832x480 LingBot landscape geometry.

The model-specific files also contain their checkpoint loading, stage assembly, request
handling, refiner lifecycle, and output encoding. Shared behavior uses TeleFuser contract
templates and video utilities directly.

Use a structured JSON caption produced by the LingBot rewriter. Dense 1.3B T2V:

```bash
python examples/lingbot_video/lingbot_video_dense_1_3b.py \
  --model_root /path/to/lingbot-video-dense-1.3b \
  --prompt "$(cat /path/to/caption.json)" --output_path result.mp4
```

Pass `--task i2v --first_image_path first_frame.png` for TI2V.

For the MoE checkpoint refiner, use the in-memory base-to-refiner path:

```bash
python examples/lingbot_video/lingbot_video_moe_30b.py \
  --model_root /path/to/lingbot-video-moe-30b-a3b --refine \
  --prompt "$(cat /path/to/caption.json)" --output_path result.mp4
```

The command releases base-stage GPU weights before loading the separate refiner.
Use `--no-refine` for MoE base-only generation. `expert_backend=auto` keeps
the validated sorted eager path on one GPU and selects native grouped GEMM for
four-GPU inference. The grouped path requires a CUDA PyTorch build that exposes
`torch._grouped_mm`; use `--expert_backend sorted` as the explicit fallback.
Use `--refiner_gpu_num` and `--refiner_batch_cfg` to override the corresponding
MoE `PPL_CONFIG` defaults for a direct CLI run.

Set `PPL_CONFIG["model_root"]` in the selected example, then serve structured-caption T2I/T2V/TI2V requests with:

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --port 8000
```

Serve MoE independently with `lingbot_video_moe_30b.py`. Refiner requests are
enabled by default and can override the `refine` contract parameter. Set
`PPL_CONFIG["refiner_parallelism"] = 4` to select its SP4/FSDP stage explicitly;
otherwise it inherits service parallelism.

```bash
telefuser serve examples/lingbot_video/lingbot_video_moe_30b.py --port 8000
```

Dense and MoE base checkpoints support TeleFuser-native four-GPU FSDP plus Ulysses sequence parallelism:

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --gpu-num 4 --port 8000
```

For MoE, set `PPL_CONFIG["model_root"]` in the MoE example and serve
`lingbot_video_moe_30b.py`. Do not use `torchrun` for either service: TeleFuser
creates and manages the workers. The refiner starts as a separate worker group
after the base workers exit. Four-GPU 1920x1088 refinement uses sequential CFG
by default to stay within H100 80 GB memory.
