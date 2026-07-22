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

On four GPUs the base and refiner DiTs remain resident together during refiner
denoising by default. Both worker groups are released before the high-resolution
VAE decode. Use `--no-refiner_co_resident` for the lower-memory sequential
lifecycle, or `--no-refine` for MoE base-only generation. `expert_backend=auto` keeps
the validated sorted eager path on one GPU and selects native grouped GEMM for
four-GPU inference. The grouped path requires a CUDA PyTorch build that exposes
`torch._grouped_mm`; use `--expert_backend sorted` as the explicit fallback.
`--expert_backend fp8` quantizes routed expert weights per output channel and
uses native dynamic W8A8 scaled GEMMs. It reduces expert residency but is an
explicit memory-oriented backend until a grouped FP8 kernel is available.

Four-GPU base and refiner stages can split the devices as CFG2 x SP2:

```bash
python examples/lingbot_video/lingbot_video_moe_30b.py \
  --gpu_num 4 --cfg_parallel_degree 2 \
  --refiner_gpu_num 4 --refiner_cfg_parallel_degree 2 \
  --refiner_co_resident --expert_backend fp8 \
  --model_root /path/to/lingbot-video-moe-30b-a3b \
  --output_path result.mp4
```

CFG parallel and batch CFG are mutually exclusive. Use
`--cfg_parallel_degree 2` for Dense or MoE base generation and
`--refiner_cfg_parallel_degree 2` for the refiner. A degree of one retains SP4.

Set `PPL_CONFIG["model_root"]` in the selected example, then serve structured-caption T2I/T2V/TI2V requests with:

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --port 8000
```

Serve MoE independently with `lingbot_video_moe_30b.py`. Refiner requests are
enabled by default and can override the `refine` contract parameter. Set
`PPL_CONFIG["refiner_parallelism"] = 4` to select its distributed FSDP stage
explicitly; otherwise it inherits service parallelism. The CFG degrees determine
whether four workers use SP4 or CFG2 x SP2.

```bash
telefuser serve examples/lingbot_video/lingbot_video_moe_30b.py --port 8000
```

Dense and MoE base checkpoints support TeleFuser-native four-GPU FSDP plus Ulysses sequence parallelism:

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --gpu-num 4 --port 8000
```

For MoE, set `PPL_CONFIG["model_root"]` in the MoE example and serve
`lingbot_video_moe_30b.py`. Do not use `torchrun` for either service: TeleFuser
creates and manages the workers. Configure `refiner_co_resident=False` when the
selected dtype, CFG layout, or GPU memory cannot hold both DiTs during refiner
denoising.
