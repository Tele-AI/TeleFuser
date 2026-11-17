# PAN 2 on Blackwell

TeleFuser exposes the public `IFM/PAN2` Diffusers modular pipeline through
`telefuser.pipelines.pan2`. PAN 2 is a 29.2B-parameter world model. BF16 does
not fit on a 24 GB RTX 5090; the supported single-card profile uses TorchAO
NVFP4 for the diffusion transformer and Diffusers block-level CPU offload.

Install `diffusers>=0.43.0`, `torchao>=0.17`, and a PyTorch build with CUDA 13.
The GPU must report compute capability SM100 or newer. The first run downloads
the PAN 2 transformer, Qwen3.5-9B text encoder, and Wan 2.2 VAE, so plan for
more than 100 GB of host storage and RAM.

```bash
python examples/pan2/pan2_t2v.py \
  --nvfp4 \
  --steps 4 \
  --prompt "A small red car drives through a rainy city street." \
  --output pan2.mp4
```

The same entrypoint accepts `--image` for image-to-video. NVFP4 quantization
is applied only to Linear layers in the PAN 2 transformer; the text encoder
and VAE stay in BF16/FP32 and are also offloaded. This profile trades latency
for fitting the model on a single 24 GB Blackwell card.

MiniMax-H3-Image is not enabled by this profile. Its 33.1B transformer is
released as approximately 66 GB of BF16 weights and requires a separate
multi-component loader; it does not fit as a resident single-card TeleFuser
pipeline on a 24 GB RTX 5090.
