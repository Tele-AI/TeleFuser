# Ascend container image

`docker/Dockerfile.ascend` layers TeleFuser on top of the public multi-arch Ascend ecosystem images maintained by `vllm-omni` and `vllm-ascend`.

## Base images and mirrors

The default base image is `quay.io/ascend/vllm-omni:v0.34.0` (`linux/arm64` and `linux/amd64`), which builds on `vllm-ascend:v0.34.0` and already includes:

- Ubuntu 22.04, CANN 9.3.0 (toolkit, AscendNPU IR, ATB, HCCL), and Python 3.12
- Matching CPU PyTorch (`torch==2.10.0+cpu`), `torch_npu==2.10.0.post7`, `torchvision`, and `torchaudio`
- `ffmpeg` / `ffprobe`, `libjemalloc2` (`LD_PRELOAD`), `diffusers`, `av`, `imageio[ffmpeg]`, `opencv-python`, `safetensors`, `scipy`, and `soundfile`

To accelerate pulls on networks with slow access to `quay.io`, pass a mirror prefix via `BASE_IMAGE`:

```bash
# DaoCloud mirror
docker build -f docker/Dockerfile.ascend \
  --build-arg BASE_IMAGE=m.daocloud.io/quay.io/ascend/vllm-omni:v0.34.0 \
  -t telefuser:ascend910b .

# NJU mirror
docker build -f docker/Dockerfile.ascend \
  --build-arg BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-omni:v0.34.0 \
  -t telefuser:ascend910b .
```

You can also pass a compatible `quay.io/ascend/vllm-ascend:<tag>` or `quay.io/atlas-ci/vllm-ascend:<tag>` image via `--build-arg BASE_IMAGE=...`; the Dockerfile automatically installs `ffmpeg` and `libjemalloc2` when missing from the base image.

## Design notes

- **Isolated application venv with shared runtime**: `/opt/telefuser/venv` is created with `--system-site-packages` so TeleFuser reuses the preinstalled `torch`, `torch_npu`, and multimedia wheels without mutating the base image's global `vllm` / `vllm-omni` packages (such as `transformers`).
- **Non-interactive CANN initialization**: `vllm-omni` clears `ENTRYPOINT` for interactive shells; `Dockerfile.ascend` restores a lightweight `ENTRYPOINT` wrapper that sources CANN `set_env.sh`, AscendNPU IR, and ATB before running commands.

See the [Ascend 910B guide](../../docs/en/ascend_910b.md) for direct container usage, device mounting, and Wan2.2 text-to-video execution.
