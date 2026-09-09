# Installation

This page covers the base TeleFuser package. Model checkpoints, LiveKit Server, and the optional `tf-kernel`
distribution are installed separately.

## Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Linux recommended |
| Python | 3.10 through 3.13 |
| PyTorch | 2.6 or newer |
| CUDA toolkit | 12.8 or newer for the maintained CUDA development path |
| GPU | Depends on the selected model; check its Cookbook guide |

An example may impose stricter versions or GPU architecture requirements. In particular, locally built `tf-kernel`
artifacts are tied to their recorded PyTorch, CUDA, ABI, and GPU-family configuration.

## Install the Package

=== "Published package"

    ```bash
    python -m pip install --upgrade pip
    python -m pip install telefuser
    ```

=== "Repository checkout"

    ```bash
    git clone https://github.com/Tele-AI/TeleFuser.git
    cd TeleFuser
    python -m pip install -e .
    ```

=== "Development"

    ```bash
    git clone https://github.com/Tele-AI/TeleFuser.git
    cd TeleFuser
    python -m pip install -e ".[dev]"
    pre-commit install
    ```

Optional dependency groups are `ui`, `distributed`, `docs`, and `dev`. Install only the groups required by the
workflow, for example `python -m pip install -e ".[distributed]"` for Ray-backed execution.

## Verify the Installation

```bash
python -c "import torch, telefuser; print(torch.__version__); print(torch.cuda.is_available())"
telefuser --help
```

Model execution expects `torch.cuda.is_available()` to print `True`. If it does not, verify the installed PyTorch
build and visible NVIDIA driver before diagnosing TeleFuser.

## Model Checkpoints

TeleFuser does not bundle model weights. The [Supported Models](supported_models.md) page links to each Cookbook
guide, where the validated Hugging Face and ModelScope identifiers, directory layout, and additional artifacts are
documented. A Hugging Face model ID can be passed directly only when the selected example explicitly supports hub
loading.

## Optional Components

- [TF-Kernel](tf_kernel.md) documents the separate Makefile-based build and compatibility checks.
- [Attention](attention.md) lists optional attention backends and their hardware requirements.
- [Stream Server](stream_server.md) covers the separately operated LiveKit service.

Continue with the [LingBot-World v2 WebRTC Core Experience](streaming_quickstart.md), or use the lighter
[Basic Inference Quickstart](quickstart.md) to verify a single-GPU pipeline and batch API.
