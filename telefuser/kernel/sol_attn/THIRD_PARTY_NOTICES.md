# Third-party notices

Sol-Attn was vendored from NVIDIA's `NVlabs/Sana` `sol-engine` branch at
commit `8a26fb0ec9e353125ead798cb2e312d5ce48cded`. The upstream repository
declares its code under the Apache License 2.0. Local changes move the
implementation under the internal `telefuser.kernel.sol_attn` namespace and
rewrite its absolute imports.

Upstream source: https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn

The files under `telefuser/kernel/sol_attn/_vendor/flash_attn/cute/` and portions of the SM90
and SM100 design scaffold derive from the FlashAttention project. Its
BSD-3-Clause license is included at
`telefuser/kernel/sol_attn/sm100/LICENSE.flash-attention`.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, PyTorch,
and Triton. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

The SM120 warp-MMA/TMA execution skeleton and online-softmax helpers are
adapted from NVIDIA cuDNN Frontend's block-sparse-attention reference at commit
`74785165de2da954a2c879a5e3e6f95411c2292d`. That source is licensed under the
Apache License 2.0; adapted files retain the corresponding SPDX header.

The vendored FlashAttention `fmax` wrapper also detects the NVVM Python
binding signature directly. CUDA toolkit version alone does not identify
whether that binding accepts an explicit result type.
