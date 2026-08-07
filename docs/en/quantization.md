# Quantization

Quantization stores or computes selected tensors at lower precision. In TeleFuser, it is used for model weights, Linear inputs, attention operands, and KV caches. These paths are independent: enabling one does not quantize the entire pipeline.

## Core ideas

`W8A16` means 8-bit weights and 16-bit activations; `W8A8` means both are 8-bit. The accumulator or output can still use BF16 or FP32, so this notation does not describe every tensor in the operation.

For symmetric integer quantization, TeleFuser's offline converter uses:

$$
s = \frac{\max |x|}{127}, \qquad
q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right), -128, 127\right),
\qquad \hat{x} = s q.
$$

The scale can cover a tensor, an output channel, a token, or a small block. Smaller groups usually reduce error, but require more scale values and more complicated kernels.

| Granularity | TeleFuser example |
| --- | --- |
| Per-tensor | ComfyUI INT8/FP8 conversion |
| Per-output-channel | Offline INT8/FP8 weights and `LinearFP8` weights |
| Per-token | `LinearFP8`, LiveAct, and LingBot FP8 activations |
| Per-block | MXFP and NVFP4 conversion kernels |

## Available paths

| Path | Precision | Entry point |
| --- | --- | --- |
| TorchAO online quantization | W8A8 or weight-only W8A16 | `QuantType.TORCHAO_FP8` |
| bitsandbytes online quantization | NF4 weight-only, W4A16 | `QuantType.BNB_NF4` |
| Scaled FP8 checkpoint | FP8 weights and dynamic FP8 activations, W8A8 | `torch_dtype=torch.float8_e4m3fn` |
| Offline checkpoint conversion | INT8, FP8, MXFP4/6/8, or NVFP4 weights | `tools/convert/converter.py` |

`QuantType` also contains formats that are not connected to generic online model loading. An enum value alone is not evidence that a model implements that path.

## Online Linear quantization

### TorchAO FP8:

TorchAO automatically selects FP8 kernels depending on your hardware paltform, between dynamic-activation and weight-only FP8s. TeleFuser determines the mode according to the selection of TorchAO:
dynamic activation and weight FP8 is W8A8, while `Float8WeightOnlyConfig` with BF16 inputs is W8A16. Check the
conversion log and run a real forward instead of inferring the mode from `QuantType.TORCHAO_FP8`.

The supported TeleFuser models are Wan, Qwen-Image, and LTX transformer blocks. The default filter skips names such as `head`, `time_embedding`, `time_projection`, and `patch_embedding`.

```python
import torch

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager

quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.TORCHAO_FP8,
    kernel_backend=QuantKernelBackend.TORCHAO,
)
manager = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
manager.load_model(
    dit_paths,
    device="cuda",
    torch_dtype=torch.bfloat16,
    quant_config=quant_config,
)
```

Run the complete Qwen-Image example with:

```bash
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --output qwen_image_fp8.png
```

FP8 reduces weight memory traffic; W8A8 additionally quantizes Linear inputs. Whether it improves latency depends on
the selected mode, matrix shapes, GPU, TorchAO version, and `torch.compile` behavior.

TorchAO and PyTorch must be version-compatible. Check the
[TorchAO release compatibility table](https://github.com/pytorch/ao/releases) instead of installing the newest
TorchAO release blindly; an import warning or failure means that configuration has not been validated.

### tf-kernel FP8: online W8A8

MiniMax H3 can use TeleFuser's tf-kernel FP8 GEMM wrapper for online W8A8 inference directly from the original BF16
checkpoint. Activations are quantized per token at each forward, weights are quantized per output channel and cached
on first use, and `tf_kernel.fp8_scaled_mm` produces BF16 output. This path requires a tf-kernel wheel built for the
runtime's PyTorch/CUDA ABI and GPU architecture; on H100, build the SM90 wheel from `tf-kernel/` with the Makefile.

```python
quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.FP8,
    kernel_backend=QuantKernelBackend.TF_KERNEL,
)
```

For MiniMax H3, use `quantization="tf-kernel-fp8"` with
`examples/minimax_h3/minimax_h3_fl2va_h100.py`. This backend is single-GPU only and keeps the FP8
weights resident after first-use conversion. It is distinct from the scaled-FP8 checkpoint path below: the latter
expects weights and scales already serialized in the checkpoint.

### bitsandbytes NF4: W4A16

NF4 uses a non-uniform 4-bit codebook designed for approximately normal weight distributions. TeleFuser replaces selected Linear layers with `bitsandbytes.nn.Linear4bit`, uses BF16 compute, and enables compressed quantization statistics.

```python
quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.BNB_NF4,
    kernel_backend=QuantKernelBackend.BITSANDBYTES,
)
```

The full example is `examples/qwen_image/qwen_image_t2i_telefuser_nf4_h100.py`. NF4 usually saves more weight memory than FP8, but 4-bit decoding does not guarantee lower latency.

## Scaled FP8 checkpoints: W8A8

This path is different from TorchAO. A compatible checkpoint already contains E4M3FN weights and a scale for each output channel. `LinearFP8` dynamically quantizes every input row to FP8, then calls a scaled GEMM through `tf_kernel`. The output returns to BF16 or FP16.

For each row, scaled FP8 follows the same absmax idea:

$$
s = \frac{\max |x|}{\mathrm{max}(\mathrm{E4M3FN})}, \qquad
q = \mathrm{cast}_{\mathrm{E4M3FN}}\left(
\mathrm{clamp}\left(\frac{x}{s}, f_{\min}, f_{\max}\right)\right).
$$

Load only a checkpoint that follows TeleFuser's expected weight and scale layout:

```python
manager.load_model(
    fp8_checkpoint,
    device="cuda",
    torch_dtype=torch.float8_e4m3fn,
)
```

Changing `torch_dtype` does not turn an arbitrary BF16 checkpoint into a scaled FP8 checkpoint. Start from the supplied Qwen-Image or Wan FP8 examples.

## Offline conversion

The converter quantizes selected two-dimensional weights and writes the scales beside them. It creates an artifact; it does not add an inference kernel.

```bash
python tools/convert/converter.py \
  --source /path/to/source \
  --output /path/to/output \
  --model_type wan_dit \
  --quantized \
  --linear_dtype fp8 \
  --non_linear_dtype torch.bfloat16 \
  --single_file
```

`--linear_dtype` accepts `int8`, `fp8`, `mxfp4`, `mxfp6`, `mxfp8`, and `nvfp4`.

- `int8` and `fp8` use one absmax scale per output row. ComfyUI mode uses one scale per tensor.
- MXFP4/6/8 use 32-value blocks and E8M0 scales through `lightx2v_kernel`.
- NVFP4 packs two values per byte and uses one E4M3 scale per 16 values plus a tensor-wide global scale.
- Non-quantized tensors are converted to `--non_linear_dtype`.

The default scale keys are `<weight>_scale`; NVFP4 also writes `<weight>_global_scale`. MXFP and NVFP4 conversion
requires CUDA and `lightx2v_kernel`. TeleFuser's generic loader does not execute these artifacts; a matching consumer
must implement the same layout and GEMM.

## Other quantized data paths

- **LiveAct FP8:** wraps Linear layers with `tf_kernel` dynamic W8A8 GEMM. Weights are cached in FP8 and activations are quantized per token.
- **LingBot-Video MoE FP8:** quantizes expert weights per output channel and routed activations per row, then uses `torch._scaled_mm`.
- **SageAttention:** all three TeleFuser variants quantize Q/K to INT8. `2_8_16` uses FP16 P/V, while `2_8_8` and its SM90 variant use FP8 P/V. These kernels do not change model weights.
- **LiveAct FP8 KV cache:** stores K/V as E4M3FN with one FP32 scale per last-dimension vector, then dequantizes to the requested attention dtype on load.

## Validation

Use the same prompt, input, seed, scheduler, and inference steps as the BF16 baseline. Check all of the following:

1. The log reports a non-zero number of converted Linear layers.
2. Optional backends pass a real `torch.inference_mode()` forward; a successful import alone is insufficient.
3. Peak loading memory and steady-state VRAM are measured separately.
4. Latency is measured after warmup with CUDA synchronization or CUDA events.
5. Generated images or videos are compared with the baseline; successful execution alone is not an accuracy test.

For operator comparisons, useful error measures are:

$$
E_{\max} = \max |y_q-y|, \qquad
E_{\mathrm{rel}} = \frac{\lVert y_q-y \rVert_2}{\lVert y \rVert_2}.
$$

Relevant tests include `tests/unit/quantize/test_quantized_linear.py` and `tests/unit/models/test_lingbot_video_moe.py`.

For TorchAO's distinction between weight-only and dynamic FP8, see its [quantized inference guide](https://docs.pytorch.org/ao/stable/workflows/inference.html).
