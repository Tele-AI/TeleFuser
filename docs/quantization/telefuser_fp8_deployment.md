# TeleFuser FP8 deployment for Qwen-Image

Telefuser users torchao as its backend of FP8 weight-only linear quantization.

First, install Telefuser as [here](https://github.com/Tele-AI/TeleFuser#install).

Then, install torchao:

```bash
pip install torchao
```

Next, download `Qwen-Image-2512` model to your `TF_MODEL_ZOO_PATH` (or specify the model path to `--model_root`), and run the NF4 `Qwen-Image-2512` example:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --aspect_ratio 1:1 \
  --num-inference-steps 16 \
  --seed 42 \
  --output qwen_image_fp8.png
```

Then generated image is saved as `qwen_image_fp8.png` at current directory.
