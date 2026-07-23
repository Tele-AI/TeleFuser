# TeleFuser NF4 deployment for Qwen-Image

Telefuser users bitsandbytes as its backend of NF4 weight-only linear quantization.

First, install Telefuser as [here](https://github.com/Tele-AI/TeleFuser#install).

Next, download `Qwen-Image-2512` model to your `TF_MODEL_ZOO_PATH` (or specify the model path to `--model_root`), and run the NF4 `Qwen-Image-2512` example:

```bash
# set the below cuda versions according to your environment
export BNB_CUDA_VERSION=128
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_nf4_h100.py   \
    --prompt "A cat playing piano"   \
    --aspect_ratio 1:1   \
    --num-inference-steps 16   \
    --seed 42   \
    --output qwen_image_nf4.png
```

Then generated image is saved as `qwen_image_nf4.png` at current directory.
