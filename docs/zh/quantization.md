# 量化

## 使用 TeleFuser 为 Qwen-Image 部署 FP8 量化

TeleFuser 使用 TorchAO 作为 FP8 仅权重线性量化的后端。

首先，按照[此处说明](https://github.com/Tele-AI/TeleFuser#install)安装 TeleFuser。

接下来，将 `Qwen-Image-2512` 模型下载到 `TF_MODEL_ZOO_PATH` 目录中（也可以通过 `--model_root` 指定模型路径），然后运行 FP8 版本的 `Qwen-Image-2512` 示例：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --aspect_ratio 1:1 \
  --num-inference-steps 16 \
  --seed 42 \
  --output qwen_image_fp8.png
```

生成的图像将以 `qwen_image_fp8.png` 为文件名保存在当前目录中。

## 使用 TeleFuser 为 Qwen-Image 部署 NF4 量化

TeleFuser 使用 bitsandbytes 作为 NF4 仅权重线性量化的后端。

首先，按照[此处说明](https://github.com/Tele-AI/TeleFuser#install)安装 TeleFuser。

接下来，将 `Qwen-Image-2512` 模型下载到 `TF_MODEL_ZOO_PATH` 目录中（也可以通过 `--model_root` 指定模型路径），然后运行 NF4 版本的 `Qwen-Image-2512` 示例：

```bash
# 请根据你的运行环境设置以下 CUDA 版本
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

生成的图像将以 `qwen_image_nf4.png` 为文件名保存在当前目录中。

