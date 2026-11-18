# 在 Blackwell 上运行 PAN 2

TeleFuser 通过 `telefuser.pipelines.pan2` 接入公开的 `IFM/PAN2`
Diffusers 模块化流水线。PAN 2 有 292 亿参数，BF16 无法放入 24 GB 的
RTX 5090；单卡配置使用 TorchAO NVFP4 量化扩散 Transformer，并启用
Diffusers block-level CPU offload。

安装 `diffusers>=0.43.0`、`torchao>=0.17` 和 CUDA 13 PyTorch。GPU 必须是
SM100 或更新架构。首次运行需要下载 PAN 2 Transformer、Qwen3.5-9B 文本
编码器和 Wan 2.2 VAE，请预留超过 100 GB 的磁盘和主机内存。

```bash
python examples/pan2/pan2_t2v.py \
  --nvfp4 \
  --steps 4 \
  --prompt "A small red car drives through a rainy city street." \
  --output pan2.mp4
```

加入 `--image` 可执行 image-to-video。NVFP4 只量化 PAN 2 Transformer 中
的 Linear 层，文本编码器和 VAE 保持 BF16/FP32 并参与 offload；代价是更高
的延迟，以换取单张 24 GB Blackwell 显卡上的可装载性。

MiniMax-H3-Image 暂未启用。它的 331 亿参数 Transformer 约 66 GB BF16，
需要独立的多组件加载器，不能作为常驻单卡 TeleFuser pipeline 放入 24 GB
RTX 5090。

## BF16 与 NVFP4 对比

PAN 2 权重已缓存后，运行匹配基准：

```bash
python tools/validation/benchmark_pan2_telefuser.py \
  --num-frames 49 --steps 4 \
  --output-dir outputs/pan2-benchmark
```

命令会生成 `official-bf16.mp4`、`telefuser-nvfp4.mp4` 和 `report.json`。
`generated_video_fps / 24` 是实时倍率，大于 `1.0` 表示生成速度超过播放速度。
两次运行使用相同的 CPU offload、seed、prompt、分辨率、帧数和采样步数。

完整 Transformer 的量化应逐个 block 执行，避免量化过程中同时驻留完整
BF16 模型：

```bash
python tools/quantization/quantize_pan2_layerwise.py \
  /path/to/IFM/PAN2 /path/to/pan2-nvfp4-layerwise
```

该工具支持中断后继续，并按 block 写出 `block-XX.pt` 文件。

使用已转换的 block 运行示例：

```bash
python examples/pan2/pan2_t2v.py \
  --model-id /path/to/IFM/PAN2 \
  --layerwise-nvfp4-dir /path/to/pan2-nvfp4-layerwise \
  --steps 4 --output pan2-nvfp4.mp4
```

运行官方 PAN2 示例时，使用 `inference/prompts/examples.json` 中的同一条
prompt，并指定 `--height 704 --width 1248 --num-frames 241 --fps 24
--seed 123 --steps 50`。实测报告和图表作为 PR 附件保存，不写入仓库，也不
把旧的 smoke/mock 输出当作质量结果。layerwise loader 在可用时使用 MSLK
Triton NVFP4 activation 路径；Python-only Triton 包可通过
`MSLK_PYTHON_ONLY=1` 启用。
