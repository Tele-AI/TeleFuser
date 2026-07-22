# LingBot-Video

LingBot-Video 支持 Dense 与 MoE base DiT 的 T2I、T2V、TI2V；MoE checkpoint
还包含独立的低噪 refiner。本集成按精度优先实现，在启用优化或分布式后端前应先完成上游参考对比。

## Checkpoint

Dense checkpoint 可直接加载 Diffusers `transformer/` 目录：

```python
from telefuser.pipelines.lingbot_video import load_lingbot_video_dense_transformer

transformer = load_lingbot_video_dense_transformer("/path/to/transformer")
```

MoE 与 refiner 使用分片 safetensors，调用
`load_lingbot_video_moe_transformer` 加载 `transformer/` 或 `refiner/`。
默认的 sorted eager expert 路径保持上游 route 顺序，并保留 `where` 诊断 fallback。
它仍是经过验证的单卡 BF16 路径。四卡 MoE 默认在 CUDA PyTorch 提供
`torch._grouped_mm` 时启用原生 grouped GEMM；可通过
`expert_backend=sorted` 显式回退。
`variant="moe"` 默认启用 stage CPU offload，避免 base DiT、text encoder、VAE 与
独立 refiner 同时驻留单张 GPU。仅在显存明确充足时才设置 `cpu_offload=False`。

## Prompt 准备

生成管线有意只消费结构化 JSON caption；prompt rewriter 是可独立部署的可选流程。
必须保留上游两阶段语义：EXPAND 使用不挂 LoRA 的 base VLM，MAP 使用同一 base VLM
并启用 LingBot rewriter LoRA。TI2V 必须将同一张首帧同时传给 rewriter 与 TeleFuser。

```bash
REWRITER_BASE_MODEL=/path/to/Qwen3.6-27B \
REWRITER_ADAPTER=/path/to/lingbot-video-rewriter-lora \
python work_dirs/lingbot-video-master/rewriter/inference.py \
  --mode t2v --prompt "<plain prompt>" --duration 5 --output prompt.json
```

将输出的 `prompt.json` 传给 `--caption-json`，或将其中的 `caption` 对象序列化为
服务请求的 `prompt`。由于 rewriter 与 DiT 的部署和显存需求独立，TeleFuser 不会在
同一 DiT 服务进程内加载 rewriter。

未显式覆盖时，pipeline、CLI 与 service 都会使用 checkpoint 的结构化负向 CFG
caption。T2I 使用上游的静态图像版本；T2V 与 TI2V 使用包含时序稳定性约束的上游视频
版本。复现上游样本时，不能以空字符串替代未传入的负向 prompt：这会改变 Qwen3-VL 的
负向 condition，并可能显著影响颜色与画质。

## 运行时组成

`LingBotVideoPipeline` 通过标准的 `init(module_manager, config)` 接口统一初始化；checkpoint 组件先加载到 `ModuleManager`，再由 `init` 创建全部 stage：

- `LingBotVideoTextEncodingStage`：Qwen3-VL structured JSON caption 编码。
- `LingBotVideoDenoisingStage`：source-order 的 two-forward CFG。
- `LingBotVideoVAEEncodeStage` / `LingBotVideoVAEDecodeStage`：使用 checkpoint 的 latent mean/std。
- `FlowUniPCMultistepScheduler`：sigma/timestep 调度。

核心 pipeline 应输入结构化 JSON caption，不应将普通自然语言 prompt 直接替代 rewriter 输出。

标准 checkpoint 的模型加载与 stage 装配直接放在公开的模型专用 example 中，
与 `PPL_CONFIG`、`CONTRACT` 以及 CLI/service 入口保持在同一层，方便阅读和修改：

```python
from examples.lingbot_video.lingbot_video_dense_1_3b import build_pipeline
from telefuser.pipelines.lingbot_video import LingBotVideoRequest

pipeline = build_pipeline("/path/to/lingbot-video-dense-1.3b", num_inference_steps=40)
frames = pipeline(LingBotVideoRequest(caption=structured_caption, height=480, width=832, num_frames=121))
```

直接使用 API 或 CLI 时，高和宽必须能被 16 整除：Wan VAE 会以 8 倍下采样，DiT 会再以 2 倍对 latent 作空间 patchify。

默认 `AttentionConfig` 使用 TeleFuser 的 SDPA dispatcher，仍属于 source-equivalent
数值路径。其他 attention backend 必须通过 `attention_config=` 显式启用并单独提供 L2
parity 报告；service 与 CLI 默认不会启用它们。

VAE decode stage 的 RGB video 范围为 `[0,1]`。视频调用方必须将这些浮点帧直接传给
Diffusers 的 `export_to_video`，由它自行转换为 uint8。若在调用前先转 uint8，就会再次
乘以 255 并使通道值溢出，生成近似负片的 MP4。

## 四卡 base 推理

Dense 与 MoE base DiT 均支持 TeleFuser 原生四卡：DiT block 使用 FSDP，联合 video/text
token 流使用 Ulysses sequence parallel。只有 token 总数不能被 rank 数整除时才补齐；输出会按
上游 token 顺序恢复。正、负 Qwen embedding 形状一致时，CFG 使用一次 batched forward；形状
不一致时会安全回退到保持上游顺序的两次 forward。

按普通方式启动 API 服务即可，TeleFuser 会管理四个 worker，不能再用 `torchrun` 包裹：

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --gpu-num 4 --port 8000
```

该模式要求 DiT 常驻 GPU，因此必须 `cpu_offload=False`。checkpoint 中保留 FP32 的 modulation
参数会作为 FSDP ignored state 在每个 rank 副本化，与上游 mixed-precision 布局一致。MoE 30B
使用独立 example；grouped expert 权重仍随 FSDP block 分片，而不是 expert parallel：

```bash
telefuser serve examples/lingbot_video/lingbot_video_moe_30b.py --gpu-num 4 --port 8000
```

在 PyTorch 2.11/CUDA 13 环境中，已验证的 grouped-GEMM MoE base 为 832x480、
121 帧、40 steps。四卡路径由 worker 持有完整 scheduler loop，并与 `torchrun`
一致地为每个 rank 使用一个 CPU intra-op 线程。四张 H100 上稳态约为 2.3 秒/step，
40 steps 约 93 秒；官方 runner 为 92 秒。TeleFuser 首次请求还包含约 15 秒的 FSDP
materialization。冷启动 CLI 总耗时为 231.9 秒，因为父进程加载和 spawn 时传输 30B
模型不属于稳态服务延迟；常驻服务的后续请求会摊销这部分开销。独立加载的 refiner
也已有经验证的四卡 FSDP/Ulysses 路径；expert parallel 与 FP8 仍需独立 parity 与
吞吐验证。

## 服务

Dense 与 MoE example 都提供 `PPL_CONFIG`、`CONTRACT`、`get_pipeline`、`run` 与
`run_with_file`，并通过 TeleFuser API 暴露 `t2i`、`t2v`、`i2v`；`prompt` 必须是
structured JSON 字符串：

启动服务前需要在对应 example 中设置 `PPL_CONFIG["model_root"]`。运行参数从 `PPL_CONFIG` 读取；直接执行 CLI 时可通过对应命令行参数覆盖。

```bash
telefuser serve examples/lingbot_video/lingbot_video_dense_1_3b.py --port 8000
```

MoE 使用 `lingbot_video_moe_30b.py`，模型路径由其中的 `PPL_CONFIG["model_root"]` 配置。它的 contract
显式提供 `refine`，默认启用。服务会在加载 refiner 前释放 base stage 权重；
`PPL_CONFIG["refiner_parallelism"] = 4` 可为 refiner 单独选择四 worker，未设置时继承
service parallelism。

```bash
telefuser serve examples/lingbot_video/lingbot_video_moe_30b.py --port 8000
```
服务请求的分辨率会向上对齐到 LingBot VAE 与 DiT 所需的 16 像素网格；例如 `480p`、`16:9` 使用已验证的 LingBot 832x480 横屏预设。
仅在确实需要覆盖上游默认值时才传入 `negative_prompt`；显式传入空字符串仍是受支持的覆盖方式。

## TI2V 与 Refiner

TI2V 接收范围为 [0,255] 的 RGB tensor（`[B, 3, H, W]` 或 `[B, 3, F, H, W]`，后者取第 0 帧）。管线会先 resize 和 center-crop，再将同一视觉帧传给正向与负向 Qwen3-VL CFG 分支，并编码为 VAE clean temporal-prefix latent；该 latent 会在每次 denoising step 前及最终 step 后重写。

`LingBotVideoRefinerStage` 直接接收 base RGB tensor，避免 MP4/decord 往返；它按 `t_thresh` 与噪声混合，并使用 low-noise sigma tail 采样。base 与 refiner 是独立运行时 stage；共享 GPU 时，请在加载 refiner 前调用 `base_pipeline.release_gpu_resources()`。

四卡模式下，refiner 使用 block FSDP 与四路 Ulysses SP。1920x1088 在四张 H100 80 GB
上默认采用正、负条件依次执行的 sequential CFG；batched CFG 会超出四卡显存，仅在额外
显存容量已经验证时设置 `PPL_CONFIG["refiner_batch_cfg"] = True`。运行时顺序为：VAE encode、
释放 VAE、分布式 refiner 去噪、关闭 refiner workers、重新加载 VAE decode，避免高显存
stage 在 rank 0 重叠。

已验证的 MoE 运行先生成 832x480、121 帧 base，再以 8 steps 输出 1920x1088、24 FPS
视频。base 耗时 372.9 秒，refiner stage 耗时 886.8 秒（不含 checkpoint 反序列化）；
最终视频为 121 帧、5.0417 秒。

示例 CLI 已为 MoE checkpoint 自动完成这个生命周期：

```bash
python examples/lingbot_video/lingbot_video_moe_30b.py \
  --model_root /path/to/lingbot-video-moe-30b-a3b --refine \
  --prompt "$(cat /path/to/caption.json)" --output_path result.mp4
```

附加 `--task i2v --first_image_path first_frame.png` 时，CLI 也会把上游 TI2V
frame-zero 几何规则应用到 refiner condition。

如需在内存中交接基座输出与 refiner，可先调用 `prepare_refiner_video(...)`，再调用
`LingBotVideoRefinerStage.refine(...)`。该函数复刻上游训练对齐抽帧与双三次缩放，
无需 MP4 写入/读取；必须显式传入基座输出 FPS，并应与对应的上游 MP4 基线进行验证。
MP4 兼容性测试使用上游 Diffusers writer；当 decord 不可用时，通过 PyAV-backed decord adapter 调用上游 loader，并逐 tensor 对比本 loader。


## 验证

```bash
python tools/validation/capture_lingbot_video_reference.py --dry-run
python tools/validation/capture_lingbot_video_reference.py --all-cases --mode t2i --mode t2v --mode ti2v --trace sampled
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-dense-1.3b --variant dense --output dense-load-report.json
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --output moe-load-report.json
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant refiner --output refiner-load-report.json
python tools/validation/compare_lingbot_video_parity.py REFERENCE CANDIDATE
python tools/validation/replay_lingbot_video_dense_reference.py --reference-dir work_dirs/lingbot_video_reference/t2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --validate-text --reference-dir work_dirs/lingbot_video_reference/ti2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --validate-text --validate-ti2v-vae --reference-dir work_dirs/lingbot_video_reference/ti2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --reference-root work_dirs/lingbot_video_reference_all_cases --assert-exact --output dense-all-cases-replay.json
PYTHONPATH=work_dirs/lingbot-video-master python tools/validation/run_lingbot_video_moe_parity.py --transformer-dir /path/to/lingbot-video-moe-30b-a3b/transformer --assert-exact
PYTHONPATH=work_dirs/lingbot-video-master python tools/validation/run_lingbot_video_refiner_core_parity.py --model-root /path/to/lingbot-video-moe-30b-a3b --assert-exact
python tools/validation/validate_lingbot_video_refiner_handoff.py --input base.mp4 --height 1088 --width 1920 --assert-exact
python tools/validation/validate_lingbot_video_refiner_output_handoff.py --model-dir /path/to/lingbot-video-moe-30b-a3b --caption-json prompt.json --height 64 --width 64 --num-frames 5 --steps 1 --refiner-height 64 --refiner-width 64 --refiner-steps 1 --output handoff-output-report.json --comparison-output handoff-comparison.mp4
python tools/validation/benchmark_lingbot_video.py --model-dir /path/to/lingbot-video-dense-1.3b --caption-json prompt.json --output result.mp4 --report benchmark.json --warmup 1 --runs 3
python tools/validation/benchmark_lingbot_video.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --refine --caption-json prompt.json --output result.mp4 --report benchmark.json --warmup 1 --runs 3
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/validation/run_lingbot_video_distributed.py --model-dir /path/to/lingbot-video-dense-1.3b --caption-json prompt.json --output dense-sp4.mp4 --report dense-sp4.json
python tools/validation/run_lingbot_video_native_parallel.py --variant moe --refine --model-dir /path/to/lingbot-video-moe-30b-a3b --caption-json prompt.json --output moe-refiner-sp4.mp4 --report moe-refiner-sp4.json
```

`validate_lingbot_video_refiner_handoff.py` 验证 TeleFuser 的 MP4 兼容 loader 与上游
loader 的输入 tensor 完全一致。可通过 `--in-memory-video` 和 `--in-memory-fps` 量化 MP4
编码带来的输入差异；该比较不替代最终 Refiner 输出质量评估。
使用 `--assert-exact` 可在 source MP4 兼容路径的 metadata、tensor shape、dtype 或值发生
漂移时使命令失败。它刻意不评判原生内存 handoff，因为它与有损 MP4 的差异是预期行为。
`validate_lingbot_video_refiner_output_handoff.py` 会先生成一个 MoE base sample，再以
完全相同的 prompt condition 与 RNG state 分别通过原生 RGB tensor 和临时 MP4 回环驱动
refiner，并报告最终输出的 L2 差异。这是质量对比而非等价性测试：上游 refiner 使用有损
MP4 回环，而原生路径去除了这一中间编码。报告还会给出 decoded-frame PSNR 与局部 SSIM；
`--comparison-output` 会写出左侧为内存输出、右侧为 MP4 回环输出的并排视频，供人工审核。

捕获工具记录 prompt/scheduler/selected denoising tensors、生成 seed、RNG state hash 和 decoded frame hash，用于 L0/L1 parity。replay 命令附加 `--validate-text` 会逐项对比 Qwen3-VL processor 输入和最终 embedding；TI2V 还会对比首帧预处理结果。附加 `--validate-ti2v-vae` 会对比采样后的 clean condition latent；仅对不含 seed metadata 的旧 capture 使用 `--seed`。
使用 `--reference-root` 可批量复放所有 Dense DiT/VAE capture。它会在所有样本间复用同一已加载的 Dense transformer 和 VAE；每个样本仍会重新实例化 scheduler，以保持各自的采样配置。
添加 `--assert-exact` 后，只要任何记录 tensor 的 shape 或值不一致，命令就会以失败状态退出，可作为 CI parity gate，而非仅输出诊断报告。
checkpoint inspection 工具会执行正常的严格加载，并记录已消费的 config 字段、checkpoint key 覆盖、component/block 参数量、dtype/device 分布、保留的 FP32 参数量及模型显存分配证据。
基准工具会将一次性 setup 与 warmup、measured 指标分开记录，覆盖 checkpoint load、text encoding、每个 denoising step、VAE、refiner、输出编码与峰值 GPU 显存。启用优化前应先用它建立基线，并将全分辨率测量与 smoke run 分开记录。
未传入 `--negative-caption` 时，benchmark 会与 pipeline、CLI、service 一样使用上游兼容的
T2I 或视频负向 caption。只有需要特意评测该语义覆盖时，才显式传入空字符串。
对于 base+refiner，它还会记录串行的 base release 与 refiner load 阶段。单卡 benchmark
使用 source-equivalent sorted eager MoE；四卡 benchmark 默认选择 grouped GEMM，仅在评测
正确性 fallback 时显式设置 `expert_backend=sorted`。
T2I/T2V 的 Refiner 会复用 base generation 生成的完全相同 CFG text condition，并在报告中
记录 `refiner_prompt_conditions_reused`；TI2V 保持上游兼容的 text-only Refiner 编码路径。
Refiner core CLI 会向上游与 TeleFuser 的 low-noise 路径注入相同的 latent、noise、prompt 与 frame-zero condition；它会先 offload 上游 DiT，再加载 TeleFuser DiT，避免两个 30B 模型同时占用 GPU。
对 MoE 或 Refiner core validator 添加 `--assert-exact`，可强制执行零漂移的 numerical-oracle gate，而非只写出指标。

## 依赖与已知限制

数值 oracle 路径需要 CUDA、PyTorch、Diffusers、Transformers 以及 checkpoint 的
`transformer/`、`text_encoder/`、`processor/`、`vae/`、`scheduler/` 组件。Dense
可在单 GPU 上按 source-equivalent 路径运行。MoE 单卡保留 sorted eager correctness 路径，
四卡使用原生 grouped GEMM；后者要求 CUDA PyTorch 提供 `torch._grouped_mm`。Dense 与
MoE base 都已有经验证的四卡 FSDP/Ulysses SP 路径；MoE refiner 也已有独立的四卡
FSDP/Ulysses SP stage。外部 FlashAttention、MoE expert parallel 与 FP8 expert 尚未启用。
base+refiner 的 stage 生命周期保持串行，确保两个 30B 权重不会同时驻留 GPU。
