# 基础推理快速上手

本指南验证维护中的单卡文生视频示例和批量 HTTP 服务。TeleFuser 的主要交互式工作流请从
[LingBot-World v2 WebRTC 核心体验](streaming_quickstart.md)开始。

## 前置条件

- Linux 环境下的 TeleFuser 仓库副本
- Python 3.10 至 3.13，以及可正常使用 CUDA 的 PyTorch
- 一张显存足以运行 Wan2.1 T2V 1.3B 480p 推理的 CUDA GPU
- 已有的本地 Wan2.1 T2V 1.3B 权重；此流程不会下载模型

请先完成[安装](installation.md)，并确认 `torch.cuda.is_available()` 返回 `True`。

## 运行流水线

在仓库根目录执行：

```bash
mkdir -p work_dirs
export WAN21_MODEL_SOURCE=/path/to/model_zoo/Wan2.1-T2V-1.3B
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/wan_video/wan21_1_3b_text_to_video_hf.py \
  --model_root "$WAN21_MODEL_SOURCE" \
  --resolution 480p \
  --prompt "A sailboat crosses a calm lake at sunrise"
```

成功时日志最后会显示 `Video saved to:`，并生成：

```text
work_dirs/wan_video_wan21_1_3b_text_to_video_hf.mp4
```

该权重也发布于
[Hugging Face 的 Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 和
[ModelScope 的 Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)。
请将 `WAN21_MODEL_SOURCE` 设为已有的本地仓库目录，并保留原始权重目录结构。以上链接用于确认权重，
并非本指南中的下载步骤。

## 启动批量服务

在已设置 `WAN21_MODEL_SOURCE` 的终端中，保持以下进程运行：

```bash
telefuser serve examples/wan_video/wan21_1_3b_text_to_video_hf.py \
  --task t2v \
  --port 8000
```

等待启动完成后，在另一个终端检查服务：

```bash
curl --fail http://127.0.0.1:8000/v1/service/health
```

创建任务：

```bash
curl --fail --request POST http://127.0.0.1:8000/v1/tasks/create \
  --header "Content-Type: application/json" \
  --data '{
    "task": "t2v",
    "prompt": "A sailboat crosses a calm lake at sunrise",
    "resolution": "480p",
    "aspect_ratio": "16:9"
  }'
```

响应包含 `task_id`、`task_status` 和 `output_path`。使用返回的任务 ID 轮询，直到 `status` 为
`completed`：

```bash
curl --fail http://127.0.0.1:8000/v1/tasks/TASK_ID/status
```

运行中的服务还会在 `http://127.0.0.1:8000/docs` 提供 Swagger UI，在
`http://127.0.0.1:8000/openapi.json` 提供 OpenAPI 文档。

## 后续步骤

- [支持的模型](supported_models.md)列出模型系列和任务指南。
- [服务与 API](serving.md)说明批量和流式服务模式。
- [配置系统](configuration.md)连接到并行、注意力和缓存等运行时优化。
- [故障排查](troubleshooting.md)提供按症状组织的诊断流程。

继续进入[LingBot-World v2 WebRTC 核心体验](streaming_quickstart.md)，验证框架的有状态双向流式链路。
