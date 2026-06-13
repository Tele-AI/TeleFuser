# TeleFuser 服务指南

本文档涵盖 TeleFuser API 服务器、CLI 命令行工具和 HTTP API 参考。关于 WebRTC 实时流式传输，请参阅[流式服务指南](stream_server.md)。

## 目录

- [快速开始](#快速开始)
- [服务模式](#服务模式)
- [支持的管线](#支持的管线)
- [CLI 命令行工具](#cli-命令行工具)
- [服务器配置](#服务器配置)
- [Service Metadata 指南](./service_metadata.md)
- [HTTP API 参考](#http-api-参考)
- [客户端 SDK](#客户端-sdk)
- [错误处理](#错误处理)
- [最佳实践](#最佳实践)
- [故障排查](#故障排查)

---

## 快速开始

### 1. 安装 TeleFuser

```bash
pip install telefuser
```

### 2. 启动服务器

```bash
# 视频生成
telefuser serve \
    --pipe_path ./examples/wan_video/wan21_14b_image_to_video_h100.py \
    --task i2v \
    --port 8000 \
    --parallelism 1

# 图像生成
telefuser serve \
    --pipe_path ./examples/qwen_image/qwen_image_pipeline.py \
    --task t2i \
    --port 8000 \
    --parallelism 1

# 实时世界模型流式推理（需要 WebRTC 支持）
pip install -e ".[webrtc]"
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/LingBot-World
telefuser stream-serve examples/stream_server/stream_lingbot_world_fast.py -p 8088 --skip-validation
```

### 3. 创建任务

```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2v",
        "prompt": "宇航员在月球上行走",
        "resolution": "720p",
        "aspect_ratio": "16:9"
    }'
```

---

## 服务模式

TeleFuser 提供两种服务命令，针对不同工作负载类型优化：

### `telefuser serve` — 批量请求-响应模式

用于批量文本生成视频、图像生成视频、图像生成和超分辨率。

- 任务 API：`/v1/tasks/*`
- OpenAI 兼容路由：`/v1/images` 和 `/v1/videos`
- 管线契约实现结构化参数暴露
- 异步调度，请求隔离

### `telefuser stream-serve` — 连续流式模式

用于实时世界模型、交互式生成、语音驱动动画和流式媒体。

- Server-push WebRTC：渐进式视频输出
- Bidirectional WebRTC：交互式控制循环（DataChannel + RTP）
- 有状态会话，连续 chunk 生成

完整流式文档请参阅[流式服务指南](stream_server.md)。

---

## 支持的管线

### 世界模型和实时推理

| 管线 | 任务 | 说明 |
|------|------|------|
| `LingBot-World-Fast` | 双向世界模型流式推理 | 交互式 WebRTC 控制循环 — 参见[流式服务指南](stream_server.md) |
| `LiveAct` | S2V（语音转视频） | 语音驱动说话人头部生成 |
| `FlashVSR` | VSR | 流式视频超分辨率 |
| `LongCat-Video` | T2V, I2V, VC | 长视频生成和续写 |

### 视频生成

| 管线 | 任务 | 说明 |
|------|------|------|
| `WanVideo`（Wan2.1 / Wan2.2） | T2V, I2V, FL2V | 主要视频生成系列 |
| `HunyuanVideo` | T2V, I2V | 通过服务示例支持 |
| `LTX Video` | I2V + Audio | 音视频统一生成 |

### 图像生成

| 管线 | 任务 | 说明 |
|------|------|------|
| `Qwen-Image` | T2I, Edit | 图像生成和编辑 |
| `Z-Image` | T2I | 图像生成 |
| `Flux2 Klein` | T2I | 图像生成 |

---

## CLI 命令行工具

TeleFuser CLI 提供启动 API 服务器、验证管道和扫描安全问题的命令。

### 可用命令

| 命令 | 描述 |
|---------|-------------|
| `serve` | 启动 TeleFuser API 服务器 |
| `validate` | 验证管道配置文件 |
| `scan` | 扫描目录中的管道文件 |

### Serve 命令

启动 TeleFuser API 服务器。

```bash
telefuser serve /path/to/pipeline --task i2v [选项]
```

#### 参数

| 参数 | 简写 | 类型 | 默认值 | 描述 |
|-----------|----------|------|---------|-------------|
| `--pipe_path` | `-pp` | string | **必需** | 管道 Python 文件路径 |
| `--task` | `-t` | choice | `i2v` | 任务类型: t2v, i2v, fl2v, vc, t2i, i2i |
| `--port` | `-p` | int | `8000` | 服务器端口 |
| `--host` | | string | `127.0.0.1` | 服务器主机地址 |
| `--cache-dir` | `-c` | string | `work_dirs/server_cache` | 缓存目录 |
| `--parallelism` | `-g` | int | `1` | 并行工作进程数 |
| `--security-level` | | choice | `strict` | 安全级别: none/basic/strict/sandbox |
| `--skip-validation` | | flag | `False` | 跳过安全验证 |
| `--validate-only` | | flag | `False` | 仅验证不启动 |

#### 示例

```bash
# 图生视频，完整参数
telefuser serve \
    --pipe_path ./examples/wan_video/wan21_14b_image_to_video_h100.py \
    --task i2v \
    --port 8080 \
    --host 0.0.0.0 \
    --parallelism 2

# 使用简写形式
telefuser serve -pp ./pipeline.py -t i2v -p 8080 -g 2

# 仅验证
telefuser serve ./pipeline.py --validate-only

# 跳过验证（生产环境不推荐）
telefuser serve ./pipeline.py --skip-validation
```

### Validate 命令

验证管道文件的安全问题。

```bash
telefuser validate /path/to/pipeline.py [选项]
```

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `pipeline_file` | **必需** | 管道 Python 文件路径 |
| `--level` | `strict` | 安全级别: none/basic/strict/sandbox |
| `--json` | `False` | 以 JSON 格式输出 |

```bash
# 默认验证
telefuser validate ./pipeline.py

# 指定安全级别
telefuser validate ./pipeline.py --level basic

# JSON 输出
telefuser validate ./pipeline.py --json
```

### Scan 命令

扫描目录中的管道文件并验证。

```bash
telefuser scan /path/to/directory [选项]
```

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `directory` | **必需** | 要扫描的目录 |
| `--level` | `strict` | 安全验证级别 |
| `--recursive` / `--no-recursive` | `True` | 递归扫描 |

```bash
# 递归扫描
telefuser scan ./examples

# 不递归扫描
telefuser scan ./examples --no-recursive
```

### 获取帮助

```bash
# 所有命令
telefuser --help

# 具体命令
telefuser serve --help
```

---

## 服务器配置

### 支持的任务类型

| 任务 | 描述 |
|------|-------------|
| `t2v` | 文生视频: 从文本生成视频 |
| `i2v` | 图生视频: 从输入图像生成视频 |
| `fl2v` | 首尾帧生视频: 从首帧和尾帧图像生成视频 |
| `vc` | 视频续写: 继续现有视频 |
| `t2i` | 文生图: 从文本生成图像 |
| `i2i` | 图生图: 从输入图像和提示生成图像 |

### 环境变量

| 变量 | 描述 | 默认值 |
|----------|-------------|---------|
| `TELEFUSER_SECURITY_LEVEL` | 安全验证级别 | `STRICT` |
| `TELEFUSER_ALLOW_UNSAFE` | 允许不安全管道 | `false` |
| `TELEFUSER_MAX_PPL_SIZE` | 最大管道文件大小（字节） | `10485760` |
| `TELEFUSER_TASK_TIMEOUT` | 任务超时（秒） | `3600` |
| `TELEFUSER_HOST` | 服务器主机 | `127.0.0.1` |
| `TELEFUSER_PORT` | 服务器端口 | `8000` |
| `TELEFUSER_RATE_LIMIT_ENABLED` | 启用速率限制 | `true` |
| `TELEFUSER_RATE_LIMIT_RPM` | 每分钟请求限制 | `60` |

### 配置文件示例

创建 `.env` 文件:

```env
TELEFUSER_SECURITY_LEVEL=STRICT
TELEFUSER_PORT=8080
TELEFUSER_HOST=0.0.0.0
TELEFUSER_RATE_LIMIT_ENABLED=true
TELEFUSER_RATE_LIMIT_RPM=100
```

### Pipeline 契约与参数定义

服务端不会只靠 Python 函数签名推断 pipeline 能力，而是优先从 example 脚本中加载显式的 pipeline contract。

#### 契约入口

服务端会按顺序查找以下定义：

- `get_pipeline_contract()`
- `get_pipeline_manifest()`
- `PIPELINE_CONTRACT`
- `PIPELINE_MANIFEST`

如果都不存在，服务端会退回到基于 CLI `--task` 的 legacy 兼容契约。

#### 契约结构

一个最小可用的 pipeline contract 应至少声明：

```python
PIPELINE_MANIFEST = {
  "contract_version": "v1",
  "pipeline_name": "wan22_A14B_i2v_h100_distill",
  "supported_tasks": ["i2v"],
  "supported_media_types": ["video"],
  "execution_mode": "serial_single_pipeline",
  "effective_max_concurrent_tasks": 1,
  "entrypoints": {
    "get_pipeline": "get_pipeline",
    "run_with_file": "run_with_file",
  },
  "task_contracts": {
    "i2v": {
      "media_type": "video",
      "required_inputs": ["first_image_path"],
      "optional_inputs": ["last_image_path"],
      "parameters": {
        "prompt": {
          "type": "string",
          "required": True,
          "default": "",
          "description": "正向提示词。",
        },
        "resolution": {
          "type": "string",
          "required": False,
          "default": "720p",
          "enum": ["480p", "720p"],
          "description": "该 example 对外暴露的输出分辨率。",
        },
      },
    },
  },
}
```

#### `task_contracts` 里应该放什么

每个 task contract 可以分成两部分：

- `required_inputs` 和 `optional_inputs`：决定任务推断和输入校验的文件类输入。
- `parameters`：服务端会实际补默认值和校验的用户可见运行参数。

只有用户需要知道、也应该能理解的参数才应放进 `parameters`。像 `num_inference_steps`、固定 distill 配置、
pipeline 内部调优项这类实现细节，应该继续保留在 `PPL_CONFIG` 或 `run()`/`run_with_file()` 里，不应暴露为
服务契约的一部分。

#### 服务端如何使用契约

对于 `/v1/tasks/create`、`/v1/tasks/form` 和 OpenAI 兼容路由，服务端大致按以下顺序处理请求：

1. 根据 `supported_tasks` 校验或推断任务类型。
2. 根据 `required_inputs` 校验文件类输入是否齐全。
3. 对调用方未提供的用户可见参数，应用 `task_contracts[task]["parameters"]` 中的默认值。
4. 校验 contract 中声明为必填的用户可见参数。
5. 仅对 contract 未声明的字段，退回到内建请求模型默认值。

也就是说，当 contract 和通用 API 默认值同时存在时，contract 默认值优先。

#### 服务元数据

`GET /v1/service/metadata` 会把当前生效的 contract 暴露给客户端。对 UI、网关或自动化层来说，这是发现当前
支持哪些 task、以及每个 task 对外暴露哪些参数的推荐入口。

如果需要前端动态表单生成、任务路由策略或网关接入方式，可继续参考
[Service Metadata 消费指南](./service_metadata.md)。

---

## HTTP API 参考

### 基础 URL

```
http://localhost:8000/v1
```

### 交互式文档

服务器运行时可访问:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### 端点概览

| 方法 | 端点 | 描述 |
|--------|----------|-------------|
| POST | `/tasks/create` | 创建新的生成任务 |
| POST | `/tasks/form` | 带文件上传创建任务 |
| GET | `/tasks/{task_id}/status` | 获取任务状态 |
| DELETE | `/tasks/{task_id}` | 取消任务 |
| GET | `/tasks/queue/status` | 获取队列状态 |
| GET | `/files/download/{file_id}` | 下载输出文件 |
| GET | `/service/health` | 健康检查 |
| GET | `/service/status` | 服务状态 |
| GET | `/service/metadata` | 服务元数据 |
| GET | `/service/metrics` | Prometheus 指标 |
| GET | `/service/metrics/json` | JSON 格式指标 |

### 创建任务

```
POST /v1/tasks/create
```

创建新的生成任务。

**请求体**: `TaskRequest`

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|-----------|------|----------|---------|-------------|
| `task` | string | 否 | `t2v` | 任务类型: t2v, i2v, fl2v, vc, t2i, i2i |
| `prompt` | string | 是 | - | 生成提示文本 |
| `aspect_ratio` | string | 否 | `16:9` | 宽高比 |
| `resolution` | string | 否 | `720p` | 分辨率（视频: 720p, 1080p; 图像: 1024x1024） |
| `seed` | int | 否 | `42` | 随机种子 |
| `negative_prompt` | string | 否 | `""` | 负面提示 |
| `first_image_path` | string | 否 | `""` | 输入图像路径（用于 I2V, I2I） |
| `last_image_path` | string | 否 | `""` | 尾帧图像（用于 FL2V） |
| `ref_video_path` | string | 否 | `""` | 参考视频（用于 VC） |
| `target_video_length` | int | 否 | `5` | 视频长度（秒） |
| `output_format` | string | 否 | `png` | 图像输出格式 |
| `output_path` | string | 否 | 自动 | 自定义输出路径 |

**响应**: `TaskResponse`

```json
{
  "task_id": "task_abc123xyz",
  "task_status": "pending",
  "output_path": "task_abc123xyz.mp4"
}
```

#### 示例

**文生视频 (T2V)**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2v",
        "prompt": "宇航员在月球上行走",
        "resolution": "720p",
        "aspect_ratio": "16:9"
    }'
```

**图生视频 (I2V)**:
```bash
# 使用 base64 编码的图片
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "i2v",
        "prompt": "宇航员行走",
        "first_image_path": "data:image/jpeg;base64,/9j/4AAQ..."
    }'

# 或者通过 form 端点同时上传图片并创建任务
curl -X POST "http://127.0.0.1:8000/v1/tasks/form" \
    -F "first_image_file=@/path/to/input.jpg" \
    -F "prompt=宇航员行走" \
    -F "task=i2v"
```

**文生图 (T2I)**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/create" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "t2i",
        "prompt": "美丽的风景",
        "resolution": "1024x1024",
        "output_format": "png"
    }'
```

### 获取任务状态

```
GET /v1/tasks/{task_id}/status
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/tasks/task_abc123xyz/status"
```

**响应**:

```json
{
  "task_id": "task_abc123xyz",
  "status": "completed",
  "start_time": "2024-01-15T08:30:00",
  "end_time": "2024-01-15T08:35:00",
  "error": null,
  "output_path": "task_abc123xyz.mp4"
}
```

### 取消任务

```
DELETE /v1/tasks/{task_id}
```

**示例**:
```bash
curl -X DELETE "http://127.0.0.1:8000/v1/tasks/task_abc123xyz"
```

### 获取队列状态

```
GET /v1/tasks/queue/status
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/tasks/queue/status"
```

**响应**:

```json
{
  "is_processing": true,
  "current_task": "task-123",
  "pending_count": 3,
  "active_count": 1,
  "queue_size": 10
}
```

### 健康检查

```
GET /v1/service/health
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/service/health"
```

**响应**:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T08:30:00Z",
  "version": "1.0.0",
  "pipeline_ready": true
}
```

### 服务状态

```
GET /v1/service/status
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/service/status"
```

### 服务元数据

```
GET /v1/service/metadata
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/service/metadata"
```

**响应**:

```json
{
  "pipeline_file": "/path/to/pipeline.py",
  "parallelism": 1,
  "task": "i2v",
  "security_level": "STRICT",
  "supported_tasks": ["t2v", "i2v", "fl2v", "vc", "t2i", "i2i"],
  "supported_media_types": ["video", "image"]
}
```

### 获取 Prometheus 指标

```
GET /v1/service/metrics
```

获取 Prometheus 兼容的文本格式指标，用于监控系统。

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/service/metrics"
```

**响应** (Prometheus 格式):
```
# HELP telefuser_tasks_created_total Total number of tasks created
# TYPE telefuser_tasks_created_total counter
telefuser_tasks_created_total 100

# HELP telefuser_task_duration_seconds Duration of task execution in seconds
# TYPE telefuser_task_duration_seconds histogram
telefuser_task_duration_seconds_bucket{le="0.005"} 10
telefuser_task_duration_seconds_bucket{le="0.01"} 25
...

# HELP telefuser_gpu_0_memory_used_bytes GPU 0 memory used in bytes
# TYPE telefuser_gpu_0_memory_used_bytes gauge
telefuser_gpu_0_memory_used_bytes 8589934592
```

**可用指标**:

| 指标 | 类型 | 描述 |
|--------|------|-------------|
| `tasks_created_total` | Counter | 创建的任务总数 |
| `tasks_completed_total` | Counter | 成功完成的任务数 |
| `tasks_failed_total` | Counter | 失败的任务数 |
| `tasks_cancelled_total` | Counter | 取消的任务数 |
| `task_duration_seconds` | Histogram | 任务执行时长 |
| `queue_size` | Gauge | 队列总大小 |
| `queue_pending` | Gauge | 等待中的任务数 |
| `queue_processing` | Gauge | 处理中的任务数 |
| `gpu_{id}_memory_used_bytes` | Gauge | GPU 显存使用量 |
| `gpu_{id}_utilization_ratio` | Gauge | GPU 利用率 (0-1) |

### 获取指标（JSON）

```
GET /v1/service/metrics/json
```

获取 JSON 格式的指标，便于解析和调试。

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/service/metrics/json"
```

**响应**:

```json
{
  "uptime_seconds": 3600,
  "tasks": {
    "created": 100,
    "completed": 95,
    "failed": 3,
    "cancelled": 2
  },
  "queue": {
    "size": 10,
    "pending": 5,
    "processing": 1
  },
  "metrics_count": 25,
  "registered_stages": ["stage_1", "stage_2"]
}
```

### 下载输出文件

```
GET /v1/files/download/{file_path}
```

```bash
curl "http://127.0.0.1:8000/v1/files/download/task_abc123xyz.mp4" \
    --output output.mp4
```

---

## OpenAI 兼容 API

TeleFuser 提供 **OpenAI 兼容**的 REST API，让您可以使用标准的 OpenAI SDK 客户端，并无缝迁移现有项目。

### 概述

| API 类型 | 端点 | 特点 |
|---------|------|------|
| **OpenAI 兼容** | `/v1/images`, `/v1/videos` | 同步/异步，行业标准格式 |
| **TeleFuser 原生** | `/v1/tasks/*` | 异步任务管理，功能更强大 |

### 何时使用哪种 API？

| 使用场景 | 推荐 API |
|------|----------|
| 快速原型开发 / 简单需求 | OpenAI API |
| 已有 OpenAI 项目迁移 | OpenAI API |
| 生产环境批处理 | TeleFuser 原生 API |
| 需要细粒度控制 | TeleFuser 原生 API |
| 长时间任务监控 | TeleFuser 原生 API |

### 图像生成 API

#### 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/images/generations` | 文本生成图像（同步） |
| POST | `/v1/images/edits` | 图像编辑（I2I） |
| GET | `/v1/images/{id}/content` | 下载生成的图像 |

#### 请求参数

```json
{
  "prompt": "一只美丽的猫",            // 必填：生成提示词
  "model": "qwen-image",              // 可选：模型名称
  "n": 1,                             // 可选：生成数量 (1-10)
  "quality": "auto",                  // 可选：图像质量 (standard/hd/auto)
  "response_format": "url",           // 可选：返回格式 url 或 b64_json
  "size": "1024x1024",                // 可选：图像尺寸
  "style": "vivid",                   // 可选：风格 (vivid/natural)
  "user": "user_id",                  // 可选：用户标识
  "seed": 42,                         // 可选：随机种子
  "negative_prompt": "模糊"           // 可选：反向提示词
}
```

#### 响应格式

```json
{
  "created": 1699000000,
  "data": [
    {
      "url": "http://localhost:8000/v1/images/task_xxx/content",
      "revised_prompt": "一只美丽的猫"
    }
  ],
  "peak_memory_mb": 4096.5,
  "inference_time_s": 2.5
}
```

#### 示例

```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "美丽的日落",
        "size": "1024x1024",
        "response_format": "url"
    }'
```

**Python SDK:**
```python
from telefuser.client.openai import OpenAICompatibleClient

client = OpenAICompatibleClient("http://localhost:8000")

# 生成图像
response = client.images.generate(
    prompt="美丽的日落",
    size="1024x1024",
    response_format="url"
)

# 保存图像
response.data[0].save("sunset.png")
```

### 视频生成 API

#### 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/videos` | 创建视频生成任务（异步） |
| GET | `/v1/videos` | 列出视频任务 |
| GET | `/v1/videos/{id}` | 获取视频状态 |
| DELETE | `/v1/videos/{id}` | 取消/删除任务 |
| GET | `/v1/videos/{id}/content` | 下载视频 |

#### 请求参数

```json
{
  "prompt": "一只猫在弹钢琴",          // 必填：生成提示词
  "input_reference": "/path/to/img",  // 可选：输入图像（I2V）
  "reference_url": "http://example.com", // 可选：输入图像 URL
  "model": "wan-video",               // 可选：模型名称
  "seconds": 5,                       // 可选：时长 (1-60)
  "size": "1024x576",                 // 可选：视频尺寸
  "seed": 1024,                       // 可选：随机种子
  "negative_prompt": "模糊",          // 可选：反向提示词
  "output_path": "/path/to/output.mp4" // 可选：自定义输出路径
}
```

#### 响应格式

```json
{
  "id": "vid_xxx",
  "object": "video",
  "model": "wan-video",
  "status": "queued",
  "progress": 0,
  "created_at": 1699000000,
  "size": "1024x576",
  "seconds": "5",
  "url": null
}
```

#### 示例

```bash
# 创建任务
curl -X POST "http://localhost:8000/v1/videos" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "一只猫在弹钢琴",
        "seconds": 5,
        "size": "720p"
    }'

# 获取状态
curl "http://localhost:8000/v1/videos/{video_id}"

# 下载视频
curl "http://localhost:8000/v1/videos/{video_id}/content" \
    -o output.mp4
```

**Python SDK:**
```python
from telefuser.client.openai import OpenAICompatibleClient

client = OpenAICompatibleClient("http://localhost:8000")

# 创建视频任务
video = client.videos.create(
    prompt="一只猫在弹钢琴",
    seconds=5,
    size="720p"
)

print(f"任务 ID: {video.id}")
print(f"状态: {video.status}")

# 等待完成
video.wait(timeout=300)

# 下载视频
video.download("output.mp4")
```

### 配置

#### 启用/禁用 OpenAI API

OpenAI 兼容 API 默认启用。您可以通过以下方式控制：

```python
from telefuser.service.api.api_server import ApiServer

# 启用 OpenAI API（默认）
server = ApiServer(enable_openai_api=True)

# 禁用 OpenAI API
server = ApiServer(enable_openai_api=False)
```

或通过命令行：
```bash
# 默认：OpenAI API 已启用
telefuser serve --pipe_path ./pipeline.py --task t2v
```

---

## 客户端 SDK

### 安装

```bash
pip install telefuser
```

### 基本用法

```python
from telefuser.client import TFClient

client = TFClient(base_url="http://127.0.0.1:8000")

# 创建视频任务
task = client.create_t2v_task(
    prompt="宇航员在月球上行走",
    resolution="720p",
    seed=42
)

print(f"任务创建: {task['task_id']}")

# 等待完成
if client.wait_for_completion(task['task_id']):
    client.download_result(task['task_id'], "./output.mp4")
```

### 任务方法

| 方法 | 描述 |
|--------|-------------|
| `create_t2v_task()` | 创建文生视频任务 |
| `create_i2v_task()` | 创建图生视频任务 |
| `create_fl2v_task()` | 创建首尾帧生视频任务 |
| `create_vc_task()` | 创建视频续写任务 |
| `create_t2i_task()` | 创建文生图任务 |
| `create_i2i_task()` | 创建图生图任务 |
| `get_task_status()` | 获取任务状态 |
| `wait_for_completion()` | 等待任务完成 |
| `download_result()` | 下载结果文件 |

### 示例

**图生视频**:
```python
task = client.create_i2v_task(
    prompt="宇航员行走",
    first_image_path="input.jpg",
    resolution="720p"
)
```

**文生图**:
```python
task = client.create_t2i_task(
    prompt="美丽的山景",
    resolution="1024x1024",
    output_format="png"
)
```

**图生图**:
```python
task = client.create_i2i_task(
    prompt="转换成油画风格",
    image_path="input.jpg",
    resolution="1024x1024",
    output_format="jpg"
)
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 描述 |
|------|-------------|
| 200 | 成功 |
| 400 | 错误请求 - 无效参数 |
| 404 | 未找到 - 任务或文件不存在 |
| 422 | 验证错误 - 无效参数 |
| 429 | 请求过多 - 超出速率限制 |
| 500 | 内部服务器错误 |
| 503 | 服务不可用 - 服务器过载 |

### 错误响应格式

```json
{
  "detail": [
    {
      "loc": ["body", "task"],
      "msg": "无效的任务类型",
      "type": "value_error"
    }
  ]
}
```

### 速率限制

- **限制**: 每个 IP 每分钟 60 个请求
- **突发**: 10 个请求
- **豁免路径**: `/v1/service/health`

---

## 最佳实践

### 资源管理

- 根据 GPU 显存指定适当的 `parallelism`
- 生产环境使用专用缓存目录
- 定期清理旧缓存文件

### 错误处理

- 使用 try-except 块包装客户端调用
- 为重试实现指数退避
- 假设完成前检查任务状态

### 安全

- 生产环境使用 `strict` 安全级别
- 上传前验证所有输入文件
- 在反向代理后运行服务器并启用 SSL

### 性能

```python
# 批处理
from telefuser.client import TFClient
import concurrent.futures

client = TFClient(base_url="http://127.0.0.1:8000")

prompts = ["提示 1", "提示 2", "提示 3"]

def generate(prompt):
    task = client.create_t2i_task(prompt=prompt, resolution="1024x1024")
    return client.wait_for_completion(task['task_id'])

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(generate, prompts))
```

---

## 故障排查

### 连接被拒绝

```bash
# 检查服务器是否运行
curl http://127.0.0.1:8000/v1/service/health

# 检查防火墙
sudo ufw allow 8000
```

### 内存不足

- 降低 `--parallelism`
- 在管道配置中启用 CPU 卸载
- 使用更小的批次大小

### 任务超时

```python
# 增加超时
client.wait_for_completion(task_id, timeout=600)
```

### 管道验证失败

```bash
# 验证管道
telefuser validate /path/to/pipeline.py --level strict

# 跳过验证（生产环境不推荐）
telefuser serve /path/to/pipeline.py --skip-validation
```

### 端口已被占用

```bash
telefuser serve /path/to/pipeline.py --port 8081
```

### 安全验证失败

1. 检查详细报告: `telefuser validate /path/to/pipeline.py`
2. 修复管道文件中的安全问题
3. 或绕过验证: `telefuser serve /path/to/pipeline.py --skip-validation`

---

*更多信息请访问 [TeleFuser 文档](https://github.com/telefuser/telefuser)。*
