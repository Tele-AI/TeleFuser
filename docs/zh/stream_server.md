# TeleFuser 流式服务器指南

本指南介绍 TeleFuser 的实时流式服务器，它通过 **WebRTC** 连接持续传输视频（以及可选的音频）—— 与 `telefuser serve` 的批量请求-响应模式不同。

---

## 快速开始

```bash
# 1. 安装
pip install -e .

# 默认安装已包含 WebRTC 支持。

# 2. 启动流式服务器
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation

# 3. 打开 WebRTC 客户端演示
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088
```

在浏览器中打开 `http://localhost:8090`，输入提示词，点击 **Connect** 即可观看实时视频。

---

## 流式模式

TeleFuser 流式服务器支持两种交互模式，均可通过 WebRTC 使用：

| 模式 | 传输方式 | 方向 | 使用场景 |
|------|----------|------|----------|
| **服务端推送** | WebRTC (RTP) | 服务端 → 客户端 | 实时预览、文生视频流式传输 |
| **双向交互** | WebRTC (RTP + DataChannel) | 客户端 ↔ 服务端 | 交互式生成、键盘/摄像头控制、语音生成视频 |

`telefuser stream-serve` 启动的是 stream-only app。它只暴露 `/v1/stream/*`、`/v1/stream/webrtc/*` 和
`/v1/service/*`；不会暴露请求-响应任务路由、文件下载路由，也不会暴露 OpenAI 兼容的 `/v1/images`
和 `/v1/videos` 路由。批量任务提交请使用 `telefuser serve`。

### 服务端推送（WebRTC）

```
客户端                          服务端
  │                               │
  │  POST /v1/stream/webrtc/offer │
  │  (SDP offer + 提示词)         │
  │──────────────────────────────►│
  │                               │  stream_task() → serve()
  │  SDP answer                   │  在后台线程运行
  │◄──────────────────────────────│
  │                               │
  │  ◄──── RTP 视频帧 ──────────  │
  │  ◄──── RTP 音频帧 ──────────  │ （可选）
  │                               │
  │  DELETE /v1/stream/webrtc/{id}│
  │──────────────────────────────►│  清理
```

管线的 `serve()` 方法产出包含 JPEG 编码帧的数据块。WebRTC 层将其解码为 `av.VideoFrame` 对象，并以目标帧率通过 RTP 流式传输到浏览器。

### 双向交互（WebRTC）

```
客户端                          服务端
  │                               │
  │  pc.createDataChannel("telefuser")
  │  pc.addTransceiver("video", recvonly)
  │  （可选添加摄像头/麦克风轨道）
  │                               │
  │  POST /v1/stream/webrtc/offer │
  │  (SDP offer + 配置)           │
  │──────────────────────────────►│
  │                               │  create_session(config)
  │                               │  pull_chunks(session_id)
  │  SDP answer                   │  启动 ChunkRouter
  │◄──────────────────────────────│
  │                               │
  │  ──── DataChannel JSON ─────► │  push_chunk(session_id, data)
  │  ◄──── RTP 视频帧 ──────────  │  ChunkRouter → FrameGeneratorTrack
  │  ◄──── RTP 音频帧 ──────────  │  ChunkRouter → AudioGeneratorTrack
  │  ◄──── DataChannel JSON ────  │  ChunkRouter → 元数据
  │                               │
  │  ──── 媒体轨道（可选）─────► │  IncomingVideoRelay / AudioRelay
  │                               │  → push_chunk(session_id, frames)
  │                               │
  │  {"type": "stop"}             │
  │  或 DELETE /v1/stream/webrtc/{}│
  │──────────────────────────────►│  close_session + 清理
```

客户端**必须**在生成 SDP offer 之前创建名为 `"telefuser"` 的 DataChannel。服务端复用该通道接收控制输入并发送元数据输出。视频和音频通过 RTP 媒体轨道双向传输。

**ChunkRouter** 是服务端的分发适配器，精确消费管线的 `pull_chunks()` 生成器一次，并进行分发：

- `frames_b64` → 解码 JPEG → `av.VideoFrame` → 推送到 `FrameGeneratorTrack`（RTP 视频）
- `audio_b64` → 解码 PCM16 → 馈入 `AudioGeneratorTrack`（RTP 音频）
- 剩余元数据 → 序列化为 `StreamChunkMessage` JSON → 通过 DataChannel 发送

---

## CLI 用法

```bash
telefuser stream-serve <pipeline_file> [选项]
```

### 选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port`, `-p` | `8088` | 服务端口 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--gpu-num`, `-g` | `1` | 管线支持时传给 `get_service(gpu_num=...)` 的 GPU 数量 |
| `--security-level` | `strict` | 管线验证级别（`none`、`basic`、`strict`、`sandbox`）。`sandbox` 是 best-effort 受限加载检查，不是运行时隔离。 |
| `--skip-validation` | `false` | 跳过管线文件安全检查 |

### 示例

```bash
# 使用默认设置启动
telefuser stream-serve examples/stream_server/stream_video_replay.py

# 自定义端口和地址
telefuser stream-serve my_pipeline.py -p 9000 --host 127.0.0.1

# 跳过验证（仅开发环境使用）
telefuser stream-serve my_pipeline.py -p 8088 --skip-validation
```

---

## 创建流式管线

流式管线是一个 Python 文件，定义一个返回服务对象的 `get_service()` 函数。该函数可以接收由
`stream-serve --gpu-num` 提供的可选 `gpu_num` 参数。服务必须实现以下两种协议之一。

### ServerPushService 协议

适用于接收单个请求并持续输出流的管线。

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyService:
    def start(self) -> None:
        """服务启动时调用一次。加载模型、打开文件等。"""
        ...

    def stop(self) -> None:
        """服务关闭时调用。释放资源。"""
        ...

    async def serve(self, request: dict) -> AsyncGenerator[dict, None]:
        """为单个请求产出输出数据块。

        request 字典包含 WebRTC offer 中的所有字段
        （prompt、task、fps、duration_s 等）。
        """
        for i in range(10):
            yield {
                "type": "chunk",
                "index": i,
                "frames_b64": [encode_frame(frame)],
                "fps": 24,
            }

def get_service() -> MyService:
    return MyService()
```

### BidirectionalService 协议

适用于接收持续输入并产生持续输出的管线。

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyBidirectionalService:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def create_session(self, config: dict) -> str:
        """创建新会话。返回 session_id。"""
        ...

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        """推送输入数据块到会话。"""
        ...

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        """产出该会话的输出数据块。"""
        ...

    def close_session(self, session_id: str) -> None:
        """关闭会话并释放资源。"""
        ...

def get_service() -> MyBidirectionalService:
    return MyBidirectionalService()
```

### 数据块格式

服务端推送模式的数据块应包含以下字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 固定为 `"chunk"` |
| `index` | `int` | 是 | 数据块序号 |
| `frames_b64` | `list[str]` | 是 | Base64 编码的 JPEG 帧列表 |
| `fps` | `int` | 是 | 目标帧率 |
| `num_frames` | `int` | 否 | 本数据块的帧数 |
| `resolution` | `str` | 否 | 帧分辨率（如 `"1920x1080"`） |
| `prompt` | `str` | 否 | 回显输入提示词 |
| `timestamp` | `float` | 否 | 数据块生成时间戳 |
| `audio_b64` | `str` | 否 | Base64 编码的 PCM16 音频数据 |
| `audio_sample_rate` | `int` | 否 | 音频采样率（默认：48000） |
| `audio_channels` | `int` | 否 | 音频通道数（默认：1） |

> 音频字段是可选的。当存在时，WebRTC 层会在视频轨道之外创建音频轨道。

---

## API 参考

### 接口总览

| 接口 | 方法 | 模式 | 说明 |
|------|------|------|------|
| `/v1/stream/webrtc/offer` | POST | 两种模式 | WebRTC SDP offer/answer 交换 |
| `/v1/stream/webrtc/{session_id}` | DELETE | 两种模式 | 关闭 WebRTC 会话 |
| `/v1/stream/sessions/{session_id}` | DELETE | 两种模式 | 关闭会话（管线 + WebRTC） |
| `/v1/stream/sessions/{session_id}/status` | GET | 两种模式 | 获取会话状态 |

stream app 也会暴露 `/v1/service/health`、`/v1/service/ready`、`/v1/service/metadata`
和 `/v1/service/metrics/json` 等服务端点。

### WebRTC: SDP Offer

**POST** `/v1/stream/webrtc/offer`

此端点同时服务于**服务端推送**和**双向交互**模式。服务器根据管线的服务类型自动检测模式。

请求：

```json
{
  "sdp": "<SDP offer 字符串>",
  "type": "offer",
  "session_id": "可选-uuid",
  "task": "t2v",
  "prompt": "海上日落",
  "fps": 24,
  "duration_s": 10,
  "config": {}
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `sdp` | `str` | 是 | 浏览器的 SDP offer |
| `type` | `str` | 是 | SDP 类型（通常为 `"offer"`） |
| `session_id` | `str` | 否 | 自定义会话 ID（不填则自动生成） |
| `task` | `str` | 是 | 任务类型（如 `t2v`、`i2v`、`bidirectional`） |
| `prompt` | `str` | 否 | 输入提示词 |
| `fps` | `int` | 否 | 目标视频帧率（默认：24） |
| `config` | `dict` | 否 | 额外配置，传递给 `BidirectionalService.create_session()` |

> 允许附加字段（`"extra": "allow"`），会被转发给管线。

响应（`200 OK`）：

```json
{
  "session_id": "abc123",
  "sdp": "<SDP answer 字符串>",
  "type": "answer"
}
```

错误响应：

| 状态码 | 条件 |
|--------|------|
| `400` | 无效的 SDP 或协商失败 |
| `503` | 流式服务未运行或已达最大会话数 |

### WebRTC: 关闭会话

**DELETE** `/v1/stream/webrtc/{session_id}`

响应（`200 OK`）：

```json
{
  "session_id": "abc123",
  "status": "closed"
}
```

**DELETE** `/v1/stream/sessions/{session_id}` 会在管线会话和 WebRTC 会话都存在时一起关闭两侧。
如果其中一侧已经不存在，但另一侧关闭成功，端点仍返回 `200 OK`；管线侧关闭失败会记录 warning，
不会再静默忽略。

### WebRTC: DataChannel 协议（双向交互）

在双向交互模式下，客户端创建的 `"telefuser"` DataChannel 双向传输 JSON 消息。

**客户端 → 服务端消息：**

```json
// 控制输入（如键盘、提示词）
{"type": "control", "key": "ArrowUp", "action": "press"}
{"type": "control", "prompt": "新的提示词"}

// 停止会话
{"type": "stop"}
```

**服务端 → 客户端消息：**

```json
// 输出数据块元数据（媒体通过 RTP 单独发送）
{
  "type": "chunk",
  "session_id": "abc123",
  "index": 0,
  "data": {"type": "chunk", "index": 0, "fps": 24, "timestamp": 1714000000.0},
  "timestamp": 1714000000.0
}

// 生成完成
{
  "type": "done",
  "session_id": "abc123",
  "total_chunks": 240,
  "timestamp": 1714000010.0
}
```

> 控制消息格式由管线定义。以上示例展示了 `ArrowOverlayService` 使用的约定。你的管线的 `push_chunk()` 接收客户端发送的任意 JSON。

### 流式服务健康检查字段

**GET** `/v1/service/health` 在流式服务运行时返回额外字段：

```json
{
  "status": "healthy",
  "stream_ready": true,
  "stream_mode": "server_push",
  "webrtc_active_sessions": 1,
  "webrtc_server_push_sessions": 1,
  "webrtc_bidirectional_sessions": 0,
  "webrtc_max_sessions": 10
}
```

### 流式服务元数据字段

**GET** `/v1/service/metadata`：

```json
{
  "service_type": "stream",
  "stream_mode": "server_push",
  "pipeline_file": "examples/stream_server/stream_video_replay.py",
  "security_level": "STRICT",
  "runner": "StreamPipelineService",
  "webrtc_active_sessions": 0,
  "webrtc_server_push_sessions": 0,
  "webrtc_bidirectional_sessions": 0,
  "webrtc_max_sessions": 10
}
```

---

## 客户端集成

### WebRTC：服务端推送（JavaScript）

服务端推送模式的最小浏览器客户端：

```javascript
// 局域网：无需 iceServers 配置（或使用默认 STUN）
// 公网：配置 STUN + TURN 服务器
const pc = new RTCPeerConnection({
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    // 生产环境 / NAT 穿透时添加 TURN 服务器：
    // { urls: "turn:your-domain.com:3478", username: "user", credential: "pass" },
  ],
});
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("http://localhost:8088/v1/stream/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    task: "t2v",
    prompt: "一只猫在弹钢琴",
  }),
});

const answer = await resp.json();
await pc.setRemoteDescription(answer);

pc.ontrack = (event) => {
  document.getElementById("video").srcObject = event.streams[0];
};
```

### WebRTC：双向交互（JavaScript）

带 DataChannel 控制和可选摄像头/麦克风输入的全双工客户端：

```javascript
const pc = new RTCPeerConnection();

// 1. 客户端必须在生成 offer 之前创建 DataChannel
const dc = pc.createDataChannel("telefuser");
dc.onopen = () => {
  // 通道打开后发送控制消息
  dc.send(JSON.stringify({ type: "control", key: "ArrowUp", action: "press" }));
};
dc.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.type === "done") console.log("生成完成:", msg.total_chunks, "个数据块");
};

// 2. 可选：向服务端发送摄像头/麦克风
// const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
// stream.getTracks().forEach(t => pc.addTrack(t, stream));

// 3. 接收服务端输出的视频/音频
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

pc.ontrack = (evt) => {
  if (evt.track.kind === "video") {
    document.getElementById("video").srcObject = evt.streams[0];
  }
};

// 4. SDP 交换（与服务端推送使用相同端点）
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("http://localhost:8088/v1/stream/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    task: "bidirectional",
    prompt: "一只狗在奔跑",
    config: { fps: 24 },
  }),
});

const answer = await resp.json();
await pc.setRemoteDescription(new RTCSessionDescription({
  sdp: answer.sdp,
  type: answer.type,
}));

// 5. 停止会话
// dc.send(JSON.stringify({ type: "stop" }));
// await fetch(`http://localhost:8088/v1/stream/webrtc/${answer.session_id}`, { method: "DELETE" });
```

### Gradio UI

TeleFuser 提供了基于 Gradio 的 Web UI 用于 WebRTC 流式传输：

```bash
# 启动流式服务器
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088

# 启动 Gradio UI（另开终端）
python webui/stream_app.py --server-url http://localhost:8088 --port 7860
```

Gradio UI 提供提示词输入、时长滑块和内嵌的 WebRTC 视频播放器。

---

## 示例

### 视频回放服务

`examples/stream_server/stream_video_replay.py` —— 一个服务端推送管线，加载本地视频文件并以帧数据块形式流式传输：

- 在 `start()` 时使用 PyAV 加载视频帧和音频
- 预编码所有帧为 JPEG（缓存后不会每次请求重新编码）
- 以可配置的节奏流式传输数据块，可选包含音频
- 支持 `duration_s`、`realtime` 和 `prompt` 请求参数

```bash
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation
```

### WebRTC 客户端演示

`examples/stream_server/webrtc_client_demo.py` —— 通过 Python HTTP 服务器提供的独立 HTML 页面：

```bash
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088 --port 8090
```

打开浏览器页面，包含视频播放器、提示词输入框，以及连接/停止/取消静音按钮。

### 方向键叠加服务（双向交互）

`examples/stream_server/stream_arrow_overlay.py` —— 一个双向管线，加载视频并根据键盘输入叠加方向键 HUD：

- 实现 `BidirectionalService` 协议，带会话状态管理
- `push_chunk()` 接收 `{"type": "control", "key": "ArrowUp", "action": "press"}` 并更新 `pressed_keys`
- `pull_chunks()` 产出带有根据 `pressed_keys` 绘制的方向键叠加的视频帧
- 循环播放视频帧直到会话关闭

```bash
# 启动服务端
telefuser stream-serve examples/stream_server/stream_arrow_overlay.py -p 8088 --skip-validation

# 启动客户端（浏览器打开 localhost:8092）
python examples/stream_server/webrtc_arrow_overlay_demo.py --server-url http://localhost:8088
```

在浏览器中按方向键，即可实时看到方向键叠加效果。

### 双向客户端演示

`examples/stream_server/webrtc_bidirectional_demo.py` —— LingBot-World-Fast 双向 WebRTC 客户端，提供：

- DataChannel 用于发送提示词和控制消息
- LingBot 输入图片路径、提示词、生成参数和方向键控制
- 服务端输出视频播放器、控制 HUD 和 DataChannel 消息日志

```bash
python examples/stream_server/webrtc_bidirectional_demo.py --server-url http://localhost:8088
```

### LingBot-World-Fast 流式生成

`examples/lingbot/stream_lingbot_world_fast.py` 提供 LingBot-World-Fast 的双向流式服务。该服务使用 WebRTC RTP 输出生成视频，通过 DataChannel 接收 prompt 和方向控制消息。当前 demo 页面不采集浏览器摄像头和麦克风；LingBot 当前仅输出视频，没有音频输出。

#### 模型文件

LingBot-World-Fast 需要两类权重：

| 配置 | 示例 | 说明 |
|----------|------|------|
| `TF_MODEL_ZOO_PATH` | `/storage/model_zoo` | 必填环境变量，用于定位两套模型目录 |
| 基础模型子目录 | `/storage/model_zoo/Wan2.2-I2V-A14B` | 基础 Wan2.2 I2V 权重，包含 VAE、T5 文本编码器和 tokenizer |
| Fast 模型子目录 | `/storage/model_zoo/lingbot/lingbot-world-fast` | LingBot-World-Fast DiT 权重 |

#### 启动服务端

启动前设置 `TF_MODEL_ZOO_PATH`，并通过 `--gpu-num` 传入 worker 数量。TeleFuser 会在服务内部创建
Ulysses worker，因此启动命令不需要 `torchrun` 或分布式环境变量。使用 `CUDA_VISIBLE_DEVICES` 选择物理 GPU。

```bash
TF_MODEL_ZOO_PATH=/storage/model_zoo \
TELEFUSER_TURN_SERVER='turn:127.0.0.1:3478' \
TELEFUSER_TURN_USERNAME=telefuser \
TELEFUSER_TURN_CREDENTIAL=your-turn-password \
telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
  --gpu-num 4 -p 8088 --host 0.0.0.0 --skip-validation
```

等待日志出现以下内容后再连接浏览器 demo：

```text
Starting stream server on 0.0.0.0:8088
```

可以用健康检查确认服务已可用：

```bash
curl --noproxy '*' http://127.0.0.1:8088/v1/service/health
```

#### 启动浏览器 demo

通过 VS Code Remote SSH 使用笔记本浏览器访问远端 demo 时，浏览器 JavaScript 实际运行在笔记本本地。此时建议使用 TURN，并将远端 `3478` 端口转发到本地 `3478`。

```bash
python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://localhost:8088 \
  --port 8091 \
  --image-path /tmp/lingbot_test_input.png \
  --action-path examples/data/lingbot_world_fast \
  --frame-num 81 \
  --chunk-size 3 \
  --fps 16 \
  --turn-url 'turn:localhost:3478?transport=tcp' \
  --turn-username telefuser \
  --turn-credential your-turn-password \
  --force-turn-relay \
  --ice-gather-timeout-ms 30000 \
  --no-open
```

浏览器打开：

```text
http://localhost:8091
```

`--image-path` 和 `--action-path` 都是服务端路径，不是笔记本本地路径。实时键盘控制只会从
`--action-path` 加载 `intrinsics.npy`，并在整个会话中固定使用第一行内参。demo 默认开启代理，浏览器只需
访问 demo 端口；`/v1/stream/webrtc/*` 请求会由 demo 进程转发到 `--server-url`。

#### 不使用 TURN：在服务器浏览器查看

如果浏览器真的运行在服务器侧，例如远程桌面、VNC 或 noVNC 中的 Chrome，可以不使用 TURN。此时不要设置 `TELEFUSER_TURN_*`，demo 也不要传 `--turn-url`。

服务端：

```bash
env -u TELEFUSER_TURN_SERVER \
-u TELEFUSER_TURN_USERNAME \
-u TELEFUSER_TURN_CREDENTIAL \
telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
  --gpu-num 4 -p 8088 --host 0.0.0.0 --skip-validation
```

demo：

```bash
env -u TELEFUSER_TURN_SERVER \
-u TELEFUSER_TURN_USERNAME \
-u TELEFUSER_TURN_CREDENTIAL \
python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8091 \
  --image-path /tmp/lingbot_test_input.png \
  --action-path examples/data/lingbot_world_fast \
  --frame-num 81 \
  --chunk-size 3 \
  --fps 16 \
  --no-open
```

在服务器浏览器打开：

```text
http://127.0.0.1:8091
```

#### 方向控制

demo 支持页面 D-pad、方向键以及与 LingBot 源码一致的 `WASD`/`IJKL` 键盘控制：

| 输入 | cam 模式含义 |
|------|--------------|
| `↑` / `W` | 向前移动 |
| `↓` / `S` | 向后移动 |
| `A` / `D` | 向左/向右横移 |
| `←` / `J` | 向左偏航 |
| `→` / `L` | 向右偏航 |
| `I` / `K` | 向上/向下俯仰 |

控制只会作用于“尚未开始生成”的后续 chunk；已经在 denoising 或 decoding 的 chunk 不会被即时改变。
长按按键会持续生成受控 chunk；松开所有按键后，服务端停止请求新 chunk，WebRTC 会重复显示最近输出的一帧。

相机位姿从单位矩阵开始。每个 latent 间隔累计 4 次视频帧控制步，与 LingBot 源码中的轨迹生成方式一致。
相机内参在创建会话时固定，绝对位姿与 pitch 状态跨 chunk 累计，并保留 chunk 边界两侧的相对位姿变化。

DataChannel 日志中出现以下状态时，表示方向控制已被服务端消费并应用到某个生成 chunk：

```text
"stage":"control_state"
"stage":"applying_direction_control"
```

demo 默认开启 `Control HUD`。该 HUD 会叠加在使用了方向控制的输出 chunk 左上角，用于确认控制链路生效。确认链路后可以在页面取消勾选，观察纯模型输出效果。

常用控制强度参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--control-move-step` | `0.05` | 每个视频帧的前进/后退位移 |
| `--control-yaw-step-degrees` | `2.0` | 每个视频帧的偏航角度 |
| `--control-lateral-step` | `0.05` | 每个视频帧的横向位移 |
| `--control-pitch-step-degrees` | `2.0` | 每个视频帧的俯仰角度 |
| `--control-pitch-limit-degrees` | `85.0` | 绝对俯仰角限制 |
| `--show-control-hud / --no-show-control-hud` | `true` | 是否在受控 chunk 上叠加方向 HUD |

#### `cam` 与 `act` 控制模式

| 模式 | 输入 | 说明 |
|------|------|------|
| `cam` | `poses + intrinsics` | 相机轨迹控制。服务端会将方向键转换为相机位姿，再构造 6 通道 camera control。 |
| `act` | `poses + intrinsics + action` | 动作控制。需要 7 通道 action-control 权重。 |

当前 demo 默认使用 `cam`。如果模型权重是 camera-control 权重，应保持：

```bash
--control-mode cam
```

只有在使用 action-control 权重时才应切换到：

```bash
--control-mode act
```

服务示例在 `examples/lingbot/stream_lingbot_world_fast.py` 中设置
`PPL_CONFIG["control_mode"]="cam"`。session 的控制模式必须与 pipeline 初始化
时的模式一致。若要使用 action-control 权重，请在启动服务前将该值改为 `"act"`，
并给 demo 传入 `--control-mode act`。

#### 帧数与控制输入契约

LingBot 只接受完整 latent chunk。`frame_num` 必须为 `4n + 1`，并且得到的
latent frame 数必须能被 `chunk_size` 整除。默认 `chunk_size=3` 时，合法输出帧数为
9、21、33、…、81；非法值会直接报错，不再向下截断。离线 control 文件必须提供覆盖
目标视频窗口的 pose 和 action。intrinsics 可以是单个静态 `(4,)` 值，也可以是逐视频帧
的 `(frames, 4)` 值。显式 control tensor 必须与 pipeline 的控制模式、device、dtype
和完整 latent-chunk shape 一致。

#### 显存与分辨率

LingBot 的 KV cache 会随 `frame_num` 和输出分辨率增长。832x480、81 帧接近 80GB H100 的显存上限，推荐先减少帧数验证链路：

```bash
# 在浏览器 demo 或 session 配置中：
--frame-num 21
```

常用调参方向：

| 参数 | 影响 |
|------|------|
| `PPL_CONFIG["resolution"]` | 选择 480p 或 720p 输出面积 |
| `--frame-num` | 降低总生成帧数和 latent chunk 数 |
| `--chunk-size` | 影响每次生成的 latent chunk 大小 |

服务当前只允许一个 LingBot active session。重新连接前请点击 Stop，或调用：

```bash
curl -X DELETE http://127.0.0.1:8088/v1/stream/webrtc/<session_id>
```

如果仍然 OOM，检查是否有其他进程占用 GPU：

```bash
nvidia-smi
```

---

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TELEFUSER_WEBRTC_MAX_SESSIONS` | `10` | 最大并发 WebRTC 会话数（1-100） |
| `TELEFUSER_STUN_SERVERS` | `["stun:stun.l.google.com:19302"]` | STUN 服务器 URL 列表（JSON 数组） |
| `TELEFUSER_TURN_SERVER` | `None` | TURN 服务器 URL（如 `turn:your-domain.com:3478`） |
| `TELEFUSER_TURN_USERNAME` | `None` | TURN 服务器用户名 |
| `TELEFUSER_TURN_CREDENTIAL` | `None` | TURN 服务器密码 |

### CORS

流式服务初始化时会自动添加 CORS 中间件（浏览器端 WebRTC 客户端需要）。默认允许所有来源。

### 安全级别

管线文件在加载前会进行验证。仅在开发环境中使用 `--skip-validation`：

| 级别 | 说明 |
|------|------|
| `none` | 禁用验证 |
| `basic` | 静态 AST 分析 |
| `strict` | 静态 AST 分析加导入限制 |
| `sandbox` | strict 检查加 best-effort 受限加载验证步骤。它不是运行时隔离。 |

---

## 公网部署

当流式服务器部署在公网上（服务端和客户端不在同一网络），WebRTC 需要额外配置以实现 NAT 穿透和满足浏览器安全要求。

### 1. HTTPS（必需）

浏览器在非 localhost 的 HTTP 页面上禁止使用 WebRTC，必须通过 HTTPS 提供服务：

```bash
# 方案 A：Let's Encrypt 免费证书（生产环境推荐）
certbot certonly --standalone -d your-domain.com

# 方案 B：自签证书（开发/测试用）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 启动时指定 TLS 证书
uvicorn telefuser.service.api.app:app \
  --host 0.0.0.0 --port 443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### 2. STUN/TURN 服务器

**STUN** 用于发现客户端的公网 IP（轻量，免费）。**TURN** 在直连失败时中继媒体流（需要带宽，建议自建）。

```bash
# 安装 coturn（常用开源 TURN 服务器）
apt install coturn
```

最小 `/etc/turnserver.conf` 配置：

```ini
listening-port=3478
tls-listening-port=5349
realm=your-domain.com
server-name=your-domain.com
fingerprint
lt-cred-mech
user=telefuser:your-secret-password
external-ip=203.0.113.10
min-port=49152
max-port=65535
```

```bash
systemctl enable coturn && systemctl start coturn
```

配置 TeleFuser 使用 TURN 服务器：

```bash
export TELEFUSER_TURN_SERVER="turn:your-domain.com:3478"
export TELEFUSER_TURN_USERNAME="telefuser"
export TELEFUSER_TURN_CREDENTIAL="your-secret-password"
telefuser stream-serve pipeline.py -p 8000
```

### 3. 防火墙端口

| 端口 | 协议 | 用途 |
|------|------|------|
| 443 | TCP | HTTPS（API + SDP 信令） |
| 3478 | TCP+UDP | STUN/TURN |
| 5349 | TCP | TURN over TLS |
| 49152-65535 | UDP | TURN 中继媒体端口 |

### 部署架构

```
┌─────────┐     HTTPS/443      ┌──────────────┐
│  客户端  │◄──────────────────►│  Nginx/CDN   │
│ （浏览器）│                    │ （TLS 终结） │
│          │     UDP            ├──────────────┤
│          │◄──────────────────►│  TeleFuser   │
│          │  （WebRTC 媒体）   │  :8000       │
│          │                    ├──────────────┤
│          │◄──────────────────►│  coturn      │
│          │   UDP 3478 +       │ （TURN 中继）│
│          │   49152-65535      │              │
└─────────┘                    └──────────────┘
```

> **局域网部署**无需配置 STUN/TURN，默认设置在同一网络内开箱即用。

---

## 故障排查

### WebRTC 连接失败（ICE 错误）

**局域网**：确保服务器绑定到 `0.0.0.0`（而非 `127.0.0.1`），且防火墙未阻止 UDP。

**公网**：

1. 确认已配置 STUN/TURN 服务器（参见[公网部署](#公网部署)）
2. 确保 TURN 服务器端口（3478、49152-65535）已开放
3. 测试 TURN 连通性：`turnutils_uclient -u user -w pass your-domain.com`
4. 检查浏览器开发者工具 → `chrome://webrtc-internals` 查看 ICE 候选详情

### 浏览器没有声音

只有管线输出 `audio_b64` 时浏览器才会有声音。`stream_video_replay.py` 可以包含音频；LingBot-World-Fast 当前只输出视频帧，不生成音频，因此 demo 中没有静音/取消静音按钮。

如果使用包含音频的管线，浏览器通常要求用户操作后才能播放音频。此时点击页面上的 **Unmute** 按钮即可。

### 端口被占用

```bash
# 查找并终止占用端口的进程
lsof -ti:8088 | xargs kill -9
```

### "Stream service is not running"（503）

管线的 `start()` 方法执行失败。检查服务器日志中的错误信息（如缺少视频文件、导入错误等）。

### 内存占用过高

每个 WebRTC 会话在队列中持有视频帧。限制 `webrtc_max_sessions` 并确保管线不缓冲过多帧。服务器在客户端断开连接时会自动清理会话。

LingBot-World-Fast 的显存主要由 DiT 权重、VAE/text encoder 和每个 session 的 runtime/KV cache 占用。如果出现 CUDA OOM：

1. 确认没有重复连接或遗留 session：`curl --noproxy '*' http://127.0.0.1:8088/v1/service/health`
2. 点击 demo 的 Stop 或调用 `DELETE /v1/stream/webrtc/{session_id}` 关闭旧会话
3. 降低 `PPL_CONFIG["max_area"]` 或 `--frame-num`
4. 使用 `nvidia-smi` 检查是否有其他任务占用 GPU
5. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片化影响
