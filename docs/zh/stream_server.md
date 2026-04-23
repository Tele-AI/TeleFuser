# TeleFuser 流式服务器指南

本指南介绍 TeleFuser 的实时流式服务器，它通过 **WebRTC** 或 **WebSocket** 连接持续传输视频（以及可选的音频）—— 与 `telefuser serve` 的批量请求-响应模式不同。

---

## 快速开始

```bash
# 1. 安装 WebRTC 支持
pip install -e ".[webrtc]"

# 2. 启动流式服务器
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation

# 3. 打开 WebRTC 客户端演示
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088
```

在浏览器中打开 `http://localhost:8090`，输入提示词，点击 **Connect** 即可观看实时视频。

---

## 流式模式

TeleFuser 流式服务器支持两种交互模式：

| 模式 | 传输方式 | 方向 | 使用场景 |
|------|----------|------|----------|
| **服务端推送** | WebRTC | 服务端 → 客户端 | 实时预览、文生视频流式传输 |
| **双向交互** | WebSocket | 客户端 ↔ 服务端 | 交互式生成、语音生成视频 |

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

### 双向交互（WebSocket）

```
客户端                          服务端
  │                               │
  │  POST /v1/stream/sessions     │
  │  (任务 + 配置)                │
  │──────────────────────────────►│
  │  { session_id, status }       │
  │◄──────────────────────────────│
  │                               │
  │  WS /v1/stream/ws/{session_id}│
  │◄════════════════════════════►│
  │  发送：输入数据块              │
  │  接收：输出数据块              │
  │                               │
  │  DELETE /v1/stream/sessions/{}│
  │──────────────────────────────►│  清理
```

客户端创建会话、连接 WebSocket，然后推送输入数据块（如语音转视频场景中的音频帧）。服务端通过同一 WebSocket 推回输出数据块。

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
| `--security-level` | `strict` | 管线验证级别（`strict`、`standard`、`permissive`） |
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

流式管线是一个 Python 文件，定义一个 `get_service()` 函数，返回一个服务对象。该服务必须实现以下两种协议之一。

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
| `/v1/stream/webrtc/offer` | POST | 服务端推送 | WebRTC SDP offer/answer 交换 |
| `/v1/stream/webrtc/{session_id}` | DELETE | 服务端推送 | 关闭 WebRTC 会话 |
| `/v1/stream/sessions` | POST | 双向交互 | 创建 WebSocket 会话 |
| `/v1/stream/ws/{session_id}` | WS | 双向交互 | WebSocket 双向连接 |
| `/v1/stream/sessions/{session_id}` | DELETE | 双向交互 | 关闭 WebSocket 会话 |
| `/v1/stream/sessions/{session_id}/status` | GET | 双向交互 | 获取会话状态 |

### WebRTC: SDP Offer

**POST** `/v1/stream/webrtc/offer`

请求：

```json
{
  "sdp": "<SDP offer 字符串>",
  "type": "offer",
  "task": "t2v",
  "prompt": "海上日落",
  "fps": 24,
  "duration_s": 10
}
```

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

### WebSocket: 创建会话

**POST** `/v1/stream/sessions`

请求：

```json
{
  "task": "s2v",
  "config": {"fps": 24}
}
```

响应（`200 OK`）：

```json
{
  "session_id": "def456",
  "stream_mode": "bidirectional",
  "status": "created"
}
```

### WebSocket: 双向连接

**WS** `/v1/stream/ws/{session_id}`

创建会话后，连接 WebSocket。以 JSON 消息发送输入数据块，接收输出数据块：

```json
// 接收的输出数据块
{
  "type": "chunk",
  "session_id": "def456",
  "index": 0,
  "data": { ... },
  "timestamp": 1714000000.0
}

// 结束消息
{
  "type": "done",
  "session_id": "def456",
  "total_chunks": 10,
  "timestamp": 1714000010.0
}
```

### 流式服务健康检查字段

**GET** `/v1/service/health` 在流式服务运行时返回额外字段：

```json
{
  "status": "healthy",
  "stream_ready": true,
  "stream_mode": "server_push",
  "webrtc_active_sessions": 1,
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
  "webrtc_max_sessions": 10
}
```

---

## 客户端集成

### WebRTC（JavaScript）

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

### WebSocket（Python）

```python
import asyncio
import json
import websockets
import httpx

async def stream_bidirectional():
    # 1. 创建会话
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8088/v1/stream/sessions",
            json={"task": "s2v", "config": {"fps": 24}},
        )
        session_id = resp.json()["session_id"]

    # 2. 连接 WebSocket
    async with websockets.connect(
        f"ws://localhost:8088/v1/stream/ws/{session_id}"
    ) as ws:
        # 发送输入
        await ws.send(json.dumps({"type": "audio", "data": "..."}))

        # 接收输出
        async for msg in ws:
            chunk = json.loads(msg)
            if chunk["type"] == "done":
                break
            print(f"数据块 {chunk['index']}")

asyncio.run(stream_bidirectional())
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

---

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TELEFUSER_WEBRTC_MAX_SESSIONS` | `10` | 最大并发 WebRTC 会话数（1-100） |
| `TELEFUSER_STREAM_WS_MAX_CONNECTIONS` | `10` | 最大并发 WebSocket 连接数（1-1000） |
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
| `strict` | 完整验证 —— 禁止危险导入、禁止文件系统访问 |
| `standard` | 适度验证 —— 允许常用库 |
| `permissive` | 最少检查 |

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

点击 **Unmute** 按钮。浏览器要求用户操作后才能播放音频。视频元素默认为静音状态。

### 端口被占用

```bash
# 查找并终止占用端口的进程
lsof -ti:8088 | xargs kill -9
```

### "Stream service is not running"（503）

管线的 `start()` 方法执行失败。检查服务器日志中的错误信息（如缺少视频文件、导入错误等）。

### WebSocket 立即关闭

- 确认已先通过 `POST /v1/stream/sessions` 创建会话
- 确认流式模式为 `bidirectional`（WebSocket 需要此模式）
- 检查 URL 中的会话 ID 是否与创建的会话匹配

### 内存占用过高

每个 WebRTC 会话在队列中持有视频帧。限制 `webrtc_max_sessions` 并确保管线不缓冲过多帧。服务器在客户端断开连接时会自动清理会话。
