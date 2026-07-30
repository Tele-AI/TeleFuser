# 流式服务

TeleFuser 仅使用 LiveKit 作为流式传输后端。`telefuser stream-serve` 接受 `get_service()` 返回
`ServerPushService` 或 `BidirectionalService` 的 pipeline 文件；不再提供 backend 选择器或直接 SDP 接口。

LiveKit 负责浏览器 WebRTC 连接、room、重连、媒体传输和可靠数据消息；TeleFuser 负责模型 worker、准入、
session 状态、pipeline 执行和 token 签发。因此必须使用 LiveKit Cloud 或自托管 LiveKit Server。

本文中的**服务实例**特指 pipeline 文件的 `get_service()` 返回的单个对象。模型 worker 只加载它一次，它可以
拥有多个用户 session；一个用户 session 并不是一个新的服务实例。

## 为什么选择 LiveKit

TeleFuser 面向高性能多模态生成模型推理，流式服务需要同时支持持续媒体输出、双向控制和长时间运行的
有状态模型 session。LiveKit 提供的实时传输能力与这些目标相契合：

- room 为模型 session 提供稳定的传输边界，使浏览器连接与 session 自有的模型状态相互独立；
- 媒体轨道承载渐进式视频和音频，data topic 承载控制、状态与有界 telemetry；
- 受限 token 区分 controller、viewer 和 worker，适配 TeleFuser 的 session 所有权与权限模型；
- `ServerPushService` 和 `BidirectionalService` 共用一个流式入口和客户端连接模型；
- LiveKit 负责连接、重连和媒体交付，TeleFuser 可以专注于模型 worker、准入、pipeline 执行和资源释放。

## 本地安装与启动

LiveKit Python SDK 已包含在 TeleFuser 基础依赖中：

```bash
pip install -e .
```

另外安装开发用 LiveKit Server，并通过当前操作系统的包管理器安装 `coturn`：

```bash
# Debian/Ubuntu；其他平台请安装对应的 coturn 软件包。
sudo apt-get update
sudo apt-get install -y coturn

curl -sSL https://get.livekit.io | bash
livekit-server --dev
```

开发服务器监听 `ws://127.0.0.1:7880`，默认凭据为 `devkey` / `secret`，生产环境不得使用这些凭据。

启动 TeleFuser：

```bash
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --port 8088 \
  --skip-validation
```

同一命令也支持 server-push pipeline：

```bash
telefuser stream-serve examples/stream_server/stream_video_replay.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --port 8088 \
  --skip-validation
```

也可使用 `TELEFUSER_LIVEKIT_*` 环境变量；显式 CLI 参数优先。

## 浏览器 Demo

仓库内页面设置了 `iceTransportPolicy: relay`，因此必须启动与其匹配的 TCP TURN 服务；生产 LiveKit 部署
可以使用不同的 TURN 配置。先启动以下仅供开发使用的 coturn 进程：

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 \
  --relay-ip=127.0.0.1 \
  --listening-port=3478 \
  --min-port=49160 --max-port=49200 \
  --user=livekit-demo:livekit-demo-password \
  --realm=livekit.local \
  --fingerprint --lt-cred-mech \
  --no-tls --no-dtls --no-cli \
  --allow-loopback-peers
```

在第四个终端启动 LingBot 控制页面：

```bash
python examples/stream_server/livekit_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8092 \
  --no-open
```

打开 `http://127.0.0.1:8092`，选择初始图片并点击 **Start**。Demo 会代理 `/v1/stream/*` 请求、获取
controller token、加入 LiveKit room、播放视频轨道，并通过 `tf.control` 发送页面或键盘相机控制消息。

使用 VS Code Remote SSH 时，需要映射 demo HTTP 端口、LiveKit signaling 端口，以及 LiveKit 使用的 TURN
listener。仓库 demo 固定使用 `turn:127.0.0.1:3478?transport=tcp` 和开发凭据 `livekit-demo` /
`livekit-demo-password`；生产环境必须同时修改浏览器配置和 LiveKit 部署。

把远端 TCP `8092`、`7880` 和 `3478` 映射到相同本地端口，然后打开 `http://127.0.0.1:8092`。Loopback
listener、静态密码、禁用 TLS 和 `--allow-loopback-peers` 只适用于通过隧道访问的可信开发主机，不能复制到
公网生产部署。

完整浏览器链路此时包含 coturn（`3478`）、LiveKit（`7880`）、TeleFuser（`8088`）和页面（`8092`）。
启动 session 前，`curl http://127.0.0.1:8088/v1/service/health` 应显示 ready 且 worker idle。成功运行时，
页面会显示视频轨道以及 `control_state`、生成 Stage 和 `chunk_sent` 等状态。关闭服务时先停止 session 或
关闭浏览器页面，再按相反顺序停止四个进程，避免 LiveKit 和模型 worker drain 时浏览器持续重连。

## 架构与生命周期

```text
Browser A ── HTTP API / LiveKit room A ── session runner A ─┐
                                                            ├─ 一个模型 worker
Browser B ── HTTP API / LiveKit room B ── session runner B ─┘       │
                                                                    └─ 一个共享服务实例
                                                                           ├─ pipeline session A 状态
                                                                           └─ pipeline session B 状态
```

| 术语 | 生命周期与所有权 |
|---|---|
| 服务进程 | 一个 `telefuser stream-serve` 进程，包含 HTTP API、准入 scheduler 和当前进程内 worker。 |
| 模型 worker | 加载一个服务实例并拥有常驻 session 容量；当前 runtime 只创建一个。 |
| 服务实例 | 一次 `get_service()` 的返回值；模型权重和 pipeline actor graph 只加载一次，由它的 session 共享。 |
| Session runner | 一个已准入 HTTP session 对应的 LiveKit room 连接和传输任务；多个 runner 共享 worker 的服务实例。 |
| Pipeline session | `BidirectionalService.create_session()` 创建的用户状态，包含该用户自有的模型/cache/control 状态。 |

`--worker-gpu-map 0,1,2,3` 是把四张卡分配给一个模型 worker 及其一个服务实例，并不会创建四个 worker 或
四个模型副本。

1. Controller 通过 `POST /v1/stream/sessions` 创建 session。
2. Scheduler 对其准入、排队或拒绝，并分配 worker 上的一个常驻 session 槽位。
3. TeleFuser 创建 LiveKit room，返回权限受限的 controller token。
4. Worker 加入 room 并启动 pipeline。
5. 视频和 PCM16 音频作为 LiveKit track 发布；状态与指标使用可靠 data topic。
6. 对 `BidirectionalService`，只有 controller 可以把规范化 control 消息送入 pipeline。
7. 删除、超时、controller 离开或 pipeline 完成时，按 actor 所有权释放状态并关闭 room。

每个流式 stage worker 只属于一个 pipeline actor。这里的 stage worker 是 pipeline 内部执行对象，不是上表中的
LiveKit 模型 worker。重连不会搬移 actor cache。Server-push pipeline 根据请求 config 启动并持续输出 chunk；
bidirectional pipeline 额外提供 create、pull、control、close。

## LingBot 多 session 时分复用

`--max-sessions-per-worker` 控制一个模型 worker 可以保留多少个 LiveKit session，不会创建更多服务实例。
容量大于 1 时必须使用 `BidirectionalService`，因为 server-push session 没有控制空闲信号。框架负责 room
runner 与常驻 session 准入；底层服务仍需自行保证 session 状态隔离，并在需要时串行化模型计算。

仓库中的 LingBot-World-Fast 和 LingBot-World v2 服务在共享服务实例内使用一个 execution lease 完成串行化。
例如，允许两个常驻 session：

```bash
telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --max-sessions-per-worker 2 --control-idle-timeout 10 --skip-validation
```

每个已准入 LingBot room 都会在共享服务实例中创建独立的 pipeline session、control queue、noise 状态、VAE
状态和模型 cache。合法的 `control_state`、`control`、`prompt` 或 `reset` 消息会刷新该 session 的 lease
活跃时间。当有其他 session 等待，且当前持有者超过 `control_idle_timeout` 秒没有收到合法控制消息时，当前
持有者停止提交新 chunk，并在正在执行的 chunk 完成后交出 lease。等待 session 随后获得 lease；被挂起的
session 在 controller 再次发送控制消息后可以重新申请 lease。

这是 chunk 边界上的时分复用，不是 continuous batching 或模型副本：LingBot 服务实例最多只有一个 session
chunk 在途。`control_idle_timeout` 不会关闭当前持有者、释放它的常驻 session 槽位或清除 cache；它只允许等待
session 在 chunk 边界取得 lease。挂起 session 仍占用该用户状态对应的 GPU 显存，因此应根据实测显存余量设置
`max_sessions_per_worker`。对于持续按住的输入，controller 必须周期性重发 `control_state`；仓库内浏览器
demo 会在有按键按下时每秒发送一次心跳。

系统中有两条相互独立的 FIFO 等待队列：

| 等待 | 适用对象 | 容量与可观测结果 |
|---|---|---|
| HTTP 准入队列 | 所有常驻 session 槽位占满后到达的请求 | 由 `queue_size` 限制；返回带 `queue_position` 的 HTTP 202，禁用或满时返回 HTTP 429。 |
| LingBot execution-lease 队列 | 已准入且正在申请模型执行权的 session | 上限为常驻 LingBot session 数；发出 `lease_queued`、`lease_granted`、`lease_parked` 状态。 |

交出 execution lease 不会让 session 回到 HTTP 准入队列；只有 session 终态清理才释放常驻槽位，并准入下一条
HTTP 排队请求。

## HTTP API

| 接口 | 方法 | 用途 |
|---|---|---|
| `/v1/stream/sessions` | POST | 创建并准入 controller session |
| `/v1/stream/sessions/{session_id}` | GET | 查询 session 状态 |
| `/v1/stream/sessions/{session_id}` | DELETE | drain 并关闭 session |
| `/v1/stream/sessions/{session_id}/tokens` | POST | 创建 viewer token |
| `/v1/stream/health` | GET | LiveKit scheduler/worker 健康状态 |
| `/v1/service/health` | GET | 通用服务健康状态 |
| `/v1/service/ready` | GET | readiness probe |
| `/v1/service/metadata` | GET | Pipeline 与 transport metadata |
| `/v1/service/metrics` | GET | Prometheus 指标 |

创建 controller session：

```bash
curl -X POST http://127.0.0.1:8088/v1/stream/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "identity": "controller-1",
    "prompt": "A first-person view moving through a forest",
    "image_path": "examples/lingbot/assets/test_1.jpeg",
    "config": {"fps": 16}
  }'
```

如需执行一分钟 LingBot-World v2 回放，启动
`examples/lingbot/lingbot_world_v2_image_to_video_h100.py` 并使用：

```json
{
  "fps": 16,
  "chunk_size": 4,
  "frame_num": 957,
  "max_duration_seconds": 60.0
}
```

完整 chunk 策略把该请求映射为 60 个 chunk 和 59.75 秒输出媒体。v2 示例使用 `local_attn_size=18` 与
`sink_size=6`，因此 KV 容量保持固定，session 自有的 noise 与 VAE 状态增量推进。可复现 LiveKit workload
和日期化的四卡实测见 [TeleFuser 与 AIPerf](benchmark_aiperf.md)。

成功响应包含 `session_id`、`room`、`livekit_url`、`token`、`worker_id` 和 `status`。session 会先占用
worker 的常驻 session 槽位；槽位全部占满后才进入 HTTP 准入队列。排队时返回 HTTP 202 和
`queue_position`；队列长度为零且槽位全满时返回 HTTP 429。

在 `/v1/stream/health` 中，`workers_busy` 统计至少保留了一个 session 的模型 worker 数，不是 session 数或
当前正在计算的 chunk 数；`workers_idle` 统计没有常驻 session 的 worker；`queued_sessions` 只表示 HTTP 准入
队列深度。尚无 session runner 连接时，`livekit_connected=false` 是正常状态，并不表示模型或服务实例加载失败。

创建没有控制权限的 viewer token：

```bash
curl -X POST http://127.0.0.1:8088/v1/stream/sessions/<session_id>/tokens \
  -H 'Content-Type: application/json' \
  -d '{"identity":"viewer-1"}'
```

主动关闭：

```bash
curl -X DELETE http://127.0.0.1:8088/v1/stream/sessions/<session_id>
```

## LiveKit 数据协议

| Topic | 方向 | 内容 |
|---|---|---|
| `tf.control` | controller 到 worker | 可靠 JSON control 消息 |
| `tf.status` | worker 到 room | 生命周期和 chunk 状态 |
| `tf.metrics` | worker 到 room | 有界 runtime 指标 |
| `tf.asset` | 保留 | 未来的有界 asset 消息 |

支持 `control_state`、`control`、`prompt`、`reset` 和 `stop`，例如：

```json
{"type":"control_state","controls":["w","j"]}
```

也支持带版本的 envelope：

```json
{"version":1,"session_id":"<id>","type":"control_state","payload":{"controls":["w"]}}
```

消息默认受 `TELEFUSER_LIVEKIT_MAX_DATA_MESSAGE_BYTES`（12 KiB）限制。未知 control、重复项、非法 JSON、
错误 topic、session 不匹配，以及 viewer 发出的 control 都会被拒绝。

## CLI 与环境变量

```text
telefuser stream-serve PIPE_PATH [OPTIONS]
```

主要选项包括 `--host`、`--port`、`--livekit-url`、`--livekit-api-key`、`--livekit-api-secret`、
`--num-workers`、`--worker-gpu-map`、`--max-sessions-per-worker`、`--control-idle-timeout`、`--queue-size`、
`--session-timeout`、`--token-ttl`、
`--controller-timeout`、`--room-empty-timeout` 和 `--worker-mode`。

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TELEFUSER_LIVEKIT_URL` | 必填 | LiveKit WebSocket URL |
| `TELEFUSER_LIVEKIT_API_KEY` | 必填 | 用于签发 token 的 API key |
| `TELEFUSER_LIVEKIT_API_SECRET` | 必填 | 用于签发 token 的 API secret |
| `TELEFUSER_LIVEKIT_HOST` | `0.0.0.0` | HTTP API 监听地址 |
| `TELEFUSER_LIVEKIT_PORT` | `8088` | HTTP API 端口 |
| `TELEFUSER_LIVEKIT_NUM_WORKERS` | `1` | 模型 worker 数；当前进程内 runtime 必须为 `1` |
| `TELEFUSER_LIVEKIT_WORKER_GPU_MAP` | 未设置 | 分配给唯一 worker 的 GPU group，如 `0,1,2,3` |
| `TELEFUSER_LIVEKIT_MAX_SESSIONS_PER_WORKER` | `1` | 每个模型 worker 常驻的双向 session 数 |
| `TELEFUSER_LIVEKIT_CONTROL_IDLE_TIMEOUT` | `10` | 有等待者时允许接管 execution lease 的控制空闲秒数 |
| `TELEFUSER_LIVEKIT_QUEUE_SIZE` | `0` | 排队数量；零表示所有常驻 session 槽位满时立即拒绝 |
| `TELEFUSER_LIVEKIT_SESSION_TIMEOUT` | `1800` | session 最大生命周期（秒） |
| `TELEFUSER_LIVEKIT_TOKEN_TTL` | `3600` | join token 生命周期（秒） |
| `TELEFUSER_LIVEKIT_CONTROLLER_TIMEOUT` | `60` | controller 离开后的宽限期 |
| `TELEFUSER_LIVEKIT_ROOM_EMPTY_TIMEOUT` | `30` | room 为空后的宽限期 |

当前 runtime 要求 `worker_mode=in-process`，并只启动一个模型 worker。在实现 process-worker 隔离前，额外模型
副本需要启动独立的 `stream-serve` 进程并在外部路由请求；各进程的准入队列与 session 状态彼此独立。
`--skip-validation` 只应用于可信本地文件，不建议生产使用。

## 生产部署

- 使用 LiveKit Cloud 或官方自托管部署方式，不要暴露 `livekit-server --dev`。
- 使用独立 API 凭据，API secret 只能保存在 TeleFuser 服务端。
- 在 LiveKit 中配置 TLS、advertised node address、UDP/TCP media port 和 TURN。
- 通过部署层的鉴权与网络策略限制 TeleFuser HTTP API。
- 监控 `/v1/service/ready`、worker failure、queue depth 和 session expiration。
- Chunk period 表示相邻输出 cadence；实时运行要求 p95 cadence 小于一个 chunk 的媒体时长，并为传输和编码
  留出余量。Pipeline residence 和客户端 delivery FPS 是不同指标。

## 故障排查

- **HTTP ready 但没有媒体：**确认浏览器和 worker 都能访问 LiveKit URL，并检查 participant/track 日志。
- **浏览器反复重连：**检查 signaling、TURN、防火墙和 LiveKit advertised node address。
- **控制无效：**确认发送者使用 controller token、topic 为 `tf.control`，且 control 受支持。
- **HTTP 429：**所有常驻 session 槽位都已占满且 `queue_size=0`，或队列已满。
- **Session 未释放：**调用 session DELETE 接口并检查 controller/room timeout。
- **本地 LiveKit 连接返回代理 HTTP 503：**部分 native SDK 路径会读取 `HTTP_PROXY`，但不会应用
  `NO_PROXY`。连接 `ws://127.0.0.1:7880` 时，启动 TeleFuser 前应取消 `HTTP_PROXY`、`HTTPS_PROXY`、
  `ALL_PROXY` 及其小写变量。
- **强制退出后残留 GPU worker：**重启前终止残留的 `spawn_main` 进程。
