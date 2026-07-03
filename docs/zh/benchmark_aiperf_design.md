# TeleFuser 与 AIPerf Benchmark 设计

本文档记录 benchmark 的设计决策、对标边界和后续扩展方向。

具体跑法、脚本、参数和结果文件说明请看：

- [TeleFuser 与 AIPerf Benchmark](benchmark_aiperf.md)

## 1. Benchmark 分层

### 1.1 批处理视频服务 benchmark

这类场景指的是：

- 用户提交一次文本或图像条件
- 服务端异步生成一个完整视频
- 客户端轮询任务状态
- 最终拿到视频文件

这一类当前最适合：

- `telefuser serve`
- OpenAI 兼容 `/v1/videos`
- `AIPerf` 端到端 HTTP benchmark

### 1.2 世界模型流式系统 benchmark

这类场景指的是：

- 长生命周期 session
- 持续输入控制信号
- 持续输出视频帧或媒体流
- 更关注控制反馈闭环，而不是单次完整视频生成

这一类更接近：

- `telefuser stream-serve`
- WebRTC + DataChannel
- 世界模型在线交互推理

这两类负载的评价指标不同，不能混在同一套 benchmark 结论里。

### 1.3 世界模型实时流式系统框架 shortlist

如果 TeleFuser 后面要做 `stream-serve`，更合适的对标对象应该是系统框架 / serving runtime，而不是模型本体。

当前更接近的开源系统层候选是：

- `SGLang-Omni`
- `vLLM-Omni`
- `vLLM`
- `SGLang`
- `Ray Serve`
- `Triton Inference Server`

如果需要实时媒体传输层，还可以单独观察：

- `LiveKit`
- `Janus`
- `mediasoup`
- `aiortc`
- `Pion`

这些是 transport 组件，不是推理框架本身。

需要明确区分的是：

- `Matrix-Game`、`MineWorld`、`LongLive`、`CausVid`、`Helios` 更偏“模型 / 研究系统 / reference system”
- 它们适合用来定义负载和行为模式
- 但不应混进“系统框架 shortlist”

真正的世界模型流式端到端 benchmark 仍需要我们自己定义请求协议、会话状态和指标。

### 1.4 `stream-serve` 第一批系统框架对比顺序

如果后面要真的落地 `stream-serve` 的端到端 benchmark，第一批建议先比：

1. `TeleFuser stream-serve`
2. `SGLang-Omni`
3. `vLLM-Omni`

优先级原因很简单：

- `SGLang-Omni` 更贴近我们现在想做的“长会话、多阶段、低延迟在线生成”系统形态
- `vLLM-Omni` 可以作为更通用的第二基线
- 两者都比通用 `Ray Serve` / `FastAPI` 更接近推理系统本体

如果需要把 transport 也纳入对比，再单独固定 WebRTC / SSE / WebSocket 实现，不要和推理框架混算。

### 1.5 批处理视频模型对标顺序

如果目标是评估 **TeleFuser 与其它视频推理框架** 的差异，那么第一轮 benchmark 必须满足下面这些约束：

1. **模型相同**
   - 例如都跑 `Wan2.1-I2V-14B-480P`
   - 或都跑 `HunyuanVideo-1.5`
   - 或都跑 `LTX-2`

2. **任务相同**
   - `t2v` 只和 `t2v` 比
   - `i2v` 只和 `i2v` 比
   - 不把 `t2v`、`i2v`、`ti2v` 混在一起

3. **输入相同**
   - 同 prompt
   - 同输入图
   - 同输出分辨率
   - 同帧数或等价生成长度
   - 同 diffusion steps

4. **硬件相同**
   - 同 GPU 型号
   - 同卡数
   - 同显存约束
   - 尽量控制一致的软件栈

5. **只变推理框架**
   - 可以比较 `TeleFuser`、`Diffusers`、`xDiT`、`LightX2V`、`FastVideo`、`SGLang Diffusion`
   - 但不能把“更换模型蒸馏版权重”误当成“只变框架”

只要不满足这些条件，就不应被视为第一轮主 benchmark。

推荐优先级：

1. `Wan2.1-I2V-14B-480P`
2. `HunyuanVideo-1.5`
3. `LTX-2`

### 1.6 当前已落地的第一版可执行基线

仓库里已经补齐了第一版“同模型、只变框架”的可执行闭环，当前不是只停留在规划层面了：

- TeleFuser 侧固定 480P 服务示例：[examples/wan_video/wan21_14b_image_to_video_480p_service.py](../../examples/wan_video/wan21_14b_image_to_video_480p_service.py)
- Diffusers 侧独立基线服务：[benchmarks/baseline/diffusers_wan_i2v/service.py](../../benchmarks/baseline/diffusers_wan_i2v/service.py)

这版首个可直接开跑的对照组合固定为：

- 模型：`Wan2.1-I2V-14B-480P`
- 分辨率：`832x480`
- 帧数：`81`
- 推理步数：`40`
- `guidance_scale=5.0`
- `fps=16`

这里刻意把 TeleFuser 侧的 480P 服务从通用面积启发式改成了固定 `832x480`，目的是避免与官方 Diffusers 基线出现宽高不一致，保证第一轮对照可比。

### 1.7 AIPerf 对批处理视频负载的支持情况

AIPerf 对 `telefuser serve` 这类批处理视频服务已经有明确支持，当前覆盖的核心能力是：

- OpenAI 兼容 `/v1/videos` 的异步提交 / 轮询 / 下载流程
- `video_generation` endpoint
- `--download-video-content` 把下载时间纳入端到端时延
- `multipart/form-data` 请求编码
- `--extra-inputs` 传递视频生成参数
- `single_turn` / `multi_turn` 这类输入组织方式

对 TeleFuser 这条线来说，AIPerf 已经能支撑两类最重要的批处理视频负载：

1. **T2V**
   - 直接用 `video_generation` endpoint
   - 用文本 prompt 驱动 `/v1/videos`

2. **I2V**
   - 同样走 `video_generation` endpoint
   - 通过 `reference_url` 或 `input_reference` 传入参考图
   - 我们当前的 `telefuser_aiperf` prompt 文件已经在用 `reference_url` 这条路

也就是说：

- AIPerf 不是只能测纯文本视频生成
- 它已经可以覆盖我们当前 TeleFuser 批处理视频 benchmark 里的 `T2V / I2V` 主线
- 但它的上游文档和教程仍然更偏通用视频生成，不是专门为 TeleFuser 的 `I2V` 场景定制

当前对我们最有用的落点是：

- `telefuser serve` + `AIPerf` 已足够做 batch video E2E benchmark
- `Wan2.1-I2V-14B-480P` 这条线已经可以直接跑
- 后续若要扩展到更复杂的 `I2V` 变体，只需要继续补 prompt/schema 适配，而不是重做 benchmark 框架

### 1.8 Baseline 接口协议标准

随着后面要逐步接入更多对照实现，例如：

- `Diffusers`
- `xDiT`
- `FastVideo`
- `LightX2V`
- `SGLang Diffusion`
- `vLLM-Omni`

仓库里需要一套统一的 **baseline benchmark contract**，否则每接一个新 baseline，就会重复发明一套：

- 目录布局
- 启动脚本
- 健康检查方式
- HTTP / WebRTC 适配层
- 指标采集字段
- 文档说明格式

这里要标准化的不是 baseline 的内部实现，而是 **benchmark 视角下的最小可测协议**。

#### 1.8.1 标准化目标

第一版建议定义成 **versioned contract**：

- `contract_version: v1`
- `mode: batch_video | stream_world`

第一版只覆盖两类主负载：

1. `batch_video`
   - 对应 `telefuser serve`
   - 对应 AIPerf 当前已经能覆盖的 `/v1/videos` 异步完整视频生成

2. `stream_world`
   - 对应 `telefuser stream-serve`
   - 对应 WebRTC 长会话、持续控制、持续输出的视频流 benchmark

后续如果再扩展：

- `audio_video_realtime`
- `multi_camera_world`
- `tool_augmented_agent_stream`

应在 `v1` 之外继续演进，而不是一开始把所有可能性都揉进同一套协议。

#### 1.8.2 目录协议

第三方或对照实现统一放在：

- `benchmarks/baseline/<baseline_name>/`

推荐最小目录结构：

- `README.md`
- `service.py` 或 `launcher.py`
- `configs/`
- `scripts/`
- `benchmark_contract.yaml`

其中：

- `telefuser_aiperf/` 不是 baseline，它是 TeleFuser 自己的 benchmark harness
- `baseline/` 下面放“只变推理框架”的对照实现
- `benchmark_contract.yaml` 用来描述 baseline 能力，而不是替代 README
- 当前 TeleFuser benchmark harness 和 Diffusers baseline 已经分别提供了第一版薄 contract：
  - [benchmarks/telefuser_aiperf/benchmark_contract.yaml](../../benchmarks/telefuser_aiperf/benchmark_contract.yaml)
  - [benchmarks/baseline/diffusers_wan_i2v/benchmark_contract.yaml](../../benchmarks/baseline/diffusers_wan_i2v/benchmark_contract.yaml)

#### 1.8.3 能力声明协议

每个 baseline 建议暴露一个能力声明文件，例如 `benchmark_contract.yaml`，最少包含：

- `contract_version`
- `baseline_name`
- `mode`
- `model_family`
- `supported_tasks`
- `transport`
- `request_encoding`
- `result_delivery`
- `metrics`

建议第一版字段示例：

```yaml
contract_version: v1
baseline_name: diffusers_wan_i2v
mode: batch_video
model_family: wan21_i2v_14b_480p
supported_tasks:
  - i2v
transport: http
request_encoding:
  - multipart_form
  - json
result_delivery:
  - poll_status
  - download_content
capabilities:
  supports_reference_url: true
  supports_input_reference_upload: true
  supports_cancel: true
metrics:
  - request_latency
  - completion_latency
  - download_latency
  - peak_memory_mb
```

这一层的价值是：

- benchmark 脚本能先检查 baseline 支持什么
- 文档和自动化部署脚本可以直接消费这份声明
- 后续接新 baseline 时，先补 contract，再补适配脚本

#### 1.8.4 `batch_video` 最小接口协议

对 `batch_video` 类型 baseline，建议统一最小 HTTP 面：

- `GET /v1/service/health`
- `POST /v1/videos`
- `GET /v1/videos/{id}`
- `GET /v1/videos/{id}/content`
- `DELETE /v1/videos/{id}` 可选

最小请求语义建议统一为：

- `prompt`
- `model`
- `size`
- `seconds`
- `seed`
- `negative_prompt`
- `reference_url` 或 `input_reference`

允许 baseline 自己额外扩展字段，但 benchmark 第一轮只依赖最小公共子集。

`GET /v1/videos/{id}` 的最小返回字段建议统一为：

- `id`
- `status`
- `progress`
- `created_at`
- `completed_at`
- `error`
- `url` 或 `file_path`

其中状态集合建议至少覆盖：

- `queued`
- `generating`
- `completed`
- `failed`
- `cancelled`

这样 AIPerf 和后续自定义 harness 都可以复用同一套 submit / poll / download 流程。

#### 1.8.5 `stream_world` 最小接口协议

对 `stream_world` 类型 baseline，建议统一最小会话面：

- `GET /v1/service/health`
- `POST /v1/stream/webrtc/offer`
- `DELETE /v1/stream/webrtc/{session_id}`

如果使用 DataChannel，建议至少约定这些消息类别：

- `control`
- `status`
- `chunk`
- `done`
- `error`

其中：

- `control` 由客户端发送，代表方向控制、动作控制或结构化控制输入
- `status` 由服务端发送，代表会话阶段和状态机推进
- `chunk` 由服务端发送，代表一批可消费的输出
- `done` 代表本轮或整个 session 正常结束
- `error` 代表 session 失败或不可恢复异常

不要求所有 baseline 都与 TeleFuser 的 DataChannel payload 完全一致，但必须有一层 adapter 能映射到这套 benchmark 语义。

#### 1.8.6 指标协议

接口统一之后，还需要统一结果侧最小指标集合。

`batch_video` 至少建议采：

- `request_latency`
- `queue_latency`
- `completion_latency`
- `download_latency`
- `success_rate`
- `throughput`

`stream_world` 至少建议采：

- `offer_rtt`
- `connected_latency`
- `first_frame_latency`
- `first_metadata_latency`
- `steady_state_fps`
- `control_ack_latency`
- `control_to_next_frame_latency`
- `session_success_rate`

如果 baseline 能提供服务端附加指标，也建议通过统一字段上报，例如：

- `inference_time_s`
- `peak_memory_mb`
- `server_metrics_url`

#### 1.8.7 非目标

这套标准当前 **不** 试图做下面这些事情：

- 不统一 baseline 的内部代码结构
- 不强制所有 baseline 使用同一种 Web 框架
- 不强制所有 baseline 使用同一种任务队列实现
- 不把 transport 协议细节过度收紧到无法接第三方系统

也就是说：

- 我们标准化的是“怎么测”
- 不是“怎么实现”

#### 1.8.8 第一批落地顺序

这套 baseline contract 第一批建议先落到：

1. `benchmarks/baseline/diffusers_wan_i2v/`
2. 后续第二个 batch baseline
3. 再考虑 `stream_world` 类型 baseline

原因是：

- `diffusers_wan_i2v` 已经是现成的第一版对照实现
- 最容易先把目录协议、能力声明和 batch_video 最小接口约束跑通
- `stream_world` 的协议复杂度更高，适合在 batch contract 稳定后再继续抽象

这部分设计一旦落地，后续新增 baseline 时应优先遵守 contract，再补实现，而不是先写一套散落脚本再回头整理。

#### 1.8.9 Batch Compare 自动化约束

对于第一版 `Wan2.1-I2V-14B-480P` 对照，不应该继续靠手工 SSH 一条条拼：

- 停 burner
- 起 TeleFuser
- 跑 AIPerf
- 停服务
- 起 baseline
- 再跑一次

这条链路现在应该固化成一个总入口：

```bash
python3 scripts/remote_bench_sync.py batch-compare
```

这条自动化入口的职责边界应该明确成下面这样：

1. 先校验远程环境
   - `telefuser` CLI 可用
   - `aiperf` CLI 可用
   - compare config 存在

2. 只管理“我们自己的 benchmark 资源”
   - 可以对指定 GPU 执行 burner `stop`
   - 可以拉起和停止本次 benchmark 启动的服务进程
   - 不应默认杀掉远程机器上其它已在运行的真实计算任务

3. GPU 资源策略默认非破坏性
   - `stop burner` 之后检查目标 GPU 是否真的空闲
   - 如果还有其它 compute process，占用信息直接报出来
   - 默认立即失败，或按显式参数进入等待模式

4. 对比执行顺序固定
   - 同一张 GPU
   - 先 TeleFuser
   - 再 Diffusers baseline
   - 使用同一组对齐后的 compare config

5. 结果输出标准化
   - 返回各自最新 `summary.json` 路径
   - 汇总 `request_latency`
   - 汇总 `request_throughput`
   - 后续如果补 `completion_latency` / `download_latency` 也沿同一路径扩展

这样做的目的不是把所有逻辑塞进一个巨型脚本，而是把：

- 资源预留
- 服务启停
- benchmark 执行
- 结果采集

变成一条可重复、可审计、默认安全的 batch compare 链路。

## 2. 设计边界

### 2.1 不把通用 serving 框架混进第一层主对标

`FastAPI`、`Ray Serve`、`BentoML`、`Xinference` 这类项目当然有价值，但它们解决的问题主要是：

- HTTP 服务封装
- 副本管理
- 路由与部署
- 多租户和线上治理

它们不是视频推理优化本身。

### 2.2 世界模型流式 benchmark 仍需要独立 harness

当前开源里仍然没有一个像 AIPerf 这样成熟、通用的 **世界模型流式系统 benchmark**。

`AIPerf` 更适合当前的完整视频生成 HTTP 服务 benchmark，`telefuser stream-serve` 最终仍然需要单独建设一套 benchmark harness。

这类 benchmark 更应该关注：

- session 建立时延
- 首帧时延
- steady-state FPS
- frame jitter
- control-to-frame latency
- 长会话稳定性
- 会话并发数
- 断线重连与恢复

当前仓库里已经补了一版最小可跑 harness：

- `benchmarks/telefuser_aiperf/scripts/run_stream_bench.py`
- `benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh`
- `benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json`

这一版的边界是：

- 先对齐 `telefuser stream-serve` 的真实 WebRTC offer / answer 协议
- 先对齐 `LingBotWorldFastService` 的 DataChannel 控制消息格式
- 先把核心时延指标打通，不强行塞进 AIPerf core

当前可直接采的指标有：

- `offer RTT`
- `connected latency`
- `first frame latency`
- `first metadata latency`
- `stream FPS`
- `control ack latency`
- `control-to-next-frame latency`

当前已知限制也要写清楚：

- benchmark harness 本身支持多 session 参数
- 但 `LingBotWorldFastService` 当前代码只允许 `1 active session`
- 所以现在这条 pipeline 还不能拿来做真正的多会话并发对比

### 2.2.1 自动化部署与运行链路

这条流式 benchmark 不应该依赖手工 SSH 拼命令。建议把准备流程固定成一条自动化链路：

1. `sync`
   - 同步 `benchmarks/telefuser_aiperf`
   - 同步 `examples/lingbot/stream_lingbot_world_fast.py`
   - 同步文档和 burner 脚本

2. `install`
   - 在独立 benchmark venv 安装 AIPerf
   - 安装 `aiortc` / `opencv-python-headless`
   - 可选安装 TeleFuser 到独立服务 venv

3. `verify`
   - 检查脚本、配置、模型入口是否存在
   - 检查 `telefuser` / `aiperf` CLI
   - 检查 WebRTC 依赖是否已装
   - 检查流式模型 zoo 路径是否可见

4. `start`
   - 由远程服务环境启动 `telefuser stream-serve`
   - 由 benchmark 环境启动 `run_stream_bench.sh`

补充说明：

- `stream-bootstrap` 会优先探测 `TF_MODEL_ZOO_PATH`，再回退到仓库内 `model_zoo` 和几个常见远程挂载点
- 一旦解析到可用根目录，会在远程仓库下创建非破坏性的 `model_zoo -> 实际路径` 软链，后续启动命令可以直接用默认相对路径
- 这样能避免每次手工记住 `TF_MODEL_ZOO_PATH`，也能保证 TeleFuser 服务和 benchmark 使用同一份权重根目录

建议的总入口是：

```bash
python3 scripts/remote_bench_sync.py stream-bootstrap
```

这会把流式 benchmark 的前置条件一次性准备好，减少手工出错。

## 3. 后续扩展方向

如果后续要 benchmark `telefuser stream-serve`，建议在同一份总文档下继续扩展为：

1. `SGLang-Omni` / `vLLM-Omni` 的系统框架接入
2. WebRTC / SSE / WebSocket transport
3. world-model session endpoint
4. interactive trace workload
5. frame-level metrics
6. control-to-frame latency metrics

这样才能覆盖 TeleFuser 最核心的实时世界模型场景，并保持和批处理视频 benchmark 在同一入口下管理。

## 4. AIPerf 需要做哪些改造

AIPerf 现在已经足够支撑 `telefuser serve` 这类批处理视频压测，但如果要把它扩展到 `stream-serve` 和更完整的世界模型流式场景，还需要补下面这些能力。

### 4.1 保留的部分

不需要重写的部分：

- benchmark runner 和 config 解析
- 并发控制、warmup、请求数、速率模型
- 结果汇总、summary JSON、原始记录导出
- HTTP trace 采集
- 远程执行、结果落盘、artifact 管理

这些都可以继续复用。

### 4.2 需要新增的抽象

AIPerf **已经支持** chat 场景里的 multi-turn conversation benchmark，包括：

- `multi_turn` 数据集
- `--conversation-turn-mean` / `--session-turns-mean`
- `--conversation-turn-delay-mean` / `--session-turn-delay-mean`
- `sticky-user-sessions`

所以这里不是“补一个多轮对话能力”这么简单，而是要把现有多轮对话能力进一步抽象成适合 `stream-serve` 的会话模型：

- `request` 之外增加更明确的 `session` 生命周期
- 支持 `session start / session step / session end`
- 支持同一 session 内多轮控制消息
- 支持流式响应里的分帧事件
- 支持把一个 session 的生命周期完整打包成一条 trace

这样它才能覆盖 `stream-serve`，而不只是覆盖 chat multi-turn。

### 4.3 需要新增的 transport adapter

现在 AIPerf 主要是 HTTP 请求模型。后续要对齐 `stream-serve`，建议加 adapter 层：

- HTTP polling adapter
- WebSocket adapter
- SSE adapter
- WebRTC adapter

每种 adapter 只负责 transport，不负责 benchmark 逻辑。

### 4.4 需要新增的 workload model

除了现在这种 batch video workload，还要再加：

- 长会话交互 workload
- 多轮控制 workload
- 连续帧输出 workload
- 动态 prompt / control message workload
- session 断开与恢复 workload

如果只保留单次请求模型，就测不到 world model 的真实交互成本。

### 4.5 需要新增的指标

批处理视频继续看：

- request latency
- throughput
- success rate
- peak memory
- tail latency

流式世界模型还要看：

- first frame latency
- steady-state FPS
- frame jitter
- control-to-frame latency
- session duration stability
- reconnect latency

### 4.6 推荐的改造顺序

建议按这个顺序做：

1. 保持现有 batch video benchmark 不动
2. 把 `session` 抽象加到 AIPerf
3. 加 WebSocket / SSE adapter
4. 再补 WebRTC adapter
5. 最后接 `stream-serve` workload 和 frame-level 指标

这样可以先不破坏现有 `telefuser serve` 压测，同时逐步覆盖实时世界模型场景。

## 5. 参考链接

- [SGLang-Omni](https://sgl-project.github.io/sglang-omni/)
- [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [TeleFuser 480P 服务示例](../../examples/wan_video/wan21_14b_image_to_video_480p_service.py)
- [Diffusers baseline 服务](../../benchmarks/baseline/diffusers_wan_i2v/service.py)
