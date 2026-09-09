# 支持的模型

本目录列出 TeleFuser 持续维护并发布的模型系列指南。各模型的 Cookbook 页面是权重 ID、Hugging Face 和
ModelScope 地址、所需文件、支持任务、硬件假设及可运行命令的事实来源。

“支持”表示 TeleFuser 提供维护中的流水线或示例契约，并不表示所有权重、精度、注意力后端和 GPU 拓扑
可以任意组合。修改优化配置前，应先使用对应指南中经过验证的配置。

## 世界模型与实时推理

| 模型 | 任务 | 执行方式 |
| --- | --- | --- |
| [LingBot-World v2](/TeleFuser/zh/cookbook/lingbot-world/) | 双向世界模型流式推理 | 单机和 LiveKit |
| [LingBot-World-Fast](/TeleFuser/zh/cookbook/lingbot-world/) | 因果交互式流推理 | 单机和 LiveKit |
| [ABot-World-0-5B-LF](/TeleFuser/zh/cookbook/abot-world/) | 交互式生成 | HTTP 和 LiveKit |

## 视频、音视频与修复

| 模型 | 任务 | 执行方式 |
| --- | --- | --- |
| [WanVideo](/TeleFuser/zh/cookbook/wan-video/) | T2V、I2V、FL2V | 单机和批量服务 |
| [LTX-2.3](/TeleFuser/zh/cookbook/ltx23/) | 带音频的 I2V | 单机和批量服务 |
| [LTX-2.5 Distilled](/TeleFuser/zh/cookbook/ltx25-distilled/) | 带音频的 T2V 和 I2V | 单机和多 GPU |
| [MiniMax H3](/TeleFuser/zh/cookbook/minimax-h3/) | T2VA、FL2VA、Ref2VA | 单机和批量服务 |
| [LingBot-Video](/TeleFuser/zh/cookbook/lingbot-video/) | T2I、T2V、TI2V、精修 | 单机和多 GPU |
| [LongCat-Video](/TeleFuser/zh/cookbook/longcat-video/) | T2V、I2V、续写 | 单机和批量服务 |
| [LiveAct](/TeleFuser/zh/cookbook/liveact/) | 语音转视频 | 单机和流式服务 |
| [FlashVSR](/TeleFuser/zh/cookbook/flashvsr/) | 视频超分辨率 | 单机和流式服务 |
| [SwiftVR](/TeleFuser/zh/cookbook/swiftvr/) | 因果视频修复 | 单机和多 GPU |

## 图像生成

| 模型 | 任务 | 执行方式 |
| --- | --- | --- |
| [Qwen-Image](/TeleFuser/zh/cookbook/qwen-image/) | T2I 和编辑 | 单机和批量服务 |
| [Z-Image](/TeleFuser/zh/cookbook/z-image/) | T2I | 单机和批量服务 |
| [Flux2 Klein](/TeleFuser/zh/cookbook/flux2-klein/) | T2I | 单机和批量服务 |

## 视觉语言动作

| 模型 | 任务 | 执行方式 |
| --- | --- | --- |
| [LingBot-VLA v2](/TeleFuser/zh/cookbook/lingbot-vla-v2/) | 机器人动作预测 | 单机推理 |

## 选择运行配置

1. 根据任务选择模型系列。
2. 打开对应 Cookbook，选择文档列出的模型来源。
3. 从经过验证的硬件和精度配置开始。
4. 启用服务或分布式执行前，先运行文档中的单机命令。
5. 每次只修改一个优化维度，并保留指南要求的输出验证。

资源要求最低的入门路径请使用[基础推理](quickstart.md)中的 Wan2.1 1.3B 配置。
