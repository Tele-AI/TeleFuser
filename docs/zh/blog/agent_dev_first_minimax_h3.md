---
title: "从 MiniMax H3 适配看 Agent Dev First"
description: 通过两次独立 Codex Goal 完成联合音视频模型集成，讨论架构、契约、验证与交接物如何为 Agent 提供方向。
date: 2026-08-20
status: validated
validated_revision: 819c238
hardware: 1, 2, and 4 x NVIDIA H100 80 GB HBM3
tags:
  - agent-development
  - architecture
  - model-integration
  - validation
  - minimax-h3
---

# 从 MiniMax H3 适配看 Agent Dev First

TeleFuser 最近完成了对 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的适配。当前实现覆盖
T2VA、FL2VA 和 Ref2VA 三类联合音视频生成任务，支持完整音视频输出、单卡 stage-level CPU offload、
两卡和四卡模型常驻、FSDP、Ulysses Sequence Parallel、Tensor Parallel，以及统一的 example 和服务入口。

这次适配主要通过两次相互独立的 Codex Goal 会话完成。按照项目过程记录，期间没有人工直接修改代码；人工参与
集中在目标设定、环境准备和最终音视频质量验收。本文不把这一过程解释为“Agent 自动完成所有软件工程”的普遍
证明，而是讨论一个更具体的问题：

> 当 Agent 已经能够自主搜索代码、比较相邻实现并根据测试反馈调整方案时，软件系统应该怎样为 Agent 提供方向？

本文将这种面向 Agent 的工程设计称为 **Agent Dev First（ADF）**。它关注的不是为一次任务写出更长的操作手册，
而是让代码库通过知识结构、模块边界、验证路径和可持续交接物，回答“在哪里做”和“怎样确认做对了”。

## 验证快照

| 字段 | 值 |
|---|---|
| 状态 | `validated` |
| 验证 revision | [`819c238`](https://github.com/Tele-AI/TeleFuser/commit/819c2388d7bbdc259821be9c6180879643a0c347) |
| 集成与验收日期 | 2026-08-04 至 2026-08-06 |
| GPU | 1、2、4 x NVIDIA H100 80 GB HBM3；最终并行验收使用 4 x H100 |
| 软件 | Python 3.11.13、PyTorch 2.11.0+cu128、CUDA 12.8、NCCL 2.28.9 |
| Reference | 固定 revision 的本地 SGLang MiniMax H3 实现 |
| Faithful-path workload | 768p，T2VA 10 秒、FL2VA 8 秒、Ref2VA 5 秒，均为 50-step |
| 输出契约 | 24 FPS 视频、32 kHz 双声道音频、同步 MP4 |
| 四卡回归 | 768p、5 秒、50-step T2VA，Ulysses2 x TP2 常驻配置 |

验证结论只覆盖上述 checkpoint、请求、软件和硬件环境。它不构成其他 checkpoint、硬件拓扑或生成内容上的质量和
性能保证。Codex Goal 的会话边界与“没有人工直接修改代码”来自项目过程记录；代码提交证明的是实现内容、验证
结果和交接物，不单独证明编辑者身份。

## 为什么 MiniMax H3 是一个合适的案例

MiniMax H3 不是只加载一个 DiT checkpoint 的普通视频模型。一次完整推理至少经过：

```text
Prompt 与参考素材
  -> Text Encoder
  -> Video / Audio Condition VAE
  -> 文本、视频和音频 token 联合打包
  -> DiT 联合去噪
  -> 视频和音频 scheduler 分别更新
  -> Video VAE 与 Audio VAE 解码
  -> 音视频同步与 MP4 封装
```

T2VA、FL2VA 和 Ref2VA 还有不同的条件路径、素材规则、时间语义和几何来源。适配同时涉及官方 checkpoint 转换、
FP32/BF16 精度边界、packed sequence、多卡并行、模型常驻、跨 stage 通信和服务请求。

因此，Agent 面对的不是“增加一个模型类”，而是一组相互关联、又必须能够独立验证的工程问题。这使 MiniMax H3
成为观察代码库能否为 Agent 提供方向的有效案例。

## 从操作手册到工程结构

为了让 Agent 稳定完成任务，常见方法是编写详细的 Skill：先读哪些文件、按什么顺序修改、遇到错误怎样处理、
最后运行哪些检查。这类操作手册仍然适合低频、专业和跨工具流程。

随着 Agent 的代码搜索、相邻实现理解和测试反馈能力增强，持续注入逐步骤指令的重要性可能下降，但工程设计的
重要性不会下降。当过程指导减少后，Agent 会更依赖系统本身回答以下问题：

- 功能属于哪个模块？
- 应该通过什么接口实现？
- 哪些依赖和修改方向是允许的？
- 怎样完成快速局部验证？
- 哪些状态才算真正完成？
- 失败后怎样定位、交接和继续？

Skill 主要回答“怎样做”；ADF 更关注“在哪里做”和“怎样确认做对了”。ADF 不要求把所有规则都变成类型检查或
CI，也不替代 Skill。它强调将长期稳定的知识放进代码结构、接口、测试和工具，而不是让每个 Agent 在每次会话中
重新记忆。

## 两次 Goal：先建立正确性，再优化执行

两次会话没有连续的对话记忆。它们通过仓库中的代码、测试、固定输入、manifest、报告生成工具和提交记录完成
交接。

| Goal | 允许改变 | 必须保持 | 关键证据 |
|---|---|---|---|
| 1. 建立 faithful path | 模型与 pipeline 的新增实现、checkpoint 转换、任务规划、验证工具 | 官方任务语义、scheduler、精度边界和输出语义 | [`82de4e9`](https://github.com/Tele-AI/TeleFuser/commit/82de4e9bb128d170c0dd8e0b769a376c06957d3b)、[`0db8807`](https://github.com/Tele-AI/TeleFuser/commit/0db8807e36d5cf278d36649d9c260617e05e230b)、[`9b4dfb7`](https://github.com/Tele-AI/TeleFuser/commit/9b4dfb72f3ae5c78e3c7b3b0c109b647541a9377) |
| 2. 优化执行 | 模型常驻、SP/TP 组合、stage 通信、静态布局、buffer 和公共 ops/kernel | Goal 1 冻结的条件布局、scheduler 语义、精度边界和输出格式 | [`b3e1672`](https://github.com/Tele-AI/TeleFuser/commit/b3e1672cd0e3e0f14402a63548b6487858e08783)、[`ecb261f`](https://github.com/Tele-AI/TeleFuser/commit/ecb261f80dde7a8062f9ee891c0b744798f6484f)、[`b629b9e`](https://github.com/Tele-AI/TeleFuser/commit/b629b9e2343decfad2c6102124a5c9d651536445) |

第一次 Goal 建立了模型组件、三类请求规划、packed joint denoising、双模态 scheduler、完整音视频输出，以及
SGLang/TeleFuser trajectory 和 artifact 比较工具。官方 T2VA、FL2VA、Ref2VA 50-step 报告通过锁定门限后，
正确性基线才成立。

第二次 Goal 在这个基线之上加入常驻多卡、Ulysses 与 TP 组合、跨 stage CUDA tensor 直传、request-static
layout、denoising buffer 复用，以及公共 ops 和 kernel 优化。随后建立的
[`a05d3ee`](https://github.com/Tele-AI/TeleFuser/commit/a05d3eec948930535a27c9a4a82225af36842d8d)
四卡回归和
[`a680135`](https://github.com/Tele-AI/TeleFuser/commit/a68013557cc37c7bc5b566e0f05a01aab7f4a6c6)
音频/service 验收，把“优化没有改变输出契约”变成了可重复检查的事实。

提交时间线是工程证据链，不是 Goal 会话的逐条转录。重要的是，第二次会话不需要读取第一次对话，也能从系统中
恢复任务语义和完成定义。

## TeleFuser 如何告诉 Agent“在哪里做”

这次适配没有使用 MiniMax H3 专用的逐步骤 Skill。Agent 主要根据目录结构、接口契约、相邻实现和测试自主推导
实现路径。

| 关注点 | 稳定落点 | 主要边界 |
|---|---|---|
| 模型数学与 checkpoint 转换 | `telefuser/models/` | 模型通过 `telefuser.ops` 使用公共算子，不直接依赖内部 kernel |
| 任务语义与素材规划 | `telefuser/pipelines/minimax_h3/` | T2VA、FL2VA、Ref2VA 差异集中在 task profile 和 request plan |
| Text Encoder、Denoising、Video/Audio VAE | 独立 stage | 每个 stage 独立拥有精度、offload、设备和生命周期配置 |
| 并行执行 | 统一 parallel config 与 distributed runtime | SP、TP、FSDP 复用现有配置，不增加模型私有框架接口 |
| 本地运行与服务 | `examples/minimax_h3/` 的标准入口和 manifest | `PPL_CONFIG`、`get_pipeline`、`run`、`run_with_file` 与服务契约共享 |
| 验证 | `tests/`、`tools/validation/`、冻结素材 | 快速局部测试与昂贵的真实生成分层执行 |

这套结构没有告诉 Agent “先改哪个文件的第几行”，但它回答了更关键的问题：模型数学、任务语义、运行时并行、
服务契约和输出呈现分别属于哪里。

稳定边界也限制了改动范围。MiniMax H3 没有引入新的框架级配置字段、环境变量、CLI 体系或服务 schema；需要的
扩展位于模型、pipeline、example 和已有公共 ops 的所有权范围内。

## ADF 还必须提供完成定义

能找到文件并不等于能判断结果正确。MiniMax H3 的验证被拆成多个层次：

| 验证对象 | 主要证明内容 | 代表性位置 |
|---|---|---|
| checkpoint 与数值测试 | 模型结构、转换和 FP32/BF16 边界正确 | `tests/unit/models/test_minimax_h3_*.py` |
| packed sequence、scheduler、stage 测试 | 局部计算和编排语义正确 | `tests/unit/pipelines/minimax_h3/` |
| trajectory parity | 固定输入下的 token 布局、初始噪声和 scheduler 边界满足基线 | `compare_minimax_h3_trajectories.py` |
| artifact parity | 视频帧、音频波形、频谱、包络、时延和容器满足门限 | `compare_minimax_h3_artifacts.py` |
| example 与 service parity | 服务封装和直接入口使用相同的参数契约 | `test_examples.py`、`test_example_service_parity.py` |
| 四卡 50-step 回归 | Ulysses2 x TP2 常驻配置能够完成端到端音视频生成 | `minimax_h3_t2va_4gpu` registry |

快速测试提供局部反馈，真实生成负责最终验收。失败因此可以定位到模型转换、pipeline 语义、并行运行时、服务入口
或最终产物，而不是全部压缩成一个服务失败。

冻结输入同样属于系统接口。[`provenance.json`](https://github.com/Tele-AI/TeleFuser/blob/819c2388d7bbdc259821be9c6180879643a0c347/examples/data/minimax-h3/provenance.json)
记录官方素材的来源、大小和 SHA-256；reference 与 candidate runner 生成包含请求、素材、checkpoint 和 artifact
hash 的 manifest。大型 tensor 和 MP4 可以保留为本地验收物，而源代码中的生成器、hash 规则和比较器使它们
能够重建。

## 复现入口

CPU 侧的模型、pipeline、契约和比较器测试可以独立运行：

```bash
pytest -q tests/unit/models/test_minimax_h3_*.py
pytest -q tests/unit/pipelines/minimax_h3
pytest -q tests/unit/service/test_example_service_parity.py
```

准备本地 checkpoint 和四张 H100 后，可建立或检查标准四卡回归：

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3 \
  --update-baseline

TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3
```

官方路径比较由以下工具组成：

```text
freeze_minimax_h3_reference.py
  -> run_minimax_h3_sglang_reference.py
  -> run_minimax_h3_telefuser_reference.py
  -> compare_minimax_h3_trajectories.py
  -> compare_minimax_h3_artifacts.py
```

这些工具要求本地 MiniMax H3 checkpoint、固定的 SGLang reference checkout 和足够的 H100 资源。完整参数可通过
各脚本的 `--help` 查看；验收门限由比较器显式写入 JSON report，而不是隐藏在人工观察中。最终音视频质量仍需要
人工验收，因为数值相似度不能证明条件遵循和感知质量的全部属性。

## 更强的 Agent 会更快暴露架构债务

这次适配也暴露了 TeleFuser 的三个改进方向。

第一，example regression、service parity 和相关测试之间仍有重复登记。虽然 pipeline manifest 已经存在，但还
没有成为 example registry、服务契约和回归矩阵的完整单一来源。

第二，SP 与 TP 的实现已经能够组合，但文档一度保留二者互斥的旧规则。代码正确并不能抵消过期文档带来的搜索和
判断成本。

第三，跨进程 CUDA tensor 的所有权和关闭顺序仍然复杂。后续提交通过独立 refcounter、channel-owned cleanup
和 cooperative shutdown 加固了生命周期，但 stage DAG 还不能完整表达 buffer、producer、consumer 和 teardown
依赖。

这些问题无法通过增加一份更长的 Skill 从根本上解决。更合理的方向是让机器可读的 pipeline manifest 驱动入口
登记和回归矩阵，并让 stage DAG 明确表达资源所有权。更强的 Agent 不会自动消除架构债务，但会更快地在组合实现
和测试反馈中遇到它们。

## Prompt、Skills、Harness 与 ADF

本文使用以下分工理解四者关系；这是从本案例提炼的工程模型，不是对某个 Agent 产品机制的规范定义。

| 层次 | 负责内容 |
|---|---|
| Prompt | 当前任务的目标、约束与验收条件 |
| Skills | 专业、低频或跨工具任务的操作方法 |
| Harness | Agent 调度、工具访问、执行隔离与反馈循环 |
| ADF | 长期稳定的知识结构、模块边界、交接物和验证路径 |
| 类型、结构测试与 CI | 任何参与者都不能破坏的不变量 |

当同一条规则开始在多个 Skill 中重复出现时，值得继续追问：它是否应该永远由 Agent 阅读和记忆，还是应该进入
架构、接口、测试和工具，成为系统本身的一部分？

## Related Work 与主张边界

[SWE-agent](https://arxiv.org/abs/2405.15793) 将 Agent-Computer Interface 作为提升软件工程 Agent 能力的重要
设计对象；[SWE-bench](https://arxiv.org/abs/2310.06770) 使用真实 GitHub issue 和测试衡量代码库级任务结果。
仓库级指令、Skills、可执行规格、architecture fitness function 和 paved path 也都在降低工具或开发者与系统
交互的成本。

ADF 不主张这些思想是新的，也不主张清晰架构只对 Agent 有价值。本文更窄的观点是：当 Agent 能够自主决定局部
实现步骤时，代码库本身需要承担更多导航、约束、验证和跨会话交接责任；MiniMax H3 提供了一个有提交记录和真实
生成验收支撑的案例。

## 限制与下一步

- 这是一个项目、一个模型族和一种 Agent 工作方式的案例，不能代表所有代码库或 Agent 配置。
- “两次 Goal”只证明在本次边界与环境下可以交接，不证明会话数量越少越好。
- 质量验收包含人工判断，尚未形成完全自动化、可公开重放的感知质量评测。
- 关键大型 trajectory 和 MP4 artifact 未进入 Git；仓库保存的是固定输入、hash、生成工具、比较逻辑和验收摘要。
- ADF 的收益尚未与重型 Skill 或其他代码库进行受控对照实验。

后续可以比较入口发现时间、首次有效验证时间、跨模块修改规模、交接成功率和失败恢复时间。TeleFuser 自身更直接
的工程任务，是让 manifest 驱动 example、service 和 regression，并让 stage DAG 表达资源及其生命周期。

## 结论

MiniMax H3 的接入不是因为 Agent 获得了一份更详细的专用操作手册。它能够分阶段完成，首先是因为 TeleFuser
已经为模型、stage、pipeline、并行、服务和验证提供了稳定落点。

过程中最难处理的问题，也出现在验收条件、重复登记和资源所有权尚未被系统清晰表达的地方。这支持了一个有限但
可操作的判断：当系统具备清晰的 stage 边界、冻结验收物和契约测试时，Agent 可以减少对连续对话记忆和逐步骤
手册的依赖，分阶段完成精度对齐与执行优化。

Skills 是给 Agent 的操作手册。Agent Dev First，则是把工程系统设计成 Agent 能够直接工作、局部验证并持续
交接的形态。操作手册会随着模型和工具变化；架构决定此后的每一个 Agent 将以多高的成本理解、修改、验证和继续
演进系统。
