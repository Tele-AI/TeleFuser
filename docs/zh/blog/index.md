---
title: TeleFuser 技术博客
description: 记录 TeleFuser 性能与运行时优化的分析、实现和验证过程。
---

# TeleFuser 技术博客

技术博客用于记录 TeleFuser 优化如何被发现、实现、测量和约束。稳定的用户指南描述受支持的行为和配置；
技术文章进一步解释这些行为背后的 profiling 证据、备选方案、实现取舍和特定硬件上的测量结果。

## 文章

| 日期 | 文章 | 状态 | 验证平台 |
|---|---|---|---|
| 2026-08-19 | [FP8 Sol-Attn：H100 视频 DiT 的量化稀疏注意力](fp8_sol_attention.md) | 已验证 | 1 x H100 80 GB |
| 2026-08-06 | [CUDA IPC Ulysses：在 H100 上重叠 Attention 通信](cuda_ipc_ulysses.md) | 已验证 | 4 x H100 80 GB |

## 发布约定

每篇文章应包含：

1. 基线实现和实际测得的瓶颈。
2. 设计目标、非目标和调研过的备选方案。
3. 实现方式和资源所有权边界。
4. 与改动风险相匹配的正确性、parity、压力和生命周期验证。
5. 可复现的 benchmark 环境、命令、workload 和指标定义。
6. 分开报告 microbenchmark、目标端计算和端到端交付结果。
7. 限制、fallback 行为以及实际完成验证的硬件。
8. Related Work，包括最接近的已有工作，并明确说明哪些内容不主张首创。

文章使用以下状态：

| 状态 | 含义 |
|---|---|
| `experimental` | 实现或证据尚不完整。 |
| `validated` | 正确性和文中 benchmark 已在指定 revision 与平台上复现。 |
| `superseded` | 已有新的实现或文章取代当前结果。 |
| `archived` | 当前源码已不再使用该实现。 |

性能数据是指定环境下的点测结果，不是可移植的性能承诺。每篇文章都记录验证 revision 和环境，避免后续代码
变化后仍将历史数据当作当前行为。

