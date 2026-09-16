# 故障排查

从最小故障层开始：安装、模型加载、单机推理、批量服务、分布式执行或 LiveKit 传输。隔离故障时不要继续
启用其他优化。

## 记录环境

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
telefuser --help
```

提交问题时应包含示例路径、完整命令、权重 ID、GPU 拓扑和完整 traceback。

## 导入失败或 CUDA 不可用

如果 `import telefuser` 失败，确认当前解释器，并在同一环境中重新安装仓库：

```bash
python -m pip show telefuser
python -m pip install -e .
```

如果 PyTorch 无法识别 CUDA，应先解决驱动和 CUDA 版 PyTorch。原生回退路径不要求安装可选的
`tf-kernel`。参见[安装](installation.md)和 [TF-Kernel](tf_kernel.md)。

## 无法加载模型权重

- 直接传入 Hub 模型 ID 前，确认示例明确支持这种加载方式。
- 使用本地权重时，保留上游仓库结构和必要的附加文件。
- 对照模型的 [Cookbook 指南](supported_models.md)检查路径。
- 对受限或远程仓库检查鉴权和可用磁盘空间。

使用[模型加载](model_loading.md)区分 Diffusers 格式与 ModuleManager 文件列表加载方式。

## CUDA 显存不足

先使用指南中经过验证的分辨率、帧数、精度和 GPU 数量复现。然后每次只减少一个显存维度：分辨率、帧数、
并发副本或常驻流式会话。CPU offload 和量化会改变性能或数值行为，应在基线成功后启用。

参见 [CPU 卸载](offload.md)、[量化](quantization.md)和[并行推理](parallel.md)。

## 注意力或内核错误

先恢复为流水线文档中的 dense 或原生注意力后端。如果基线正常，再根据[注意力机制](attention.md)检查优化
后端要求的 GPU 架构、PyTorch、CUDA 和依赖版本。对于 `tf-kernel`，应读取记录的构建信息，不要绕过
兼容性检查。

## 分布式启动卡住

检查所有 rank 是否看到预期 GPU，以及旧 worker 是否仍占用端口或设备。记录 `CUDA_VISIBLE_DEVICES`、
world size、各并行度和 NCCL 输出。先在单卡复现，再恢复到经过验证的拓扑。参见[并行推理](parallel.md)
和[通信架构](communication.md)。

## 批量服务未就绪

先直接运行流水线，再检查 CLI 和服务接口：

```bash
telefuser validate /path/to/pipeline.py
telefuser serve /path/to/pipeline.py --task t2v --port 8000
curl --fail http://127.0.0.1:8000/v1/service/health
curl --fail http://127.0.0.1:8000/v1/service/status
```

验证错误、端口冲突、任务失败和错误响应参见[批量服务与 API 参考](service.md)。

## LiveKit 会话已连接但没有媒体

分别确认 TeleFuser health 接口、LiveKit URL 和凭据、room 参与者、worker 准入及 TURN 可达性。增加代理或
远程网络前先在 loopback 环境验证。拓扑和诊断顺序参见[流式服务](stream_server.md)，worker 与会话所有权
参见[流式调度器](stream_scheduler.md)。

## 文档构建失败

```bash
python -m unittest discover -s tests/docs -v
python scripts/docs/prepare_cookbook.py build
```

不要编辑 `.build/` 或 `site/`。应修改源文档、`docs/cookbook.yml` 或已注册的示例 README。
