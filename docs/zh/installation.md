# 安装

本页介绍 TeleFuser 基础包。模型权重、LiveKit Server 和可选的 `tf-kernel` 分发包需要单独安装。

## 环境要求

| 组件 | 要求 |
| --- | --- |
| 操作系统 | 推荐 Linux |
| Python | 3.10 至 3.13 |
| PyTorch | 2.6 或更高版本 |
| CUDA Toolkit | 当前 CUDA 开发路径要求 12.8 或更高版本 |
| GPU | 取决于所选模型，以对应 Cookbook 为准 |

具体示例可能要求更严格的软件版本或 GPU 架构。特别是本地构建的 `tf-kernel` 产物与其记录的 PyTorch、
CUDA、ABI 和 GPU 架构族绑定。

## 安装 TeleFuser

=== "已发布包"

    ```bash
    python -m pip install --upgrade pip
    python -m pip install telefuser
    ```

=== "仓库源码"

    ```bash
    git clone https://github.com/Tele-AI/TeleFuser.git
    cd TeleFuser
    python -m pip install -e .
    ```

=== "开发环境"

    ```bash
    git clone https://github.com/Tele-AI/TeleFuser.git
    cd TeleFuser
    python -m pip install -e ".[dev]"
    pre-commit install
    ```

可选依赖组包括 `ui`、`distributed`、`docs` 和 `dev`。只安装当前工作流需要的依赖，例如使用 Ray
分布式执行时运行 `python -m pip install -e ".[distributed]"`。

## 验证安装

```bash
python -c "import torch, telefuser; print(torch.__version__); print(torch.cuda.is_available())"
telefuser --help
```

模型执行要求 `torch.cuda.is_available()` 输出 `True`。否则应先检查 PyTorch CUDA 构建和 NVIDIA 驱动，
再排查 TeleFuser。

## 模型权重

TeleFuser 不随软件包分发模型权重。[支持的模型](supported_models.md)页面会链接到各模型的 Cookbook，
其中记录经过验证的 Hugging Face、ModelScope 地址、目录布局和附加权重。只有明确支持 Hub 加载的示例
才能直接使用 Hugging Face 模型 ID。

## 可选组件

- [TF-Kernel](tf_kernel.md)介绍独立的 Makefile 构建流程和兼容性检查。
- [注意力机制](attention.md)列出可选注意力后端及硬件要求。
- [流式服务](stream_server.md)介绍需要单独运行的 LiveKit 服务。

下一步进入[LingBot-World v2 WebRTC 核心体验](streaming_quickstart.md)，或使用更轻量的
[基础推理快速上手](quickstart.md)验证单卡流水线和批量 API。
