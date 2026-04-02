# 测试指南

TeleFuser 使用 pytest 进行单元/集成测试，并提供批量回归测试框架用于示例 pipeline 测试。

## 单元测试与集成测试

### 测试结构

```
tests/
├── conftest.py              # 共享 fixtures 和 pytest 配置
├── unit/                    # 按模块组织的单元测试
│   ├── core/               # 核心模块测试
│   ├── distributed/        # 分布式通信测试
│   ├── feature_cache/      # 特征缓存测试
│   ├── kernel/             # Triton 内核测试
│   ├── models/             # 模型架构测试
│   ├── ops/                # 自定义算子测试
│   ├── schedulers/         # 扩散调度器测试
│   ├── service/            # API 服务测试
│   └── utils/              # 工具函数测试
└── integration/             # 集成测试
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行指定测试文件
pytest tests/unit/core/test_config.py

# 详细输出
pytest tests/ -v

# 运行匹配模式的测试
pytest tests/ -k "attention"

# 并行运行测试（需要 pytest-xdist）
pytest tests/ -n auto
```

### 测试标记

TeleFuser 为硬件相关测试定义了自定义标记：

| 标记 | 描述 | 使用场景 |
|------|------|----------|
| `@pytest.mark.gpu` | 需要 GPU | 无 CUDA 时跳过 |
| `@pytest.mark.multi_gpu` | 需要多 GPU | GPU 数量 < 2 时跳过 |
| `@pytest.mark.slow` | 长时间运行测试 | 使用 `-m "not slow"` 跳过 |
| `@pytest.mark.distributed` | 需要分布式环境 | 需要特殊环境配置 |

```python
import pytest

@pytest.mark.gpu
def test_attention_forward():
    """需要 GPU 的测试"""
    ...

@pytest.mark.multi_gpu
def test_parallel_inference():
    """需要多 GPU 的测试"""
    ...
```

### 常用 Fixtures

定义在 `tests/conftest.py` 中：

#### 硬件检测

```python
def test_with_device(device):
    """使用适当的设备（CUDA 或 CPU）"""
    tensor = torch.randn(1, 3, 512, 512, device=device)

def test_gpu_count(gpu_count):
    """检查可用 GPU 数量"""
    assert gpu_count >= 0
```

#### 样本数据

```python
def test_image_processing(sample_image_pil, sample_image_tensor):
    """使用样本图像 fixtures"""
    # sample_image_pil: 512x512 RGB PIL 图像
    # sample_image_tensor: (1, 3, 512, 512) 张量
```

#### CUDA 清理

```python
def test_memory_intensive(clear_cuda_cache):
    """测试后清理 CUDA 缓存"""
    # 测试代码...
    # 测试结束后自动清理 CUDA 缓存
```

#### 随机种子

```python
def test_reproducible(set_seed):
    """设置固定随机种子以确保可重复性"""
    # 已应用 torch.manual_seed(42) 和 np.random.seed(42)
    # 测试结束后重置为随机状态
```

### 编写测试

#### GPU 相关测试

对于需要 GPU 的测试，在模块级别检查可用性：

```python
import pytest
import torch

# 如果 CUDA 不可用，跳过整个模块
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    pytest.skip("Triton 不可用", allow_module_level=True)

@pytest.mark.gpu
def test_triton_kernel():
    """测试 Triton 内核"""
    ...
```

#### Mock Fixtures

使用提供的 mock fixtures 进行隔离测试：

```python
def test_pipeline(mock_model_manager, mock_pipeline_config):
    """使用模拟依赖测试 pipeline"""
    pipeline = MyPipeline(config=mock_pipeline_config)
    pipeline.model_manager = mock_model_manager
```

### CI 集成

CI 中运行不同配置的测试：

```bash
# 仅 CPU 测试（默认）
pytest tests/ -m "not gpu and not multi_gpu"

# GPU 测试（需要 GPU runner）
pytest tests/ -m "gpu"

# 完整测试套件
pytest tests/
```

#### CI 测试脚本

位于 `scripts/run_ci_tests.sh`：

```bash
#!/bin/bash
# 运行完整 CI 测试套件
bash scripts/run_ci_tests.sh
```

### 最佳实践

1. **合理使用标记** - 标记 GPU 相关测试，在 CPU 环境中跳过
2. **清理资源** - GPU 测试使用 `clear_cuda_cache` fixture
3. **设置种子确保可重复** - 涉及随机性时使用 `set_seed` fixture
4. **模拟外部依赖** - 模型加载、API 调用使用 mock fixtures
5. **保持测试隔离** - 每个测试应独立于其他测试
6. **命名清晰** - 使用 `test_<功能>_<场景>_<预期>` 模式

### 示例测试

```python
import pytest
import torch

from telefuser.ops.normalization import RMSNorm


class TestRMSNorm:
    """测试 RMSNorm 算子"""

    @pytest.mark.gpu
    def test_forward_cuda(self, device):
        """测试 GPU 前向传播"""
        norm = RMSNorm(hidden_size=64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_forward_cpu(self):
        """测试 CPU 前向传播"""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_reproducibility(self, set_seed):
        """测试输出确定性"""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out1 = norm(x.clone())
        out2 = norm(x.clone())
        assert torch.allclose(out1, out2)
```

---

## 回归测试

TeleFuser 提供批量回归测试框架，用于运行示例 pipeline、对比 baseline 输出、生成测试报告。

### 快速开始

```bash
# 列出所有配置的 pipeline
python examples/run_examples.py --list

# 运行指定 pipeline
python examples/run_examples.py --pipeline wan21_1_3b_t2v

# 运行所有启用的 pipeline
python examples/run_examples.py --all

# 实时显示日志输出
python examples/run_examples.py --all --verbose

# 更新 baseline
python examples/run_examples.py --all --update-baseline
```

### CLI 参考

```
python examples/run_examples.py [选项]

选项:
  --list                 列出配置的 pipeline 并退出
  --pipeline NAME        运行指定的 pipeline
  --all                  运行所有启用的 pipeline
  --update-baseline      成功运行后更新 baseline
  --config PATH          配置文件路径（默认: example_config.yaml）
  -v, --verbose          实时显示每个 pipeline 的日志输出
```

### 配置说明

通过 `examples/example_config.yaml` 配置：

```yaml
defaults:
  seed: 42
  timeout_seconds: 1800
  psnr_min: 25.0          # 视频回归测试最低 PSNR
  ssim_min: 0.85          # 视频回归测试最低 SSIM
  pixel_diff_max: 0.02    # 图像回归测试最大像素差

output_root: work_dirs/example_outputs

pipelines:
  wan21_1_3b_t2v:
    script: wan_video/wan21_1_3b_text_to_video_h100.py
    gpu_count: 1
    output_type: video
    model_root: /path/to/model
    ppl_config_overrides:
      attn_impl: FLASH_ATTN_2
```

### Pipeline 配置字段

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| script | str | 必填 | 示例脚本路径（相对于 `examples/`） |
| enabled | bool | true | 为 false 时跳过 |
| gpu_count | int | 1 | 分配 GPU 数量 |
| output_type | str | video | `video` 或 `image` |
| timeout_seconds | int | 1800 | 最大执行时间（秒） |
| seed | int | 42 | 随机种子 |
| model_root | str\|null | null | 覆盖模型目录 |
| prompt | str\|null | null | 覆盖生成提示词 |
| input_image_path | str\|null | null | I2V/编辑 pipeline 的输入图像 |
| input_video_path | str\|null | null | VSR/续写 pipeline 的输入视频 |
| ppl_config_overrides | dict | {} | 覆盖 PPL_CONFIG 配置 |
| psnr_min | float | 25.0 | 视频：最低 PSNR 阈值 |
| ssim_min | float | 0.85 | 视频：最低 SSIM 阈值 |
| pixel_diff_max | float | 0.02 | 图像：最大像素差异 |
| max_elapsed_seconds | float\|null | null | 性能阈值（秒） |
| max_gpu_memory_mb | float\|null | null | GPU 内存阈值（MB） |

### 输出结构

```
work_dirs/example_outputs/
├── 2026-04-02/                                    # 按日期组织
│   ├── wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
│   └── qwen_image__qwen_t2i_1gpu_1024x1024.png
├── baseline/                                      # baseline 输出
│   └── wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
├── logs/                                          # 日志文件
│   ├── 20260402_120000_wan_video__wan21_1_3b_t2v_1gpu.log
│   └── 20260402_130000_qwen_image__qwen_t2i_1gpu.log
└── example_report.json                            # 总结报告
```

#### 输出命名规范

**输出文件：**
```
{示例目录}__{示例名称}_{GPU数量}gpu_{分辨率}.{扩展名}
```

示例：`wan_video__wan21_1_3b_text_to_video_h100_1gpu_480x832.mp4`

**日志文件：**
```
{时间戳}_{示例目录}__{示例名称}_{GPU数量}gpu.log
```

示例：`20260402_120000_wan_video__wan21_1_3b_text_to_video_h100_1gpu.log`

### 回归指标

Runner 使用以下指标对比 baseline：

- **视频**：PSNR（峰值信噪比）和 SSIM（结构相似度）
- **图像**：平均像素差异

#### 指标阈值

在 YAML 中配置或每个 pipeline 单独设置：

```yaml
psnr_min: 25.0      # 越高越严格
ssim_min: 0.85      # 范围 [0, 1]，越高越严格
pixel_diff_max: 0.02 # 范围 [0, 1]，越低越严格
```

#### Baseline 管理

- 首次运行：输出自动保存为 baseline
- 后续运行：与 baseline 对比
- 更新 baseline：使用 `--update-baseline` 参数

### 错误分类

| 类别 | 描述 | 分析提示 |
|------|------|----------|
| MODEL_LOAD_ERROR | 模型加载失败 | 检查 model_root 路径和模型文件完整性 |
| INFERENCE_ERROR | 推理过程出错 | 查看 log_path 中的 traceback 定位具体模块 |
| OUTPUT_ERROR | 输出保存失败 | 检查输出目录权限和磁盘空间 |
| OOM_ERROR | GPU 内存不足 | 考虑减少 batch_size 或使用更低分辨率 |
| TIMEOUT | 执行超时 | 考虑增加 timeout_seconds 或检查是否有死循环 |

### 报告结构

`example_report.json` 包含：

```json
{
  "generated_at": "2026-04-02T12:00:00",
  "environment": {
    "pytorch_version": "2.6.0",
    "cuda_version": "12.8",
    "gpu_count": 8
  },
  "summary": {
    "total": 20,
    "pass": 18,
    "fail": 1,
    "error": 1,
    "timeout": 0
  },
  "results": { ... },
  "failed_details": [
    {
      "name": "wan21_1_3b_t2v",
      "status": "ERROR",
      "error_category": "INFERENCE_ERROR",
      "error_message": "...",
      "reproduce_command": "python examples/run_examples.py --pipeline wan21_1_3b_t2v",
      "log_path": "work_dirs/example_outputs/logs/20260402_120000_wan_video__wan21_1_3b_t2v_1gpu.log",
      "last_50_lines_log": "...",
      "analysis_hint": "推理过程出错，查看 log_path 中的 traceback 定位具体模块"
    }
  ],
  "reproduce_all_failed": "python examples/run_examples.py --pipeline wan21_1_3b_t2v && ..."
}
```

### 功能特性

- **进程隔离**：每个 pipeline 在独立进程中运行，GPU 绑定
- **Baseline 管理**：首次运行自动保存，支持更新
- **回归指标**：视频使用 PSNR/SSIM，图像使用像素差异
- **GPU 内存追踪**：记录每个 pipeline 的峰值显存
- **输出验证**：NaN/Inf 检测
- **增强报告**：失败案例包含复现命令和分析提示

### 添加新 Pipeline

1. 在 `examples/` 适当目录创建示例脚本
2. 在 `example_config.yaml` 添加配置：

```yaml
pipelines:
  my_new_pipeline:
    script: my_category/my_script.py
    gpu_count: 1
    output_type: video
    model_root: /path/to/model
```

3. 运行生成 baseline：
```bash
python examples/run_examples.py --pipeline my_new_pipeline
```