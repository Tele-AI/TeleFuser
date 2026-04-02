# Testing Guide

TeleFuser uses pytest for unit/integration testing and provides a batch regression testing framework for example pipelines.

## Unit & Integration Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests by module
│   ├── core/               # Core module tests
│   ├── distributed/        # Distributed communication tests
│   ├── feature_cache/      # Feature cache tests
│   ├── kernel/             # Triton kernel tests
│   ├── models/             # Model architecture tests
│   ├── ops/                # Custom operations tests
│   ├── schedulers/         # Diffusion scheduler tests
│   ├── service/            # API service tests
│   └── utils/              # Utility function tests
└── integration/             # Integration tests
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/core/test_config.py

# Run with verbose output
pytest tests/ -v

# Run tests matching a pattern
pytest tests/ -k "attention"

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### Test Markers

TeleFuser defines custom markers for hardware-dependent tests:

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.gpu` | Requires GPU | Skipped if CUDA unavailable |
| `@pytest.mark.multi_gpu` | Requires multiple GPUs | Skipped if < 2 GPUs |
| `@pytest.mark.slow` | Long-running tests | Use `-m "not slow"` to skip |
| `@pytest.mark.distributed` | Requires distributed setup | Needs special environment |

```python
import pytest

@pytest.mark.gpu
def test_attention_forward():
    """Test that requires a GPU."""
    ...

@pytest.mark.multi_gpu
def test_parallel_inference():
    """Test that requires multiple GPUs."""
    ...
```

### Common Fixtures

Defined in `tests/conftest.py`:

#### Hardware Detection

```python
def test_with_device(device):
    """Use the appropriate device (CUDA or CPU)."""
    tensor = torch.randn(1, 3, 512, 512, device=device)

def test_gpu_count(gpu_count):
    """Check number of available GPUs."""
    assert gpu_count >= 0
```

#### Sample Data

```python
def test_image_processing(sample_image_pil, sample_image_tensor):
    """Use sample image fixtures."""
    # sample_image_pil: 512x512 RGB PIL Image
    # sample_image_tensor: (1, 3, 512, 512) tensor
```

#### CUDA Cleanup

```python
def test_memory_intensive(clear_cuda_cache):
    """Clear CUDA cache after test."""
    # Test code here...
    # CUDA cache automatically cleared after test
```

#### Random Seed

```python
def test_reproducible(set_seed):
    """Set fixed random seed for reproducibility."""
    # torch.manual_seed(42) and np.random.seed(42) applied
    # Reset to random state after test
```

### Writing Tests

#### GPU-Aware Tests

For tests that require GPU, check availability at module level:

```python
import pytest
import torch

# Skip entire module if CUDA unavailable
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    pytest.skip("Triton not available", allow_module_level=True)

@pytest.mark.gpu
def test_triton_kernel():
    """Test Triton kernel."""
    ...
```

#### Mock Fixtures

Use provided mock fixtures for isolation:

```python
def test_pipeline(mock_model_manager, mock_pipeline_config):
    """Test pipeline with mocked dependencies."""
    pipeline = MyPipeline(config=mock_pipeline_config)
    pipeline.model_manager = mock_model_manager
```

### CI Integration

Tests run in CI with different configurations:

```bash
# CPU-only tests (default)
pytest tests/ -m "not gpu and not multi_gpu"

# GPU tests (requires GPU runner)
pytest tests/ -m "gpu"

# Full test suite
pytest tests/
```

#### CI Test Script

Located at `scripts/run_ci_tests.sh`:

```bash
#!/bin/bash
# Run full CI test suite
bash scripts/run_ci_tests.sh
```

### Best Practices

1. **Use markers appropriately** - Mark GPU-dependent tests to skip in CPU environments
2. **Clean up resources** - Use `clear_cuda_cache` fixture for GPU tests
3. **Set seeds for reproducibility** - Use `set_seed` fixture when randomness is involved
4. **Mock external dependencies** - Use mock fixtures for model loading, API calls
5. **Keep tests isolated** - Each test should be independent of others
6. **Name tests descriptively** - Use `test_<function>_<scenario>_<expected>` pattern

### Example Test

```python
import pytest
import torch

from telefuser.ops.normalization import RMSNorm


class TestRMSNorm:
    """Test RMSNorm operation."""

    @pytest.mark.gpu
    def test_forward_cuda(self, device):
        """Test forward pass on GPU."""
        norm = RMSNorm(hidden_size=64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_forward_cpu(self):
        """Test forward pass on CPU."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_reproducibility(self, set_seed):
        """Test deterministic output."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out1 = norm(x.clone())
        out2 = norm(x.clone())
        assert torch.allclose(out1, out2)
```

---

## Regression Testing

TeleFuser provides a batch regression testing framework for running example pipelines, comparing outputs against baselines, and generating reports.

### Quick Start

```bash
# List all configured pipelines
python examples/run_examples.py --list

# Run a specific pipeline
python examples/run_examples.py --pipeline wan21_1_3b_t2v

# Run all enabled pipelines
python examples/run_examples.py --all

# Run with real-time log output
python examples/run_examples.py --all --verbose

# Update baselines after successful runs
python examples/run_examples.py --all --update-baseline
```

### CLI Reference

```
python examples/run_examples.py [OPTIONS]

Options:
  --list                 List configured pipelines and exit
  --pipeline NAME        Run a specific pipeline by name
  --all                  Run all enabled pipelines
  --update-baseline      Update baseline outputs after successful runs
  --config PATH          Path to config YAML (default: example_config.yaml)
  -v, --verbose          Show real-time log output from each pipeline
```

### Configuration

The runner is configured via `examples/example_config.yaml`:

```yaml
defaults:
  seed: 42
  timeout_seconds: 1800
  psnr_min: 25.0          # Minimum PSNR for video regression
  ssim_min: 0.85          # Minimum SSIM for video regression
  pixel_diff_max: 0.02    # Max pixel diff for image regression

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

### Pipeline Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| script | str | required | Path to example script (relative to `examples/`) |
| enabled | bool | true | Skip if false |
| gpu_count | int | 1 | GPUs to allocate |
| output_type | str | video | `video` or `image` |
| timeout_seconds | int | 1800 | Max execution time |
| seed | int | 42 | Random seed |
| model_root | str\|null | null | Override model directory |
| prompt | str\|null | null | Override generation prompt |
| input_image_path | str\|null | null | Input image for I2V/edit pipelines |
| input_video_path | str\|null | null | Input video for VSR/continue pipelines |
| ppl_config_overrides | dict | {} | Override PPL_CONFIG keys |
| psnr_min | float | 25.0 | Video: minimum PSNR vs baseline |
| ssim_min | float | 0.85 | Video: minimum SSIM vs baseline |
| pixel_diff_max | float | 0.02 | Image: max mean pixel difference |
| max_elapsed_seconds | float\|null | null | Performance threshold |
| max_gpu_memory_mb | float\|null | null | GPU memory threshold |

### Output Structure

```
work_dirs/example_outputs/
├── 2026-04-02/                                    # Date-based output directory
│   ├── wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
│   └── qwen_image__qwen_t2i_1gpu_1024x1024.png
├── baseline/                                      # Baseline outputs
│   └── wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
├── logs/                                          # Log files
│   ├── 20260402_120000_wan_video__wan21_1_3b_t2v_1gpu.log
│   └── 20260402_130000_qwen_image__qwen_t2i_1gpu.log
└── example_report.json                            # Summary report
```

#### Output Naming Convention

**Output files:**
```
{example_dir}__{example_name}_{gpu_count}gpu_{resolution}.{ext}
```

Example: `wan_video__wan21_1_3b_text_to_video_h100_1gpu_480x832.mp4`

**Log files:**
```
{timestamp}_{example_dir}__{example_name}_{gpu_count}gpu.log
```

Example: `20260402_120000_wan_video__wan21_1_3b_text_to_video_h100_1gpu.log`

### Regression Metrics

The runner compares outputs against baselines using:

- **Video**: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity)
- **Image**: Mean pixel difference

#### Metrics Thresholds

Configure in YAML or per-pipeline:

```yaml
psnr_min: 25.0      # Higher = stricter
ssim_min: 0.85      # Range [0, 1], higher = stricter
pixel_diff_max: 0.02 # Range [0, 1], lower = stricter
```

#### Baseline Management

- First run: Output automatically saved as baseline
- Subsequent runs: Compared against baseline
- Update baseline: `--update-baseline` flag

### Error Classification

| Category | Description | Analysis Hint |
|----------|-------------|---------------|
| MODEL_LOAD_ERROR | Failed to load model | Check model_root path and file integrity |
| INFERENCE_ERROR | Error during inference | Check traceback in log_path |
| OUTPUT_ERROR | Failed to save output | Check directory permissions and disk space |
| OOM_ERROR | GPU out of memory | Reduce batch_size or resolution |
| TIMEOUT | Execution exceeded time limit | Increase timeout_seconds or check for deadlock |

### Report Structure

`example_report.json` contains:

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
      "analysis_hint": "Check traceback in log_path to locate the specific module"
    }
  ],
  "reproduce_all_failed": "python examples/run_examples.py --pipeline wan21_1_3b_t2v && ..."
}
```

### Features

- **Subprocess isolation**: Each pipeline runs in isolated process with pinned GPUs
- **Baseline management**: Auto-save first run, update with flag
- **Regression metrics**: PSNR/SSIM for video, pixel diff for image
- **GPU memory tracking**: Peak VRAM usage per pipeline
- **Output validation**: NaN/Inf detection
- **Enhanced reporting**: Reproduce commands and analysis hints for failures

### Adding New Pipelines

1. Create example script in appropriate directory under `examples/`
2. Add entry to `example_config.yaml`:

```yaml
pipelines:
  my_new_pipeline:
    script: my_category/my_script.py
    gpu_count: 1
    output_type: video
    model_root: /path/to/model
```

3. Run to generate baseline:
```bash
python examples/run_examples.py --pipeline my_new_pipeline
```