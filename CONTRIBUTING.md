# Contributing to TeleFuser

Thank you for your interest in contributing to TeleFuser! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Clone the repository** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/telefuser.git
   cd telefuser
   ```

2. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/telefuser.git
   ```

## Development Setup

### Prerequisites

- Python >= 3.10, < 3.14
- CUDA Toolkit 12.8+
- PyTorch >= 2.6.0

### Setup Development Environment

#### Install with development dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Also build the in-tree tf-kernel project for joint development
scripts/install_dev.sh --kernel

# Install pre-commit hooks
pip install -U pre-commit
pip install ruff==0.9.4
pre-commit install
```

#### Dependency Groups

| Group | Description | Packages |
|-------|-------------|----------|
| `dev` | All development dependencies | pytest, pre-commit, ruff |
| `docs` | Documentation dependencies | mkdocs, mkdocs-material, mkdocstrings-python |
| `kernel` | Released CUDA extension package | tf-kernel |

`pip install -e ".[kernel]"` installs the released kernel wheel. Use `scripts/install_dev.sh --kernel` when a change
touches both TeleFuser and the local `tf-kernel/` source tree.

**Note**: Since the `ruff-pre-commit` is currently maintained in the open-source community of the R&D Cloud, you will need to set up SSH key authentication in the open-source community before running `pre-commit install`.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the existing issues to avoid duplicates.

When filing a bug report, please include:

- **System information**: OS, CUDA version, GPU architecture
- **Python version** and **PyTorch version**
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Error messages** or stack traces
- **Minimal code example** that reproduces the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed feature
- Explain why this enhancement would be useful
- List some examples of how the feature would be used

### Adding New Pipelines

To add a new pipeline:

1. **Create pipeline directory** under `telefuser/pipelines/your_pipeline/`
2. **Implement pipeline class** inheriting from `BasePipeline`
3. **Define configuration dataclass** for your pipeline
4. **Implement required stages**: text encoding, denoising, VAE, etc.
5. **Add model architectures** to `telefuser/models/` if needed
6. **Add example script** in `examples/your_pipeline/`
7. **Add tests** in `tests/`

See [AGENTS.md](AGENTS.md) for more detailed technical information.

## Coding Standards

### Python Code

We adopt PEP8 as our code style with the following configurations:

- **ruff**: Linting and formatting (line length: 120)
- **pre-commit hooks**: Automated code quality checks

```bash
# Run linting and formatting checks
pre-commit run --all-files
```

### Pre-commit Hooks

All commits are checked by pre-commit hooks. The pre-commit configuration is stored in `.pre-commit-config.yaml`.

Once successfully installed, code style checks and automatic formatting will be executed automatically each time you commit code.

You can also manually run the checks:

```bash
pre-commit run --all-files
```

## Commit Message Guidelines

Use clear and meaningful commit messages:

- **Use the imperative mood** ("Add feature" not "Added feature")
- **Keep the first line under 72 characters**
- **Reference issues and PRs** where appropriate

Example:
```
Add FP8 quantization support for WanVideo pipeline

- Implements FP8 weight quantization for DiT models
- Adds memory-efficient attention with FP8
- Updates pipeline configs to support quantization

Fixes #123
```

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the coding standards

4. **Add or update tests** as necessary

5. **Run the test suite** locally (see [Testing Guide](#testing-guide) below)

6. **Update documentation** if needed (README, AGENTS.md, code comments)

7. **Commit your changes** with clear messages

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request**

### PR Review Process

- Maintainers will review your PR as soon as possible
- Address review comments by pushing additional commits
- Once approved, a maintainer will merge your PR

### PR Checklist

- [ ] Code follows the project's coding standards (PEP8, ruff)
- [ ] All pre-commit hooks pass
- [ ] Tests pass locally (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, AGENTS.md, inline comments)
- [ ] Commit messages are clear and descriptive

## Testing Guide

This section provides comprehensive guidance on running and writing tests for TeleFuser.

### Running Tests

#### Run all tests
```bash
pytest -v
```

#### Run only unit tests (excluding GPU/distributed tests)
```bash
pytest tests/unit -m "not gpu and not distributed"
```

#### Run with coverage
```bash
pytest tests/unit --cov=telefuser --cov-report=term-missing
```

#### Run specific test file
```bash
pytest tests/unit/core/test_config.py -v
```

#### Run tests matching a pattern
```bash
pytest -k "test_default_values"
```

### Test Markers

We use pytest markers to categorize tests:

| Marker | Description |
|--------|-------------|
| `gpu` | Tests requiring a GPU (skipped on CPU-only environments) |
| `multi_gpu` | Tests requiring multiple GPUs |
| `distributed` | Tests requiring distributed setup (Ray, multiprocessing) |
| `quant` | Tests requiring quantization support |
| `slow` | Tests that take a long time to run |

#### Examples

```bash
# Skip GPU and distributed tests (default for CI)
pytest -m "not gpu and not distributed and not quant"

# Run only GPU tests
pytest -m "gpu"

# Run only unit tests
pytest tests/unit -v
```

### Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── unit/                          # Unit tests
│   ├── core/                      # Core module tests
│   ├── distributed/               # Distributed processing tests
│   ├── entrypoints/               # CLI entrypoint tests
│   ├── feature_cache/             # Feature cache tests
│   ├── models/                    # Model architecture tests
│   ├── offload/                   # CPU offloading tests
│   ├── ops/                       # Operation tests (attention, norms)
│   ├── quantize/                  # Quantization tests
│   ├── schedulers/                # Diffusion scheduler tests
│   ├── service/                   # API service tests
│   └── worker/                    # Worker tests
└── integration/                   # Integration tests
```

### Writing Tests

When adding new tests, follow these principles:

1. **Place tests correctly**: Unit tests go in `tests/unit/<module>/`
2. **Use descriptive names**: `test_<function_name>_<scenario>`
3. **Add docstrings**: Explain what the test verifies
4. **Use fixtures**: Common setup in `conftest.py`
5. **Apply markers**: Use `@pytest.mark.gpu` for GPU tests
6. **Do NOT use sys.path.insert**: The project should be installed in development mode (`pip install -e ".[dev]"`), so imports work naturally without modifying sys.path

#### Example

```python
# Good: Direct imports after pip install -e ".[dev]"
from telefuser.core.config import AttentionConfig
from telefuser.distributed.device_mesh import create_device_mesh_from_config

# Bad: Do NOT do this in test files
# sys.path.insert(0, os.path.dirname(...))


def test_function_name_valid_input():
    """Test function_name with valid input parameters."""
    result = function_name(valid_param)
    assert result == expected_value


@pytest.mark.gpu
def test_gpu_function():
    """Test GPU-specific functionality."""
    # This test will be skipped if no GPU is available
    pass
```

#### Writing GPU/Distributed Tests for CPU CI

GitHub Actions CI runs on CPU-only environments without GPU support. When writing tests that import GPU-dependent modules (e.g., `telefuser.distributed` which depends on `triton`, or `telefuser.ops.attention` with Flash Attention), you must handle import failures gracefully at the module level:

```python
# tests/unit/ops/test_gpu_feature.py
import pytest
import torch

# Skip entire module if GPU dependencies not available (CPU-only CI environment)
try:
    from telefuser.distributed.device_mesh import create_device_mesh_from_config
    from telefuser.ops.attention.attention_impl import long_context_attention
except ImportError as e:
    pytest.skip(f"Skipping test module due to missing dependencies: {e}", allow_module_level=True)

# Mark all tests in this module as requiring GPU/distributed
pytestmark = [
    pytest.mark.distributed,
    pytest.mark.gpu,
    pytest.mark.multi_gpu,
]

class TestGPUFeature:
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_feature(self):
        # Test implementation
        pass
```

**Why this is needed:**
- During test collection, pytest imports all test modules to discover tests
- If a module imports `telefuser.distributed` or other GPU-dependent modules at the top level, and `triton`/CUDA is not available, the import will fail
- Without `allow_module_level=True`, the entire test suite will fail with an `ImportError`
- With proper handling, the module is skipped and other tests can run normally

**Modules that require this pattern:**
- `telefuser.distributed` (depends on `triton`)
- `telefuser.ops.attention.attention_impl` (may depend on Flash Attention)
- `telefuser.ops.quantized_linear` (depends on FP8 kernels)

### Test Statistics

Current test coverage:
- **Total tests**: 300+ tests
- **Unit tests**: 290+ tests
- **Integration tests**: 10+ tests
- **Test files**: 20+ files

### CI Integration

Tests run automatically in GitHub Actions:
- **lint.yml**: Code style checks
- **test.yml**: Unit tests on Python 3.10/3.11/3.12
- **server-test.yml**: Server and integration tests
- GPU, distributed, and quantization tests are skipped in CI

### Local CI Testing

We provide a local CI script that mirrors the GitHub Actions workflows exactly:

```bash
# Run the full CI suite locally
bash scripts/run_ci_tests.sh
```

This script runs the same checks as the remote CI:
1. **Lint checks**: ruff check, ruff format check, import checks
2. **Unit tests**: All unit tests with appropriate markers excluded
3. **Server integration tests**: Tests the FastAPI server with a test pipeline
4. **Integration tests**: API-level integration tests

**Important**: If you pass local CI tests (`bash scripts/run_ci_tests.sh`), GitHub Actions CI will also pass. The local script is kept in sync with the remote workflows to ensure consistency.

**Prerequisites for local CI**:
- Python >= 3.10
- No GPU required (CPU-only PyTorch is sufficient for CI tests)

## Questions?

If you have questions or need help, feel free to:

- Open an issue in the repository
- Check [AGENTS.md](AGENTS.md) for technical details

Thank you for contributing to TeleFuser!
