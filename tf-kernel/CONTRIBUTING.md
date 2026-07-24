# Contributing to tf-kernel

Thank you for your interest in contributing to tf-kernel! This document provides guidelines and instructions for contributing to this project.

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

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TeleFuser.git
   cd TeleFuser/tf-kernel
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Tele-AI/TeleFuser.git
   ```

## Development Setup

### Prerequisites

- Python >= 3.10
- CMake >= 3.26
- CUDA Toolkit 12.8+
- PyTorch == 2.11.0

### Setup Development Environment

#### Option 1: Install with all dev dependencies (Recommended)

This installs all development, testing, documentation, and linting dependencies:

```bash
# Install with all development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Build the project (requires CUDA)
make build-auto

# Run tests
make test
```

#### Option 2: Install specific dependency groups

If you only need specific dependencies:

```bash
# For running tests
pip install -e ".[test]"

# For building documentation
pip install -e ".[docs]"

# For code linting and formatting
pip install -e ".[lint]"

# Install pre-commit hooks (after installing lint dependencies)
pre-commit install
```

#### Dependency Groups

| Group | Description | Packages |
|-------|-------------|----------|
| `dev` | All development dependencies | Includes test, docs, and lint |
| `test` | Testing dependencies | pytest, pytest-cov |
| `docs` | Documentation dependencies | sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints |
| `lint` | Code formatting and linting | pre-commit, black, isort, ruff |

**Note**: `clang-format` for C++/CUDA formatting should be installed via system package manager:
```bash
# Ubuntu/Debian
sudo apt-get install clang-format

# macOS
brew install clang-format
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/ORIGINAL_OWNER/tf-kernel/issues) to avoid duplicates.

When filing a bug report, please include:

- **System information**: OS, CUDA version, GPU architecture
- **Python version** and **PyTorch version**
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Error messages** or stack traces
- **Minimal code example** that reproduces the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed feature
- Explain why this enhancement would be useful
- List some examples of how the feature would be used

### Adding New Kernels

To add a new CUDA kernel:

1. **Implement the kernel** in `csrc/<category>/your_kernel.cu`
2. **Declare the interface** in `include/tf_kernel_ops.h`
3. **Register with PyTorch** in `csrc/common_extension.cc`:
   ```cpp
   m.def("your_kernel(Tensor input, Tensor! output) -> ()");
   m.impl("your_kernel", torch::kCUDA, &your_kernel);
   ```
4. **Update CMakeLists.txt**: Add source file to the appropriate `SOURCES` list
5. **Create Python wrapper** in `tf_kernel/<category>.py`
6. **Export in `tf_kernel/__init__.py`**
7. **Add tests** in `tests/test_your_kernel.py`
8. **Add benchmarks** in `benchmark/` (if applicable)

See [AGENTS.md](AGENTS.md) for more detailed technical information.

## Coding Standards

### C++/CUDA Code

- Use **clang-format** with the provided `.clang-format` config (Google style based)
- 2-space indentation
- 120 column limit
- Left pointer alignment (`int* ptr` not `int *ptr`)

```bash
# Format C++/CUDA files
make format
```

### Python Code

- **isort**: Import sorting
- ~~**black**: Code formatting~~ (Disabled)
- **ruff**: Linting (F821 rule, F401 disabled)

```bash
# Format Python files
make format
```

### Pre-commit Hooks

All commits are checked by pre-commit hooks. You can run them manually:

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
Add FP8 blockwise quantization kernel

- Implements per-token-group quantization for FP8
- Optimized for SM90 (Hopper) architecture
- Adds corresponding tests and benchmarks

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

5. **Run the test suite** locally:
   ```bash
   make test
   ```

6. **Update documentation** if needed (README, AGENTS.md, code comments)

7. **Commit your changes** with clear messages

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request** on GitHub

### PR Review Process

- Maintainers will review your PR as soon as possible
- Address review comments by pushing additional commits
- Once approved, a maintainer will merge your PR

### PR Checklist

- [ ] Code follows the project's coding standards
- [ ] All pre-commit hooks pass
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

## Questions?

If you have questions or need help, feel free to:

- Open a [GitHub Discussion](https://github.com/ORIGINAL_OWNER/tf-kernel/discussions)
- Join our community channels (if available)

Thank you for contributing to tf-kernel!
