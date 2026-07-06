"""Pytest shared fixtures and configuration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from PIL import Image

# ============================================================================
# Collection Ignore - Skip modules that require unavailable dependencies
# ============================================================================

# Check if triton is available (required by telefuser.distributed and telefuser.kernel.triton)
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

TRITON_REQUIRED_TESTS = [
    # Distributed tests require triton
    "unit/distributed/test_device_mesh.py",
    "unit/distributed/test_parallel_shard.py",
    "unit/distributed/test_pp_comm.py",
    "unit/distributed/test_ulysses_comm.py",
    # Kernel tests require triton
    "unit/kernel/test_rmsnorm.py",
    "unit/kernel/test_rotary.py",
    "unit/kernel/test_scale_shift.py",
    # Ops tests that require triton
    "unit/ops/test_long_context_attention.py",
    "unit/ops/test_parallel_shard_attention.py",
]

GPU_ONLY_TESTS = [
    # Kernel tests import CUDA-only Triton kernels during collection.
    "unit/kernel/test_rmsnorm.py",
    "unit/kernel/test_rotary.py",
    "unit/kernel/test_scale_shift.py",
    # These modules are marked entirely as GPU/multi-GPU tests.
    "unit/ops/test_long_context_attention.py",
    "unit/ops/test_parallel_shard_attention.py",
    "unit/offload/test_async_offload.py",
]

collect_ignore = []
TESTS_ROOT = Path(__file__).parent.resolve()


def _extend_collect_ignore(paths):
    """Add collection ignores while preserving order and avoiding duplicates."""
    for path in paths:
        if path not in collect_ignore:
            collect_ignore.append(path)


def _relative_test_path(collection_path):
    """Return a collection path relative to tests/, or None if outside tests/."""
    path = Path(str(collection_path)).resolve()
    try:
        return path.relative_to(TESTS_ROOT).as_posix()
    except ValueError:
        return None


# Skip tests that require triton if it's not available
if not HAS_TRITON:
    _extend_collect_ignore(TRITON_REQUIRED_TESTS)

# Skip GPU-only test modules in CPU-only environments before pytest imports them.
if not torch.cuda.is_available():
    _extend_collect_ignore(GPU_ONLY_TESTS)


class _IgnoredTestModule(pytest.Module):
    """Empty collector used for explicitly requested ignored test modules."""

    def collect(self):
        return []


def _should_ignore_test_path(collection_path):
    relative_path = _relative_test_path(collection_path)
    return relative_path in collect_ignore


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    """Avoid importing GPU-only test modules when their dependencies are unavailable."""
    if _should_ignore_test_path(collection_path):
        return True
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    """Return an empty collector when an ignored module is passed explicitly."""
    if _should_ignore_test_path(module_path):
        return _IgnoredTestModule.from_parent(parent, path=module_path)
    return None


# ============================================================================
# Hardware Detection Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(has_cuda):
    """Return the appropriate device."""
    return torch.device("cuda" if has_cuda else "cpu")


@pytest.fixture(scope="session")
def gpu_count():
    """Return the number of available GPUs."""
    return torch.cuda.device_count()


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "multi_gpu: marks tests that require multiple GPUs")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "distributed: marks tests that require distributed setup")


# ============================================================================
# Common Fixtures
# ============================================================================


@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (512, 512), color="red")


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 512, 512)


@pytest.fixture
def sample_noise_tensor():
    """Create a sample noise tensor for testing."""
    return torch.randn(1, 4, 64, 64)


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    manager.load_model = Mock(return_value=None)
    manager.get_model = Mock(return_value=Mock())
    return manager


@pytest.fixture
def mock_pipeline_config():
    """Create a mock pipeline configuration."""
    return {
        "name": "test_pipeline",
        "model_root": "/tmp/test_model",
        "num_inference_steps": 10,
        "height": 512,
        "width": 512,
    }


@pytest.fixture
def clear_cuda_cache():
    """Clear CUDA cache after test if CUDA is available."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset to random state
    torch.seed()
    np.random.seed()


# ============================================================================
# Skip Decorators Helpers
# ============================================================================

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA but CUDA is not available")

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason=f"Test requires multiple GPUs but only {torch.cuda.device_count()} available"
)


# Make these available as fixtures too
@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_single_gpu():
    """Skip test if less than 2 GPUs available."""
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need 2+ GPUs, found {torch.cuda.device_count()}")
