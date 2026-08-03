#!/bin/bash
# Run CI tests locally before pushing
# This script simulates what CI would run

set -e

skip_install=false
if [ "${1:-}" = "--skip-install" ]; then
    skip_install=true
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--skip-install]"
    exit 2
fi

echo "=========================================="
echo "Running CI Tests Locally"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Function to check if command succeeded
check_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 passed${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the project root${NC}"
    exit 1
fi

# Install dependencies if needed
if [ "$skip_install" = true ]; then
    print_section "Skipping dependency installation"
    echo "Using dependencies from the active Python environment"
else
    print_section "Installing dependencies"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    pip install -e ".[dev]" -q
    check_result "Dependencies installation"
fi

# Run lint checks
print_section "Running lint checks"
ruff check telefuser tests --output-format=full
check_result "Ruff check"

ruff format --check telefuser tests
check_result "Ruff format check"

ruff check --select I telefuser tests
check_result "Import check"

# A CUDA-enabled local PyTorch build is valid for local CI, but the tests below
# must observe the same CPU-only runtime contract as the GitHub runners.
print_section "Enforcing CPU-only test execution"
export CUDA_VISIBLE_DEVICES=""
python -c 'import torch; assert not torch.cuda.is_available(), "CUDA must be hidden during CPU CI tests"'
check_result "CPU-only runtime"

# Run unit tests
print_section "Running unit tests"
python -m pytest tests/unit -v \
    -m "not gpu and not distributed and not slow and not quant" \
    --tb=short
check_result "Unit tests"

# Run server pytest tests (includes OpenAI API tests)
print_section "Running server pytest tests (includes OpenAI API)"
python -m pytest tests/server/ -v \
    -m "not gpu and not distributed and not slow" \
    --tb=short
check_result "Server pytest tests (including OpenAI API)"

print_section "All CI tests passed!"
echo -e "${GREEN}✓ Ready to push${NC}"
