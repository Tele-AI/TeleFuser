#!/bin/bash
# Run CI tests locally before pushing
# This script simulates what CI would run

set -e

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
print_section "Installing dependencies"
pip install -e ".[dev,webrtc]" -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
check_result "Dependencies installation"

# Run lint checks
print_section "Running lint checks"
ruff check telefuser tests --output-format=full
check_result "Ruff check"

ruff format --check telefuser tests
check_result "Ruff format check"

ruff check --select I telefuser tests
check_result "Import check"

# Run unit tests
print_section "Running unit tests"
pytest tests/unit -v \
    -m "not gpu and not distributed and not slow and not quant" \
    --tb=short
check_result "Unit tests"

# Run server pytest tests (includes OpenAI API tests)
print_section "Running server pytest tests (includes OpenAI API)"
pytest tests/server/ -v \
    -m "not gpu and not distributed and not slow" \
    --tb=short
check_result "Server pytest tests (including OpenAI API)"

print_section "All CI tests passed!"
echo -e "${GREEN}✓ Ready to push${NC}"
