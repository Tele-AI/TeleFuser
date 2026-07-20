#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml}"
SERVER_URL="${TELEFUSER_AIPERF_URL:-http://127.0.0.1:8000}"
HEALTH_URL="${TELEFUSER_AIPERF_HEALTH_URL:-${SERVER_URL}/v1/service/health}"
AIPERF_DIR="${ROOT_DIR}/benchmarks/aiperf"
UV_BIN="${AIPERF_UV_BIN:-uv}"
NOFILE_LIMIT="${TELEFUSER_BENCH_NOFILE_LIMIT:-8192}"

if ! ulimit -n "${NOFILE_LIMIT}" >/dev/null 2>&1; then
    echo "Warning: failed to raise open-file limit to ${NOFILE_LIMIT}" >&2
fi

if [[ ! -f "${AIPERF_DIR}/pyproject.toml" ]]; then
    echo "AIPerf checkout not found. Run: bash scripts/setup_aiperf_repo.sh" >&2
    exit 1
fi
if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    echo "Checking TeleFuser health: ${HEALTH_URL}"
    curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null
fi

echo "Running AIPerf with config: ${CONFIG_PATH}"
exec "${UV_BIN}" run --frozen --no-dev --project "${AIPERF_DIR}" aiperf profile --config "${CONFIG_PATH}"
