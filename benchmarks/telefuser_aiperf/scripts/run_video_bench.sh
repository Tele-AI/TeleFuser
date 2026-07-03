#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml}"
SERVER_URL="${TELEFUSER_AIPERF_URL:-http://127.0.0.1:8000}"
HEALTH_URL="${TELEFUSER_AIPERF_HEALTH_URL:-${SERVER_URL}/v1/service/health}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
NOFILE_LIMIT="${TELEFUSER_BENCH_NOFILE_LIMIT:-8192}"

if ! ulimit -n "${NOFILE_LIMIT}" >/dev/null 2>&1; then
    echo "Warning: failed to raise open-file limit to ${NOFILE_LIMIT}" >&2
fi

if ! command -v "${AIPERF_BIN}" >/dev/null 2>&1; then
    echo "aiperf is not installed. Set AIPERF_BIN or run: pip install -e ./benchmarks/aiperf" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    echo "Checking TeleFuser health: ${HEALTH_URL}"
    curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null
fi

echo "Running AIPerf with config: ${CONFIG_PATH}"
"${AIPERF_BIN}" profile --config "${CONFIG_PATH}"
