#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-benchmarks/baseline/diffusers_wan_i2v/configs/video_generation_e2e.yaml}"
SERVER_URL="${DIFFUSERS_WAN_AIPERF_URL:-http://127.0.0.1:8010}"
HEALTH_URL="${DIFFUSERS_WAN_AIPERF_HEALTH_URL:-${SERVER_URL}/v1/service/health}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"

if ! command -v "${AIPERF_BIN}" >/dev/null 2>&1; then
    echo "aiperf is not installed. Set AIPERF_BIN or run: bash scripts/setup_aiperf_repo.sh" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    echo "Checking baseline service health: ${HEALTH_URL}"
    curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null
fi

echo "Running AIPerf with config: ${CONFIG_PATH}"
"${AIPERF_BIN}" profile --config "${CONFIG_PATH}"
