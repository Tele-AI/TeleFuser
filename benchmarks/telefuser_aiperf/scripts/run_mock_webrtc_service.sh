#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
AIPERF_SRC="${AIPERF_SRC:-${ROOT_DIR}/benchmarks/aiperf/src}"
if [[ ! -d "${AIPERF_SRC}/aiperf" ]]; then
    echo "AIPerf source checkout not found at ${AIPERF_SRC}." >&2
    echo "Run: bash scripts/setup_aiperf_repo.sh" >&2
    exit 1
fi
export PYTHONPATH="${AIPERF_SRC}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${TELEFUSER_MOCK_WEBRTC_PYTHON:-python3}"
HOST="${TELEFUSER_MOCK_WEBRTC_HOST:-127.0.0.1}"
PORT="${TELEFUSER_MOCK_WEBRTC_PORT:-8088}"

exec "${PYTHON_BIN}" benchmarks/telefuser_aiperf/scripts/run_mock_webrtc_service.py \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
