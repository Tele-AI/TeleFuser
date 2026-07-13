#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
AIPERF_SRC="${AIPERF_SRC:-${ROOT_DIR}/benchmarks/aiperf/src}"
if [[ ! -d "${AIPERF_SRC}/aiperf" ]]; then
    echo "AIPerf source checkout not found at ${AIPERF_SRC}." >&2
    echo "Run: bash scripts/setup_aiperf_repo.sh" >&2
    exit 1
fi
export PYTHONPATH="${AIPERF_SRC}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${SGLANG_MOCK_STREAM_PYTHON:-python3}"
HOST="${SGLANG_MOCK_STREAM_HOST:-127.0.0.1}"
PORT="${SGLANG_MOCK_STREAM_PORT:-30000}"

exec "${PYTHON_BIN}" benchmarks/baseline/sglang_lingbot_stream/scripts/run_mock_stream_service.py \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
