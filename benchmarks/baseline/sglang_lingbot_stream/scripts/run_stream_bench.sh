#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

AIPERF_DIR="${ROOT_DIR}/benchmarks/aiperf"
UV_BIN="${AIPERF_UV_BIN:-uv}"
CONFIG_PATH="${1:-benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_quick.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "${AIPERF_DIR}/pyproject.toml" ]]; then
    echo "AIPerf checkout not found. Run: bash scripts/setup_aiperf_repo.sh" >&2
    exit 1
fi
if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

SERVER_URL="${SGLANG_STREAM_BENCH_URL:-http://127.0.0.1:30000}"
SERVER_ARGS=(--stream-server-url "${SERVER_URL}")
for argument in "$@"; do
    if [[ "${argument}" == "--stream-server-url" || "${argument}" == --stream-server-url=* ]]; then
        SERVER_ARGS=()
        break
    fi
done

RESOURCE_ARGS=()
RESOURCE_HISTORY_URL="${AIPERF_HISTORY_URL:-}"
RESOURCE_TARGET_PID="${AIPERF_RESOURCE_TARGET_PID:-${SGLANG_STREAM_BENCH_PID:-}}"
if [[ -n "${RESOURCE_HISTORY_URL}" || -n "${RESOURCE_TARGET_PID}" ]]; then
    if [[ -z "${RESOURCE_HISTORY_URL}" || -z "${RESOURCE_TARGET_PID}" ]]; then
        echo "AIPERF_HISTORY_URL and AIPERF_RESOURCE_TARGET_PID must be set together" >&2
        exit 2
    fi
    RESOURCE_ARGS+=(--stream-resource-history-url "${RESOURCE_HISTORY_URL}")
    RESOURCE_ARGS+=(--stream-resource-target-pid "${RESOURCE_TARGET_PID}")
fi

exec "${UV_BIN}" run --frozen --no-dev --project "${AIPERF_DIR}" \
    aiperf profile \
    --stream-config "${CONFIG_PATH}" \
    "${SERVER_ARGS[@]}" \
    "${RESOURCE_ARGS[@]}" \
    "$@"
