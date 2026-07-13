#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

AIPERF_REPO="${AIPERF_REPO:-${ROOT_DIR}/benchmarks/aiperf}"
UV_BIN="${AIPERF_UV_BIN:-uv}"
CONFIG_PATH="${1:-benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_quick.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

SERVER_URL="${SGLANG_STREAM_BENCH_URL:-http://127.0.0.1:30000}"
SERVER_ARGS=(--stream-server-url "${SERVER_URL}")
for argument in "$@"; do
    if [[ "${argument}" == "--stream-server-url" || "${argument}" == --stream-server-url=* ]]; then
        SERVER_ARGS=()
        break
    fi
done

exec "${UV_BIN}" run --project "${AIPERF_REPO}" \
    aiperf profile \
    --stream-config "${CONFIG_PATH}" \
    "${SERVER_ARGS[@]}" \
    "$@"
