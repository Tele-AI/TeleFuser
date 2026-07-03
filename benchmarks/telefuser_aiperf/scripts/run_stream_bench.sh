#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH="${1:-benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

NOFILE_LIMIT="${TELEFUSER_BENCH_NOFILE_LIMIT:-8192}"
if ! ulimit -n "${NOFILE_LIMIT}" >/dev/null 2>&1; then
    echo "Warning: failed to raise open-file limit to ${NOFILE_LIMIT}" >&2
fi

SERVER_URL="${TELEFUSER_STREAM_BENCH_URL:-http://127.0.0.1:8088}"
HEALTH_URL="${TELEFUSER_STREAM_BENCH_HEALTH_URL:-${SERVER_URL}/v1/service/health}"
PYTHON_BIN="${TELEFUSER_STREAM_BENCH_PYTHON:-python3}"
ICE_HOST_IPS="${TELEFUSER_STREAM_BENCH_ICE_HOST_IPS:-}"
ICE_HOST_ARGS=()
if [[ -n "${ICE_HOST_IPS}" ]]; then
    IFS=',' read -r -a _ICE_HOST_IP_ARRAY <<< "${ICE_HOST_IPS}"
    for ice_host_ip in "${_ICE_HOST_IP_ARRAY[@]}"; do
        if [[ -n "${ice_host_ip}" ]]; then
            ICE_HOST_ARGS+=(--ice-host-ip "${ice_host_ip}")
        fi
    done
fi

if command -v curl >/dev/null 2>&1; then
    echo "Checking TeleFuser stream health: ${HEALTH_URL}"
    curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null
fi

echo "Running stream benchmark with config: ${CONFIG_PATH}"
"${PYTHON_BIN}" benchmarks/telefuser_aiperf/scripts/run_stream_bench.py \
    --config "${CONFIG_PATH}" \
    --server-url "${SERVER_URL}" \
    "${ICE_HOST_ARGS[@]}" \
    "$@"
