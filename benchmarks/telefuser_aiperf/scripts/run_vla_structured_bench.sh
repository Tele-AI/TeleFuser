#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-benchmarks/telefuser_aiperf/configs/vla_structured_e2e.yaml}"
SERVER_URL="${TELEFUSER_AIPERF_URL:-http://127.0.0.1:18080}"
HEALTH_URL="${TELEFUSER_AIPERF_HEALTH_URL:-${SERVER_URL}/v1/service/ready}"
DEFAULT_PYTHON="${ROOT_DIR}/.venv-aiperf/bin/python"
ADAPTER_ROOT="${ROOT_DIR}/benchmarks/telefuser_aiperf"
AIPERF_PYTHON="${TELEFUSER_AIPERF_PYTHON:-${DEFAULT_PYTHON}}"

if [[ ! -x "${AIPERF_PYTHON}" ]]; then
    echo "The isolated AIPerf environment is unavailable. Run: bash scripts/setup_aiperf.sh" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    echo "Checking TeleFuser VLA readiness: ${HEALTH_URL}"
    curl --noproxy '*' --fail --silent --show-error "${HEALTH_URL}" >/dev/null
fi

export PYTHONPATH="${ADAPTER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${TELEFUSER_AIPERF_SERVICE_PID:-}" ]]; then
    RESOURCE_PYTHON="${TELEFUSER_VLA_PYTHON:-${ROOT_DIR}/.venv-vla/bin/python}"
    if [[ ! -x "${RESOURCE_PYTHON}" ]]; then
        echo "The VLA resource sampler interpreter is unavailable: ${RESOURCE_PYTHON}" >&2
        exit 1
    fi
    exec "${RESOURCE_PYTHON}" benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.py \
        --config "${CONFIG_PATH}" \
        --service-pid "${TELEFUSER_AIPERF_SERVICE_PID}" \
        --output "${TELEFUSER_AIPERF_RESOURCE_OUTPUT:-artifacts/telefuser_aiperf/vla_structured/resource_summary.json}"
fi
exec "${AIPERF_PYTHON}" -m telefuser_aiperf.cli profile --config "${CONFIG_PATH}"
