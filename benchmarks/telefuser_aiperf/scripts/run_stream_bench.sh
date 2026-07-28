#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-benchmarks/telefuser_aiperf/configs/stream_lingbot_world_v2_1min.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

DEFAULT_PYTHON="${ROOT_DIR}/.venv-aiperf/bin/python"
ADAPTER_ROOT="${ROOT_DIR}/benchmarks/telefuser_aiperf"
if [[ -n "${TELEFUSER_AIPERF_PYTHON:-}" ]]; then
    AIPERF_PYTHON="${TELEFUSER_AIPERF_PYTHON}"
elif [[ -x "${DEFAULT_PYTHON}" ]]; then
    AIPERF_PYTHON="${DEFAULT_PYTHON}"
else
    AIPERF_PYTHON="$(command -v python || true)"
fi
if [[ -z "${AIPERF_PYTHON}" ]] || ! command -v "${AIPERF_PYTHON}" >/dev/null 2>&1 \
    || ! PYTHONPATH="${ADAPTER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${AIPERF_PYTHON}" -c 'import livekit, telefuser_aiperf' >/dev/null 2>&1; then
    echo "The pinned streaming-capable AIPerf or LiveKit is not installed. Run: bash scripts/setup_aiperf.sh" >&2
    exit 1
fi

export PYTHONPATH="${ADAPTER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${AIPERF_PYTHON}" -m telefuser_aiperf.cli profile --stream-config "${CONFIG_PATH}" "$@"
