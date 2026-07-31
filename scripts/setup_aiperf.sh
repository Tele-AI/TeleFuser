#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AIPERF_PYTHON_BIN:-python3}"
AIPERF_ENV_DIR="${AIPERF_ENV_DIR:-${ROOT_DIR}/.venv-aiperf}"
AIPERF_SPEC="${AIPERF_SPEC:-aiperf @ git+https://github.com/ActivePeter/aiperf.git@e977ffbb1648510acec431b2a3fbd1a0f7bb8a35}"
LIVEKIT_SPEC="${LIVEKIT_SPEC:-livekit>=1.1.13,<2.0.0}"
MSGSPEC_SPEC="${MSGSPEC_SPEC:-msgspec>=0.18,<1.0}"
WEBSOCKETS_SPEC="${WEBSOCKETS_SPEC:-websockets>=15,<17}"
ADAPTER_ROOT="${ROOT_DIR}/benchmarks/telefuser_aiperf"

usage() {
    echo "Usage: scripts/setup_aiperf.sh"
    echo ""
    echo "Environment overrides: AIPERF_SPEC, LIVEKIT_SPEC, MSGSPEC_SPEC, WEBSOCKETS_SPEC,"
    echo "                       AIPERF_ENV_DIR, AIPERF_PYTHON_BIN"
}

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
        usage
        exit 0
    fi
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python is required to install AIPerf." >&2
    exit 1
fi

if [[ ! -x "${AIPERF_ENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${AIPERF_ENV_DIR}"
fi

if "${AIPERF_ENV_DIR}/bin/python" -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("aiperf") is None)'; then
    "${AIPERF_ENV_DIR}/bin/python" -m pip uninstall -y aiperf
fi

"${AIPERF_ENV_DIR}/bin/python" -m pip install \
    "${AIPERF_SPEC}" \
    "${LIVEKIT_SPEC}" \
    "${MSGSPEC_SPEC}" \
    "${WEBSOCKETS_SPEC}"

mkdir -p "${ROOT_DIR}/artifacts"

echo "AIPerf environment ready: ${AIPERF_ENV_DIR}"
PYTHONPATH="${ADAPTER_ROOT}" "${AIPERF_ENV_DIR}/bin/python" -c \
    'import aiperf, importlib.metadata as m, json, telefuser_aiperf; d=m.distribution("aiperf"); u=json.loads(d.read_text("direct_url.json") or "{}"); c=u.get("vcs_info", {}).get("commit_id", "unknown"); print(f"aiperf={aiperf.__version__} commit={c} adapter={telefuser_aiperf.__file__}")'
