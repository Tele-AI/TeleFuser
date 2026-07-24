#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
WITH_KERNEL=0

usage() {
    cat <<'EOF'
Usage: scripts/install_dev.sh [--kernel]

Install TeleFuser in editable development mode.

Options:
  --kernel  Also build and install the local tf-kernel project in editable mode.
  -h, --help
            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)
            WITH_KERNEL=1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${WITH_KERNEL}" == "1" ]]; then
    exec "${PYTHON_BIN}" -m pip install \
        --editable "${ROOT_DIR}/tf-kernel" \
        --editable "${ROOT_DIR}[dev]"
fi

exec "${PYTHON_BIN}" -m pip install --editable "${ROOT_DIR}[dev]"
