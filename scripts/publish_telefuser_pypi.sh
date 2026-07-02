#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PYTHON_BIN="${PYTHON:-python3}"
REPOSITORY="${TWINE_REPOSITORY:-pypi}"

if [[ ! -d "${DIST_DIR}" ]] || ! compgen -G "${DIST_DIR}/telefuser-*" >/dev/null; then
    echo "No TeleFuser distributions found in ${DIST_DIR}. Run scripts/build_telefuser_dist.sh first." >&2
    exit 1
fi

"${PYTHON_BIN}" -m twine check "${DIST_DIR}"/telefuser-*
"${PYTHON_BIN}" -m twine upload --repository "${REPOSITORY}" "${DIST_DIR}"/telefuser-*
