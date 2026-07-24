#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PYTHON_BIN="${PYTHON:-python3}"

if ! "${PYTHON_BIN}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "TeleFuser package builds require Python 3.10 or newer. Set PYTHON=/path/to/python3.10+ if needed." >&2
    exit 1
fi

RELEASE_TAG="$(git -C "${ROOT_DIR}" describe --exact-match --tags HEAD 2>/dev/null || true)"
if [[ ! "${RELEASE_TAG}" =~ ^v[0-9] ]]; then
    echo "Refusing to build a PyPI release: HEAD is not exactly on a git tag." >&2
    if [[ -n "${RELEASE_TAG}" ]]; then
        echo "Tag ${RELEASE_TAG} is not a TeleFuser release tag; expected v<version>." >&2
    fi
    echo "Create a release tag first, for example: git tag -a v0.1.0 -m 'Release v0.1.0'" >&2
    echo "For a local smoke build only, run: SKIP_TAG_CHECK=1 ${0}" >&2
    if [[ "${SKIP_TAG_CHECK:-}" != "1" ]]; then
        exit 1
    fi
fi

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

"${PYTHON_BIN}" -m build --sdist --wheel --outdir "${DIST_DIR}" "${ROOT_DIR}"

if "${PYTHON_BIN}" -m twine --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m twine check "${DIST_DIR}"/*
else
    echo "twine is not installed; skipping twine check. Install it with: ${PYTHON_BIN} -m pip install twine" >&2
fi

echo "Built TeleFuser distributions:"
ls -1 "${DIST_DIR}"
