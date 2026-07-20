#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIPERF_REPO_URL="${AIPERF_REPO_URL:-https://github.com/ActivePeter/aiperf.git}"
AIPERF_DIR="${ROOT_DIR}/benchmarks/aiperf"
AIPERF_BRANCH="${AIPERF_BRANCH:-teleai}"
AIPERF_REF="${AIPERF_REF:-}"
UV_BIN="${AIPERF_UV_BIN:-uv}"
UPDATE_REPO=0

usage() {
    cat <<'EOF'
Usage: scripts/setup_aiperf_repo.sh [options]

Clone or update the external AIPerf checkout and install its runtime environment.

Options:
  --repo-url URL   Git URL (default: https://github.com/ActivePeter/aiperf.git)
  --branch NAME    Branch used for a new clone (default: teleai)
  --ref REF        Branch, tag, or commit to check out after fetching
  --update         Fast-forward the current branch; refuses a dirty checkout
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url)
            AIPERF_REPO_URL="$2"
            shift 2
            ;;
        --branch)
            AIPERF_BRANCH="$2"
            shift 2
            ;;
        --ref)
            AIPERF_REF="$2"
            shift 2
            ;;
        --update)
            UPDATE_REPO=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv is required to install AIPerf: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [[ ! -d "${AIPERF_DIR}/.git" ]]; then
    if [[ -e "${AIPERF_DIR}" ]]; then
        echo "Path exists and is not an AIPerf checkout: ${AIPERF_DIR}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${AIPERF_DIR}")"
    git clone --branch "${AIPERF_BRANCH}" "${AIPERF_REPO_URL}" "${AIPERF_DIR}"
fi

if [[ "${UPDATE_REPO}" -eq 1 || -n "${AIPERF_REF}" ]]; then
    if [[ -n "$(git -C "${AIPERF_DIR}" status --porcelain)" ]]; then
        echo "AIPerf checkout has local changes: ${AIPERF_DIR}" >&2
        exit 1
    fi
    git -C "${AIPERF_DIR}" fetch origin
fi

if [[ "${UPDATE_REPO}" -eq 1 ]]; then
    current_branch="$(git -C "${AIPERF_DIR}" symbolic-ref --quiet --short HEAD || true)"
    if [[ -z "${current_branch}" ]]; then
        echo "Cannot fast-forward a detached AIPerf checkout" >&2
        exit 1
    fi
    git -C "${AIPERF_DIR}" pull --ff-only origin "${current_branch}"
fi

if [[ -n "${AIPERF_REF}" ]]; then
    git -C "${AIPERF_DIR}" checkout "${AIPERF_REF}"
fi

"${UV_BIN}" sync --no-dev --project "${AIPERF_DIR}" --extra streaming-webrtc
mkdir -p "${ROOT_DIR}/artifacts"

echo "AIPerf ready: ${AIPERF_DIR}"
git -C "${AIPERF_DIR}" rev-parse HEAD
