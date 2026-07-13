#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AIPERF_REPO_URL="${AIPERF_REPO_URL:-https://github.com/ActivePeter/aiperf.git}"
AIPERF_REPO_DIR="${AIPERF_REPO_DIR:-${ROOT_DIR}/benchmarks/aiperf}"
AIPERF_BRANCH="${AIPERF_BRANCH:-teleai}"
AIPERF_REF="${AIPERF_REF:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_EDITABLE=1
UPDATE_REPO=0

usage() {
    cat <<'EOF'
Usage: scripts/setup_aiperf_repo.sh [options]

Clone or update the external AIPerf dependency repository and optionally install
it in editable mode.

Options:
  --repo-url URL    AIPerf git URL. Default: https://github.com/ActivePeter/aiperf.git
  --dir PATH       Checkout directory. Default: benchmarks/aiperf
  --branch NAME    Branch used for a new clone. Default: teleai
  --ref REF        Branch, tag, or commit to checkout after clone/fetch.
  --update         Fetch and fast-forward an existing checkout. Refuses dirty trees.
  --no-install     Skip "python -m pip install -e <checkout>".
  -h, --help       Show this help.

Environment overrides:
  AIPERF_REPO_URL, AIPERF_REPO_DIR, AIPERF_BRANCH, AIPERF_REF, PYTHON_BIN
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url)
            AIPERF_REPO_URL="$2"
            shift 2
            ;;
        --dir)
            AIPERF_REPO_DIR="$2"
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
        --no-install)
            INSTALL_EDITABLE=0
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

ensure_clean_checkout() {
    local repo_dir="$1"
    if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
        echo "AIPerf checkout has local changes: ${repo_dir}" >&2
        echo "Refusing to update or checkout a ref. Commit/stash them, or run without --update/--ref." >&2
        exit 1
    fi
}

clone_repo() {
    local repo_url="$1"
    local repo_dir="$2"
    local branch="$3"

    mkdir -p "$(dirname "${repo_dir}")"
    if [[ -e "${repo_dir}" ]]; then
        if [[ -d "${repo_dir}" && -z "$(ls -A "${repo_dir}")" ]]; then
            rmdir "${repo_dir}"
        else
            echo "Path exists and is not an AIPerf git checkout: ${repo_dir}" >&2
            exit 1
        fi
    fi

    echo "Cloning AIPerf from ${repo_url} (${branch}) -> ${repo_dir}"
    git clone --branch "${branch}" "${repo_url}" "${repo_dir}"
}

checkout_ref_if_needed() {
    local repo_dir="$1"
    local ref="$2"

    if [[ -z "${ref}" ]]; then
        return
    fi

    ensure_clean_checkout "${repo_dir}"
    echo "Fetching and checking out AIPerf ref: ${ref}"
    git -C "${repo_dir}" fetch origin "${ref}" || git -C "${repo_dir}" fetch origin
    git -C "${repo_dir}" checkout "${ref}"
}

update_repo_if_requested() {
    local repo_dir="$1"

    if [[ "${UPDATE_REPO}" -ne 1 ]]; then
        return
    fi

    ensure_clean_checkout "${repo_dir}"
    echo "Fetching AIPerf updates"
    git -C "${repo_dir}" fetch origin

    local branch
    branch="$(git -C "${repo_dir}" symbolic-ref --short HEAD 2>/dev/null || true)"
    if [[ -n "${branch}" ]]; then
        echo "Fast-forwarding AIPerf branch: ${branch}"
        git -C "${repo_dir}" pull --ff-only origin "${branch}"
    else
        echo "AIPerf checkout is detached; fetched origin but did not pull."
    fi
}

if [[ -d "${AIPERF_REPO_DIR}/.git" ]]; then
    remote_url="$(git -C "${AIPERF_REPO_DIR}" config --get remote.origin.url || true)"
    echo "AIPerf checkout already exists: ${AIPERF_REPO_DIR}"
    if [[ -n "${remote_url}" ]]; then
        echo "origin: ${remote_url}"
    fi
else
    clone_repo "${AIPERF_REPO_URL}" "${AIPERF_REPO_DIR}" "${AIPERF_BRANCH}"
fi

update_repo_if_requested "${AIPERF_REPO_DIR}"
checkout_ref_if_needed "${AIPERF_REPO_DIR}" "${AIPERF_REF}"

if [[ "${INSTALL_EDITABLE}" -eq 1 ]]; then
    echo "Installing AIPerf in editable mode"
    "${PYTHON_BIN}" -m pip install -e "${AIPERF_REPO_DIR}"
fi

echo "AIPerf dependency is ready at: ${AIPERF_REPO_DIR}"
