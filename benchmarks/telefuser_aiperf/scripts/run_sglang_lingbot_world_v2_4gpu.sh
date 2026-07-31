#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

SGLANG_BIN="${SGLANG_BIN:-}"
SGLANG_SOURCE_DIR="${SGLANG_SOURCE_DIR:-${ROOT_DIR}/work_dirs/sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SGLANG_FLOW_SHIFT="${SGLANG_FLOW_SHIFT:-10}"

if [[ -n "${SGLANG_BIN}" ]]; then
    if ! command -v "${SGLANG_BIN}" >/dev/null 2>&1; then
        echo "SGLANG_BIN is not executable: ${SGLANG_BIN}" >&2
        exit 1
    fi
    sglang_command=("${SGLANG_BIN}")
elif command -v sglang >/dev/null 2>&1; then
    sglang_command=("$(command -v sglang)")
elif [[ -n "${SGLANG_PYTHON}" && -x "${SGLANG_PYTHON}" \
    && -f "${SGLANG_SOURCE_DIR}/python/sglang/cli/main.py" ]]; then
    export PYTHONPATH="${SGLANG_SOURCE_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
    sglang_command=("${SGLANG_PYTHON}" "-c" "from sglang.cli.main import main; main()")
else
    echo "SGLang is unavailable. Set SGLANG_BIN, or set SGLANG_SOURCE_DIR and a compatible SGLANG_PYTHON." >&2
    exit 1
fi

IFS=',' read -r -a gpu_ids <<< "${SGLANG_CUDA_VISIBLE_DEVICES}"
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
    echo "SGLANG_CUDA_VISIBLE_DEVICES must contain exactly four comma-separated GPU IDs." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES}"
export SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES="${SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES:-60}"

exec "${sglang_command[@]}" serve \
    --model-path "${SGLANG_MODEL_PATH}" \
    --pipeline-class-name LingBotWorldCausalDMDPipeline \
    --host "${SGLANG_HOST}" \
    --port "${SGLANG_PORT}" \
    --num-gpus 4 \
    --ulysses-degree 4 \
    --flow-shift "${SGLANG_FLOW_SHIFT}" \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload false \
    --vae-config.use-parallel-decode true \
    --vae-config.parallel-decode-mode spatial \
    --enable-torch-compile false \
    "$@"
