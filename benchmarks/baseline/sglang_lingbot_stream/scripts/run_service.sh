#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
fi

if [[ -n "${SGLANG_EXTRA_PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SGLANG_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

SGLANG_BIN="${SGLANG_BIN:-sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-}"
SERVICE_PORT="${SGLANG_LINGBOT_PORT:-30000}"
MODEL_PATH="${SGLANG_LINGBOT_MODEL_PATH:-robbyant/lingbot-world-fast-diffusers}"
MODEL_ID="${SGLANG_LINGBOT_MODEL_ID:-lingbot-world-fast-diffusers}"
MODEL_TYPE="${SGLANG_LINGBOT_MODEL_TYPE:-diffusion}"
PIPELINE_CLASS="${SGLANG_LINGBOT_PIPELINE_CLASS:-LingBotWorldCausalDMDPipeline}"
PERFORMANCE_MODE="${SGLANG_LINGBOT_PERFORMANCE_MODE:-speed}"
ATTENTION_BACKEND_CONFIG="${SGLANG_LINGBOT_ATTENTION_BACKEND_CONFIG:-VSA_sparsity=0.0}"
NUM_GPUS="${SGLANG_LINGBOT_NUM_GPUS:-1}"
ULYSSES_DEGREE="${SGLANG_LINGBOT_ULYSSES_DEGREE:-1}"
DIT_CPU_OFFLOAD="${SGLANG_LINGBOT_DIT_CPU_OFFLOAD:-false}"
TEXT_ENCODER_CPU_OFFLOAD="${SGLANG_LINGBOT_TEXT_ENCODER_CPU_OFFLOAD:-false}"

if [[ -n "${SGLANG_PYTHON}" ]]; then
    SGLANG_CMD=("${SGLANG_PYTHON}" -c "from sglang.cli.main import main; main()")
else
    read -r -a SGLANG_CMD <<< "${SGLANG_BIN}"
fi

exec "${SGLANG_CMD[@]}" serve \
    --model-type "${MODEL_TYPE}" \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --pipeline-class-name "${PIPELINE_CLASS}" \
    --performance-mode "${PERFORMANCE_MODE}" \
    --attention-backend-config "${ATTENTION_BACKEND_CONFIG}" \
    --port "${SERVICE_PORT}" \
    --num-gpus "${NUM_GPUS}" \
    --ulysses-degree "${ULYSSES_DEGREE}" \
    --dit-cpu-offload "${DIT_CPU_OFFLOAD}" \
    --text-encoder-cpu-offload "${TEXT_ENCODER_CPU_OFFLOAD}" \
    "$@"
