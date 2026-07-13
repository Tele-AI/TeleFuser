#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

SHIM_DIR="${ROOT_DIR}/benchmarks/baseline/sglang_lingbot_stream/python_shims"
PYTHONPATH_PREFIX="${SHIM_DIR}"
if [[ -n "${SGLANG_EXTRA_PYTHONPATH:-}" ]]; then
    PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX}:${SGLANG_EXTRA_PYTHONPATH}"
fi
export PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}"

SGLANG_BIN="${SGLANG_BIN:-sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-}"
SERVICE_PORT="${SGLANG_LINGBOT_PORT:-30000}"
MODEL_PATH="${SGLANG_LINGBOT_MODEL_PATH:-robbyant/lingbot-world-fast-diffusers}"
MODEL_ID="${SGLANG_LINGBOT_MODEL_ID:-lingbot-world-fast-diffusers}"
MODEL_TYPE="${SGLANG_LINGBOT_MODEL_TYPE:-diffusion}"
PIPELINE_CLASS="${SGLANG_LINGBOT_PIPELINE_CLASS:-LingBotWorldCausalDMDPipeline}"
PERFORMANCE_MODE="${SGLANG_LINGBOT_PERFORMANCE_MODE:-speed}"
NUM_GPUS="${SGLANG_LINGBOT_NUM_GPUS:-4}"
ULYSSES_DEGREE="${SGLANG_LINGBOT_ULYSSES_DEGREE:-4}"
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
    --port "${SERVICE_PORT}" \
    --num-gpus "${NUM_GPUS}" \
    --ulysses-degree "${ULYSSES_DEGREE}" \
    --dit-cpu-offload "${DIT_CPU_OFFLOAD}" \
    --text-encoder-cpu-offload "${TEXT_ENCODER_CPU_OFFLOAD}" \
    "$@"
