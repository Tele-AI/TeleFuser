#!/usr/bin/env bash
# Run the public 30-minute ABot lifecycle trace on four physical GPUs.
#
# Default: physical GPUs 4,5,6,7 -> logical worker GPUs 0,1,2,3.
# Override, for example:
#   GPU_IDS=0,1,2,3 CUDA_GRAPH_ENABLED=0 \
#     tools/validation/run_abot_4gpu_30min_trace.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/public/fanyk1/lwb/envs/telefuser_sage291/bin/python}"
MODEL_ZOO_PATH="${TF_MODEL_ZOO_PATH:-/public/fanyk1/lwb/model_zoo}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
PORT="${PORT:-8088}"
LIVEKIT_URL="${LIVEKIT_URL:-ws://127.0.0.1:7880}"
LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
SCENARIO="${SCENARIO:-tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16.json}"
TRACE_DURATION_SECONDS="${TRACE_DURATION_SECONDS:-1800}"
METRICS_DURATION_SECONDS="${METRICS_DURATION_SECONDS:-1860}"
CUDA_GRAPH_ENABLED="${CUDA_GRAPH_ENABLED:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-3}"
MAX_DEADLINE_WAIT_MS="${MAX_DEADLINE_WAIT_MS:-1000}"
FRAME_CREDIT_TARGET_FRAMES="${FRAME_CREDIT_TARGET_FRAMES:-36}"
BATCH_SAFETY_FACTOR="${BATCH_SAFETY_FACTOR:-1.05}"
RUN="${RUN:-results/experiments/abot_4gpu_lf3_12fps_publicdemo_b${MAX_BATCH_SIZE}_f${FRAME_CREDIT_TARGET_FRAMES}_30min_$(date -u +%Y%m%dT%H%M%SZ)}"

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated physical GPU IDs; got: ${GPU_IDS}" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID: ${gpu}" >&2
    exit 2
  fi
  owners="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits | sed '/^$/d')"
  if [[ -n "${owners}" ]]; then
    echo "Refusing to share physical GPU ${gpu}; active compute process(es):" >&2
    echo "${owners}" >&2
    exit 1
  fi
done

if ss -ltn "( sport = :${PORT} )" | tail -n +2 | grep -q .; then
  echo "Refusing to start: TCP port ${PORT} is already listening." >&2
  exit 1
fi
if [[ ! -f "${SCENARIO}" ]]; then
  echo "Scenario not found: ${SCENARIO}" >&2
  exit 2
fi
if [[ -e "${RUN}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN}" >&2
  exit 2
fi

mkdir -p "${RUN}"
cp "${SCENARIO}" "${RUN}/scenario.json"
printf '%s\n' \
  "GPU_IDS=${GPU_IDS}" \
  "CUDA_GRAPH_ENABLED=${CUDA_GRAPH_ENABLED}" \
  "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}" \
  "MAX_DEADLINE_WAIT_MS=${MAX_DEADLINE_WAIT_MS}" \
  "FRAME_CREDIT_TARGET_FRAMES=${FRAME_CREDIT_TARGET_FRAMES}" \
  "BATCH_SAFETY_FACTOR=${BATCH_SAFETY_FACTOR}" \
  "SCENARIO=${SCENARIO}" \
  >"${RUN}/run-config.env"

SERVER_PID=""
SERVING_METRICS_PID=""
GPU_METRICS_PID=""
cleanup() {
  local status=$?
  for pid in "${SERVING_METRICS_PID}" "${GPU_METRICS_PID}" "${SERVER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait "${SERVING_METRICS_PID}" 2>/dev/null || true
  wait "${GPU_METRICS_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "Starting 4-GPU service on physical GPUs ${GPU_IDS}; artifacts: ${RUN}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
PYTHONPATH="${REPO_ROOT}" \
TF_MODEL_ZOO_PATH="${MODEL_ZOO_PATH}" \
TELEFUSER_ABOT_CUDA_GRAPH_ENABLED="${CUDA_GRAPH_ENABLED}" \
TELEFUSER_ABOT_SCHEDULER_MODE=batched \
TELEFUSER_ABOT_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
TELEFUSER_ABOT_BATCHING_WINDOW_MS=2 \
TELEFUSER_ABOT_MAX_DEADLINE_BATCH_WAIT_MS="${MAX_DEADLINE_WAIT_MS}" \
TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_ENABLED=1 \
TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS=3.0 \
TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES="${FRAME_CREDIT_TARGET_FRAMES}" \
TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES=4 \
TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_GUARD_MS=50 \
TELEFUSER_ABOT_BATCH_COMPUTE_PROFILE=h100_lf3_eager_full_pipeline_v1 \
TELEFUSER_ABOT_BATCH_COMPUTE_SAFETY_FACTOR="${BATCH_SAFETY_FACTOR}" \
TELEFUSER_LIVEKIT_DISPATCH_TRACE_PATH="${REPO_ROOT}/${RUN}/dispatch-trace.jsonl" \
TELEFUSER_LIVEKIT_DISPATCH_TRACE_MAX_EVENTS=100000 \
"${PYTHON_BIN}" -m telefuser.entrypoints.cli.main stream-serve \
  examples/abot_world/abot_world_livekit_service.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --livekit-url "${LIVEKIT_URL}" \
  --livekit-api-key "${LIVEKIT_API_KEY}" \
  --livekit-api-secret "${LIVEKIT_API_SECRET}" \
  --num-workers 4 \
  --worker-gpu-map '0;1;2;3' \
  --worker-mode process-nccl \
  --max-sessions-per-worker 4 \
  --queue-size 0 \
  --skip-validation \
  >"${RUN}/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${RUN}/server.pid"

for _ in $(seq 1 240); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/service/ready" >"${RUN}/ready.json"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before becoming ready; see ${RUN}/server.log" >&2
    exit 1
  fi
  sleep 1
done
test -s "${RUN}/ready.json"
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/service/metadata" >"${RUN}/metadata-before.json"

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/capture_abot_serving_metrics.py \
  --server-url "http://127.0.0.1:${PORT}" \
  --duration "${METRICS_DURATION_SECONDS}" \
  --interval 1 \
  --output-dir "${RUN}/serving_metrics" \
  >"${RUN}/serving-metrics.log" 2>&1 &
SERVING_METRICS_PID=$!

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/capture_gpu_nvml_metrics.py \
  --gpu-indices "${GPU_IDS}" \
  --duration "${METRICS_DURATION_SECONDS}" \
  --interval 1 \
  --output-dir "${RUN}/gpu_metrics" \
  >"${RUN}/gpu-metrics.log" 2>&1 &
GPU_METRICS_PID=$!

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/replay_abot_livekit_lifecycle_trace.py \
  --scenario "${RUN}/scenario.json" \
  --output "${RUN}/result.json" \
  2>&1 | tee "${RUN}/replay.log"

wait "${SERVING_METRICS_PID}"
SERVING_METRICS_PID=""
wait "${GPU_METRICS_PID}"
GPU_METRICS_PID=""

curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/service/metadata" >"${RUN}/metadata-after.json"

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/render_abot_dispatch_timeline.py \
  --dispatch-trace "${RUN}/dispatch-trace.jsonl" \
  --result "${RUN}/result.json" \
  --output-dir "${RUN}/dispatch_analysis"

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/analyze_abot_serving_trace.py \
  --serving-metrics-dir "${RUN}/serving_metrics" \
  --gpu-metrics "${RUN}/gpu_metrics/gpu-metrics.jsonl" \
  --output-dir "${RUN}/serving_analysis"

echo "Completed: ${RUN}"
