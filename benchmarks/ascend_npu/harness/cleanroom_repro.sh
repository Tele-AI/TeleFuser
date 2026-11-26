#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Gate G4: Execute a clean-room reproduction script with stripped env and zero source-tree mounts.
set -euo pipefail

IMAGE_REF=""
MODEL_DIR=""
DEVICES_CSV="0"
ALLOWED_CSV=""
RUN_DIR=""
ENTRYPOINT_SCRIPT="/opt/telefuser/examples/wan_video/wan22_t2v_5b.py"
EXPECTED_MP4_GLOB="wan_video_wan22_t2v_5b*.mp4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE_REF="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --devices) DEVICES_CSV="$2"; shift 2 ;;
    --allowed-devices) ALLOWED_CSV="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --script) ENTRYPOINT_SCRIPT="$2"; shift 2 ;;
    --mp4-glob) EXPECTED_MP4_GLOB="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$IMAGE_REF" || -z "$MODEL_DIR" || -z "$RUN_DIR" ]]; then
  echo "Usage: $0 --image <tag_or_id> --model-dir <abs_path> --devices <0,1,...> --run-dir <dir>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_ARGS=(--devices "$DEVICES_CSV")
[[ -n "$ALLOWED_CSV" ]] && PREFLIGHT_ARGS+=(--allowed-devices "$ALLOWED_CSV")
bash "$SCRIPT_DIR/preflight_check.sh" "${PREFLIGHT_ARGS[@]}" > /dev/null

mkdir -p "$RUN_DIR"
REPRO_SH="$RUN_DIR/REPRODUCE.sh"

cat > "$REPRO_SH" <<EOF
#!/usr/bin/env bash
# Standalone Clean-Room Reproduction Script (Zero Source Mounts)
set -euo pipefail

DEVICES="\${DEVICES:-$DEVICES_CSV}"
TF_IMAGE="\${TF_IMAGE:-$IMAGE_REF}"
TF_MODEL_DIR="\${TF_MODEL_DIR:-$MODEL_DIR}"
OUT_DIR="\${OUT_DIR:-\$(mktemp -d /tmp/telefuser-repro.XXXXXX)}"

IFS=',' read -r -a DEV_IDS <<< "\$DEVICES"
DEV_ARGS=()
for id in "\${DEV_IDS[@]}"; do
  DEV_ARGS+=(--device "/dev/davinci\${id}")
done
CARD_COUNT="\${#DEV_IDS[@]}"

mkdir -p "\$OUT_DIR"
docker run --rm --shm-size=64g \\
  "\${DEV_ARGS[@]}" \\
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \\
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \\
  --mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \\
  --mount type=bind,src=/usr/local/Ascend/driver/lib64,dst=/usr/local/Ascend/driver/lib64,readonly \\
  --mount type=bind,src=/usr/local/Ascend/driver/version.info,dst=/usr/local/Ascend/driver/version.info,readonly \\
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \\
  --mount "type=bind,src=\$TF_MODEL_DIR,dst=/models/checkpoint,readonly" \\
  --mount "type=bind,src=\$OUT_DIR,dst=/out" \\
  -e TELEAI_EXAMPLE_OUTPUT_DIR=/out -e HF_HUB_OFFLINE=1 \\
  "\$TF_IMAGE" python "$ENTRYPOINT_SCRIPT" \\
    --model_root /models/checkpoint \\
    --gpu_num "\$CARD_COUNT" \\
    --resolution 480p --aspect_ratio 16:9 --num_frames 121 \\
    --prompt 'A red kite flying over a green field'

MP4_FILE="\$(ls -1 "\$OUT_DIR"/$EXPECTED_MP4_GLOB | head -n 1)"
docker run --rm --entrypoint ffprobe \\
  --mount "type=bind,src=\$OUT_DIR,dst=/out,readonly" \\
  "\$TF_IMAGE" -v error -select_streams v:0 -count_frames \\
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \\
  -of default=noprint_wrappers=1 "/out/\$(basename "\$MP4_FILE")"
echo "CLEANROOM_PASS: \$MP4_FILE"
EOF
chmod +x "$REPRO_SH"

CLEAN_OUT="$RUN_DIR/cleanroom_${DEVICES_CSV//,/_}card"
rm -rf "$CLEAN_OUT"
mkdir -p "$CLEAN_OUT"

env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="/tmp" \
  OUT_DIR="$CLEAN_OUT" DEVICES="$DEVICES_CSV" TF_IMAGE="$IMAGE_REF" TF_MODEL_DIR="$MODEL_DIR" \
  bash "$REPRO_SH" | tee "$RUN_DIR/g4_cleanroom_${DEVICES_CSV//,/_}card.log"
