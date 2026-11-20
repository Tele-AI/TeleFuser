# Run Wan2.2 on Ascend 910B with Docker

## Build the image

From the TeleFuser checkout containing `docker/Dockerfile.ascend`, run the following commands in one Bash session on a Linux arm64 Atlas 910B2 host. Check that the assigned cards are idle and that the host driver and firmware match CANN. The container supplies Ubuntu 22.04, Python 3.11, CANN 9.3.0-beta.1, PyTorch 2.11.0+cpu and torch_npu 2.11.0.post4. [Ascend release notes and wheels](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/v26.3.0-beta.1-pytorch2.11.0)

```bash
uname -m
npu-smi info
docker version
TF_IMAGE=telefuser:ascend910b-local
docker build --platform linux/arm64 -f docker/Dockerfile.ascend \
  --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
  --build-arg "TELEFUSER_VERSION=0.0.0+g$(git rev-parse --short HEAD).local" \
  -t "$TF_IMAGE" .
```

The build defaults to AscendHub in China. For Docker Hub add `--build-arg CANN_REPOSITORY=docker.io/ascendai/cann`. Both registries serve the same arm64 manifest.

After a qualified image is published, you can install it directly using the TeleFuser digest from its release page. The intended release tag is `ghcr.io/tele-ai/telefuser:<release>-ascend910b`.

```bash
read -r -p 'TeleFuser image digest from release page > ' TF_IMAGE
docker pull "$TF_IMAGE"
```

## Prepare checkpoints and outputs

Allow about 34 GB for the model in addition to the image. Model-level CPU offload and host-staged worker queues also use host memory.

```bash
TF_MODEL_DIR="$PWD/models/Wan2.2-TI2V-5B"
TF_OUTPUT_DIR=$(mktemp -d "$PWD/telefuser-results.XXXXXX")
mkdir -p "$TF_MODEL_DIR" "$TF_OUTPUT_DIR"/{1,2,4}card
```

Download the pinned Wan weights once, or set `TF_MODEL_DIR` to the absolute path of an existing complete checkpoint.

```bash
docker run --rm --entrypoint hf \
  --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
  --mount "type=bind,src=$TF_MODEL_DIR,dst=/download" \
  "$TF_IMAGE" download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir /download
```

## Check the devices

The example below maps physical devices starting at 0. Set `TF_DEVICE_IDS` to your assigned physical IDs. For example, use `TF_DEVICE_IDS=(5)` for physical card 5. Workers use container-local indices starting at 0. Confirm each visible device can execute a tensor operation. The weight mount is read-only.

```bash
TF_DEVICE_IDS=(0 1 2 3)

run_910b() {
  local count=$1 i
  shift
  case "$count" in
    1|2|4) ;;
    *) echo 'Choose 1, 2 or 4 cards' >&2; return 2 ;;
  esac
  (( count <= ${#TF_DEVICE_IDS[@]} )) || { echo "Assigned devices: ${TF_DEVICE_IDS[*]}" >&2; return 2; }
  local -a devices=() ids=()
  for ((i=0; i<count; i++)); do
    devices+=(--device "/dev/davinci${TF_DEVICE_IDS[i]}")
    ids+=("${TF_DEVICE_IDS[i]}")
  done
  local IFS=,
  docker run --rm -i --shm-size=64g \
    "${devices[@]}" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
    --mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \
    --mount type=bind,src=/usr/local/Ascend/driver/lib64,dst=/usr/local/Ascend/driver/lib64,readonly \
    --mount type=bind,src=/usr/local/Ascend/driver/version.info,dst=/usr/local/Ascend/driver/version.info,readonly \
    --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
    --mount "type=bind,src=$TF_MODEL_DIR,dst=/models/Wan2.2-TI2V-5B,readonly" \
    --mount "type=bind,src=$TF_OUTPUT_DIR,dst=/out" \
    -e "ASCEND_RT_VISIBLE_DEVICES=${ids[*]}" \
    -e "TELEAI_EXAMPLE_OUTPUT_DIR=/out/${count}card" -e HF_HUB_OFFLINE=1 \
    "$TF_IMAGE" "$@"
}

check_910b() {
  run_910b "$1" python - "$1" <<'PY'
import sys
from importlib.metadata import version

import torch
import torch_npu
from telefuser.platforms import current_platform

expected = int(sys.argv[1])
assert current_platform.device_type == "npu", current_platform.device_type
assert current_platform.dist_backend == "hccl", current_platform.dist_backend
assert torch.npu.is_available(), "Ascend runtime unavailable"
actual = torch.npu.device_count()
assert actual == expected, (actual, expected)
print("telefuser", version("telefuser"), "torch", torch.__version__, "torch_npu", version("torch-npu"))
for i in range(actual):
    value = torch.tensor([1, 2], dtype=torch.float32, device=f"npu:{i}")
    assert value.square().sum().item() == 5
    print(f"npu:{i}", torch.npu.get_device_name(i), "OK")
PY
}

run_910b 1 npu-smi info
check_910b 1
```

## Generate a video

Run the single-card offline text-to-video recipe with eager BF16, PyTorch SDPA, UniPC, model CPU offload and seed 42.

```bash
generate_wan() {
  local count=$1
  check_910b "$count" || return 1
  run_910b "$count" python /opt/telefuser/examples/wan_video/wan22_t2v_5b.py \
    --model_root /models/Wan2.2-TI2V-5B \
    --gpu_num "$count" --resolution 480p --aspect_ratio 16:9 --num_frames 121 \
    --prompt 'A red kite flying over a green field'
}

inspect_video() {
  local count=$1
  docker run --rm --entrypoint ffprobe \
    --mount "type=bind,src=$TF_OUTPUT_DIR,dst=/out,readonly" \
    "$TF_IMAGE" -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,r_frame_rate,nb_read_frames \
    -of default=noprint_wrappers=1 \
    "/out/${count}card/wan_video_wan22_t2v_5b.mp4"
}

generate_wan 1 && inspect_video 1
```

On a host with enough free devices, use two cards for CFG 2 or four cards for CFG 2 × Ulysses 2.

```bash
generate_wan 2 && inspect_video 2
generate_wan 4 && inspect_video 4
```

Expect 864 × 480, 24 fps, 121 decoded frames and an MP4 in each selected output directory. Retain the source revision, image ID or digest, host driver/firmware, device IDs, model revision, timing and decoded video properties with each result.

If HCCL initialization reports EJ0003, check rank placement and other jobs using the same TCP range. Multi-rank groups reserve blocks of 16, starting at the first aligned port at or above `HCCL_IF_BASE_PORT` (default 60000). Processes sharing a network namespace must also share their temporary lock directory (normally `/tmp`) or have isolated allocations. A worker bus error can indicate exhausted shared memory. For an ACL import or device error, compare host and container `npu-smi info`, check the device/driver mounts, and rerun the tensor check.

PR #42 reports results for 1, 2 and 4 cards on CANN 8.2. The 9.3.0-beta.1 image awaits hardware qualification for each card count. [Previous hardware results](https://github.com/Tele-AI/TeleFuser/pull/42)
