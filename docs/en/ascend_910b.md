# Run Wan2.2 on Ascend 910B with Public Docker Images

This guide runs offline `Wan2.2-TI2V-5B` text-to-video on Atlas 910B2 64 GB by reusing the public multi-arch container
images maintained by the `vllm-omni` and `vllm-ascend` ecosystem instead of assembling a custom CANN runtime from
scratch. [vLLM-Omni NPU container guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/npu/) ·
[vLLM-Ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)

## Requirements

| Component | Target configuration |
| --- | --- |
| Host | Linux (`arm64` or `amd64`), Atlas 910B2 64 GB, compatible Ascend driver/firmware, and Docker. Confirm idle assigned cards with `npu-smi info`. |
| Public base image | Defaults to `quay.io/ascend/vllm-omni:v0.34.0` (built on `vllm-ascend:v0.34.0` with Ubuntu 22.04, CANN 9.3.0, Python 3.12, `torch==2.10.0+cpu`, `torch_npu==2.10.0.post7`, `ffmpeg`, `libjemalloc2`, `diffusers`, and `av`). |
| Workload | Offline `Wan-AI/Wan2.2-TI2V-5B` T2V with eager BF16, PyTorch SDPA, UniPC, and model-level CPU offload on 1, 2, or 4 cards. |
| Storage and memory | Allow about 34 GB for checkpoints plus image and output space. Host RAM holds offloaded weights and worker tensor queues. |

Choose either setup option below, then keep the same **Bash** session for checkpoint preparation, device checks, and
video generation.

## Set up the container

### Option 1: Build a thin TeleFuser image on the public base (recommended)

Run from the TeleFuser repository root. `docker/Dockerfile.ascend` reuses CANN, PyTorch, `torch_npu`, and `ffmpeg` from
`quay.io/ascend/vllm-omni:v0.34.0`, installs TeleFuser into `/opt/telefuser/venv` (`--system-site-packages`), and
initializes the CANN environment in its entrypoint.

```bash
uname -m
npu-smi info
docker version

# If direct access to quay.io is slow, switch to a registry mirror prefix:
#   BASE_IMAGE=m.daocloud.io/quay.io/ascend/vllm-omni:v0.34.0
#   BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-omni:v0.34.0
BASE_IMAGE=quay.io/ascend/vllm-omni:v0.34.0
TF_IMAGE=telefuser:ascend910b

docker build -f docker/Dockerfile.ascend \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
  --build-arg "TELEFUSER_VERSION=0.0.0+g$(git rev-parse --short HEAD).local" \
  -t "$TF_IMAGE" .
```

You can also pass a compatible `quay.io/ascend/vllm-ascend:<tag>` or `quay.io/atlas-ci/vllm-ascend:<tag>` image via
`--build-arg BASE_IMAGE=...`; the Dockerfile automatically installs `ffmpeg` when missing.

### Option 2: Run directly inside `vllm-omni` without `docker build`

Pull `quay.io/ascend/vllm-omni:v0.34.0` (or one of the mirrors above), mount the local TeleFuser checkout together with
the checkpoint and output directories, and install TeleFuser editable in an isolated venv sharing the base site-packages:

```bash
export VLLM_OMNI_IMAGE=quay.io/ascend/vllm-omni:v0.34.0
TF_MODEL_DIR="$PWD/models/Wan2.2-TI2V-5B"
TF_OUTPUT_DIR=$(mktemp -d "$PWD/telefuser-results.XXXXXX")
mkdir -p "$TF_MODEL_DIR" "$TF_OUTPUT_DIR"/{1,2,4}card
docker pull "$VLLM_OMNI_IMAGE"

docker run --rm -it --shm-size=64g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
  --mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \
  --mount type=bind,src=/usr/local/Ascend/driver/lib64,dst=/usr/local/Ascend/driver/lib64,readonly \
  --mount type=bind,src=/usr/local/Ascend/driver/version.info,dst=/usr/local/Ascend/driver/version.info,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount "type=bind,src=$PWD,dst=/workspace/TeleFuser" \
  --mount "type=bind,src=$TF_MODEL_DIR,dst=/models/Wan2.2-TI2V-5B,readonly" \
  --mount "type=bind,src=$TF_OUTPUT_DIR,dst=/out" \
  -e TELEAI_EXAMPLE_OUTPUT_DIR=/out/1card -e HF_HUB_OFFLINE=1 \
  "$VLLM_OMNI_IMAGE" bash -lc '
    python3 -m venv --system-site-packages /opt/telefuser/venv &&
    source /opt/telefuser/venv/bin/activate &&
    pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu -e /workspace/TeleFuser &&
    exec bash --norc -i
  '
```

The stateless commands below use `TF_IMAGE=telefuser:ascend910b` from Option 1.

## Prepare checkpoints and outputs

```bash
TF_MODEL_DIR="$PWD/models/Wan2.2-TI2V-5B"
TF_OUTPUT_DIR=$(mktemp -d "$PWD/telefuser-results.XXXXXX")
mkdir -p "$TF_MODEL_DIR" "$TF_OUTPUT_DIR"/{1,2,4}card
```

Download the pinned Wan weights once, or set `TF_MODEL_DIR` to the absolute path of an existing complete checkpoint:

```bash
docker run --rm --entrypoint hf \
  --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
  --mount "type=bind,src=$TF_MODEL_DIR,dst=/download" \
  "$TF_IMAGE" download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir /download
```

## Check the devices

The example below maps physical devices starting at 0. Set `TF_DEVICE_IDS` to your assigned physical IDs (for example,
`TF_DEVICE_IDS=(5)` for physical card 5). Docker restricts visibility through the mounted device nodes, and CANN
numbers the visible set starting at 0 (`npu:0`). Passing host index 5 again in `ASCEND_RT_VISIBLE_DEVICES` causes ACL
error `207004`.

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
  local -a devices=()
  for ((i=0; i<count; i++)); do
    devices+=(--device "/dev/davinci${TF_DEVICE_IDS[i]}")
  done
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

## Generate and inspect a video

Run the offline text-to-video recipe with eager BF16, PyTorch SDPA, UniPC, model CPU offload, and seed 42:

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

On a host with enough free devices, use two cards for CFG 2 or four cards for CFG 2 × Ulysses 2:

```bash
generate_wan 2 && inspect_video 2
generate_wan 4 && inspect_video 4
```

Expect `width=864`, `height=480`, `r_frame_rate=24/1`, `nb_read_frames=121`, and a playable MP4 in each output folder.

If HCCL initialization reports `EJ0003`, check rank placement and other jobs using the same TCP range. Multi-rank
groups reserve 16-port blocks starting at the first aligned port at or above `HCCL_IF_BASE_PORT` (default 60000).
Containers sharing a network namespace must share their temporary lock directory (`/tmp`) or use isolated port ranges.
A worker bus error usually indicates exhausted `/dev/shm` or container RAM. For ACL import or device errors, compare
host and container `npu-smi info`, check the device/driver mounts, and rerun `check_910b`.

## Hardware validation baseline

Measured offline `Wan-AI/Wan2.2-TI2V-5B` results on Atlas 910B2 64 GB (50 UniPC steps, CFG 5, BF16, PyTorch SDPA,
model-level CPU offload, 121 frames):

| Software stack | Resolution | Parallel configuration | Generation time | Status |
| --- | --- | --- | ---: | --- |
| CANN 9.3.0-beta.1 (driver 25.2.0, firmware 7.7.0.6.236, `torch` 2.11.0+cpu, `torch_npu` 2.11.0.post4) | 480p (864 × 480) | 1 card | 157.07 s | Passed |
| Same stack | 480p (864 × 480) | 2 cards (CFG 2) | 92.05 s | Passed; all 121 decoded frames matched 1-card pixel-for-pixel |
| CANN 8.2 (`torch` 2.9.0, `torch_npu` 2.9.0.post1) | 480p (864 × 480) | 1 / 2 / 4 cards (CFG 2 × Ulysses 2) | 180.9 / 120.8 / 87.0 s | Passed |
| Same stack | 720p | 1 / 4 cards | 477.8 / 173.9 s | Passed |

Separate two-rank hardware checks also verified HCCL all-reduce, default NPU mesh selection, pipeline send/receive with
implicit receive buffers, and host-staged NPU tensor channel sharding and reassembly. For head-to-head latency and
multi-card scaling comparisons against `vLLM-Omni`, see [Ascend NPU Benchmarks](benchmark_ascend_npu.md).
