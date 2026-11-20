# 使用镜像在 Ascend 910B 上运行 Wan2.2 文生视频

本手册面向 Linux arm64 的 Atlas 910B2 64 GB。镜像将 CANN 9.3.0-beta.1 与昇腾同次发布的 TorchNPU 配套。该候选镜像已通过单卡与双卡 Wan 实机验收。可加载候选镜像归档，或从包含 `docker/Dockerfile.ascend` 的 checkout 构建。四卡仍待验收。[昇腾 CANN](https://www.hiascend.com/cann/download?versionId=822&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=0) · [TorchNPU 发布与附件](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/v26.3.0-beta.1-pytorch2.11.0)

## 运行条件

| 项目 | 本手册的目标 |
| --- | --- |
| 宿主机 | Linux arm64，Atlas 910B2 64 GB，已安装该 CANN 版本配套的昇腾驱动、固件及 Docker。`npu-smi info` 能看到分配给作业的空闲卡。 |
| 容器 | Ubuntu 22.04、Python 3.11、CANN 9.3.0-beta.1、Torch 2.11.0+cpu、torch_npu 2.11.0.post4。候选镜像支持本地构建。 |
| 工作负载 | `Wan-AI/Wan2.2-TI2V-5B` 的离线 T2V，BF16、PyTorch SDPA、UniPC、模型级 CPU offload。单机 1 卡和 2 卡已通过，4 卡待验收。 |
| 本地文件 | 官方模型目录约 34 GB，另预留镜像和输出空间。主存需容纳 offload 权重与 worker 队列。模型目录以只读方式传入，输出目录应对容器运行用户可写。 |

以下安装方式任选其一。准备权重、设备自检与生成使用同一个 **Bash** 会话。设备映射示例假定宿主分配的物理设备从 0 开始。设备节点、驱动库和 `npu-smi` 的宿主路径参照昇腾标准安装。[昇腾软件与固件下载](https://www.hiascend.com/developer/download) · [上游容器挂载示例](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/npu/)

## 安装镜像与准备权重

### 从离线归档安装

拿到已验收的镜像归档和同名校验文件时，在目标机的同一个目录执行下列命令，然后直接准备权重。Docker image ID 应与本页末尾的实机记录一致。该归档为 6,137,716,301 字节，SHA-256 为 `8a85baafb991f8640af6f3dba9f38ac8a76376ca669e7e64363c7eee8e9f2369`。

```bash
sha256sum -c telefuser-ascend910b-c27fabd.tar.gz.sha256
docker load -i telefuser-ascend910b-c27fabd.tar.gz
TF_IMAGE=telefuser:ascend910b-c27fabd
docker image inspect --format '{{.Id}}' "$TF_IMAGE"
```

### 从源码构建

在含本次改动的 TeleFuser checkout 根目录执行。Dockerfile 固定 910B 的 arm64 基础镜像，逐个校验 Python wheel 并安装同一份源码的示例。权重独立挂载。使用 Bash 执行下列命令。

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

默认从昇腾境内仓下载 CANN 底座。使用 Docker Hub 时，在构建命令中加 `--build-arg CANN_REPOSITORY=docker.io/ascendai/cann`，两处清单的 arm64 摘要相同。

### 从发布仓拉取

发布后，可使用发布页给出的完整 TeleFuser 镜像引用。镜像将使用 `ghcr.io/tele-ai/telefuser:<release>-ascend910b`，按 digest 拉取可复现同一批验收结果。

```bash
read -r -p '粘贴发布页的 TeleFuser 镜像完整引用 > ' TF_IMAGE
docker pull "$TF_IMAGE"
```

### 准备权重与输出目录

```bash
TF_MODEL_DIR="$PWD/models/Wan2.2-TI2V-5B"
TF_OUTPUT_DIR=$(mktemp -d "$PWD/telefuser-results.XXXXXX")
mkdir -p "$TF_MODEL_DIR" "$TF_OUTPUT_DIR"/{1,2,4}card
```

已有完整权重时，将 `TF_MODEL_DIR` 设为已有目录的绝对路径。首次下载可用镜像内的 `hf` 命令，下载这一步无需挂卡。

```bash
docker run --rm --entrypoint hf \
  --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
  --mount "type=bind,src=$TF_MODEL_DIR,dst=/download" \
  "$TF_IMAGE" download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir /download
```

`--model_root` 指向同时含有三个 DiT `.safetensors` 分片、`Wan2.2_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth` 和 `google/umt5-xxl/` tokenizer 的目录。权重就绪后，后续步骤使用本地文件。[固定版本的官方模型目录](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e)

## 挂载设备并自检

以下步骤共用同一个 Bash 会话。下面的 `run_910b` 是当前 shell 的辅助函数。单卡传入 `davinci0`，双卡传入 `davinci0–1`，四卡传入 `davinci0–3`。把 `TF_DEVICE_IDS` 改成宿主分配的物理卡号，例如单用物理 5 号卡时设为 `TF_DEVICE_IDS=(5)`。Docker 通过设备节点限制可见范围，CANN 再从 0 编号。只挂载物理 5 号卡时，应用使用 `npu:0`。此时把宿主编号 5 写入 `ASCEND_RT_VISIBLE_DEVICES` 会触发 `207004`。`--shm-size=64g` 是此多卡配方的共享内存上限，供 worker 经主存交换张量。

```bash
TF_DEVICE_IDS=(0 1 2 3)

run_910b() {
  local count=$1 i
  shift
  case "$count" in
    1|2|4) ;;
    *) echo '卡数取 1、2 或 4' >&2; return 2 ;;
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

应看到期望数量的逻辑卡，每张卡的实际运算都返回 `OK`。自检覆盖设备与同步取回结果，多卡之间的 HCCL 通信在下一步的并行生成中验证。

## 生成和检查视频

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

有足够空闲设备时，继续执行对应行。示例自行启动 workers，双卡使用 CFG 2，四卡使用 CFG 2 × Ulysses 2。

```bash
generate_wan 2 && inspect_video 2
generate_wan 4 && inspect_video 4
```

日志应包含 `Video generation time` 和 `Video saved to`。对于上述 16:9、480p 配方，尺寸函数按 32 对齐，预期 `width=864`、`height=480`、`r_frame_rate=24/1`、`nb_read_frames=121`。在宿主 `$TF_OUTPUT_DIR` 的 `1card`、`2card`、`4card` 下分别查看同名 MP4。可播放视频、卡数正确和 worker 正常退出共同构成一次完整运行。[尺寸计算](https://github.com/Tele-AI/TeleFuser/blob/9c36c159e140c4831115a9689d4c139f8e699bf8/telefuser/utils/video.py#L38-L78) · [输出命名](https://github.com/Tele-AI/TeleFuser/blob/9c36c159e140c4831115a9689d4c139f8e699bf8/telefuser/utils/utils.py#L45-L49)

## 按现象排查

| 现象 | 先检查 |
| --- | --- |
| 宿主 `npu-smi info` 失败 | 检查宿主驱动、固件和设备状态，使用与该版本发布页匹配的组合。 |
| 宿主能看到卡，容器 `import torch_npu` 或小张量运算失败 | 对照该镜像 digest 核对驱动 `lib64`、管理节点、设备权限和 CANN 环境初始化。保留原始 import / ACL 错误。 |
| 卡数与 `--gpu_num` 不符 | 核对分配的 `/dev/davinciN`，清除从宿主带入的 `ASCEND_RT_VISIBLE_DEVICES`，在同一组挂载上执行 `check_910b`。并行 rank 使用容器可见的逻辑索引。 |
| 单卡成功，多卡卡在初始化，日志出现 `EJ0003` | 检查 rank、设备映射和连续 16 个 TCP 端口。TeleFuser 将 `HCCL_IF_BASE_PORT`（默认 60000）作为搜索下限，按 16 对齐。共用网络命名空间的容器需共享临时锁目录（默认 `/tmp`），或由调度器隔离网络并预留互不重叠的范围。`MASTER_PORT` 与 HCCL host 端口段分别用于 rendezvous 和 HCCL 初始化。 |
| worker 出现 `Bus error` 或 `/dev/shm` 空间不足 | 运行 `run_910b 4 df -h /dev/shm`（按实际卡数调整），同时检查容器内存上限、宿主可用 RAM 和 CPU offload。 |
| 找不到权重或 tokenizer | 检查 `TF_MODEL_DIR` 的层级和模型 revision，目录根应直接含有 VAE、T5 和分片文件。 |

## 验收记录

每个卡数单独记录 TeleFuser 源码提交、镜像 ID 或仓库摘要、CANN、Torch、TorchNPU、驱动/固件、权重 revision、卡号、并行度、生成时间和 `ffprobe` 输出。本次候选源码为 `c27fabdcdf6f1156126e1bc22699f1af4580c97e`，910B2 上实测的 Docker image ID 为 `sha256:8c49e26c804c7691725506086aedf7bcf1770afdcd73b90947d243128ab058da`。宿主驱动为 25.2.0，固件为 7.7.0.6.236。

| CANN 9.3.0-beta.1 | 并行配置 | 生成时间 | 结果 |
| --- | --- | ---: | --- |
| 1 卡 | 单卡，物理 5 | 157.07 秒 | 通过 |
| 2 卡 | CFG 2，物理 2、3 | 92.05 秒 | 通过，两个 worker 正常退出 |
| 4 卡 | CFG 2 × Ulysses 2 | — | 待验收 |

2026-11-20 用本页红风筝提示词、seed 42 和 50 步 UniPC 在同一节点分卡同时运行。生成时间取自示例的 `Video generation time`。两次 VAE 输出在转为 RGB8 前均为有限值，`ffprobe` 均读出 864 × 480、24 fps、121 帧。两个 MP4 的 SHA-256 都是 `d0ad0b5452024e2a6373c6f670f4bad8e7e520a292720843d0d8eaac71c0895d`，独立解码后 121 帧逐像素相同。双卡还单独验证了 HCCL all-reduce、默认 NPU mesh、PP 发送和缺省接收缓冲区，以及真实 NPU 张量经主存分片并在目标设备重组。

公开 PR 的作者在 **CANN 8.2**、Torch 2.9.0、torch_npu 2.9.0.post1 上报告了 1 / 2 / 4 卡生成。[#42 的实机报告](https://github.com/Tele-AI/TeleFuser/pull/42)
