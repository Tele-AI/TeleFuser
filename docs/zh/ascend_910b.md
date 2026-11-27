# 使用公共镜像在 Ascend 910B 上运行 Wan2.2 文生视频

本手册面向 Atlas 910B2 64 GB（推荐 Linux arm64 或 amd64），直接复用昇腾开源生态维护的 `vllm-omni` / `vllm-ascend` 公共多架构镜像运行 `Wan2.2-TI2V-5B` 离线文生视频，无需从零组装 CANN 与多媒体环境。[vLLM-Omni NPU 镜像文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/npu/) · [vLLM-Ascend 安装指南](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)

## 运行条件

| 项目 | 本手册的目标 |
| --- | --- |
| 宿主机 | Linux（arm64 / amd64），Atlas 910B2 64 GB，已安装与容器 CANN 版本兼容的昇腾驱动、固件及 Docker。`npu-smi info` 能看到分配给作业的空闲卡。 |
| 公共底座 | 默认复用 `quay.io/ascend/vllm-omni:v0.34.0`（基于 `vllm-ascend:v0.34.0`，含 Ubuntu 22.04、CANN 9.3.0、Python 3.12、Torch 2.10.0+cpu、`torch_npu` 2.10.0.post7、`ffmpeg`、`libjemalloc2`、`diffusers` 与 `av`）。 |
| 工作负载 | `Wan-AI/Wan2.2-TI2V-5B` 离线 T2V，BF16、PyTorch SDPA、UniPC、模型级 CPU offload，支持单机 1、2、4 卡运行。 |
| 本地文件 | 官方模型目录约 34 GB，另预留镜像和输出空间。主存需容纳 offload 权重与 worker 队列。模型目录以只读方式传入，输出目录应对容器运行用户可写。 |

以下两种准备方式任选其一。准备权重、设备自检与生成使用同一个 **Bash** 会话。设备映射示例假定宿主分配的物理设备从 0 开始，设备节点、驱动库和 `npu-smi` 的宿主路径参照昇腾标准安装。[昇腾软件与固件下载](https://www.hiascend.com/developer/download)

## 准备镜像与模型权重

### 方式一：基于公共底座构建轻量 TeleFuser 镜像（推荐批处理运行）

在 TeleFuser 仓库根目录执行。`docker/Dockerfile.ascend` 直接复用 `quay.io/ascend/vllm-omni:v0.34.0` 内的 CANN、PyTorch、`torch_npu` 与 `ffmpeg`，仅在独立虚拟环境 `/opt/telefuser/venv`（`--system-site-packages`）中增量安装当前源码与少量轻量依赖，并自动在入口加载 CANN 环境。

```bash
uname -m
npu-smi info
docker version

# 直连 Quay 较慢时，可改设为国内镜像前缀：
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

若宿主机已缓存 `quay.io/ascend/vllm-ascend:<tag>` 或 `quay.io/atlas-ci/vllm-ascend:<tag>`，也可通过 `--build-arg BASE_IMAGE=...` 直接传入；Dockerfile 会在缺少 `ffmpeg` 时自动补装。

### 方式二：免构建直接进入 `vllm-omni` 公共容器运行

不希望执行 `docker build` 时，可直接拉取 `quay.io/ascend/vllm-omni:v0.34.0`（或上述国内镜像源地址），挂载本地 TeleFuser 源码目录、权重目录与输出目录进入容器安装可编辑包后直接运行：

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

下文命令默认以方式一构建出的 `TF_IMAGE=telefuser:ascend910b` 演示无状态单卡与多卡执行。

### 准备权重与输出目录

```bash
TF_MODEL_DIR="$PWD/models/Wan2.2-TI2V-5B"
TF_OUTPUT_DIR=$(mktemp -d "$PWD/telefuser-results.XXXXXX")
mkdir -p "$TF_MODEL_DIR" "$TF_OUTPUT_DIR"/{1,2,4}card
```

已有完整权重时，将 `TF_MODEL_DIR` 设为已有目录的绝对路径。首次下载可用镜像内的 `hf` 命令，下载这一步无需挂卡：

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

以下步骤共用同一个 Bash 会话。下面的 `run_910b` 是当前 shell 的辅助函数。单卡传入 1 张设备，双卡传入 2 张，四卡传入 4 张。把 `TF_DEVICE_IDS` 改成宿主分配的物理卡号，例如单用物理 5 号卡时设为 `TF_DEVICE_IDS=(5)`。Docker 通过设备节点限制可见范围，CANN 在容器内从 0 重新编号；只挂载物理 5 号卡时应用使用 `npu:0`，此时若再把宿主编号 5 写入 `ASCEND_RT_VISIBLE_DEVICES` 会触发 ACL 错误 `207004`。`--shm-size=64g` 供多卡 worker 经主存交换张量。

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

有足够空闲设备时，继续执行对应行。示例自行启动 workers，双卡使用 CFG 2，四卡使用 CFG 2 × Ulysses 2：

```bash
generate_wan 2 && inspect_video 2
generate_wan 4 && inspect_video 4
```

日志应包含 `Video generation time` 和 `Video saved to`。对于上述 16:9、480p 配方，尺寸函数按 32 对齐，预期 `width=864`、`height=480`、`r_frame_rate=24/1`、`nb_read_frames=121`。在宿主 `$TF_OUTPUT_DIR` 的 `1card`、`2card`、`4card` 下分别查看同名 MP4。可播放视频、卡数正确和 worker 正常退出共同构成一次完整运行。[尺寸计算](https://github.com/Tele-AI/TeleFuser/blob/main/telefuser/utils/video.py) · [输出命名](https://github.com/Tele-AI/TeleFuser/blob/main/telefuser/utils/utils.py)

## 按现象排查

| 现象 | 先检查 |
| --- | --- |
| 宿主 `npu-smi info` 失败 | 检查宿主驱动、固件和设备状态，确保与容器内 CANN 版本兼容。 |
| 宿主能看到卡，容器 `import torch_npu` 或小张量运算失败 | 核对驱动 `lib64`、管理节点、设备权限和 CANN `set_env.sh` 环境初始化，保留原始 import / ACL 错误信息。 |
| 卡数与 `--gpu_num` 不符 | 核对分配的 `/dev/davinciN`，清除从宿主带入的 `ASCEND_RT_VISIBLE_DEVICES`，在同一组挂载上执行 `check_910b`。并行 rank 使用容器可见的逻辑索引。 |
| 单卡成功，多卡卡在初始化，日志出现 `EJ0003` | 检查 rank、设备映射和连续 16 个 TCP 端口。TeleFuser 将 `HCCL_IF_BASE_PORT`（默认 60000）作为搜索下限，按 16 对齐。共用网络命名空间的容器需共享临时锁目录（默认 `/tmp`），或由调度器隔离网络并预留互不重叠的范围。`MASTER_PORT` 与 HCCL host 端口段分别用于 rendezvous 和 HCCL 初始化。 |
| worker 出现 `Bus error` 或 `/dev/shm` 空间不足 | 运行 `run_910b 4 df -h /dev/shm`（按实际卡数调整），同时检查容器内存上限、宿主可用 RAM 和 CPU offload。 |
| 找不到权重或 tokenizer | 检查 `TF_MODEL_DIR` 的层级和模型 revision，目录根应直接含有 VAE、T5 和分片文件。 |

## 硬件实测基线

下表汇总 `Wan-AI/Wan2.2-TI2V-5B`（50 步 UniPC，CFG 5，BF16，PyTorch SDPA，模型级 CPU offload，121 帧）在 Atlas 910B2 64 GB 上的实机验证记录：

| 软件栈 | 分辨率 | 并行配置 | 生成时间 | 验证状态 |
| --- | --- | --- | ---: | --- |
| CANN 9.3.0-beta.1（驱动 25.2.0 / 固件 7.7.0.6.236，Torch 2.11.0+cpu，`torch_npu` 2.11.0.post4） | 480p（864 × 480） | 1 卡 | 157.07 秒 | 通过 |
| 同上 | 480p（864 × 480） | 2 卡（CFG 2） | 92.05 秒 | 通过，与单卡 121 帧逐像素一致，worker 正常退出 |
| CANN 8.2（Torch 2.9.0，`torch_npu` 2.9.0.post1） | 480p（864 × 480） | 1 / 2 / 4 卡（CFG 2 × Ulysses 2） | 180.9 / 120.8 / 87.0 秒 | 通过 |
| 同上 | 720p | 1 / 4 卡 | 477.8 / 173.9 秒 | 通过 |

双卡实机还单独验证了 HCCL all-reduce、默认 NPU mesh、流水线并行（PP）发送与缺省接收缓冲区，以及真实 NPU 张量经主存分片并在目标设备重组。与 `vLLM-Omni` 的同底座延迟对标及多卡扩展结果见[《昇腾 NPU 性能基准》](benchmark_ascend_npu.md)。
