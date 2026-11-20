# Ascend image maintenance

The 910B image targets Linux arm64 and CPython 3.11. The Dockerfile pins the architecture-specific CANN manifest. The runtime lock combines the base dependencies from `pyproject.toml` and the stack from `requirements.in`. Every entry names one compatible wheel and its SHA-256. The Ascend release provides the wheel for torch_npu 2.11.0.post4. [Ascend release artifacts](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/v26.3.0-beta.1-pytorch2.11.0)

The default registry is AscendHub in China. For public pulls use `--build-arg CANN_REPOSITORY=quay.io/ascend/cann` or `--build-arg CANN_REPOSITORY=docker.io/ascendai/cann`. All three registries serve the same pinned arm64 manifest. [Ascend image repositories](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.zh.md)

If the upstream base was transferred by `docker save` / `docker load`, Docker may retain only its local tag. Verify the loaded arm64 image ID against `sha256:92ceaadada7011d9179f5854b08fc66981d3d604cfd72445bcbae756c777dee1`, the config referenced by the pinned manifest. Pass that verified local tag via `--build-arg CANN_IMAGE=<local-tag>` to reuse the cached layers.

For a build host with limited access to Python package servers, prepare the wheels on a connected Linux arm64 machine with Python 3.11. Transfer the directory to the same path in the build checkout. The Dockerfile detects it, installs from disk, checks every hash in the release lock and leaves the wheel archives in the build stage. The system stage still needs Ubuntu packages for FFmpeg.

```bash
python3.11 -m pip download --only-binary=:all: --require-hashes \
  --index-url https://pypi.org/simple -r docker/ascend/requirements.lock \
  --dest docker/ascend/wheelhouse
```

## Update the dependency lock

Run from the repository root in a clean CPython 3.11 environment on Ubuntu 22.04 arm64. Match each direct wheel URL and hash in `requirements.in` to its publisher before resolving. For TorchCodec, use the pinned arm64 CPU wheel. Its decoder must load against the system FFmpeg libraries during the image build. The pip report includes the source checkout as a separate local distribution. [TorchCodec CPU releases](https://download.pytorch.org/whl/cpu/torchcodec/)

```bash
python -m pip install --dry-run --ignore-installed --only-binary=:all: \
  --index-url https://pypi.org/simple --report /tmp/telefuser-ascend-resolution.json \
  -r docker/ascend/requirements.in .
python - <<'PY'
import json
from pathlib import Path

from packaging.utils import canonicalize_name

directory = Path("docker/ascend")
direct = {
    line.split(" @ ")[0]: line.split(" @ ")[1].partition("#")[0]
    for line in (directory / "requirements.in").read_text().splitlines() if " @ " in line
}
entries = []
for record in json.loads(Path("/tmp/telefuser-ascend-resolution.json").read_text())["install"]:
    name = canonicalize_name(record["metadata"]["name"])
    if name == "telefuser":
        continue
    digest = record["download_info"]["archive_info"]["hashes"]["sha256"]
    requirement = f"{name} @ {direct[name]}" if name in direct else f"{name}=={record['metadata']['version']}"
    entries.append((name, f"{requirement} --hash=sha256:{digest}"))
header = "# Linux arm64, Ubuntu 22.04, CPython 3.11. One wheel hash per distribution."
(directory / "requirements.lock").write_text("\n".join([header, *(line for _, line in sorted(entries))]) + "\n")
PY
```

## Publish the TeleFuser application image

Build the `runtime-payload` target from a clean release checkout. This target packages the Python environment and Wan example in a scratch image. The runnable image uses CANN layers pulled from the official Ascend registry. Record the source revision, checksums, tested driver and firmware, and validated card counts in the release. [CANN software license](https://www.hiascend.com/en/legal/cannua-download?isNewCon=true) · [Qualification procedure](../../docs/en/ascend_910b.md)

```bash
read -r -p 'TeleFuser package version > ' TF_VERSION
TF_REV=$(git rev-parse HEAD)
TF_PAYLOAD="telefuser:ascend910b-${TF_REV:0:7}-payload"
docker build --platform linux/arm64 --target runtime-payload -f docker/Dockerfile.ascend \
  --build-arg "TELEFUSER_VERSION=$TF_VERSION" --build-arg "SOURCE_REVISION=$TF_REV" \
  -t "$TF_PAYLOAD" .
```

Assemble the runnable image with `Dockerfile.ascend.install`. Pass `--build-arg TELEFUSER_PAYLOAD="$TF_PAYLOAD"` and the same version and revision arguments. Run the hardware checks on that image. For a GitHub Release, upload the compressed `docker save` archive, `Dockerfile.ascend.install` and their SHA-256 checksums. Verify anonymous download and repeat the documented install from those assets.

## Save a complete image for an existing installation

Use the tag on the qualification host so the recipient loads the exact image that ran the tests. Transfer both files together. The first build qualified on one and two 910B2 devices used the tag below.

```bash
set -o pipefail
TF_IMAGE=telefuser:ascend910b-c27fabd
TF_ARCHIVE=telefuser-ascend910b-c27fabd.tar.gz
docker save "$TF_IMAGE" | gzip -1 > "$TF_ARCHIVE"
sha256sum "$TF_ARCHIVE" > "$TF_ARCHIVE.sha256"
```

On the target host, verify the archive, load it with `docker load -i` and compare the resulting image ID with the hardware qualification record. [Tested image and run instructions](../../docs/en/ascend_910b.md)
