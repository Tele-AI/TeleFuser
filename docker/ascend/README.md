# Ascend image maintenance

The 910B image targets Linux arm64 and CPython 3.11. The Dockerfile pins the architecture-specific CANN manifest. The runtime lock combines the base dependencies from `pyproject.toml` and the stack from `requirements.in`. Every entry names one compatible wheel and its SHA-256. The Ascend release provides the wheel for torch_npu 2.11.0.post4. [Ascend release artifacts](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/v26.3.0-beta.1-pytorch2.11.0)

The default registry is AscendHub in China. For Docker Hub, pass `--build-arg CANN_REPOSITORY=docker.io/ascendai/cann`. Both registries serve the same pinned arm64 manifest. [Ascend image repositories](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.zh.md)

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

## Build, qualify and publish

Build from a clean release checkout. Replace the version below with its PEP 440 version. Tag the candidate with the commit. Record the digest printed by `docker push`. Run the hardware checks on that digest, including separate 1-, 2- and 4-card Wan runs, before promoting a release. [Qualification procedure](../../docs/en/ascend_910b.md)

```bash
read -r -p 'TeleFuser package version > ' TF_VERSION
TF_REV=$(git rev-parse HEAD)
TF_REPO=ghcr.io/tele-ai/telefuser
TF_CANDIDATE="$TF_REPO:sha-$TF_REV-ascend910b"
docker build --platform linux/arm64 -f docker/Dockerfile.ascend \
  --build-arg "TELEFUSER_VERSION=$TF_VERSION" --build-arg "SOURCE_REVISION=$TF_REV" \
  -t "$TF_CANDIDATE" .
docker push "$TF_CANDIDATE"
```

After qualification, point the public release tag at the tested manifest. Confirm anonymous access and publish the digest, driver and firmware versions and validated card counts alongside the release.

```bash
read -r -p 'Tested TeleFuser sha256 digest > ' TF_DIGEST
docker buildx imagetools create --prefer-index=false \
  --tag "$TF_REPO:$TF_VERSION-ascend910b" "$TF_REPO@$TF_DIGEST"
docker buildx imagetools inspect "$TF_REPO:$TF_VERSION-ascend910b"
```
