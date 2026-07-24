#!/usr/bin/env bash
set -euxo pipefail

WHEEL_DIR="${WHEEL_DIR:-dist}"
PYTHON_BIN="${PYTHON:-python3}"
MANYLINUX_TAG="${MANYLINUX_TAG:-manylinux_2_28}"

torch_version=$("${PYTHON_BIN}" -c "import torch; print(torch.__version__.split('+')[0])")
torch_tag="${torch_version//[^a-zA-Z0-9]/_}"

torch_cuda_version=$("${PYTHON_BIN}" -c "import torch; print(torch.version.cuda or '')")
case "${torch_cuda_version}" in
    12.4*) cuda_tag="cu124" ;;
    12.8* | 12.9*) cuda_tag="cu128" ;;
    13.0*) cuda_tag="cu130" ;;
    "") cuda_tag="cpu" ;;
    *)
        echo "Unsupported PyTorch CUDA version: ${torch_cuda_version}" >&2
        exit 1
        ;;
esac

# Keep the project version identical to METADATA. Encode build compatibility in
# the PEP 427 build tag, and let wheel update WHEEL and RECORD while retagging.
build_tag="0torch${torch_tag}_${cuda_tag}"

wheel_files=("${WHEEL_DIR}"/*.whl)
for wheel in "${wheel_files[@]}"; do
    [ -f "${wheel}" ] || continue

    case "${wheel}" in
        *-linux_x86_64.whl) platform_tag="${MANYLINUX_TAG}_x86_64" ;;
        *-linux_aarch64.whl) platform_tag="${MANYLINUX_TAG}_aarch64" ;;
        *)
            echo "Unsupported wheel platform tag: ${wheel}" >&2
            exit 1
            ;;
    esac

    "${PYTHON_BIN}" -m wheel tags \
        --remove \
        --build "${build_tag}" \
        --platform-tag "${platform_tag}" \
        "${wheel}"
done

echo "Wheel retagging completed."
echo "Public version is managed in pyproject.toml; Torch/CUDA compatibility uses the wheel build tag."
