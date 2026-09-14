#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_python="${repo_root}/.venv/bin/python"
local_lib_root="${repo_root}/.venv-mujoco-libs"

if [[ ! -x "${venv_python}" ]]; then
  echo "Expected the TeleFuser virtual environment at ${repo_root}/.venv" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to install into the existing virtual environment" >&2
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "apt-get and dpkg-deb are required to stage repository-local OSMesa libraries" >&2
  exit 1
fi

uv pip install --python "${venv_python}" "mujoco==3.13.0"
mkdir -p "${local_lib_root}/root"
pushd "${local_lib_root}" >/dev/null
apt-get download libegl1 libopengl0 libosmesa6
for package in ./*.deb; do
  dpkg-deb -x "${package}" root
done
popd >/dev/null

echo "MuJoCo and repository-local OSMesa are ready under ${repo_root}"
