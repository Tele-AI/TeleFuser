#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVICE_HOST="${DIFFUSERS_WAN_I2V_HOST:-127.0.0.1}"
SERVICE_PORT="${DIFFUSERS_WAN_I2V_PORT:-8010}"

exec "${PYTHON_BIN}" benchmarks/baseline/diffusers_wan_i2v/service.py
