#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Gate G0 (Host): Verify target Ascend NPU devices are within allowed bounds, healthy, and idle.
set -euo pipefail

DEVICES_CSV="0"
ALLOWED_CSV=""
OUT_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --devices) DEVICES_CSV="$2"; shift 2 ;;
    --allowed-devices) ALLOWED_CSV="$2"; shift 2 ;;
    --out) OUT_JSON="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a REQ_CARDS <<< "$DEVICES_CSV"
if [[ "${#REQ_CARDS[@]}" -eq 0 ]]; then
  echo "ERROR: --devices must list at least one physical NPU ID" >&2
  exit 2
fi

declare -A ALLOWED_MAP=()
if [[ -n "$ALLOWED_CSV" ]]; then
  IFS=',' read -r -a ALLOWED_LIST <<< "$ALLOWED_CSV"
  for a in "${ALLOWED_LIST[@]}"; do ALLOWED_MAP[$a]=1; done
fi

declare -A SEEN=()
for c in "${REQ_CARDS[@]}"; do
  if ! [[ "$c" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid NPU ID '$c'" >&2
    exit 2
  fi
  if [[ -n "$ALLOWED_CSV" && -z "${ALLOWED_MAP[$c]:-}" ]]; then
    echo "FENCE VIOLATION: Physical NPU '$c' is outside --allowed-devices={$ALLOWED_CSV}" >&2
    exit 2
  fi
  if [[ -n "${SEEN[$c]:-}" ]]; then
    echo "ERROR: Duplicate NPU ID '$c'" >&2
    exit 2
  fi
  SEEN[$c]=1
done

SMI_OUT="$(npu-smi info)"

python3 - "$DEVICES_CSV" "$OUT_JSON" "$SMI_OUT" <<'PY'
import json
import platform
import re
import sys

cards = [int(x) for x in sys.argv[1].split(",")]
out_path = sys.argv[2]
smi = sys.argv[3]

driver_match = re.search(r"Version:\s*([0-9A-Za-z._-]+)", smi)
driver_ver = driver_match.group(1) if driver_match else "unknown"

card_info = {}
lines = smi.splitlines()
for idx, line in enumerate(lines):
    m = re.match(r"^\|\s*(\d+)\s+(910[A-Za-z0-9_-]+)\s*\|\s*([A-Za-z]+)\s*\|", line)
    if m and idx + 1 < len(lines):
        cid = int(m.group(1))
        raw_name = m.group(2)
        health = m.group(3)
        hbm_m = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", lines[idx + 1])
        card_info[cid] = {
            "raw_name": raw_name,
            "health": health,
            "hbm_used_mb": int(hbm_m.group(1)) if hbm_m else 0,
            "hbm_total_mb": int(hbm_m.group(2)) if hbm_m else 65536,
        }

busy_by_card = {c: [] for c in cards}
in_proc = False
for line in lines:
    if "Process id" in line and "Process name" in line:
        in_proc = True
        continue
    if in_proc:
        pm = re.match(r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*([^\s|]+)\s*\|\s*(\d+)", line)
        if pm and int(pm.group(1)) in busy_by_card:
            busy_by_card[int(pm.group(1))].append(
                {"pid": int(pm.group(2)), "name": pm.group(3), "mem_mb": int(pm.group(4))}
            )

for c in cards:
    if c not in card_info:
        print(f"ERROR: Physical NPU {c} not present in npu-smi", file=sys.stderr)
        sys.exit(2)
    if card_info[c]["health"].upper() != "OK":
        print(f"ERROR: Physical NPU {c} health is {card_info[c]['health']}", file=sys.stderr)
        sys.exit(75)
    if busy_by_card[c]:
        print(f"BUSY: Physical NPU {c} has active processes: {busy_by_card[c]}", file=sys.stderr)
        sys.exit(75)

first = card_info[cards[0]]
gb = round(first["hbm_total_mb"] / 1024)
raw_sku = first["raw_name"]
chip_model = f"Ascend 910B4 ({gb}GB)" if "910B4" in raw_sku else f"Ascend 910B2 ({gb}GB)"

payload = {
    "status": "READY",
    "card_count": len(cards),
    "chip_model": chip_model,
    "cpu_arch": platform.machine(),
    "driver_version": driver_ver,
    "baseline_hbm_used_mb": [card_info[c]["hbm_used_mb"] for c in cards],
}
text = json.dumps(payload, indent=2)
if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
print(text)
PY
