# TurboServe public-demo-derived ABot workloads

These two scenarios are deterministic, 30-minute ABot LiveKit workload
projections of TurboServe's public simulator trace:
`../../../TurboServe/traces/example_8gpu.json`.

They are **not** TurboServe production traces and are **not** reproductions of
the private paper T1--T6 traces. The source records session lifecycle events,
not real ABot keyboard actions. The adapter maps its selected events as:

- `session_arrival` → create an ABot LiveKit session;
- `user_active` → resume its action heartbeat;
- `user_idle` → pause its action heartbeat while retaining the session/state;
- `session_departure` → stop and delete the session.

The source wall clock is retained exactly (1,800 seconds; no time compression).
Its observed retained-session peak is 186. The capacity transform uses
half-up proportional scaling to an ABot peak of `workers × 4`, then keeps a
selected source session sticky until source departure or scaled capacity
decrease. Scale-up selection uses a stable SHA-256 rank with seed `20260815`.
The full transform, source SHA-256, and event provenance are embedded in each
JSON's `trace_contract`.

| Scenario | ABot workers | Peak retained sessions | Duration | Arrivals / departures | Active / idle transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abot_livekit_1gpu_lf3_12fps_turboserve_public_demo_trace_peak4.json` | 1 | 4 | 1,800 s | 61 / 61 | 66 / 69 |
| `abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16.json` | 4 | 16 | 1,800 s | 300 / 300 | 282 / 323 |

Regenerate and verify deterministic files:

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world
TF_PY=/public/fanyk1/lwb/envs/telefuser_sage291/bin/python
$TF_PY tools/validation/derive_abot_turboserve_trace.py --check
```

Validate, then replay through the normal public serving interfaces:

```bash
$TF_PY tools/validation/replay_abot_livekit_lifecycle_trace.py \
  --scenario tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16.json \
  --dry-run

$TF_PY tools/validation/replay_abot_livekit_lifecycle_trace.py \
  --scenario tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16.json \
  --output results/experiments/abot_turboserve_public_demo_4gpu/result.json
```

The replay client never names or selects a GPU; placement, batching, and any
migration remain black-box serving-system behavior.
