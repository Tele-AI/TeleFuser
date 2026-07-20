# SGLang-Diffusion LingBot Stream Baseline

This target compares TeleFuser LingBot streaming with the diffusion runtime in
`sgl-project/sglang` (`sglang.multimodal_gen`). It uses WebSocket + MessagePack while
TeleFuser uses WebRTC + DataChannel; AIPerf normalizes both into the same session and
control timeline.

## Requirements

- a version-pinned SGLang checkout that provides `LingBotWorldCausalDMDPipeline`;
- `robbyant/lingbot-world-fast-diffusers` or an equivalent local model path;
- the AIPerf checkout prepared by `scripts/setup_aiperf_repo.sh`.

The launcher does not monkeypatch SGLang internals. Missing dependencies or incompatible
CUDA kernels must be fixed in the SGLang environment or recorded as a failed
qualification, not hidden behind an unversioned shim.

## Start the target

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

Common overrides:

```bash
SGLANG_PYTHON=/path/to/venv/bin/python \
SGLANG_LINGBOT_MODEL_PATH=/path/to/model \
SGLANG_LINGBOT_NUM_GPUS=1 \
SGLANG_LINGBOT_ULYSSES_DEGREE=1 \
  bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

The default address is `http://127.0.0.1:30000`; readiness is checked at `/health`.

## Run AIPerf

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh

bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh \
  benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_compare.json
```

The baseline reuses
`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`. AIPerf maps the shared
directional controls to SGLang camera actions and keeps implementation-specific fields
as raw evidence.

For a valid performance comparison, use the same accelerator count, prompt, first
frame, FPS, session window, control trace, dtype, attention/cache geometry, and offload
policy. Record the exact SGLang and model revisions. Mock, native fallback, CPU offload,
and layerwise offload runs require separate qualifications.
