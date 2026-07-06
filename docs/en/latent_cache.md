# Latent Cache (Cross-Request Approximate Cache for Diffusion)

Latent cache reuses **the intermediate latent from a previous inference**
when an incoming prompt is similar enough to a prompt already served, so the
first N denoising steps can be skipped. TeleFuser integrates this feature
through the external **CacheSeek** package:
<https://github.com/Tele-AI/CacheSeek>. The legacy in-tree
`telefuser/cache_mem/` implementation has been removed.

## Latent Cache vs. Feature Cache

The two solve different problems:

|                  | Feature cache (see `feature_cache.md`)      | Latent cache (this doc)                              |
| ---------------- | ------------------------------------------- | ---------------------------------------------------- |
| Granularity      | Within a single inference, across timesteps | Across requests                                      |
| Reuse key        | Step index                                  | Prompt embedding similarity                          |
| Acceleration     | Skip approximable blocks                    | Skip the first N denoising steps                     |
| Module           | `telefuser/feature_cache/`                  | External `cacheseek` package                         |
| Persistence      | None (request lifetime only)                | KV on disk / distributed store + vector DB + metadata |

Use feature cache to speed up *one* inference; use latent cache to speed up
the **next** inference whose prompt is similar to a cached one. The two can
be enabled at the same time without interfering.

---

## Service Interface

TeleFuser imports CacheSeek only after latent cache resolves to enabled. When
the resolved value is enabled, a missing CacheSeek install fails service startup
immediately with an installation hint. See "Startup and Runtime Behavior" for
configuration precedence.

When enabled, the service layer uses CacheSeek's TeleFuser adapter:

```python
adapter.build_query(task_request)
cache_service.lookup(cache_query)
adapter.apply_resume(lookup_result, engine_ctx=task_data)
adapter.on_response(task_request, latent_payload)
cache_service.save(cache_query, outputs)
```

Lookup / resume / save failures for an individual request are best-effort and
degrade to the uncached path; startup failures are not ignored.

### Use in Model Forward

The pipeline forwards the `latent_data` injected by the service layer down
to the denoise stage. The denoising loop uses `skip_step` to decide where
to start, and snapshots intermediate latents at the steps listed in
`saved_steps`:

```python
# In the denoise stage (see telefuser/pipelines/wan_video/moe_dit_denoising.py)
cached_latent, effective_start_step, saved_steps = parse_latent_data(
    latent_data,
    expected_shape=tuple(latents.shape),
    total_steps=total_steps,
)
if cached_latent is not None:
    latents = cached_latent.to(device=latents.device, dtype=latents.dtype)

saved_steps_set = frozenset(saved_steps)
latent_states_dict: dict[int, torch.Tensor] = {}

for progress_id, timestep in enumerate(timesteps[effective_start_step:]):
    absolute_step = effective_start_step + progress_id
    # snapshot BEFORE scheduler.step: step k stores the latent that enters step k
    if absolute_step in saved_steps_set:
        latent_states_dict[absolute_step] = latents.detach().cpu()
    noise_pred = self.predict_noise_with_cfg(...)
    latents = self.scheduler.step(noise_pred, timesteps[absolute_step], latents)

# pipeline returns the payload alongside the latent so the service layer
# can write it back asynchronously
latent_payload = {
    "latent_states_dict": latent_states_dict,
    "saved_steps": saved_steps,
    "final_step": total_steps - 1,
}
return latents, latent_payload
```

`parse_latent_data` (`telefuser/pipelines/wan_video/latent_data_utils.py`)
performs shape and range validation: if the shape mismatches or `skip_step`
is out of range, the cache is silently dropped and the pipeline falls back
to full denoising, so the main path is never poisoned by a bad cache entry.

---

## Factory Function

The production path builds a `(CacheService, TeleFuserCacheAdapter)` pair from
CLI arguments and the `CACHE_CONFIG` declared in the pipeline file:

```python
from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory

cache_service, cache_adapter = CacheServiceFactory.create_cache_service(
    ppl_file="examples/wan_video/wan22_14b_text_to_video_service.py",
    enable_latent_cache=True,
    cache_mode="read_write",  # "read_write" / "read_only" / "write_only"
)
```

TeleFuser only relies on this factory's input/output contract:

- Pass the current pipeline file path as `ppl_file`.
- Explicit CLI values for `enable_latent_cache` / `cache_mode` are passed to
  CacheSeek as overrides; omitted CLI values are passed as `None`.
- Expect `(cache_service, cache_adapter)` back, where the adapter provides
  `build_query`, `apply_resume`, and `on_response`.

If you need to construct the cache service outside TeleFuser service startup,
use the CacheSeek documentation and `cacheseek/service/config.py` as the
source of truth. TeleFuser no longer provides the old in-tree `LatentCache`
facade.

---

## Pipeline Configuration Example

Declare `CACHE_CONFIG` in the pipeline file:

```python
# examples/wan_video/wan22_14b_text_to_video_service.py
CACHE_CONFIG = dict(
    enable_latent_cache=True,
    latent_cache_dir=os.getenv("TELEFUSER_LATENT_CACHE_DIR", "./latent_cache/wan22_t2v"),
    cache_mode="write_only",
    kv_store_type="local_file",
    vector_store_type="faiss",
    # Qwen3-VL-Embedding-2B hidden_size=2048; must match the vector_store dim.
    vector_dim=2048,
    key_steps=[5, 10, 15, 20, 25],
    video_embedding_enabled=True,
    video_embedding_model_path=os.getenv("QWEN3VL_EMBEDDING_PATH", ""),
    video_embedding_max_frames=16,
    text_embedding_device_id=1,
    video_embedding_device_id=1,
    video_vector_collection="video",
    rerank_enabled=True,
    rerank_model_path=os.getenv("QWEN3VL_RERANKER_PATH", "/storage/model_zoo/Qwen3-VL-Reranker-2B"),
    rerank_device_id=int(os.getenv("TELEFUSER_RERANK_DEVICE_ID", "0")),
    rerank_top_k=5,
    rerank_score_threshold=0.85,
)
```

The table below describes common fields in the Wan2.2 service example's
`CACHE_CONFIG` and the example defaults. Here "default" means the value the
example file passes to CacheSeek when no matching environment variable or CLI
override is set. For CacheSeek's complete field list and built-in defaults,
refer to the CacheSeek documentation and `cacheseek/service/config.py`.

| Field | Example default | Description |
|---|---|---|
| `enable_latent_cache` | `True` | The example pipeline enables latent cache by default. |
| `cache_mode` | `write_only` | Write-only by default, useful for warming or building cache first; override with `TELEFUSER_CACHE_MODE`. |
| `latent_cache_dir` | `./latent_cache/wan22_t2v` | Cache root directory; override with `TELEFUSER_LATENT_CACHE_DIR`. |
| `kv_store_type` | `local_file` | KV backend type; override with `TELEFUSER_KV_STORE_TYPE`. |
| `vector_store_type` | `faiss` | Vector-search backend type; override with `TELEFUSER_VECTOR_STORE_TYPE`. |
| `vector_dim` | `2048` | Vector dimension; must match the embedding model output dimension. |
| `key_steps` | `[5, 10, 15, 20, 25]` | Denoise steps at which the pipeline is asked to snapshot latents. |
| `video_embedding_enabled` | `True` | Enables video-frame embedding; the example save path backfills `embedding_video_frames`. |
| `video_embedding_model_path` | `""` | Video embedding model path; override with `QWEN3VL_EMBEDDING_PATH`. How an empty value is resolved is owned by CacheSeek. |
| `video_embedding_max_frames` | `16` | Maximum number of video frames sampled before cache writeback. |
| `text_embedding_device_id` / `video_embedding_device_id` | `1` / `1` | Logical GPU ids for embedding models; adjust for `CUDA_VISIBLE_DEVICES` and parallelism. |
| `video_vector_collection` | `video` | Video vector collection name. |
| `rerank_enabled` / `rerank_top_k` / `rerank_score_threshold` | `True` / `5` / `0.85` | Example rerank configuration; CacheSeek executes the actual hit policy. |

---

## Example Scripts

| Pipeline                          | Script                                                            | Notes                                |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------ |
| Wan2.2 14B T2V (cache enabled)    | `examples/wan_video/wan22_14b_text_to_video_service.py`           | Full latent cache configuration example |

Start the service:

```bash
# Install CacheSeek into the same Python environment as TeleFuser first.
# cacheseek is not yet published to public PyPI. Prepare a CacheSeek checkout
# that contains the TeleFuser adapter support, then install from its local path:
# git clone https://github.com/Tele-AI/CacheSeek.git
# git -C CacheSeek checkout <commit-or-branch>
CACHESEEK_REPO=/path/to/CacheSeek
python -m pip install "${CACHESEEK_REPO}"
#
# If you are developing CacheSeek and need source edits to take effect
# immediately, use an editable install:
# python -m pip install -e "${CACHESEEK_REPO}"
#
# If the matching CacheSeek commit or branch has been pushed to GitHub, you can
# install it directly:
# python -m pip install "cacheseek @ git+https://github.com/Tele-AI/CacheSeek.git@<commit-or-branch>"
#
# Once cacheseek is published to the current pip package index, this can be:
# python -m pip install cacheseek

python -c "import cacheseek, torch; print(cacheseek.__file__, torch.__version__)"

telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py \
    --port 8000 \
    --cache-mode read_write
```

TeleFuser no longer provides a `cache` extra and does not vendor CacheSeek
backend dependencies. The `cacheseek` package declares the dependencies needed
by the default TeleFuser Wan2.2 path, including the local FAISS vector backend
and Qwen3-VL embedding / rerank dependencies. Install `cacheseek[qdrant]` only
when using the Qdrant vector backend, for example:

```bash
CACHESEEK_REPO=/path/to/CacheSeek
python -m pip install "${CACHESEEK_REPO}[qdrant]"
# Or from a pushed GitHub commit / branch:
# python -m pip install "cacheseek[qdrant] @ git+https://github.com/Tele-AI/CacheSeek.git@<commit-or-branch>"
```

When installing from a GitHub branch, remember that branches are moving targets.
For reproducible experiments or production deployments, use a known-good commit.
If `python -m pip install cacheseek` reports that no matching distribution was
found, the package has not been published to the current pip index or that
private index is not configured.

---

## Startup and Runtime Behavior

### The Three Cache Modes

| Mode | Effect |
|---|---|
| `read_write` | Read existing cache entries; write new cache entries after each request completes. |
| `read_only` | Read existing cache entries; do not write new cache entries. |
| `write_only` | Do not use cache hits; only write cache entries after each request completes. |

`enable_latent_cache` and `cache_mode` can be set in the pipeline file's
`CACHE_CONFIG` or overridden by CLI flags. The precedence is CLI >
`CACHE_CONFIG` > CacheSeek defaults. `--cache-mode` only accepts the three
values above. When `--cache-mode` is omitted, TeleFuser does not override the
`CACHE_CONFIG` value. The Wan2.2 service example defaults `cache_mode` to
`write_only`.

### Startup and Failure Semantics

- When no cache CLI flag is passed and the pipeline `CACHE_CONFIG` does not
  enable latent cache, CacheSeek is not loaded.
- When pipeline `CACHE_CONFIG` enables latent cache, or `--enable-latent-cache`
  is passed, a missing CacheSeek install or initialization failure fails service
  startup immediately.
- `--disable-latent-cache` prevents CacheSeek initialization even if the
  pipeline `CACHE_CONFIG` enables cache.
- Per-request failures in `build_query` / `lookup` / `apply_resume` / `save`
  are logged as warnings and the request continues through the uncached path.
