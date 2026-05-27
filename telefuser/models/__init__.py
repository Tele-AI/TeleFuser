"""TeleFuser model implementations for various diffusion models.

Model registration is handled automatically by ``ModelRegistry.autodiscover()``
which scans all modules containing ``register_model_config`` calls.
No manual imports are needed here for single-file registrations.
"""

from __future__ import annotations

# Cross-file registration: LTX 2.3 shared checkpoint contains 3 models
# that live in separate files with circular import dependencies, so it
# cannot be registered inside any one of them.
from telefuser.core.model_registry import register_model_config
from telefuser.models.ltx_dit import LTXVideoTransformer
from telefuser.models.ltx_gemma_text_encoder import LTXEmbeddingsProcessor
from telefuser.models.ltx_video_vae import LTXVideoVAE

register_model_config(
    None,
    "f3a83ecf3995dcc4fae2d27e08ad5767",
    ["ltx_embeddings_processor", "ltx_video_vae", "ltx_dit"],
    [LTXEmbeddingsProcessor, LTXVideoVAE, LTXVideoTransformer],
    "official",
)
