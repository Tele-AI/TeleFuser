"""LiveAct: Audio-conditioned Image-to-Video Generation Pipeline.

This pipeline generates talking head videos from an input image and audio.
It extends the Wan I2V architecture with audio cross-attention for lip-sync generation.
"""

from .audio_encoding import AudioEncodingStage
from .denoising import LiveActDenoisingStage
from .pipeline import LiveActPipeline, LiveActPipelineConfig

__all__ = [
    "LiveActPipeline",
    "LiveActPipelineConfig",
    "LiveActDenoisingStage",
    "AudioEncodingStage",
]
