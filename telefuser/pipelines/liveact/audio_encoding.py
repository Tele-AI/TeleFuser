"""Audio encoding stage for LiveAct pipeline.

Handles audio feature extraction using Wav2Vec2 model.
The audio projection (AudioProjModel) is part of WanModel (DiT) and
is called inside WanModel.forward().

Supports stream_audio mode: re-encode audio segment for each iteration.

Note: We do NOT apply tempo effect (rate=25/fps) when resampling audio.
This differs from original SoulX-LiveAct which stretched audio to match
expected video frames. We keep original audio duration for accurate
audio-video alignment.
"""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class AudioEncodingStage(BaseStage):
    """Audio encoding stage for LiveAct pipeline.

    Extracts audio features using Wav2Vec2 model.

    Note: The audio_proj is inside WanModel (DiT) and is called during WanModel.forward().
    This stage only extracts windowed raw embeddings using Wav2Vec2.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        audio_window: int = 5,
        vae_scale: int = 4,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
    ):
        super().__init__(name, model_runtime_config)
        self.audio_encoder = module_manager.fetch_module("wav2vec2")
        self.feature_extractor = module_manager.fetch_module("wav2vec2_feature_extractor")
        self.audio_window = audio_window
        self.vae_scale = vae_scale
        self.model_names = ["audio_encoder"]

        # Pre-create indices tensor for windowing (matches original get_audio_emb)
        # indices = [-2, -1, 0, 1, 2] for window_size=5
        self._window_indices = (torch.arange(2 * 2 + 1) - 2) * 1  # shape: [5]

        # Store original audio for stream_audio mode
        self._original_audio: torch.Tensor | None = None
        self._original_sr: int | None = None
        self._fps: int = 24

        # Store pre-encoded embedding for non-stream mode
        self._pre_encoded_embedding: torch.Tensor | None = None

    def resample_audio_to_16k(
        self,
        audio: torch.Tensor,
        sr: int,
    ) -> tuple[torch.Tensor, int]:
        """Resample audio to 16kHz for wav2vec2 (no tempo effect).

        Note: We do NOT apply tempo effect (rate=25/fps) here.
        The original SoulX-LiveAct used tempo to stretch audio to match expected video frames,
        but this causes audio distortion when fps != 25.
        Instead, we keep original audio duration and let video frames naturally align with audio.

        Args:
            audio: Audio waveform [channels, samples] (CPU tensor)
            sr: Original sample rate

        Returns:
            Tuple of (resampled_audio, target_sr=16000)
        """
        # Keep audio on CPU for resampling operations
        audio = audio.cpu()

        # Resample to 16kHz for wav2vec2 (no tempo effect)
        resampler = T.Resample(sr, 16000)
        audio = resampler(audio)
        audio = audio * 3.0  # Scale factor from original implementation

        return audio, 16000

    def set_original_audio(
        self,
        audio: torch.Tensor,
        sr: int,
        fps: int = 24,
    ) -> None:
        """Store original audio for stream_audio mode.

        Args:
            audio: Original audio waveform [channels, samples]
            sr: Original sample rate
            fps: Target video fps
        """
        self._original_audio = audio
        self._original_sr = sr
        self._fps = fps

    def get_original_duration(self) -> float:
        """Get original audio duration in seconds.

        Returns:
            Duration in seconds
        """
        if self._original_audio is not None and self._original_sr is not None:
            return self._original_audio.size(1) / self._original_sr
        return 0.0

    def extract_features(
        self,
        audio: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Extract wav2vec2 features from audio.

        Args:
            audio: Audio waveform [samples]
            seq_len: Expected sequence length for video frames

        Returns:
            Audio embeddings [T, blocks, channels]
        """
        # Prepare input for wav2vec2
        audio_feature = self.feature_extractor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values
        audio_feature = audio_feature.to(self.device, self.audio_encoder.dtype)

        # Extract features
        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature,
                seq_len=seq_len,
                output_hidden_states=True,
            )

        # Stack hidden states from layers 1-12 (excluding layer 0)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        return audio_emb

    def extract_features_for_segment(
        self,
        audio_start_idx: int,
        audio_end_idx: int,
        fps: int,
        frame_num: int,
    ) -> torch.Tensor:
        """Extract wav2vec2 features for a specific audio segment (stream_audio mode).

        Steps:
        1. Slice original audio at audio_start_idx/audio_end_idx (in frame units)
        2. Resample to 16kHz (no tempo effect)
        3. Encode with wav2vec2
        4. Get windowed embeddings starting from index 0

        Args:
            audio_start_idx: Start frame index (video frame units)
            audio_end_idx: End frame index (video frame units, +2 for window)
            fps: Target video fps
            frame_num: Number of frames for this segment (determines seq_len)

        Returns:
            Audio embeddings [T, blocks, channels] for this segment
        """
        if self._original_audio is None or self._original_sr is None:
            raise ValueError("Original audio not set. Call set_original_audio() first.")

        sr_ori = self._original_sr
        audio_ori = self._original_audio

        # Slice original audio: +2 offset for window boundary (indices [-2,-1,0,1,2])
        slice_start = int(sr_ori * (audio_start_idx / fps))
        slice_end = int(sr_ori * ((audio_end_idx + 2) / fps))

        # Clamp to valid range
        slice_end = min(slice_end, audio_ori.size(1))

        audio_slice = audio_ori[:1, slice_start:slice_end]

        # Resample to 16kHz (no tempo effect)
        audio_resampled, sr = self.resample_audio_to_16k(audio_slice, sr_ori)

        # Use frame_num as seq_len to match expected video frames
        seq_len = frame_num

        # Extract features
        audio_embedding = self.extract_features(audio_resampled[0], seq_len=seq_len)

        return audio_embedding

    def get_audio_emb_with_window(
        self,
        audio_embedding: torch.Tensor,
        audio_start_idx: int,
        audio_end_idx: int,
    ) -> torch.Tensor:
        """Get audio embeddings with sliding window for temporal alignment.

        Args:
            audio_embedding: Full audio embedding [T, blocks, channels]
            audio_start_idx: Start frame index
            audio_end_idx: End frame index

        Returns:
            Windowed audio embeddings [1, T_window, window_size, blocks, channels]
        """
        # Use pre-created indices tensor
        indices = self._window_indices.to(audio_embedding.device)
        center_indices = torch.arange(audio_start_idx, audio_end_idx, 1, device=audio_embedding.device).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embedding.shape[0] - 1)
        audio_emb = audio_embedding[center_indices][None, ...]
        return audio_emb

    def process_stream_audio_segment(
        self,
        audio_start_idx: int,
        audio_end_idx: int,
        fps: int,
        frame_num: int,
    ) -> torch.Tensor:
        """Process audio segment for stream_audio mode.

        This is the main method called during each iteration in stream_audio mode.
        Matches original generate.py behavior: re-encode audio for each segment.

        Args:
            audio_start_idx: Start frame index (video frame units)
            audio_end_idx: End frame index (video frame units)
            fps: Target video fps
            frame_num: Number of frames for this segment

        Returns:
            Windowed audio embeddings [1, T, window, blocks, channels]
        """
        # Ensure audio_encoder is on device
        self.audio_encoder.to(self.device)

        # Extract features for this specific segment (re-encoding)
        audio_embedding = self.extract_features_for_segment(audio_start_idx, audio_end_idx, fps, frame_num)

        # Get windowed embeddings starting from index 0 (stream_audio mode)
        # Original generate.py: audio_embs = get_audio_emb(audio_embedding, 0, frame_num, device)
        audio_embs = self.get_audio_emb_with_window(audio_embedding, 0, frame_num)

        return audio_embs.to(self.torch_dtype)

    def process_pre_encoded_segment(
        self,
        audio_start_idx: int,
        audio_end_idx: int,
    ) -> torch.Tensor:
        """Process audio segment for pre-encoded mode (stream_audio=False).

        Slice pre-encoded embedding and apply sliding window.
        Matches original generate.py behavior when --steam_audio=False.

        Args:
            audio_start_idx: Start frame index (video frame units)
            audio_end_idx: End frame index (video frame units)

        Returns:
            Windowed audio embeddings [1, T, window, blocks, channels]
        """
        if self._pre_encoded_embedding is None:
            raise ValueError("Pre-encoded embedding not set. Call process() first.")

        # Ensure embedding is on device
        audio_embedding = self._pre_encoded_embedding.to(self.device)

        # Get windowed embeddings from pre-encoded embedding
        # Original generate.py: audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device)
        audio_embs = self.get_audio_emb_with_window(audio_embedding, audio_start_idx, audio_end_idx)

        return audio_embs.to(self.torch_dtype)

    @with_model_offload(["audio_encoder"])  # Only audio_encoder, audio_proj comes from Dit
    @ProfilingContext4Debug("audio_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        audio_path: str | None = None,
        audio: torch.Tensor | None = None,
        sr: int = 16000,
        fps: int = 24,
        video_length: int | None = None,
        stream_audio: bool = True,
    ) -> tuple[torch.Tensor | None, int]:
        """Load and process audio based on stream_audio mode.

        Args:
            audio_path: Path to audio file (alternative to audio tensor)
            audio: Audio waveform tensor [channels, samples]
            sr: Sample rate of input audio
            fps: Target video fps
            video_length: Expected video length for pre-encoding (used when stream_audio=False)
            stream_audio: If True, store original audio for per-iteration encoding.
                          If False, pre-encode full audio immediately.

        Returns:
            Tuple of (audio_embedding or None, audio_duration_seconds)
            - stream_audio=True: Returns (None, original_duration), embedding extracted per-iteration
            - stream_audio=False: Returns (pre-encoded embedding, original_duration)
              Note: No tempo effect, so duration equals original audio duration
        """
        # Load audio if path provided (returns CPU tensor)
        original_audio = None
        original_sr = sr
        if audio_path is not None:
            audio, sr = torchaudio.load(audio_path)
            original_audio = audio
            original_sr = sr

        # Calculate original audio duration
        if original_audio is not None:
            original_duration = original_audio.size(1) / original_sr
        elif audio is not None:
            original_duration = audio.size(1) / sr
        else:
            raise ValueError("Either audio_path or audio must be provided")

        # Store original audio for stream_audio mode (per-iteration encoding)
        self.set_original_audio(original_audio, original_sr, fps)

        if stream_audio:
            # stream_audio=True: Only store original audio, encoding happens per-iteration
            # Return original_duration for iter_total_num calculation
            self._pre_encoded_embedding = None
            return None, original_duration
        else:
            # stream_audio=False: Pre-encode full audio immediately
            # Resample to 16kHz (no tempo effect)
            audio_resampled, sr = self.resample_audio_to_16k(original_audio, original_sr)

            # Calculate video_length for wav2vec2 seq_len based on original duration
            if video_length is None:
                video_length = int(original_duration * fps)

            # Pre-encode full audio
            self._pre_encoded_embedding = self.extract_features(audio_resampled[0], seq_len=video_length)

            # Return original_duration (no tempo effect applied)
            return self._pre_encoded_embedding.detach(), original_duration
