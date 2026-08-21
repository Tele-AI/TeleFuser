"""Eager neighborhood-attention building blocks for the isolated DiffVAE."""

from .attention import NeighborhoodAttention3D
from .blocks import DiffusionNABlock, NABlock
from .combined.block import CombinedDiffusionNABlock
from .config import DiffVAEMode
from .layers import AdaLNZero, ChannelLinear, LinearPixelShuffleUpsample

__all__ = [
    "AdaLNZero",
    "ChannelLinear",
    "CombinedDiffusionNABlock",
    "DiffVAEMode",
    "DiffusionNABlock",
    "LinearPixelShuffleUpsample",
    "NABlock",
    "NeighborhoodAttention3D",
]
