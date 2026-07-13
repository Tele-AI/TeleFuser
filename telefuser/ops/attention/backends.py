"""Attention backend management.

Centralizes all attention backend imports, availability flags,
and backend selection logic.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Callable

import torch
from torch import Tensor

from telefuser.utils.logging import logger

# Availability flags
FLASH_ATTN_4_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
SDPA_AVAILABLE = False
SAGE_ATTN_AVAILABLE = False
SPARGE_ATTN_AVAILABLE = False
FLASHINFER_AVAILABLE = False

# Backend function references (populated on successful import)
flash_attn2: Callable | None = None
flash_attn3: Callable | None = None
flash_attn4: Callable | None = None
sageattention: object | None = None
spas_sage2_attn_meansim_cuda: Callable | None = None
flashinfer: object | None = None


def _try_import_flash_attn() -> None:
    """Import Flash Attention 2/3/4."""
    global FLASH_ATTN_4_AVAILABLE, FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE, flash_attn4, flash_attn3, flash_attn2

    if importlib.util.find_spec("flash_attn") is None:
        return

    # Flash Attention 4 (Cute interface)
    try:
        from flash_attn.cute import flash_attn_func as flash_attn4_impl

        flash_attn4 = flash_attn4_impl
        FLASH_ATTN_4_AVAILABLE = True
        logger.debug("Flash Attention 4 available")
    except (ModuleNotFoundError, ImportError):
        pass

    # Flash Attention 2
    try:
        from flash_attn import flash_attn_func as flash_attn2_impl

        flash_attn2 = flash_attn2_impl
        FLASH_ATTN_2_AVAILABLE = True
        logger.debug("Flash Attention 2 available")
    except (ModuleNotFoundError, ImportError):
        pass

    # Flash Attention 3
    try:
        if importlib.util.find_spec("flash_attn_interface") is not None:
            from flash_attn_interface import flash_attn_func as flash_attn3_impl

            flash_attn3 = flash_attn3_impl
            FLASH_ATTN_3_AVAILABLE = True
            logger.debug("Flash Attention 3 available")
    except (ModuleNotFoundError, ImportError):
        pass


def _try_import_sdpa() -> None:
    """Import PyTorch SDPA."""
    global SDPA_AVAILABLE

    try:
        SDPA_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if SDPA_AVAILABLE:
            logger.debug("PyTorch SDPA available")
    except AttributeError:
        pass


def _try_import_sage_attn() -> None:
    """Import Sage Attention."""
    global SAGE_ATTN_AVAILABLE, sageattention

    for module_name in ["tf_kernel.sageattn2", "sageattention"]:
        try:
            if importlib.util.find_spec(module_name) is not None:
                sageattention = importlib.import_module(module_name)
                SAGE_ATTN_AVAILABLE = True
                logger.debug(f"Sage Attention loaded from {module_name}")
                return
        except (ModuleNotFoundError, ImportError):
            continue


def _try_import_sparge_attn() -> None:
    """Import Sparge Attention."""
    global SPARGE_ATTN_AVAILABLE, spas_sage2_attn_meansim_cuda

    try:
        if importlib.util.find_spec("spas_sage_attn") is not None:
            spas_sage2_attn_meansim_cuda = importlib.import_module("spas_sage_attn").spas_sage2_attn_meansim_cuda
            SPARGE_ATTN_AVAILABLE = True
            logger.debug("Sparge Attention loaded from spas_sage_attn")
    except (ModuleNotFoundError, ImportError, AttributeError):
        pass


def _try_import_flashinfer() -> None:
    """Import FlashInfer."""
    global FLASHINFER_AVAILABLE, flashinfer

    try:
        import flashinfer as flashinfer_impl

        flashinfer = flashinfer_impl
        FLASHINFER_AVAILABLE = True
        logger.debug("FlashInfer available")
    except ImportError:
        pass


# Initialize all backends
_try_import_flash_attn()
_try_import_sdpa()
_try_import_sage_attn()
_try_import_sparge_attn()
_try_import_flashinfer()


def supports_return_lse(attn_impl: str) -> bool:
    """Check if attention implementation supports log-sum-exp return.

    Required for Ring Attention's online softmax merging.
    """
    if attn_impl == "FLASH_ATTN_2" and FLASH_ATTN_2_AVAILABLE:
        return True
    if attn_impl == "FLASH_ATTN_3" and FLASH_ATTN_3_AVAILABLE:
        return True
    if attn_impl == "FLASH_ATTN_4" and FLASH_ATTN_4_AVAILABLE:
        return True
    if attn_impl in ("SAGE_ATTN_2_8_8", "SAGE_ATTN_2_8_16", "SAGE_ATTN_2_8_8_SM90") and SAGE_ATTN_AVAILABLE:
        return True
    return False


def get_lse_fallback_impl() -> str | None:
    """Get best available attention implementation with LSE support."""
    if FLASH_ATTN_4_AVAILABLE:
        return "FLASH_ATTN_4"
    if FLASH_ATTN_3_AVAILABLE:
        return "FLASH_ATTN_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "FLASH_ATTN_2"
    if SAGE_ATTN_AVAILABLE:
        return "SAGE_ATTN_2_8_8_SM90"
    return None


def sdpa_attn_cudnn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> Tensor:
    """SDPA with CUDNN backend."""
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale, is_causal=is_causal
        )


def sparge_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    """Sparge attention wrapper."""
    return spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=attn_mask, scale=scale)


__all__ = [
    "FLASH_ATTN_4_AVAILABLE",
    "FLASH_ATTN_3_AVAILABLE",
    "FLASH_ATTN_2_AVAILABLE",
    "SDPA_AVAILABLE",
    "SAGE_ATTN_AVAILABLE",
    "SPARGE_ATTN_AVAILABLE",
    "FLASHINFER_AVAILABLE",
    "flash_attn2",
    "flash_attn3",
    "flash_attn4",
    "sageattention",
    "flashinfer",
    "supports_return_lse",
    "get_lse_fallback_impl",
    "sdpa_attn_cudnn",
    "sparge_attn",
]
