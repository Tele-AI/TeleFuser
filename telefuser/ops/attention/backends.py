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

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

# Availability flags
FLASH_ATTN_4_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
SDPA_AVAILABLE = False
SAGE_ATTN_AVAILABLE = False
SPARGE_ATTN_AVAILABLE = False
FLASHINFER_AVAILABLE = False
SOL_ATTN_AVAILABLE = False
MINDIE_ATTN_AVAILABLE = False

# Backend function references (populated on successful import)
flash_attn2: Callable | None = None
flash_attn3: Callable | None = None
flash_attn4: Callable | None = None
flash_attn4_varlen: Callable | None = None
sageattention: object | None = None
spas_sage2_attn_meansim_cuda: Callable | None = None
flashinfer: object | None = None
sol_attn: Callable | None = None
mindiesd_attention_forward: Callable | None = None


def _try_import_flash_attn() -> None:
    """Import Flash Attention 2/3/4."""
    global FLASH_ATTN_4_AVAILABLE, FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE
    global flash_attn4, flash_attn4_varlen, flash_attn3, flash_attn2

    if importlib.util.find_spec("flash_attn") is None:
        return

    # Flash Attention 4 (Cute interface)
    try:
        from flash_attn.cute import flash_attn_func as flash_attn4_impl

        flash_attn4 = flash_attn4_impl
        FLASH_ATTN_4_AVAILABLE = True
        logger.debug("Flash Attention 4 available")
        try:
            from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_impl

            flash_attn4_varlen = flash_attn4_varlen_impl
        except (ModuleNotFoundError, ImportError):
            logger.debug("Flash Attention 4 varlen interface unavailable")
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
    """Import SageAttention, preferring the optional tf-kernel package."""
    global SAGE_ATTN_AVAILABLE, sageattention

    SAGE_ATTN_AVAILABLE = False
    sageattention = None
    for module_name in ("tf_kernel", "sageattention"):
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
            sageattention = importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError) as error:
            logger.debug("SageAttention backend %s unavailable: %s", module_name, error)
            continue

        SAGE_ATTN_AVAILABLE = True
        logger.debug("SageAttention loaded from %s", module_name)
        return


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


def _try_import_sol_attn() -> None:
    """Import TeleFuser's built-in Sol-Attn kernel."""
    global SOL_ATTN_AVAILABLE, sol_attn

    SOL_ATTN_AVAILABLE = False
    sol_attn = None
    module_name = "telefuser.kernel.sol_attn"
    try:
        if importlib.util.find_spec(module_name) is None:
            return
        candidate = getattr(importlib.import_module(module_name), "sol_attn", None)
    except (ModuleNotFoundError, ImportError, RuntimeError) as error:
        logger.debug("Built-in Sol-Attn backend unavailable: %s", error)
        return
    if callable(candidate):
        sol_attn = candidate
        SOL_ATTN_AVAILABLE = True
        logger.debug("Built-in Sol-Attn kernel available")


# Initialize all backends
def _try_import_mindie_attn() -> None:
    """Import the optional MindIE-SD attention backend (Ascend NPU only)."""
    global MINDIE_ATTN_AVAILABLE, mindiesd_attention_forward

    MINDIE_ATTN_AVAILABLE = False
    mindiesd_attention_forward = None
    if current_platform.device_type != "npu":
        return
    try:
        if importlib.util.find_spec("mindiesd") is None:
            return
        from mindiesd import attention_forward
    except (ModuleNotFoundError, ImportError) as error:
        logger.debug("MindIE-SD attention backend unavailable: %s", error)
        return
    mindiesd_attention_forward = attention_forward
    MINDIE_ATTN_AVAILABLE = True
    logger.debug("MindIE-SD attention available")


def mindie_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> Tensor:
    """MindIE-SD attention wrapper for BNSD inputs on Ascend NPU."""
    if is_causal:
        raise ValueError("MINDIE_ATTN does not support causal attention")
    return mindiesd_attention_forward(q, k, v, attn_mask=attn_mask, scale=scale, head_first=True)


_try_import_flash_attn()
_try_import_sdpa()
_try_import_sage_attn()
_try_import_sparge_attn()
_try_import_flashinfer()
_try_import_sol_attn()
_try_import_mindie_attn()


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
    "SOL_ATTN_AVAILABLE",
    "MINDIE_ATTN_AVAILABLE",
    "flash_attn2",
    "flash_attn3",
    "flash_attn4",
    "flash_attn4_varlen",
    "sageattention",
    "flashinfer",
    "sol_attn",
    "supports_return_lse",
    "get_lse_fallback_impl",
    "sdpa_attn_cudnn",
    "sparge_attn",
    "mindie_attn",
]
