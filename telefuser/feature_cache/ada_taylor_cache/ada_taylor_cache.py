"""AdaTaylorCache: Hybrid Strategy with Fallback to Residual Reuse.

This module implements AdaTaylorCache with a hybrid strategy:
- When elapsed <= taylor_threshold: Use Taylor series expansion
- When elapsed > taylor_threshold: Fall back to simple residual reuse

The key insight is that Taylor expansion errors accumulate for large elapsed values,
so we switch to residual reuse for better stability.

When n_derivatives=0, AdaTaylorCache reduces to simple residual caching (no Taylor expansion).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from telefuser.utils.logging import logger

from ..base import BaseFeatureCache


def _get_safe_model_name(model_name: str) -> str:
    """Convert model name to safe filename by replacing special characters."""
    return model_name.replace(".", "_").replace("/", "_")


def load_cache_params(model_name: str) -> dict:
    """
    Load cache parameters for a given model name from JSON file.

    Args:
        model_name: The model name/key (e.g., "Wan2.1-T2V-1.3B")

    Returns:
        dict: The parameters containing K, retention_ratio, thresh,
              cond_mag_ratios, uncond_mag_ratios

    Raises:
        FileNotFoundError: If the parameter file for the model doesn't exist
    """
    safe_name = _get_safe_model_name(model_name)
    params_dir = Path(__file__).parent / "params"
    params_file = params_dir / f"{safe_name}.json"

    if not params_file.exists():
        raise FileNotFoundError(f"Parameters not found for model '{model_name}'. Expected file: {params_file}")

    with open(params_file, "r") as f:
        params = json.load(f)

    return params


def nearest_interp(src_array: np.ndarray | list, target_length: int) -> np.ndarray:
    src_length = len(src_array)
    src_array = np.array(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


@dataclass
class AdaTaylorCacheConfig:
    """Configuration for AdaTaylorCache feature caching.

    Attributes:
        enabled: Whether to enable AdaTaylorCache.
        model_type: Model type for loading cache parameters.
        n_derivatives: Order of Taylor series expansion (default: 1).
        num_inference_steps: Total number of inference steps.
        taylor_threshold: Threshold for switching to residual reuse (default: 2).
            When elapsed <= threshold: use Taylor expansion.
            When elapsed > threshold: use residual reuse.
        init_step: Initial step number (default: 0).
    """

    enabled: bool = True
    model_type: str = ""
    n_derivatives: int = 1
    num_inference_steps: int = 50
    taylor_threshold: int = 2
    init_step: int = 0

    def __post_init__(self):
        if self.n_derivatives < 0:
            raise ValueError("n_derivatives must be non-negative")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if self.taylor_threshold < 1:
            raise ValueError("taylor_threshold must be at least 1")

    def __repr__(self) -> str:
        return (
            f"AdaTaylorCacheConfig(enabled={self.enabled}, "
            f"model_type='{self.model_type}', "
            f"n_derivatives={self.n_derivatives}, "
            f"taylor_threshold={self.taylor_threshold}, "
            f"steps={self.num_inference_steps})"
        )


class AdaTaylorCacheState:
    """State container for AdaTaylorCache with hybrid strategy.

    Uses adaptive skip logic with hybrid approximation:
    - Small elapsed: Taylor series expansion for higher accuracy
    - Large elapsed: Simple residual reuse for stability

    Args:
        num_inference_steps: Total number of inference steps.
        thresh: Error threshold for skip decision.
        K: Maximum number of consecutive skip steps.
        mag_ratios: Magnitude ratios array for error accumulation.
        retention_ratio: Ratio of initial steps to always compute.
        n_derivatives: Order of Taylor series expansion.
        taylor_threshold: Threshold for switching to residual reuse.
        init_step: Initial step number.
    """

    def __init__(
        self,
        num_inference_steps: int,
        thresh: float,
        K: int,
        mag_ratios: np.ndarray,
        retention_ratio: float,
        n_derivatives: int = 1,
        taylor_threshold: int = 2,
        init_step: int = 0,
    ):
        self.num_inference_steps = num_inference_steps
        self.init_step = init_step
        self.thresh = thresh
        self.start_step = int(num_inference_steps * retention_ratio)
        self.K = K
        self.n_derivatives = n_derivatives
        self.order = n_derivatives + 1
        self.mag_ratios = mag_ratios
        self.taylor_threshold = taylor_threshold

        # Pre-compute compute steps at initialization (stored as set for O(1) lookup)
        self._compute_steps_set = self._precompute_compute_steps_set()

        # Internal step tracking - each path manages its own step counter
        # Incremented on each pre_forward call, independent of external mark_step_begin
        self._internal_step = init_step - 1  # Will be incremented to init_step on first call
        self.last_compute_step = init_step - 1

        # Derivative storage
        self.derivatives: Dict[str, List[Optional[torch.Tensor]]] = {
            "dR_prev": [None] * self.order,
            "dR_current": [None] * self.order,
        }

        # Cache the last computed residual for fallback
        self.last_residual: Optional[torch.Tensor] = None

        self.reset()

    def _precompute_compute_steps_set(self) -> set:
        """Pre-compute which steps require real computation.

        Since the decision is deterministic given fixed parameters (mag_ratios, K, thresh),
        we can compute this once at initialization instead of repeatedly during inference.

        Returns:
            Set of step indices that require real computation.
        """
        compute_steps = set()
        accumulated_err = 0.0
        accumulated_steps = 0
        accumulated_ratio = 1.0

        for step in range(self.num_inference_steps):
            # Always compute during warmup period, at last step, or at init step
            if step <= self.start_step or step == self.num_inference_steps - 1 or step == self.init_step:
                compute_steps.add(step)
                accumulated_err = 0.0
                accumulated_steps = 0
                accumulated_ratio = 1.0
            else:
                cur_mag_ratio = self.mag_ratios[step]
                accumulated_ratio = accumulated_ratio * cur_mag_ratio
                accumulated_steps += 1
                cur_skip_err = np.abs(1 - accumulated_ratio)
                accumulated_err += cur_skip_err

                if accumulated_err < self.thresh and accumulated_steps <= self.K:
                    # Skip this step
                    pass
                else:
                    # Compute this step and reset accumulators
                    compute_steps.add(step)
                    accumulated_err = 0.0
                    accumulated_steps = 0
                    accumulated_ratio = 1.0

        return compute_steps

    def reset(self):
        """Reset all cached state."""
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.accumulated_ratio = 1.0
        self.derivatives = {
            "dR_prev": [None] * self.order,
            "dR_current": [None] * self.order,
        }
        self.last_residual = None
        self._internal_step = self.init_step - 1
        self.last_compute_step = self.init_step - 1

    def _increment_step(self) -> int:
        """Increment internal step counter and return the new step.

        Called at the beginning of each pre_forward to mark a new diffusion step.

        Returns:
            The new internal step number.
        """
        self._internal_step += 1
        return self._internal_step

    @property
    def current_step(self) -> int:
        """Get the current internal step number."""
        return self._internal_step

    def should_compute(self) -> bool:
        """Determine if real computation is needed for this step.

        Uses pre-computed compute steps for O(1) lookup instead of recalculating
        error accumulation on every call.

        Returns:
            True if real computation is needed, False if can skip and use approximation.
        """
        # O(1) lookup using pre-computed set
        return self._internal_step in self._compute_steps_set

    @property
    def compute_steps(self) -> set:
        """Get the set of steps that require real computation.

        Returns:
            Set of step indices that require real computation.
        """
        return self._compute_steps_set

    def compute_derivatives(self, residual: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """Compute derivatives of residual using finite differences.

        Uses window (_internal_step - last_compute_step) as the difference interval,
        consistent with TaylorSeer approach.

        Args:
            residual: Current residual tensor (output - input).

        Returns:
            List of residual derivative tensors [dR_0, dR_1, ..., dR_n].
        """
        dR_current: List[Optional[torch.Tensor]] = [None] * self.order
        dR_current[0] = residual

        # Use window as the difference interval
        window = self._internal_step - self.last_compute_step
        if window <= 0:
            window = 1

        # Check shape compatibility
        if self.derivatives["dR_prev"][0] is not None:
            if dR_current[0].shape != self.derivatives["dR_prev"][0].shape:
                self.reset()
                return dR_current

        # Compute higher-order derivatives using finite differences
        for i in range(self.n_derivatives):
            if self.derivatives["dR_prev"][i] is not None and dR_current[i] is not None and self._internal_step > 0:
                dR_current[i + 1] = dR_current[i] - self.derivatives["dR_prev"][i]
                dR_current[i + 1] = dR_current[i + 1] / window
            else:
                break

        return dR_current

    def update(self, output: torch.Tensor, ori_input: torch.Tensor):
        """Update derivative cache with newly computed output and original input.

        This should be called after real computation to store the residual
        and compute its derivatives for Taylor approximation.

        Args:
            output: Newly computed output tensor from DiT block.
            ori_input: Original input tensor to DiT block.
        """
        # Compute residual: R = output - input
        residual = output - ori_input

        # Cache the residual for fallback
        self.last_residual = residual.detach().clone()

        # Store previous derivatives and compute new ones
        self.derivatives["dR_prev"] = self.derivatives["dR_current"]
        self.derivatives["dR_current"] = self.compute_derivatives(residual)
        self.last_compute_step = self._internal_step

    def approximate_residual(self) -> Optional[torch.Tensor]:
        """Approximate residual using hybrid strategy.

        When elapsed <= taylor_threshold: Use Taylor series expansion
        When elapsed > taylor_threshold: Fall back to simple residual reuse

        Returns:
            Approximated residual tensor, or None if no derivatives available.
        """
        elapsed = self._internal_step - self.last_compute_step

        # Check if we should use Taylor or fallback to residual reuse
        if elapsed > self.taylor_threshold:
            # Fallback: Use simple residual reuse
            # This avoids Taylor expansion errors for large elapsed
            if self.last_residual is not None:
                return self.last_residual
            return self.derivatives["dR_current"][0] if self.derivatives["dR_current"][0] is not None else None

        # Use Taylor expansion for small elapsed
        output: Optional[torch.Tensor] = None

        for i, derivative in enumerate(self.derivatives["dR_current"]):
            if derivative is not None:
                term = (1.0 / math.factorial(i)) * derivative * (elapsed**i)
                if output is None:
                    output = term
                else:
                    output = output + term
            else:
                break

        return output

    def approximate(self, current_input: torch.Tensor) -> torch.Tensor:
        """Approximate output using hybrid strategy.

        When elapsed <= taylor_threshold: Use Taylor expansion of residual
        When elapsed > taylor_threshold: Use simple residual reuse

        Args:
            current_input: Current input tensor to DiT block.

        Returns:
            Approximated output tensor.
        """
        residual_approx = self.approximate_residual()
        if residual_approx is None:
            return current_input
        return current_input + residual_approx


class AdaTaylorCache(BaseFeatureCache):
    """Adaptive Taylor Cache with hybrid strategy.

    Manages AdaTaylorCache states for both conditional (cond) and unconditional (uncond)
    paths in classifier-free guidance (CFG) diffusion models.

    This implementation uses a hybrid strategy:
    - When elapsed <= taylor_threshold: Use Taylor series expansion (higher accuracy)
    - When elapsed > taylor_threshold: Fall back to residual reuse (more stable)

    Args:
        model_type: Model type for loading cache parameters (e.g., "Wan2.1-T2V-1.3B").
        num_inference_steps: Total number of inference steps.
        n_derivatives: Order of Taylor series expansion (default: 1).
        taylor_threshold: Threshold for switching to residual reuse (default: 2).
        init_step: Initial step number (default: 0).
    """

    def __init__(
        self,
        model_type: str,
        num_inference_steps: int,
        n_derivatives: int = 1,
        taylor_threshold: int = 2,
        init_step: int = 0,
    ):
        # Load cache parameters
        params = load_cache_params(model_type)

        self.num_inference_steps = num_inference_steps
        self.n_derivatives = n_derivatives
        self.taylor_threshold = taylor_threshold
        self.init_step = init_step
        self.model_type = model_type

        # Interpolate mag_ratios if needed
        cond_mag_ratios = params["cond_mag_ratios"]
        uncond_mag_ratios = params["uncond_mag_ratios"]
        if len(cond_mag_ratios) != num_inference_steps:
            cond_mag_ratios = nearest_interp(cond_mag_ratios, target_length=num_inference_steps)
        if len(uncond_mag_ratios) != num_inference_steps:
            uncond_mag_ratios = nearest_interp(uncond_mag_ratios, target_length=num_inference_steps)

        # Initialize cond and uncond states - each manages its own step counter
        self.cond_state = AdaTaylorCacheState(
            num_inference_steps=num_inference_steps,
            thresh=params["thresh"],
            K=params["K"],
            mag_ratios=cond_mag_ratios,
            retention_ratio=params["retention_ratio"],
            n_derivatives=n_derivatives,
            taylor_threshold=taylor_threshold,
            init_step=init_step,
        )
        self.uncond_state = AdaTaylorCacheState(
            num_inference_steps=num_inference_steps,
            thresh=params["thresh"],
            K=params["K"],
            mag_ratios=uncond_mag_ratios,
            retention_ratio=params["retention_ratio"],
            n_derivatives=n_derivatives,
            taylor_threshold=taylor_threshold,
            init_step=init_step,
        )

        logger.info(
            f"Initialized AdaTaylorCache: model_type='{model_type}', "
            f"num_steps={num_inference_steps}, n_derivatives={n_derivatives}, "
            f"taylor_threshold={taylor_threshold}, thresh={params['thresh']}, K={params['K']}"
        )

    def reset(self) -> None:
        """Reset all states."""
        self.cond_state.reset()
        self.uncond_state.reset()

    def should_compute(self, is_cond: bool) -> bool:
        """Check if real computation is needed for this step.

        This method automatically increments the internal step counter
        for the corresponding path before checking.

        Args:
            is_cond: True for conditional path, False for unconditional path

        Returns:
            True if real computation is needed, False if can use approximation
        """
        state = self.cond_state if is_cond else self.uncond_state
        state._increment_step()
        return state.should_compute()

    def update(self, output: torch.Tensor, ori_input: torch.Tensor, is_cond: bool) -> None:
        """Update cache with newly computed output and original input.

        Stores the residual (output - input) and computes its derivatives
        for Taylor series approximation.

        Args:
            output: Newly computed output tensor from DiT block.
            ori_input: Original input tensor to DiT block.
            is_cond: True for conditional path, False for unconditional path.
        """
        state = self.cond_state if is_cond else self.uncond_state
        state.update(output, ori_input)

    def approximate(self, input: torch.Tensor, is_cond: bool) -> torch.Tensor:
        """Get approximated output using hybrid strategy.

        Args:
            input: Current input tensor to DiT block.
            is_cond: True for conditional path, False for unconditional path.

        Returns:
            Approximated output tensor.
        """
        state = self.cond_state if is_cond else self.uncond_state
        return state.approximate(input)

    def get_compute_steps(self) -> List[int]:
        """Get list of steps where real computation will occur (for debugging).

        Returns:
            Sorted list of step indices that require real computation.
        """
        return sorted(self.cond_state.compute_steps)

    def __repr__(self) -> str:
        return f"AdaTaylorCache(n_derivatives={self.n_derivatives}, \
            taylor_threshold={self.taylor_threshold}, steps={self.num_inference_steps})"


class _AdaTaylorCacheCalibratorState:
    """Internal state container for calibrating a single path (cond or uncond)."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        self.cnt = 0
        self.norm_ratio = []
        self.residual_cache = None

    def store(self, x: torch.Tensor, ori_x: torch.Tensor) -> None:
        """Store residual and compute norm_ratio."""
        residual_x = x - ori_x
        if self.cnt >= 1:
            # Compute norm_ratio between current and previous residual
            norm_ratio = ((residual_x.norm(dim=-1) / self.residual_cache.norm(dim=-1)).mean()).item()
            self.norm_ratio.append(round(norm_ratio, 5))
            print(f"  step {self.cnt + 1}: norm_ratio={norm_ratio:.5f}")
        self.residual_cache = residual_x
        self.cnt += 1

    def get_mag_ratios(self) -> list:
        """Get mag_ratios with 1.0 prepended."""
        return [1.0] + self.norm_ratio

    def is_finished(self) -> bool:
        return self.cnt >= self.num_steps


class AdaTaylorCacheCalibrator(BaseFeatureCache):
    """Calibrator for generating AdaTaylorCache parameters.

    This calibrator runs the pipeline once and collects norm_ratio data
    for both cond and uncond paths, then outputs a JSON file with parameters.

    Usage:
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=50,
            sigma_shift=8.0,
            model_name="Wan2.1-T2V-1.3B",
            output_path="path/to/params.json"
        )
        # During inference, call update() after each block forward
        # After all steps, calibration file is auto-saved
    """

    def __init__(
        self,
        num_inference_steps: int,
        sigma_shift: float,
        model_name: str,
        output_path: str | None = None,
    ):
        print(f"Init AdaTaylorCacheCalibrator: num_steps={num_inference_steps}, sigma_shift={sigma_shift}, model={model_name}")
        self.num_inference_steps = num_inference_steps
        self.sigma_shift = sigma_shift
        self.model_name = model_name

        # Set output path
        if output_path is None:
            safe_name = _get_safe_model_name(model_name)
            params_dir = Path(__file__).parent / "params"
            output_path = str(params_dir / f"{safe_name}.json")
        self.output_path = output_path

        # Initialize cond and uncond calibrators
        self.cond_calibrator = _AdaTaylorCacheCalibratorState(num_inference_steps)
        self.uncond_calibrator = _AdaTaylorCacheCalibratorState(num_inference_steps)

        self._finished = False

    def reset(self) -> None:
        """Reset all calibrator states."""
        self.cond_calibrator.reset()
        self.uncond_calibrator.reset()
        self._finished = False

    def should_compute(self, is_cond: bool) -> bool:
        """Always returns True - calibrator needs to compute all steps."""
        return True

    def update(self, x: torch.Tensor, ori_x: torch.Tensor, is_cond: bool) -> None:
        """Store residual for calibration.

        Args:
            x: Output tensor from DiT block.
            ori_x: Original input tensor.
            is_cond: True for conditional path, False for unconditional path.
        """
        calibrator = self.cond_calibrator if is_cond else self.uncond_calibrator
        calibrator.store(x, ori_x)
        self._check_and_save()

    def approximate(self, input: torch.Tensor, is_cond: bool) -> torch.Tensor:
        """Calibrator always computes, so this returns input unchanged."""
        return input

    def _check_and_save(self):
        """Check if calibration is finished and save results."""
        if self._finished:
            return

        if self.cond_calibrator.is_finished() and self.uncond_calibrator.is_finished():
            self.save()
            self._finished = True

    def save(self):
        """Save calibration results to JSON file."""
        # Calculate smart defaults based on num_inference_steps
        default_K = min(4, max(1, self.num_inference_steps // 10))
        default_retention = 0.2
        default_thresh = 0.12

        result = {
            "K": default_K,
            "retention_ratio": default_retention,
            "thresh": default_thresh,
            "sigma_shift": self.sigma_shift,
            "num_inference_steps": self.num_inference_steps,
            "cond_mag_ratios": self.cond_calibrator.get_mag_ratios(),
            "uncond_mag_ratios": self.uncond_calibrator.get_mag_ratios(),
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        logger.info(f"Saving calibration results to {self.output_path}")
        logger.info(f"  cond_mag_ratios: {len(result['cond_mag_ratios'])} values")
        logger.info(f"  uncond_mag_ratios: {len(result['uncond_mag_ratios'])} values")

        with open(self.output_path, "w") as f:
            json.dump(result, f, indent=4)

        print("\nCalibration completed!")
        print(f"Output file: {self.output_path}")
        print(f"Default parameters: K={default_K}, retention_ratio={default_retention}, thresh={default_thresh}")
        print("Adjust these values based on your quality/speed requirements:")
        print("  - Higher K: More aggressive skipping, faster inference")
        print("  - Higher retention_ratio: More initial steps computed, better quality")
        print("  - Higher thresh: More tolerant to errors, faster inference")
