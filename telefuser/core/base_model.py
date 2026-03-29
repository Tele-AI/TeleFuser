"""Base model class with device management and distributed training support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed.device_mesh as device_mesh

from telefuser.utils.logging import logger

from .config import AttentionConfig, AttnImplType, OffloadConfig

if TYPE_CHECKING:
    from telefuser.feature_cache import (
        AdaTaylorCacheCalibratorHook,
        AdaTaylorCacheHook,
        FeatureCacheHookManager,
        ResidualAnalyzerHook,
    )


class BaseModel(torch.nn.Module):
    """Base model with quantization, distributed training and memory management support.

    Provides functionality for device management (load/unload), quantization,
    distributed training (FSDP, TP), and asynchronous offloading.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quant_type: str | torch.dtype | None = None
        self.device_mesh: device_mesh.DeviceMesh | None = None
        self.async_offload_flag: bool = False
        self.usp_flag: bool = False  # Unified Sparse Processing
        self.pp_flag: bool = False  # Pipeline Parallelism
        self.fsdp_flag: bool = False
        self.attention_config: AttentionConfig = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        # Layer/block names for async offload (e.g., ["layers", "blocks"])
        self.layer_name_list: list[str] = []
        # Whether to use pinned CPU memory for faster H2D transfer
        self._use_pinned_memory: bool = True
        # Feature cache hook manager (lazy initialization)
        self._feature_cache_hook: FeatureCacheHookManager | None = None

    @property
    def feature_cache_hook(self) -> FeatureCacheHookManager:
        """Get or create the feature cache hook manager."""
        if self._feature_cache_hook is None:
            from telefuser.feature_cache import FeatureCacheHookManager

            self._feature_cache_hook = FeatureCacheHookManager()
        return self._feature_cache_hook

    def onload_device(self, device: torch.device, non_blocking: bool | None = None) -> None:
        """Load model to specified device.

        Args:
            device: Target device.
            non_blocking: Use non-blocking transfer. If None, uses _use_pinned_memory.
        """
        # Handle sequential offload: delegate to wrapped modules
        if getattr(self, "sequential_cpu_offload_enabled", False):
            for module in self.modules():
                if hasattr(module, "onload"):
                    module.onload()
            return

        # Handle async offload: only load non-layer modules
        if self.async_offload_flag:
            if not self.layer_name_list:
                logger.warning("model's layers name is not set while async offload is enabled")
            for name, module in self.named_children():
                if name not in self.layer_name_list:
                    module.to(device)
            return

        # Standard offload: move all tensors to device
        use_non_blocking = (
            non_blocking if non_blocking is not None else self._use_pinned_memory
        ) and device.type == "cuda"
        from telefuser.offload import move_tensors_to_device

        move_tensors_to_device(self.parameters(), device, use_non_blocking)
        move_tensors_to_device(self.buffers(), device, use_non_blocking)
        if use_non_blocking:
            torch.cuda.current_stream().synchronize()

    def offload_device(self, pin_memory: bool | None = None) -> None:
        """Offload model to CPU to free GPU memory.

        Args:
            pin_memory: Use pinned CPU memory for faster transfer back. If None, uses _use_pinned_memory.
        """
        # Handle sequential offload: delegate to wrapped modules
        if getattr(self, "sequential_cpu_offload_enabled", False):
            for module in self.modules():
                if hasattr(module, "offload"):
                    module.offload()
            return

        # Handle async offload: only offload non-layer modules
        if self.async_offload_flag:
            if not self.layer_name_list:
                logger.warning("model's layers name is not set while async offload is enabled")
            for name, module in self.named_children():
                if name not in self.layer_name_list:
                    module.to("cpu")
            return

        # Standard offload: move all tensors to CPU
        use_pin_memory = pin_memory if pin_memory is not None else self._use_pinned_memory
        if use_pin_memory:
            import gc

            from telefuser.offload import move_tensors_to_pinned_cpu

            move_tensors_to_pinned_cpu(self.parameters())
            move_tensors_to_pinned_cpu(self.buffers())
            gc.collect()
        else:
            self.to("cpu")

    def get_fsdp_module_names(self) -> list[str]:
        """Get module names that require FSDP sharding. Empty list means all modules."""
        raise NotImplementedError()

    def enable_quant(self, quant_type: str) -> None:
        """Enable model quantization (e.g., 'int8', 'int4')."""
        self.quant_type = quant_type

    def get_tp_plan(self) -> dict:
        """Get tensor parallelism plan configuration."""
        raise NotImplementedError()

    @staticmethod
    def state_dict_converter() -> callable:
        """Get state dict converter for model format conversion."""
        raise NotImplementedError()

    def enable_usp(self) -> None:
        """Enable Unified Sparse Processing for sparse attention models."""
        raise NotImplementedError()

    def enable_async_offload(self, device: torch.device, offload_config: OffloadConfig) -> None:
        """Enable asynchronous memory offloading to optimize memory usage."""
        raise NotImplementedError()

    def enable_sequential_cpu_offload(
        self, device: torch.device, torch_dtype: torch.dtype, *args: any, **kwargs: any
    ) -> None:
        """Enable sequential CPU offloading for VRAM management."""
        raise NotImplementedError()

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        """Set attention configuration (dense/sparse implementations)."""
        self.attention_config = attention_config

    def set_ada_taylor_cache(
        self,
        num_inference_steps: int,
        model_type: str,
        n_derivatives: int = 1,
        taylor_threshold: int = 2,
        init_step: int = 0,
    ) -> None:
        """Set up AdaTaylorCache feature caching.

        Args:
            num_inference_steps: Total number of inference steps.
            model_type: Model type for loading cache parameters.
            n_derivatives: Order of Taylor series expansion (default: 1).
            taylor_threshold: Threshold for switching to residual reuse (default: 2).
            init_step: Initial step number (default: 0).
        """
        from telefuser.feature_cache import AdaTaylorCacheHook

        self.feature_cache_hook.set_hook(
            AdaTaylorCacheHook(
                model_type=model_type,
                num_inference_steps=num_inference_steps,
                n_derivatives=n_derivatives,
                taylor_threshold=taylor_threshold,
                init_step=init_step,
            )
        )
        logger.info(f"AdaTaylorCache enabled: model_type={model_type}, num_steps={num_inference_steps}")

    def set_ada_taylor_cache_calibrator(
        self,
        num_inference_steps: int,
        sigma_shift: float,
        model_name: str,
        output_path: str | None = None,
    ) -> None:
        """Set up calibrator mode for generating AdaTaylorCache parameters.

        Args:
            num_inference_steps: Number of inference steps.
            sigma_shift: Sigma shift value used in the scheduler.
            model_name: Model name for the output file.
            output_path: Output path for the JSON file.
        """
        from telefuser.feature_cache import AdaTaylorCacheCalibratorHook

        self.feature_cache_hook.set_hook(
            AdaTaylorCacheCalibratorHook(
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                model_name=model_name,
                output_path=output_path,
            )
        )
        logger.info(f"AdaTaylorCache calibrator enabled for {model_name}")

    def set_residual_analyzer(self, analyzer) -> None:
        """Set up residual analyzer for Taylor approximation error analysis."""
        from telefuser.feature_cache import ResidualAnalyzerHook

        self.feature_cache_hook.set_hook(ResidualAnalyzerHook(analyzer))
        logger.info("Residual analyzer enabled")

    def reset_feature_cache(self) -> None:
        """Reset feature cache hook."""
        if self._feature_cache_hook is not None:
            self._feature_cache_hook.clear_hook()

    def mark_compile_static(self) -> None:
        """Mark module runtime states as static for torch.compile compatibility.

        This method should be called before torch.compile() to prevent
        dynamic conditions on runtime flags from causing graph breaks.

        Marks the module class as static, indicating that instance attributes
        will not be modified after compilation. This ensures integer attributes
        remain constant (not symbolic) during tracing.

        Example:
            model.mark_compile_static()
            model.compile()
        """
        # Import mark_static from torch._dynamo
        try:
            from torch._dynamo import mark_static
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")
            return

        # Mark this module class as static (instance attributes won't change after compile)
        # This prevents integer attributes like flags from becoming symbolic integers
        mark_static(self.__class__)

        logger.info(f"Model class {self.__class__.__name__} marked as static for torch.compile")

    def compile(self, **kwargs) -> None:
        """Compile model forward for better performance.

        This method marks runtime states as static before compilation
        to prevent graph breaks from dynamic conditions.

        Args:
            **kwargs: Arguments passed to torch.compile()
        """
        # Mark static states before compilation
        self.mark_compile_static()
        # Subclasses should override to specify what to compile
        logger.info("Model compilation enabled")
