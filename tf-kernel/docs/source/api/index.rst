API Reference
=============

This section contains the API reference for tf-kernel.

.. toctree::
   :maxdepth: 2

   elementwise
   gemm
   attention

Module Overview
---------------

- :mod:`tf_kernel.elementwise` - fused activations, normalization, RoPE, and casting
- :mod:`tf_kernel.gemm` - quantized matrix multiplication and quantization helpers
- :mod:`tf_kernel.sageattn2` - architecture-selected SageAttention v2
- :mod:`tf_kernel.sageattn3` - Blackwell FP4 SageAttention v3
- :mod:`tf_kernel.block_sparse_attn` - block-sparse and streaming attention kernels
- :mod:`tf_kernel.memory` - CUDA memory helpers
