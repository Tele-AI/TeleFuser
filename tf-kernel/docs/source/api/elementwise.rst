Elementwise Operations
======================

.. automodule:: tf_kernel.elementwise

Activation Functions
--------------------

.. autofunction:: tf_kernel.silu_and_mul
.. autofunction:: tf_kernel.gelu_and_mul
.. autofunction:: tf_kernel.gelu_tanh_and_mul

Normalization
-------------

.. autofunction:: tf_kernel.rmsnorm
.. autofunction:: tf_kernel.fused_add_rmsnorm
.. autofunction:: tf_kernel.gemma_rmsnorm

Positional Encoding
-------------------

.. autofunction:: tf_kernel.apply_rope_with_cos_sin_cache_inplace
.. autofunction:: tf_kernel.rotary_embedding

Casting
-------

.. autofunction:: tf_kernel.downcast_fp8

Memory
------

.. autofunction:: tf_kernel.copy_to_gpu_no_ce
