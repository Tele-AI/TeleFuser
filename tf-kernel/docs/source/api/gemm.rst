GEMM Operations
===============

.. automodule:: tf_kernel.gemm

Quantized Matrix Multiplication
-------------------------------

INT8
^^^^

.. autofunction:: tf_kernel.int8_scaled_mm

FP8
^^^

.. autofunction:: tf_kernel.fp8_scaled_mm
.. autofunction:: tf_kernel.fp8_blockwise_scaled_mm
.. autofunction:: tf_kernel.bmm_fp8

FP4 (Blackwell)
^^^^^^^^^^^^^^^

.. autofunction:: tf_kernel.gemm.cutlass_scaled_fp4_mm

Quantization Functions
----------------------

Per-Token
^^^^^^^^^

.. autofunction:: tf_kernel.tf_per_token_quant_fp8

Per-Tensor
^^^^^^^^^^

.. autofunction:: tf_kernel.tf_per_tensor_quant_fp8

Per-Token Group
^^^^^^^^^^^^^^^

.. autofunction:: tf_kernel.tf_per_token_group_quant_fp8
.. autofunction:: tf_kernel.tf_per_token_group_quant_int8
