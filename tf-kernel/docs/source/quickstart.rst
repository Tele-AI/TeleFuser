Quickstart
==========

Install the published package and verify it as described in :doc:`installation`:

.. code-block:: bash

   python -m pip install --upgrade tf-kernel

TeleFuser Usage
---------------

TeleFuser model code should import from ``telefuser.ops``. The ops layer uses
the optimized extension for supported eager CUDA paths and retains framework
fallbacks where implemented.

.. code-block:: python

   import torch
   from telefuser.ops.activations import silu_and_mul

   # SwiGLU input stores gate and value in the final dimension.
   x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
   output = silu_and_mul(x)
   assert output.shape == (4, 1024)

Standalone Elementwise Operations
---------------------------------

Direct imports are appropriate for standalone kernel use, diagnostics, and
kernel development.

.. code-block:: python

   import torch
   import tf_kernel

   x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
   weight = torch.ones(1024, device="cuda", dtype=torch.float16)
   normalized = tf_kernel.rmsnorm(x, weight, eps=1e-6)

   # silu_and_mul splits the final dimension into equal gate/value tensors.
   gated_input = torch.randn(8, 2048, device="cuda", dtype=torch.float16)
   gated_output = tf_kernel.silu_and_mul(gated_input)
   assert gated_output.shape == (8, 1024)

FP8 Quantization
----------------

Per-token quantization writes into caller-provided tensors; it does not return
a tuple.

.. code-block:: python

   import torch
   import tf_kernel

   x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
   x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
   x_scale = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
   tf_kernel.tf_per_token_quant_fp8(x, x_q, x_scale)

SageAttention
-------------

Query, key, and value tensors must be CUDA FP16 or BF16 tensors. With ``HND``
layout, their dimensions are ``[batch, heads, sequence, head_dim]``. The
following generic FP8 path passes the current H100 smoke test:

.. code-block:: python

   import torch
   import tf_kernel

   q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
   k = torch.randn_like(q)
   v = torch.randn_like(q)
   output = tf_kernel.sageattn_qk_int8_pv_fp8_cuda(
       q,
       k,
       v,
       tensor_layout="HND",
       is_causal=False,
       pv_accum_dtype="fp32",
   )
   assert output.shape == q.shape

.. warning::

   In the currently validated H100 build, the architecture-selected
   ``tf_kernel.sageattn()`` path chooses the SM90-specific FP8 implementation
   and can fail with ``CUDA error: misaligned address``. Do not enable that
   backend in production until its focused GPU test passes on the deployed
   wheel.

Performance Tips
----------------

1. Build only the target architecture with ``make build-auto`` or an explicit
   ``build-sm*`` target.
2. Keep tensors contiguous and use the documented dtype and layout contracts.
3. For repeated static shapes, evaluate CUDA Graphs to reduce launch overhead.
4. Benchmark end-to-end TeleFuser ops as well as isolated kernels before
   selecting a backend for production.

Next Steps
----------

- Read :doc:`installation` for source builds and troubleshooting.
- Check :doc:`api/index` for detailed API documentation.
- See :doc:`development` for contributing guidelines.
- Browse the `tf-kernel source <https://github.com/Tele-AI/TeleFuser/tree/main/tf-kernel>`_.
