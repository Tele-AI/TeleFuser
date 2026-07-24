Installation
============

``tf-kernel`` is an independently released CUDA extension stored in the
TeleFuser monorepo. TeleFuser may consume the published wheel through an
optional extra or compile the in-tree project for joint development.

Requirements
------------

- Python 3.10 or newer
- PyTorch 2.11.0
- An NVIDIA GPU in the SM80, SM90, or SM100 target families
- CUDA Toolkit 12.8 or newer and CMake 3.26 or newer for source builds

A PyTorch local version such as ``2.11.0+cu128`` satisfies the package's
``torch==2.11.0`` requirement. FP4 operations require SM100 or newer.

Install a Published Wheel
-------------------------

Install ``tf-kernel`` independently:

.. code-block:: bash

   python -m pip install --upgrade tf-kernel

Or install it through TeleFuser's optional extra:

.. code-block:: bash

   python -m pip install "telefuser[kernel]"

From a TeleFuser checkout, the equivalent editable TeleFuser command is:

.. code-block:: bash

   python -m pip install -e ".[kernel]"

The extra resolves ``tf-kernel`` from the configured package index. It does
not compile the sibling ``tf-kernel/`` directory.

Joint Editable Development
--------------------------

Clone the TeleFuser monorepo and install both projects with the same Python
interpreter:

.. code-block:: bash

   git clone https://github.com/Tele-AI/TeleFuser.git
   cd TeleFuser
   PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel

The direct equivalent is:

.. code-block:: bash

   /path/to/venv/bin/python -m pip install -e ./tf-kernel -e ".[dev]"

Build an Architecture-Specific Wheel
------------------------------------

Enter the kernel project and select an architecture. Every target writes a
wheel to ``dist/`` and installs it into the interpreter selected by
``PYTHON``.

.. code-block:: bash

   cd TeleFuser/tf-kernel
   make build-auto PYTHON=/path/to/venv/bin/python
   make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere and Ada
   make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper
   make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell

``make build`` compiles all supported targets. A resource-bounded H100 build
can be run with:

.. code-block:: bash

   PATH=/usr/local/cuda-12.8/bin:$PATH \
   CUDA_HOME=/usr/local/cuda-12.8 \
   make build-sm90 \
     PYTHON=/path/to/venv/bin/python \
     MAX_JOBS=2 \
     CMAKE_BUILD_PARALLEL_LEVEL=2 \
     CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"

The first source build needs network access for pinned CUTLASS, FlashInfer,
and other CMake dependencies.

Verify Installation
-------------------

Run the smoke test with the interpreter that will load the package:

.. code-block:: bash

   python - <<'PY'
   from pathlib import Path
   import torch
   import tf_kernel

   print("tf-kernel:", tf_kernel.__version__)
   print("PyTorch:", torch.__version__)
   print("CUDA runtime:", torch.version.cuda)
   print("GPU:", torch.cuda.get_device_name())
   print("extension:", Path(tf_kernel.common_ops.__file__).resolve())

   x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
   weight = torch.ones(1024, device="cuda", dtype=torch.float16)
   output = tf_kernel.rmsnorm(x, weight)
   assert output.shape == x.shape and torch.isfinite(output).all()
   print("RMSNorm smoke test: OK")
   PY

Run ``python -m pip check`` afterward to find dependency conflicts. An H100
wheel should load its common extension from an ``sm90`` package directory.

Troubleshooting
---------------

Wrong Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``python -m pip`` consistently and pass the intended interpreter as the
Make ``PYTHON`` variable. Check it with ``python -m pip show tf-kernel`` and
``python -c "import sys; print(sys.executable)"``.

CUDA Toolkit Not Found
^^^^^^^^^^^^^^^^^^^^^^

Check ``nvcc --version`` and set ``CUDA_HOME`` to the CUDA 12.8+ toolkit. Put
``$CUDA_HOME/bin`` before older toolkits in ``PATH``.

Wrong GPU Architecture
^^^^^^^^^^^^^^^^^^^^^^

Rebuild with ``make build-auto`` on the target machine or explicitly select
``build-sm80``, ``build-sm90``, or ``build-sm100``.

High Build Resource Use
^^^^^^^^^^^^^^^^^^^^^^^

Reduce ``MAX_JOBS`` and ``CMAKE_BUILD_PARALLEL_LEVEL`` and set
``CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"``. Building one SM target also
reduces compilation time and wheel size.

FP4 Warning on Ampere or Hopper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The import-time message that FP4 operators are unavailable is expected on
SM80 and SM90. FP4 kernels are built only for SM100 and newer GPUs.

SageAttention Misaligned Address on H100
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The architecture-selected ``sageattn`` entry point currently routes H100 to
the SM90-specific FP8 implementation, which can fail with a CUDA misaligned
address in the validated build. Select another TeleFuser attention backend
until the focused SM90 GPU test passes on the deployed wheel. Restart the
process after this asynchronous CUDA error before running other kernels.
