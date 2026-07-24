Development Guide
=================

This guide is for developers who want to contribute to tf-kernel.

Setting up Development Environment
----------------------------------

1. Clone the TeleFuser monorepo and enter the kernel project:

   .. code-block:: bash

      git clone https://github.com/Tele-AI/TeleFuser.git
      cd TeleFuser/tf-kernel

2. Install development dependencies:

   .. code-block:: bash

      python -m pip install -e ".[dev]"

3. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

4. Build the project:

   .. code-block:: bash

      make build-auto PYTHON=/path/to/venv/bin/python

Project Structure
-----------------

.. code-block:: text

   tf-kernel/
   ├── csrc/              # C++/CUDA source files
   │   ├── elementwise/   # Elementwise operations
   │   ├── gemm/          # GEMM operations
   │   ├── sageattn2/     # SageAttention v2
   │   ├── sageattn3/     # SageAttention v3
   │   └── block_sparse_attn/  # Block sparse attention
   ├── tf_kernel/         # Python package
   ├── include/           # C++ headers
   ├── tests/             # Test suite
   ├── benchmark/         # Benchmarks
   └── docs/              # Documentation

Adding a New Kernel
-------------------

1. **Implement the kernel** in ``csrc/<category>/your_kernel.cu``

2. **Declare the interface** in ``include/tf_kernel_ops.h``

3. **Register with PyTorch** in ``csrc/common_extension.cc``:

   .. code-block:: cpp

      m.def("your_kernel(Tensor input, Tensor! output) -> ()");
      m.impl("your_kernel", torch::kCUDA, &your_kernel);

4. **Update CMakeLists.txt**: Add source file to ``SOURCES``

5. **Create Python wrapper** in ``tf_kernel/<category>.py``

6. **Export in** ``tf_kernel/__init__.py``

7. **Add tests** in ``tests/test_your_kernel.py``

8. **Add benchmarks** in ``benchmark/`` (if applicable)

Coding Standards
----------------

C++/CUDA
^^^^^^^^

- Use clang-format with the provided ``.clang-format`` config
- 2-space indentation
- 120 column limit
- Left pointer alignment (``int* ptr`` not ``int *ptr``)

Format C++/CUDA files:

.. code-block:: bash

   make format

Python
^^^^^^

- **isort**: Import sorting
- ~~**black**: Code formatting~~ (Disabled)
- **ruff**: Linting

Format Python files:

.. code-block:: bash

   make format

Running Tests
-------------

Run all tests:

.. code-block:: bash

   make test

Run with coverage:

.. code-block:: bash

   make test-cov

Run tests that don't require GPU:

.. code-block:: bash

   make test-cpu

Building Documentation
----------------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install ".[docs]"

2. Build the documentation:

   .. code-block:: bash

      cd docs
      make html

3. View the documentation:

   .. code-block:: bash

      open build/html/index.html
