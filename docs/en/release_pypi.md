# Releasing TeleFuser to PyPI

This release path publishes only the pure Python `telefuser` package. The CUDA
`tf-kernel` package should be released separately after prebuilt kernel wheels
are available.

## Prerequisites

- Python 3.10 or newer
- PyPI project owner or maintainer access
- A PyPI API token
- `build` and `twine`

```bash
python -m pip install --upgrade build twine
```

## Release Steps

1. Ensure the tree is clean and tests pass.

   ```bash
   git status --short
   pytest tests/
   ```

2. Create an annotated release tag. TeleFuser uses `setuptools-scm`, so the
   package version comes from the git tag.

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   ```

3. Build the source distribution and wheel.

   ```bash
   PYTHON=python3.10 scripts/build_telefuser_dist.sh
   ```

4. Upload to TestPyPI first.

   ```bash
   TWINE_REPOSITORY=testpypi scripts/publish_telefuser_pypi.sh
   ```

5. Install from TestPyPI in a clean environment and run a smoke test.

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple telefuser
   telefuser --help
   ```

6. Upload to PyPI.

   ```bash
   scripts/publish_telefuser_pypi.sh
   ```

## Token Setup

For interactive upload, `twine` will prompt for credentials. For CI or scripted
upload, use a PyPI API token:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
scripts/publish_telefuser_pypi.sh
```

## Notes

- Do not publish `tf-kernel` as part of the first TeleFuser release.
- Do not publish from an untagged commit. The build script blocks this by default.
- If a local smoke build is needed before tagging, run:

  ```bash
  SKIP_TAG_CHECK=1 PYTHON=python3.10 scripts/build_telefuser_dist.sh
  ```
