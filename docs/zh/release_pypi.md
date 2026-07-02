# 发布 TeleFuser 到 PyPI

这个流程只发布纯 Python 的 `telefuser` 主包。CUDA `tf-kernel` 包应在预编译 kernel wheel 准备好之后单独发布。

## 前置条件

- Python 3.10 或更新版本
- PyPI 项目的 owner 或 maintainer 权限
- PyPI API token
- `build` 和 `twine`

```bash
python -m pip install --upgrade build twine
```

## 发布步骤

1. 确认工作区干净，并通过测试。

   ```bash
   git status --short
   pytest tests/
   ```

2. 创建 annotated release tag。TeleFuser 使用 `setuptools-scm`，包版本号来自 git tag。

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   ```

3. 构建 sdist 和 wheel。

   ```bash
   PYTHON=python3.10 scripts/build_telefuser_dist.sh
   ```

4. 先上传到 TestPyPI。

   ```bash
   TWINE_REPOSITORY=testpypi scripts/publish_telefuser_pypi.sh
   ```

5. 在干净环境中从 TestPyPI 安装并做 smoke test。

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple telefuser
   telefuser --help
   ```

6. 上传到正式 PyPI。

   ```bash
   scripts/publish_telefuser_pypi.sh
   ```

## Token 配置

交互式上传时，`twine` 会提示输入凭据。CI 或脚本上传时，使用 PyPI API token：

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
scripts/publish_telefuser_pypi.sh
```

## 注意事项

- 第一版不要把 `tf-kernel` 和 `telefuser` 一起发布。
- 不要从未打 tag 的 commit 发布。构建脚本默认会阻止这种操作。
- 如果只想在本地做未打 tag 的 smoke build，可以运行：

  ```bash
  SKIP_TAG_CHECK=1 PYTHON=python3.10 scripts/build_telefuser_dist.sh
  ```
