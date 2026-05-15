"""Tests for utility functions."""

import os
import sys
import tempfile

import pytest

from telefuser.utils.utils import (
    get_example_name,
    import_function_from_file,
    split_list,
)


class TestSplitList:
    """Test split_list function."""

    @pytest.mark.parametrize(
        "lst,n,expected",
        [
            ([1, 2, 3, 4, 5, 6], 2, [[1, 2, 3], [4, 5, 6]]),  # Evenly divisible
            ([1, 2, 3, 4, 5], 2, [[1, 2, 3], [4, 5]]),  # Uneven - first gets extra
            ([1], 1, [[1]]),  # Single element
            ([1, 2, 3], 3, [[1], [2], [3]]),  # n equals length
            ([1, 2], 5, [[1], [2], [], [], []]),  # n greater than length
            ([], 3, [[], [], []]),  # Empty list
        ],
    )
    def test_split_list_cases(self, lst, n, expected):
        """Test split_list with various input cases."""
        result = split_list(lst, n)
        assert result == expected

    def test_split_into_three_distribution(self):
        """Test splitting into three parts has correct distribution."""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = split_list(lst, 3)

        assert len(result) == 3
        assert sum(len(part) for part in result) == 10

    def test_split_preserves_order(self):
        """Test that original order is preserved."""
        lst = list(range(100))
        result = split_list(lst, 5)

        reconstructed = []
        for part in result:
            reconstructed.extend(part)

        assert reconstructed == lst


class TestImportFunctionFromFile:
    """Test import_function_from_file function."""

    def test_import_simple_function(self, tmp_path):
        """Test importing a simple function."""
        # Create a temporary Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def test_function():
    return "hello"
""")

        func = import_function_from_file(str(test_file), "test_function")

        assert callable(func)
        assert func() == "hello"

    def test_import_function_with_args(self, tmp_path):
        """Test importing a function with arguments."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def add(a, b):
    return a + b
""")

        func = import_function_from_file(str(test_file), "add")

        assert func(2, 3) == 5

    def test_import_nonexistent_function(self, tmp_path):
        """Test importing a non-existent function."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def existing_function():
    pass
""")

        with pytest.raises(AttributeError):
            import_function_from_file(str(test_file), "nonexistent_function")

    def test_import_from_nonexistent_file(self):
        """Test importing from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            import_function_from_file("/nonexistent/path/file.py", "function")

    def test_import_multiple_functions(self, tmp_path):
        """Test importing multiple functions from same file."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def func1():
    return 1

def func2():
    return 2
""")

        func1 = import_function_from_file(str(test_file), "func1")
        func2 = import_function_from_file(str(test_file), "func2")

        assert func1() == 1
        assert func2() == 2

    def test_module_added_to_sys_modules(self, tmp_path):
        """Test that module is added to sys.modules."""
        test_file = tmp_path / "my_test_module.py"
        test_file.write_text("""
def func():
    pass
""")

        import_function_from_file(str(test_file), "func")

        # Module is registered with a path-derived unique name prefix
        module_names = [name for name in sys.modules if "my_test_module" in name]
        assert len(module_names) >= 1


class TestGetExampleName:
    """Test get_example_name function."""

    def test_basic_usage(self):
        """Test basic usage."""
        path = "/home/user/data/images/photo.jpg"
        result = get_example_name(path)

        # Function removes original extension and adds .mp4
        assert "images" in result
        assert "photo" in result
        assert result.endswith(".mp4")

    def test_different_extension(self):
        """Test with different output extension."""
        path = "/home/user/data/videos/movie.avi"
        result = get_example_name(path, ext=".png")

        assert "videos" in result
        assert "movie" in result
        assert result.endswith(".png")

    def test_no_extension_in_input(self):
        """Test input without extension."""
        path = "/home/user/data/files/document"
        result = get_example_name(path)

        assert "files" in result
        assert "document" in result
        assert result.endswith(".mp4")

    def test_single_directory(self):
        """Test with single directory path."""
        path = "folder/file.txt"
        result = get_example_name(path)

        assert "folder" in result
        assert "file" in result
        assert result.endswith(".mp4")

    def test_absolute_path(self):
        """Test with absolute path."""
        path = "/absolute/path/to/video/input.mp4"
        result = get_example_name(path)

        assert "video" in result
        assert "input" in result
        assert result.endswith(".mp4")

    def test_deeply_nested_path(self):
        """Test with deeply nested path."""
        path = "/a/b/c/d/e/file.name.jpg"
        result = get_example_name(path)

        # Should use immediate parent directory 'e'
        assert "e_" in result or "e." in result
        assert "file" in result
        assert result.endswith(".mp4")

    def test_multiple_dots_in_filename(self):
        """Test filename with multiple dots."""
        path = "/data/my.file.name.jpg"
        result = get_example_name(path)

        assert "data" in result
        assert "my" in result
        assert result.endswith(".mp4")
