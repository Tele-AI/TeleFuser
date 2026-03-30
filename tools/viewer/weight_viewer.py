"""
Model weight information viewer tool
Supports pt files and safetensors format, supports wildcards and split model loading
"""

import glob
import json
import os
import re
from collections import defaultdict
from typing import Any

import torch
from safetensors import safe_open

from telefuser.utils.model_weight import hash_state_dict_keys


class WeightViewer:
    """Model weight viewer"""

    def __init__(self):
        self.weights_info = {}
        self.total_params = 0
        self.file_paths = []

    def load_weights(self, file_pattern: str) -> None:
        """
        Load weight files, supports wildcards and split models

        Args:
            file_pattern: File path pattern, supports wildcards (*.pt, model_part*.safetensors, etc.)
        """
        self.file_paths = self._expand_file_pattern(file_pattern)

        if not self.file_paths:
            raise FileNotFoundError(f"No matching files found: {file_pattern}")

        # Merge weights from all files
        all_weights = {}
        for file_path in self.file_paths:
            weights = self._load_single_file(file_path)
            all_weights.update(weights)

        # Build nested structure
        self.weights_info = self._build_nested_structure(all_weights)

        # Calculate total parameters
        self.total_params = self._calculate_total_params(all_weights)
        # Calculate hash value
        self.weight_hash = hash_state_dict_keys(all_weights)

    def _expand_file_pattern(self, pattern: str) -> list[str]:
        """Expand file wildcard patterns"""
        # Handle split model files (e.g., model-00001-of-00002.safetensors)
        if "*" in pattern or "?" in pattern:
            files = sorted(glob.glob(pattern))
            # Sort split model files
            files = self._sort_split_files(files)
            return files
        else:
            # Single file
            if os.path.exists(pattern):
                return [pattern]
            else:
                return []

    def _sort_split_files(self, files: list[str]) -> list[str]:
        """Intelligently sort split model files"""

        def extract_split_info(filename):
            # Match format like model-00001-of-00002.safetensors
            match = re.search(r"(\d+)-of-(\d+)", filename)
            if match:
                return int(match.group(1)), int(match.group(2))
            return (0, 0)

        # Sort by split number
        return sorted(files, key=lambda x: extract_split_info(x))

    def _load_single_file(self, file_path: str) -> dict[str, torch.Tensor]:
        """Load a single weight file"""
        if file_path.endswith(".safetensors"):
            return self._load_safetensors(file_path)
        elif file_path.endswith((".pt", ".pth", ".ckpt", ".bin")):
            return self._load_torch(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _load_safetensors(self, file_path: str) -> dict[str, torch.Tensor]:
        """Load safetensors file"""
        weights = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = tensor
        return weights

    def _load_torch(self, file_path: str) -> dict[str, torch.Tensor]:
        """Load PyTorch file"""
        try:
            state_dict = torch.load(file_path, map_location="cpu", weights_only=True)

            # Handle different file formats
            if isinstance(state_dict, dict):
                # Flatten nested OrderedDict structures
                return self._flatten_state_dict(state_dict)
            elif hasattr(state_dict, "state_dict"):
                # Model object
                return self._flatten_state_dict(state_dict.state_dict())
            else:
                # Other formats, try to convert to dict
                return self._flatten_state_dict(dict(state_dict))
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch file: {e}")

    def _flatten_state_dict(self, state_dict: dict, prefix: str = "") -> dict[str, torch.Tensor]:
        """
        Flatten nested state_dict structure

        Some model files (like RealESRGAN) may contain nested OrderedDict structures,
        this method flattens them into a single level dictionary with dotted keys.
        """
        flattened = {}
        for key, value in state_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and not isinstance(value, torch.Tensor):
                # Recursively flatten nested dictionaries
                nested = self._flatten_state_dict(value, full_key)
                flattened.update(nested)
            elif isinstance(value, torch.Tensor):
                flattened[full_key] = value
            else:
                # Skip non-tensor, non-dict values (e.g., metadata)
                pass
        return flattened

    def _build_nested_structure(self, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Build nested structure based on prefix relationships"""
        nested = {}

        for key, tensor in weights.items():
            parts = key.split(".")
            current_level = nested

            # Traverse each part of the key to build nested structure
            for i, part in enumerate(parts[:-1]):
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Add weight information
            last_part = parts[-1]
            current_level[last_part] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "requires_grad": tensor.requires_grad if hasattr(tensor, "requires_grad") else False,
                "device": str(tensor.device),
            }

        return nested

    def _calculate_total_params(self, weights: dict[str, torch.Tensor]) -> int:
        """Calculate total number of parameters"""
        total = 0
        for tensor in weights.values():
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel()
        return total

    def get_summary(self) -> dict[str, Any]:
        """Get weight information summary"""
        if not self.weights_info:
            return {}

        # Count parameters by different data types
        dtype_stats = defaultdict(int)
        shape_stats = defaultdict(int)

        def traverse_stats(node, prefix=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict) and "shape" in value:
                        # This is a weight node
                        dtype_stats[value["dtype"]] += value["numel"]
                        shape_key = str(tuple(value["shape"]))
                        shape_stats[shape_key] += 1
                    else:
                        traverse_stats(value, new_prefix)

        traverse_stats(self.weights_info)

        return {
            "hash_with_shape": self.weight_hash,
            "total_parameters": self.total_params,
            "total_parameters_formatted": self._format_number(self.total_params),
            "files_loaded": len(self.file_paths),
            "file_paths": self.file_paths,
            "dtype_distribution": dict(dtype_stats),
            "shape_distribution": dict(shape_stats),
        }

    def _format_number(self, num: int) -> str:
        """Format number display"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)

    def _group_similar_modules(self, node: dict) -> list[dict]:
        """
        Group modules with identical structures

        Args:
            node: Current level node dictionary

        Returns:
            List of grouping information, each element contains:
            - keys: List of keys with identical structure
            - count: Number of identical structures
            - is_leaf: Whether it's a leaf node (weight parameter)
        """
        if not isinstance(node, dict):
            return []

        # Group by naming patterns
        pattern_groups = defaultdict(list)

        for key in node.keys():
            # Extract naming pattern (remove numeric suffix)
            base_pattern = re.sub(r"\d+$", "", key)
            pattern_groups[base_pattern].append(key)

        grouped_results = []

        for base_pattern, keys in pattern_groups.items():
            if len(keys) > 1:
                # Check if values corresponding to these keys have identical structure
                first_value = node[keys[0]]

                # Determine if it's a leaf node (weight parameter)
                is_leaf = isinstance(first_value, dict) and "shape" in first_value

                # For non-leaf nodes, check if nested structures are identical
                similar_keys = [keys[0]]
                for key in keys[1:]:
                    if self._is_same_structure(first_value, node[key]):
                        similar_keys.append(key)

                if len(similar_keys) > 1:
                    grouped_results.append(
                        {
                            "keys": similar_keys,
                            "count": len(similar_keys),
                            "is_leaf": is_leaf,
                        }
                    )
                    # Remove grouped keys from original list
                    remaining_keys = [k for k in keys if k not in similar_keys]
                    if remaining_keys:
                        pattern_groups[base_pattern] = remaining_keys
                else:
                    # No modules with identical structure found, process individually
                    for key in keys:
                        grouped_results.append(
                            {
                                "keys": [key],
                                "count": 1,
                                "is_leaf": isinstance(node[key], dict) and "shape" in node[key],
                            }
                        )
            else:
                # Single key
                key = keys[0]
                grouped_results.append(
                    {
                        "keys": [key],
                        "count": 1,
                        "is_leaf": isinstance(node[key], dict) and "shape" in node[key],
                    }
                )

        return grouped_results

    def _is_same_structure(self, obj1: Any, obj2: Any) -> bool:
        """
        Determine if two objects have identical nested structure

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            Whether they have identical structure
        """
        # Different types mean different structures
        if not isinstance(obj1, type(obj2)):
            return False

        # If dictionaries, recursively compare key and value structures
        if isinstance(obj1, dict):
            # Check if it's a weight parameter node
            if "shape" in obj1 and "shape" in obj2:
                # Weight parameter node: compare data type and shape
                return obj1["dtype"] == obj2["dtype"] and obj1["shape"] == obj2["shape"]
            else:
                # Nested node: compare key sets and value structures
                if set(obj1.keys()) != set(obj2.keys()):
                    return False

                for key in obj1.keys():
                    if not self._is_same_structure(obj1[key], obj2[key]):
                        return False
                return True

        # 其他类型直接比较
        return obj1 == obj2

    def print_detailed_info(self, max_depth: int = 3, show_all: bool = False, merge_similar: bool = True):
        """Print detailed weight information"""
        if not self.weights_info:
            print("No weight information loaded")
            return

        summary = self.get_summary()
        print("=" * 80)
        print("Model Weight Information Overview")
        print("=" * 80)
        print(f"Total parameters: {summary['total_parameters_formatted']} ({summary['total_parameters']:,})")
        print(f"hash with shape: {summary['hash_with_shape']}")
        print(f"Files loaded: {summary['files_loaded']}")
        print(f"File list: {summary['file_paths']}")
        print()

        print("Data type distribution:")
        for dtype, count in summary["dtype_distribution"].items():
            formatted_count = self._format_number(count)
            percentage = (count / summary["total_parameters"]) * 100
            print(f"  {dtype}: {formatted_count} ({percentage:.2f}%)")

        print()
        print("Detailed weight structure:")
        if merge_similar and not show_all:
            print("(Structure-identical modules merged, use --show-all to view full structure)")
        self._print_nested_structure(
            self.weights_info,
            max_depth=max_depth,
            show_all=show_all,
            merge_similar=merge_similar,
        )

    def _print_nested_structure(
        self,
        node: dict,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 3,
        show_all: bool = False,
        merge_similar: bool = True,
    ):
        """Recursively print nested structure, automatically merging structure-identical modules"""
        if depth > max_depth and not show_all:
            print(f"{prefix}... (深度限制，使用 --show-all 查看完整结构)")
            return

        # Analyze child nodes of current node to find structure-identical modules
        grouped_nodes = self._group_similar_modules(node)

        for group_info in grouped_nodes:
            if group_info["count"] > 1 and merge_similar and not show_all:
                # Display merged module information
                first_key = group_info["keys"][0]
                current_prefix = f"{prefix}.{first_key}" if prefix else first_key

                if group_info["is_leaf"]:
                    # Leaf node (weight parameter)
                    value = node[first_key]
                    shape_str = str(tuple(value["shape"]))
                    numel_str = self._format_number(value["numel"])
                    print(
                        f"{current_prefix:60} | {shape_str:20} | {value['dtype']:15} | {numel_str:>10} x{group_info['count']}"
                    )
                else:
                    # Nested node
                    print(f"{current_prefix} x{group_info['count']}")
                    # Only recursively display structure of first module
                    self._print_nested_structure(
                        node[first_key],
                        current_prefix,
                        depth + 1,
                        max_depth,
                        show_all,
                        merge_similar,
                    )
            else:
                # Display each node individually
                for key in group_info["keys"]:
                    current_prefix = f"{prefix}.{key}" if prefix else key
                    value = node[key]

                    if isinstance(value, dict) and "shape" in value:
                        # This is a weight node
                        shape_str = str(tuple(value["shape"]))
                        numel_str = self._format_number(value["numel"])
                        print(f"{current_prefix:60} | {shape_str:20} | {value['dtype']:15} | {numel_str:>10}")
                    elif isinstance(value, dict):
                        # This is a nested node
                        print(f"{current_prefix}")
                        self._print_nested_structure(
                            value,
                            current_prefix,
                            depth + 1,
                            max_depth,
                            show_all,
                            merge_similar,
                        )
                    else:
                        print(f"{current_prefix}: {value}")

    def export_to_json(self, output_path: str):
        """Export weight information to JSON file"""
        if not self.weights_info:
            raise ValueError("No weight information to export")

        export_data = {
            "summary": self.get_summary(),
            "weights_structure": self.weights_info,
            "metadata": {
                "tool_version": "1.0.0",
                "export_timestamp": torch.tensor(0).numpy().tolist(),  # Placeholder
            },
        }

        # 转换Tensor为可序列化的格式
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return {
                    "__tensor__": True,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "requires_grad": obj.requires_grad,
                }
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj

        export_data = convert_tensors(export_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Weight information exported to: {output_path}")


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Model weight information viewer tool")
    parser.add_argument("file_pattern", help="Weight file path pattern (supports wildcards)")
    parser.add_argument("--max-depth", type=int, default=5, help="显示的最大嵌套深度")
    parser.add_argument("--show-all", action="store_true", help="显示完整的权重结构")
    parser.add_argument("--no-merge", action="store_true", help="Disable merging of structure-identical modules")
    parser.add_argument("--export", type=str, help="导出到JSON文件")
    parser.add_argument("--quiet", action="store_true", help="仅显示摘要信息")

    args = parser.parse_args()

    viewer = WeightViewer()

    try:
        # Load weights
        viewer.load_weights(args.file_pattern)

        if args.quiet:
            # Only show summary
            summary = viewer.get_summary()
            print(f"Total parameters: {summary['total_parameters_formatted']}")
            print(f"Files: {summary['files_loaded']}")
        else:
            # Show detailed information
            merge_similar = not args.no_merge
            viewer.print_detailed_info(
                max_depth=args.max_depth,
                show_all=args.show_all,
                merge_similar=merge_similar,
            )

        # Export functionality
        if args.export:
            viewer.export_to_json(args.export)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
