from __future__ import annotations

from typing import Any


class Trimesh:
    def __init__(self, vertices: Any = None, faces: Any = None, **kwargs: Any) -> None:
        self.vertices = vertices
        self.faces = faces
        self.kwargs = kwargs
        self.visual = None


class Scene:
    def dump(self, *, concatenate: bool = False) -> Any:
        raise RuntimeError("trimesh is not installed; 3D mesh utilities are unavailable")


class _SimpleMaterial:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _TextureVisuals:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _MaterialModule:
    SimpleMaterial = _SimpleMaterial


class _VisualModule:
    TextureVisuals = _TextureVisuals
    material = _MaterialModule()


visual = _VisualModule()
