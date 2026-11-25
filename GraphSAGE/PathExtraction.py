"""加载同级 CktPath 工程的公共图与路径提取实现。"""

import sys
from pathlib import Path


CKTPATH_ROOT = Path(__file__).resolve().parents[2] / "path"
if not (CKTPATH_ROOT / "core" / "Graph.py").is_file() or not (
    CKTPATH_ROOT / "extraction" / "Manager.py"
).is_file():
    raise ImportError(f"CktPath 工程不存在或结构不完整: {CKTPATH_ROOT}")

if str(CKTPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(CKTPATH_ROOT))

from core import CircuitGraph, PortType, VertexType  # noqa: E402
from extraction import PathExtractConfig, PathExtractManager  # noqa: E402
from extraction.PostProcess import VPATH_LINK_TYPE_LABELS  # noqa: E402


__all__ = [
    "CKTPATH_ROOT", "CircuitGraph", "PortType", "VertexType",
    "PathExtractConfig", "PathExtractManager", "VPATH_LINK_TYPE_LABELS",
]
