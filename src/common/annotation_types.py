"""Shared in-memory annotation types produced by the auto-labeling stage and
consumed by every dataset exporter."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BoxAnnotation:
    class_id: int
    class_name: str
    # Absolute pixel coordinates, top-left / bottom-right.
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0


@dataclass
class ImageAnnotation:
    image_path: str
    width: int
    height: int
    boxes: List[BoxAnnotation] = field(default_factory=list)
