"""Deterministic train/val split shared by every dataset exporter."""

import random
from typing import List, Tuple

from src.common.annotation_types import ImageAnnotation


def train_val_split(annotations: List[ImageAnnotation],
                     split_ratio: float,
                     seed: int) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
    shuffled = list(annotations)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * split_ratio)) if shuffled else 0
    train_set = shuffled[:split_index]
    val_set = shuffled[split_index:]
    if not val_set and len(shuffled) > 1:
        val_set = [train_set.pop()]
    return train_set, val_set
