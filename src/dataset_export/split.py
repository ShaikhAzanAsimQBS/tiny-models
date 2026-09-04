"""Deterministic train/val split shared by every dataset exporter."""

import os
import random
from collections import defaultdict
from typing import List, Tuple

import config
from src.common.annotation_types import ImageAnnotation


def _random_split(annotations: List[ImageAnnotation], split_ratio: float,
                   seed: int) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
    shuffled = list(annotations)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * split_ratio)) if shuffled else 0
    train_set = shuffled[:split_index]
    val_set = shuffled[split_index:]
    if not val_set and len(shuffled) > 1:
        val_set = [train_set.pop()]
    return train_set, val_set


def _sequential_split(annotations: List[ImageAnnotation], split_ratio: float,
                       seed: int) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
    """Splits WITHIN each source folder (e.g. each video's own subfolder
    under extracted_frames/), taking the first `split_ratio` fraction of
    each group in path order (chronological order for extracted video
    frames) as train and the rest as val.

    Why: consecutive video frames are near-duplicates of each other. A
    plain random shuffle (as `_random_split` does) will scatter
    near-identical frames across BOTH the train and val sets - the model
    then gets evaluated on images that are almost pixel-identical to ones
    it trained on ("leakage"), so val mAP looks great while telling you
    almost nothing about real generalization, and it hides overfitting
    instead of catching it. Splitting chronologically within each
    video/folder means only the few frames right at the split boundary are
    similar across train/val, giving a much more honest validation signal
    while every source video/folder still contributes to both splits.
    """
    groups = defaultdict(list)
    for ann in annotations:
        groups[os.path.dirname(ann.image_path)].append(ann)

    train_set: List[ImageAnnotation] = []
    val_set: List[ImageAnnotation] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda a: a.image_path)
        split_index = max(1, int(len(group) * split_ratio)) if group else 0
        group_train, group_val = group[:split_index], group[split_index:]
        if not group_val and len(group) > 1:
            group_val = [group_train.pop()]
        train_set.extend(group_train)
        val_set.extend(group_val)

    # Shuffle each split's ORDER (not membership) so a training dataloader
    # doesn't see whole videos back to back - membership/train-vs-val
    # assignment above stays fully chronological/leak-free.
    random.Random(seed).shuffle(train_set)
    random.Random(seed + 1).shuffle(val_set)
    return train_set, val_set


def train_val_split(annotations: List[ImageAnnotation],
                     split_ratio: float,
                     seed: int,
                     strategy: str = None) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
    """strategy:
        "sequential" (recommended default, see config.TRAIN_VAL_SPLIT_STRATEGY) -
            leak-free split for video-frame-heavy datasets, see `_sequential_split`.
        "random" - plain random shuffle-then-slice. Fine for datasets made
            entirely of independent photos, but leaks near-duplicate video
            frames across train/val if any of the data came from
            extract_frames().
    """
    strategy = config.TRAIN_VAL_SPLIT_STRATEGY if strategy is None else strategy

    if strategy == "sequential":
        return _sequential_split(annotations, split_ratio, seed)
    if strategy == "random":
        return _random_split(annotations, split_ratio, seed)
    raise ValueError(f"Unknown TRAIN_VAL_SPLIT_STRATEGY: {strategy!r} (expected 'sequential' or 'random')")
