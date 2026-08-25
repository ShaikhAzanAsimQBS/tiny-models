"""Helpers for figuring out what kind of media INPUT_PATH points at."""

import os
from typing import List, Tuple

import config


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in config.IMAGE_EXTENSIONS


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in config.VIDEO_EXTENSIONS


def collect_media(input_path: str) -> Tuple[List[str], List[str]]:
    """Resolve INPUT_PATH into (image_paths, video_paths).

    - If input_path is a single video file -> ([], [input_path])
    - If input_path is a single image file -> ([input_path], [])
    - If input_path is a folder -> recursively walk it and bucket every file
      found into images / videos, ignoring anything else.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"INPUT_PATH does not exist: {input_path}")

    if os.path.isfile(input_path):
        if is_video_file(input_path):
            return [], [input_path]
        if is_image_file(input_path):
            return [input_path], []
        raise ValueError(
            f"INPUT_PATH points at a file with an unsupported extension: {input_path}"
        )

    image_paths: List[str] = []
    video_paths: List[str] = []
    for root, _dirs, files in os.walk(input_path):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if is_video_file(full_path):
                video_paths.append(full_path)
            elif is_image_file(full_path):
                image_paths.append(full_path)

    image_paths.sort()
    video_paths.sort()
    return image_paths, video_paths
