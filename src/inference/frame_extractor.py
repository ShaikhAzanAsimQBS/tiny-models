"""Extracts every Nth frame from a video file - or a remote video URL,
streamed/decoded on the fly and NEVER downloaded to disk first - saving the
kept frames to disk as .jpg images."""

import os
from typing import List
from urllib.parse import urlparse

import cv2

import config
from src.common.media_utils import is_url


def _video_stem(video_path: str) -> str:
    """A filesystem-safe name to use for this video's output subfolder,
    for both local paths and remote URLs (query strings stripped)."""
    relevant_path = urlparse(video_path).path if is_url(video_path) else video_path
    stem = os.path.splitext(os.path.basename(relevant_path))[0]
    return stem or "url_video"


def extract_frames(video_path: str, output_dir: str, frame_skip: int = None,
                    jpeg_quality: int = None) -> List[str]:
    """Decode video_path (a local file path OR a remote http(s)/rtsp/rtmp
    URL - opened directly with cv2.VideoCapture/ffmpeg and streamed frame by
    frame, never downloaded to disk first) and save every `frame_skip`-th
    frame as a .jpg.

    Returns the list of saved frame file paths (in order).
    """
    frame_skip = config.FRAME_SKIP if frame_skip is None else frame_skip
    jpeg_quality = config.FRAME_JPEG_QUALITY if jpeg_quality is None else jpeg_quality
    frame_skip = max(1, int(frame_skip))

    video_name = _video_stem(video_path)
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # cv2.CAP_FFMPEG is required (not just the default auto-detected
    # backend) to reliably open remote http(s)/rtsp/rtmp URLs for streaming
    # decode; it also works fine for local files.
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        source_kind = "URL stream" if is_url(video_path) else "video"
        raise IOError(
            f"Could not open {source_kind}: {video_path}"
            + (" (check the URL is reachable/public and points directly at a video file)" if is_url(video_path) else "")
        )

    saved_paths: List[str] = []
    frame_index = 0
    saved_index = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_skip == 0:
                frame_path = os.path.join(
                    video_output_dir, f"{video_name}_frame_{saved_index:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame, encode_params)
                saved_paths.append(frame_path)
                saved_index += 1
            frame_index += 1
    finally:
        cap.release()

    print(f"[frame_extractor] {video_path}: decoded {frame_index} frames, "
          f"kept {saved_index} (skip={frame_skip}) -> {video_output_dir}")
    return saved_paths
