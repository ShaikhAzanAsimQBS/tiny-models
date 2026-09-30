"""
Label new videos with general_product_detection.pt and APPEND the results
onto the existing YOLOX product dataset (datasets/yolox_product).

Does NOT rebuild or delete train2017 / val2017. New frames get a random
75/25 split, then new 12-digit filenames after the current max image id.

Settings live in config.py:
    APPEND_PRODUCT_VIDEO_PATHS
    APPEND_PRODUCT_FRAME_SKIP
    YOLOX_DATASET_DIR
    TRAIN_VAL_SPLIT_RATIO / SPLIT_RANDOM_SEED

Run:
    python tools\\append_yolox_product_videos.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.inference.frame_extractor import extract_frames
from src.inference.auto_labeler import YoloAutoLabeler
from src.dataset_export.coco_export import append_coco_dataset


def main():
    video_paths = list(config.APPEND_PRODUCT_VIDEO_PATHS)
    missing = [p for p in video_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Video(s) not found: {missing}")

    all_frames = []
    for video_path in video_paths:
        print(f"[append_yolox_product] extracting frames from {video_path} "
              f"(skip={config.APPEND_PRODUCT_FRAME_SKIP})")
        frames = extract_frames(
            video_path=video_path,
            output_dir=config.EXTRACTED_FRAMES_DIR,
            frame_skip=config.APPEND_PRODUCT_FRAME_SKIP,
            jpeg_quality=config.FRAME_JPEG_QUALITY,
        )
        print(f"[append_yolox_product] {len(frames)} frames from {os.path.basename(video_path)}")
        all_frames.extend(frames)

    if not all_frames:
        print("[append_yolox_product] no frames extracted. aborting.")
        return

    print(f"[append_yolox_product] labeling {len(all_frames)} frames with {config.PRODUCT_MODEL_PATH}")
    labeler = YoloAutoLabeler(
        model_path=config.PRODUCT_MODEL_PATH,
        class_names=list(config.PRODUCT_CLASS_NAMES),
    )
    annotations = labeler.label_images(all_frames)
    n_boxes = sum(len(a.boxes) for a in annotations)
    print(f"[append_yolox_product] {len(annotations)} images with detections, {n_boxes} product boxes")

    if not annotations:
        print("[append_yolox_product] nothing to append.")
        return

    append_coco_dataset(
        annotations,
        config.YOLOX_DATASET_DIR,
        class_names=config.DATASET_CLASS_NAMES,
        split_ratio=config.TRAIN_VAL_SPLIT_RATIO,
        seed=config.SPLIT_RANDOM_SEED,
    )
    print("[append_yolox_product] done. dataset is at", config.YOLOX_DATASET_DIR)


if __name__ == "__main__":
    main()
