"""
Main entry point: run this to go from raw media (a video OR a folder of
images/videos) all the way to 4 ready-to-train datasets:

    datasets/mobilenet_ssd/   (Pascal VOC layout)
    datasets/nanodet/         (Pascal VOC layout)
    datasets/yolox/           (COCO layout)
    datasets/rtmdet/          (COCO layout)

Steps:
    1. Resolve INPUT_PATH into a flat list of images to label
       (videos are decoded, keeping every FRAME_SKIP-th frame).
    2. Run the fine-tuned YOLO11n teacher model (general_product_detection.pt)
       over every image/frame to auto-label the "product" class.
    3. Export the resulting annotations into the 4 dataset formats above.

Everything is controlled via config.py - no command line arguments.
Run with the CUDA-enabled venv, e.g.:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.common.media_utils import collect_media
from src.inference.frame_extractor import extract_frames
from src.inference.auto_labeler import YoloAutoLabeler
from src.dataset_export.voc_export import export_voc_dataset
from src.dataset_export.coco_export import export_coco_dataset


def gather_images_to_label():
    """Resolve config.INPUT_PATH -> flat list of image paths to run the
    teacher model on (extracting video frames to disk as needed)."""
    image_paths, video_paths = collect_media(config.INPUT_PATH)
    print(f"[pipeline] Found {len(image_paths)} image(s) and {len(video_paths)} video(s) in INPUT_PATH")

    all_images = list(image_paths)
    for video_path in video_paths:
        frame_paths = extract_frames(
            video_path=video_path,
            output_dir=config.EXTRACTED_FRAMES_DIR,
            frame_skip=config.FRAME_SKIP,
            jpeg_quality=config.FRAME_JPEG_QUALITY,
        )
        all_images.extend(frame_paths)

    print(f"[pipeline] Total images to label: {len(all_images)}")
    return all_images


def main():
    all_images = gather_images_to_label()
    if not all_images:
        print("[pipeline] No images/frames found. Check config.INPUT_PATH. Aborting.")
        return

    labeler = YoloAutoLabeler(
        model_path=config.YOLO_MODEL_PATH,
        class_names=config.YOLO_CLASS_NAMES,
        confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold=config.YOLO_IOU_THRESHOLD,
        img_size=config.YOLO_IMG_SIZE,
        device=config.YOLO_DEVICE,
    )
    annotations = labeler.label_images(all_images, batch_size=config.YOLO_INFERENCE_BATCH_SIZE)

    total_boxes = sum(len(a.boxes) for a in annotations)
    print(f"[pipeline] Auto-labeled {len(annotations)} images with {total_boxes} total 'product' boxes")

    if not annotations:
        print("[pipeline] No annotated images to export. Aborting dataset export.")
        return

    print("[pipeline] Exporting MobileNet-SSD dataset (Pascal VOC layout)...")
    export_voc_dataset(annotations, config.MOBILENET_SSD_DATASET_DIR)

    print("[pipeline] Exporting NanoDet dataset (Pascal VOC layout)...")
    export_voc_dataset(annotations, config.NANODET_DATASET_DIR)

    print("[pipeline] Exporting YOLOX-Tiny dataset (COCO layout)...")
    export_coco_dataset(annotations, config.YOLOX_DATASET_DIR, class_names=config.DATASET_CLASS_NAMES)

    print("[pipeline] Exporting RTMDet-tiny dataset (COCO layout)...")
    export_coco_dataset(annotations, config.RTMDET_DATASET_DIR, class_names=config.DATASET_CLASS_NAMES)

    print("[pipeline] Done. Datasets are ready under:", config.DATASETS_ROOT)


if __name__ == "__main__":
    main()
