"""
Main entry point: run this to go from raw media (a video OR a folder of
images/videos) all the way to ready-to-train datasets for up to 4 models:

    datasets/mobilenet_ssd/   (Pascal VOC layout)
    datasets/nanodet/         (Pascal VOC layout)
    datasets/yolox/           (COCO layout)
    datasets/rtmdet/          (COCO layout)

Which of the 4 are actually built is controlled by config.CREATE_DATASET_*
(set any to False to skip it).

Steps:
1. Resolve INPUT_PATH into a flat list of images to label
   (videos are decoded, keeping every FRAME_SKIP-th frame).
2. Auto-label according to config.AUTO_LABEL_MODE:
   - "product": run general_product_detection.pt on each full image.
   - "person_face": yolo11x (person) then face11n on each person crop.
3. Export annotations into whichever of the 4 dataset formats are enabled.

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
from src.inference.person_face_labeler import PersonFaceAutoLabeler
from src.dataset_export.voc_export import export_voc_dataset
from src.dataset_export.coco_export import export_coco_dataset
from src.dataset_export.dedupe_faces import dedupe_coco_dataset_dir


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

    mode = getattr(config, "AUTO_LABEL_MODE", "person_face")
    print(f"[pipeline] AUTO_LABEL_MODE={mode}")

    if mode == "product":
        labeler = YoloAutoLabeler(
            model_path=config.PRODUCT_MODEL_PATH,
            class_names=list(config.PRODUCT_CLASS_NAMES),
        )
        annotations = labeler.label_images(all_images)
        product_boxes = sum(len(a.boxes) for a in annotations)
        print(f"[pipeline] Auto-labeled {len(annotations)} images with "
              f"{product_boxes} 'product' boxes")
    elif mode == "person_face":
        labeler = PersonFaceAutoLabeler(
            person_model_path=config.PERSON_MODEL_PATH,
            face_model_path=config.FACE_MODEL_PATH,
        )
        annotations = labeler.label_images(all_images)
        person_boxes = sum(1 for a in annotations for b in a.boxes if b.class_name == config.PERSON_CLASS_NAME)
        face_boxes = sum(1 for a in annotations for b in a.boxes if b.class_name == config.FACE_CLASS_NAME)
        print(f"[pipeline] Auto-labeled {len(annotations)} images with "
              f"{person_boxes} 'person' + {face_boxes} 'face' boxes")
    else:
        raise ValueError(f"Unknown AUTO_LABEL_MODE={mode!r} (expected 'product' or 'person_face')")

    if not annotations:
        print("[pipeline] No annotated images to export. Aborting dataset export.")
        return

    if config.CREATE_DATASET_MOBILENET_SSD:
        print("[pipeline] Exporting MobileNet-SSD dataset (Pascal VOC layout)...")
        export_voc_dataset(annotations, config.MOBILENET_SSD_DATASET_DIR)
    else:
        print("[pipeline] Skipping MobileNet-SSD dataset (CREATE_DATASET_MOBILENET_SSD=False)")

    if config.CREATE_DATASET_NANODET:
        print("[pipeline] Exporting NanoDet dataset (Pascal VOC layout)...")
        export_voc_dataset(annotations, config.NANODET_DATASET_DIR)
    else:
        print("[pipeline] Skipping NanoDet dataset (CREATE_DATASET_NANODET=False)")

    if config.CREATE_DATASET_YOLOX:
        print("[pipeline] Exporting YOLOX-Tiny dataset (COCO layout)...")
        export_coco_dataset(annotations, config.YOLOX_DATASET_DIR, class_names=config.DATASET_CLASS_NAMES)
        if config.YOLOX_DEDUPE_FACES and mode == "person_face":
            print("[pipeline] Deduping extra face boxes in the YOLOX dataset (1 face per person)...")
            removed = dedupe_coco_dataset_dir(config.YOLOX_DATASET_DIR)
            print(f"[pipeline] YOLOX face dedupe removed {removed} extra face box(es)")
    else:
        print("[pipeline] Skipping YOLOX-Tiny dataset (CREATE_DATASET_YOLOX=False)")

    if config.CREATE_DATASET_RTMDET:
        print("[pipeline] Exporting RTMDet-tiny dataset (COCO layout)...")
        export_coco_dataset(annotations, config.RTMDET_DATASET_DIR, class_names=config.DATASET_CLASS_NAMES)
    else:
        print("[pipeline] Skipping RTMDet-tiny dataset (CREATE_DATASET_RTMDET=False)")

    print("[pipeline] Done. Datasets are ready under:", config.DATASETS_ROOT)


if __name__ == "__main__":
    main()
