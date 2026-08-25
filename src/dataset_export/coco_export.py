"""Exports ImageAnnotation objects into a COCO-style dataset:

    <output_dir>/
        train2017/*.jpg
        val2017/*.jpg
        annotations/instances_train2017.json
        annotations/instances_val2017.json

This matches the exact layout YOLOX's COCODataset expects (and is what
NanoDet's CocoDataset expects too), so `data_dir=<output_dir>` just works.
"""

import json
import os
import shutil
from typing import List

import config
from src.common.annotation_types import ImageAnnotation
from src.dataset_export.split import train_val_split


def _build_coco_dict(annotations: List[ImageAnnotation], class_names: List[str]) -> dict:
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "none"}
        for idx, name in enumerate(class_names)
    ]
    name_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}

    images = []
    coco_annotations = []
    ann_id = 1
    for image_id, ann in enumerate(annotations, start=1):
        file_name = f"{image_id:012d}.jpg"
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": ann.width,
            "height": ann.height,
        })
        for box in ann.boxes:
            width = max(0.0, box.xmax - box.xmin)
            height = max(0.0, box.ymax - box.ymin)
            if width <= 0 or height <= 0:
                continue
            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": name_to_id.get(box.class_name, 1),
                "bbox": [box.xmin, box.ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return {"images": images, "annotations": coco_annotations, "categories": categories}


def export_coco_dataset(annotations: List[ImageAnnotation],
                         output_dir: str,
                         class_names: List[str] = None,
                         split_ratio: float = None,
                         seed: int = None) -> None:
    class_names = config.DATASET_CLASS_NAMES if class_names is None else class_names
    split_ratio = config.TRAIN_VAL_SPLIT_RATIO if split_ratio is None else split_ratio
    seed = config.SPLIT_RANDOM_SEED if seed is None else seed

    train_dir = os.path.join(output_dir, "train2017")
    val_dir = os.path.join(output_dir, "val2017")
    ann_dir = os.path.join(output_dir, "annotations")
    for d in (train_dir, val_dir, ann_dir):
        os.makedirs(d, exist_ok=True)

    train_set, val_set = train_val_split(annotations, split_ratio, seed)

    def write_split(split_annotations: List[ImageAnnotation], image_dir: str, json_name: str) -> None:
        coco_dict = _build_coco_dict(split_annotations, class_names)
        for image_entry, ann in zip(coco_dict["images"], split_annotations):
            shutil.copyfile(ann.image_path, os.path.join(image_dir, image_entry["file_name"]))
        with open(os.path.join(ann_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

    write_split(train_set, train_dir, "instances_train2017.json")
    write_split(val_set, val_dir, "instances_val2017.json")

    print(f"[coco_export] {output_dir}: {len(train_set)} train / {len(val_set)} val images")
