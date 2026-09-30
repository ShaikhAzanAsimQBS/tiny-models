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


def _empty_coco(class_names: List[str]) -> dict:
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name, "supercategory": "none"}
            for idx, name in enumerate(class_names)
        ],
    }


def _load_coco_json(json_path: str, class_names: List[str]) -> dict:
    if not os.path.isfile(json_path):
        return _empty_coco(class_names)
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    coco.setdefault("categories", _empty_coco(class_names)["categories"])
    return coco


def _append_split(split_annotations: List[ImageAnnotation],
                   image_dir: str,
                   json_path: str,
                   class_names: List[str]) -> tuple:
    """Copy new images into an existing COCO split and rewrite its JSON.

    Image / annotation ids continue from the current max so existing
    train2017 / val2017 files are never overwritten.
    """
    os.makedirs(image_dir, exist_ok=True)
    coco = _load_coco_json(json_path, class_names)
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    for idx, name in enumerate(class_names):
        name_to_id.setdefault(name, idx + 1)
        if not any(c["name"] == name for c in coco["categories"]):
            coco["categories"].append({"id": name_to_id[name], "name": name, "supercategory": "none"})

    next_image_id = (max((img["id"] for img in coco["images"]), default=0) + 1)
    next_ann_id = (max((ann["id"] for ann in coco["annotations"]), default=0) + 1)
    added_images = 0
    added_boxes = 0

    for ann in split_annotations:
        file_name = f"{next_image_id:012d}.jpg"
        dest = os.path.join(image_dir, file_name)
        shutil.copyfile(ann.image_path, dest)
        coco["images"].append({
            "id": next_image_id,
            "file_name": file_name,
            "width": ann.width,
            "height": ann.height,
        })
        for box in ann.boxes:
            width = max(0.0, box.xmax - box.xmin)
            height = max(0.0, box.ymax - box.ymin)
            if width <= 0 or height <= 0:
                continue
            coco["annotations"].append({
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": name_to_id.get(box.class_name, 1),
                "bbox": [box.xmin, box.ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],
            })
            next_ann_id += 1
            added_boxes += 1
        next_image_id += 1
        added_images += 1

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return added_images, added_boxes, len(coco["images"]), len(coco["annotations"])


def append_coco_dataset(annotations: List[ImageAnnotation],
                         output_dir: str,
                         class_names: List[str] = None,
                         split_ratio: float = None,
                         seed: int = None) -> None:
    """Add newly labeled images onto an existing YOLOX COCO folder.

    Existing train2017 / val2017 jpgs and their JSON entries stay as-is.
    New images get a random train/val split, then new 12-digit filenames
    starting after the current max image id.
    """
    class_names = config.DATASET_CLASS_NAMES if class_names is None else class_names
    split_ratio = config.TRAIN_VAL_SPLIT_RATIO if split_ratio is None else split_ratio
    seed = config.SPLIT_RANDOM_SEED if seed is None else seed

    train_dir = os.path.join(output_dir, "train2017")
    val_dir = os.path.join(output_dir, "val2017")
    ann_dir = os.path.join(output_dir, "annotations")
    train_json = os.path.join(ann_dir, "instances_train2017.json")
    val_json = os.path.join(ann_dir, "instances_val2017.json")

    train_set, val_set = train_val_split(annotations, split_ratio, seed, strategy="random")
    print(f"[coco_export] appending {len(train_set)} train / {len(val_set)} val images to {output_dir}")

    t_added, t_boxes, t_total_img, t_total_box = _append_split(
        train_set, train_dir, train_json, class_names
    )
    v_added, v_boxes, v_total_img, v_total_box = _append_split(
        val_set, val_dir, val_json, class_names
    )
    print(f"[coco_export] appended train +{t_added} images / +{t_boxes} boxes "
          f"(now {t_total_img} images / {t_total_box} boxes)")
    print(f"[coco_export] appended val   +{v_added} images / +{v_boxes} boxes "
          f"(now {v_total_img} images / {v_total_box} boxes)")
