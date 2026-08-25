"""A torch.utils.data.Dataset that reads the COCO layout produced by
src/dataset_export/coco_export.py (<dataset_dir>/{train,val}2017,
<dataset_dir>/annotations/instances_{train,val}2017.json). Used by the
RTMDet-tiny training script.

Images are read as BGR (RTMDet's official data_preprocessor uses
bgr_to_rgb=False, so this matches the official pretrained weights' expected
input) and letterbox-resized (aspect-ratio-preserving resize + pad) to a
fixed square input size, mirroring RTMDet's own test_pipeline.
"""

import json
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox(image: np.ndarray, boxes: np.ndarray, target_size: Tuple[int, int], pad_value: int = 114):
    """Resizes `image` (BGR, HxWxC) to fit inside target_size (h, w) while
    preserving aspect ratio, pads the rest with `pad_value`, and rescales
    `boxes` (N,4 xyxy, pixel coords in the original image) to match."""
    target_h, target_w = target_size
    h0, w0 = image.shape[:2]
    scale = min(target_h / h0, target_w / w0)
    new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    if boxes.size > 0:
        boxes = boxes.copy()
        boxes *= scale
    return canvas, boxes, scale


class CocoDetectionDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, input_size: Tuple[int, int] = (640, 640),
                 train: bool = True, pixel_mean=(103.53, 116.28, 123.675), pixel_std=(57.375, 57.12, 58.395),
                 hflip_prob: float = 0.5):
        assert split in ("train", "val")
        self.image_dir = os.path.join(dataset_dir, f"{split}2017")
        ann_path = os.path.join(dataset_dir, "annotations", f"instances_{split}2017.json")
        self.input_size = input_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32)
        self.pixel_std = np.array(pixel_std, dtype=np.float32)

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_ids = sorted(cat["id"] for cat in coco["categories"])
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.class_names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]

        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.images = coco["images"]
        self.anns_by_image = anns_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_info = self.images[index]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)  # BGR
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        anns = self.anns_by_image.get(image_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        canvas, boxes, _ = letterbox(image, boxes, self.input_size)

        if self.train and random.random() < self.hflip_prob:
            canvas = np.ascontiguousarray(canvas[:, ::-1, :])
            if boxes.size > 0:
                w = canvas.shape[1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        normed = (canvas.astype(np.float32) - self.pixel_mean) / self.pixel_std
        image_tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float()

        target = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
        }
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
