"""A torch.utils.data.Dataset that reads the Pascal-VOC layout produced by
src/dataset_export/voc_export.py (<dataset_dir>/<split>/JPEGImages,
<dataset_dir>/<split>/Annotations). Used by the MobileNet-SSD training script."""

import glob
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class VocDetectionDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, class_names: List[str], transforms=None):
        """
        Args:
            dataset_dir: root folder containing <split>/JPEGImages, <split>/Annotations
            split: "train" or "val"
            class_names: ordered list of class names, background is implicit index 0
        """
        self.jpeg_dir = os.path.join(dataset_dir, split, "JPEGImages")
        self.ann_dir = os.path.join(dataset_dir, split, "Annotations")
        self.class_names = class_names
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}  # 0 = background
        self.transforms = transforms

        xml_paths = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in xml_paths]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image_path = os.path.join(self.jpeg_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._parse_annotation(ann_path)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _parse_annotation(self, ann_path: str) -> Tuple[List[List[float]], List[int]]:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_idx:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))
