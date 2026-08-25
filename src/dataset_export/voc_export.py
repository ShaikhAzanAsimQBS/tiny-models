"""Exports ImageAnnotation objects into a Pascal-VOC style dataset:

    <output_dir>/
        train/JPEGImages/*.jpg
        train/Annotations/*.xml
        val/JPEGImages/*.jpg
        val/Annotations/*.xml

Separate per-split folders (rather than one shared folder + an ImageSets
split-list) so this layout works directly, unmodified, with:
  - the custom VocDetectionDataset used by train_mobilenet_ssd.py
  - NanoDet's built-in `XMLDataset`, which just walks a given img/ann folder
    (it has no concept of a shared-folder split list).
"""

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List

import config
from src.common.annotation_types import ImageAnnotation
from src.dataset_export.split import train_val_split


def _prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough_string).toprettyxml(indent="  ")


def _write_voc_xml(annotation: ImageAnnotation, image_file_name: str, xml_path: str) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = image_file_name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(annotation.width)
    ET.SubElement(size, "height").text = str(annotation.height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for box in annotation.boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(round(box.xmin)))
        ET.SubElement(bndbox, "ymin").text = str(int(round(box.ymin)))
        ET.SubElement(bndbox, "xmax").text = str(int(round(box.xmax)))
        ET.SubElement(bndbox, "ymax").text = str(int(round(box.ymax)))

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(_prettify(root))


def _write_split(split_annotations: List[ImageAnnotation], split_dir: str, split_name: str) -> None:
    jpeg_dir = os.path.join(split_dir, "JPEGImages")
    ann_dir = os.path.join(split_dir, "Annotations")
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for idx, ann in enumerate(split_annotations):
        image_id = f"{split_name}_{idx:06d}"
        image_file_name = f"{image_id}.jpg"
        shutil.copyfile(ann.image_path, os.path.join(jpeg_dir, image_file_name))
        _write_voc_xml(ann, image_file_name, os.path.join(ann_dir, f"{image_id}.xml"))


def export_voc_dataset(annotations: List[ImageAnnotation],
                        output_dir: str,
                        split_ratio: float = None,
                        seed: int = None) -> None:
    split_ratio = config.TRAIN_VAL_SPLIT_RATIO if split_ratio is None else split_ratio
    seed = config.SPLIT_RANDOM_SEED if seed is None else seed

    train_set, val_set = train_val_split(annotations, split_ratio, seed)

    _write_split(train_set, os.path.join(output_dir, "train"), "train")
    _write_split(val_set, os.path.join(output_dir, "val"), "val")

    print(f"[voc_export] {output_dir}: {len(train_set)} train / {len(val_set)} val images")
