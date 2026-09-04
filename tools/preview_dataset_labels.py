"""
Sanity-check tool: draws the exported bounding boxes on top of a random
sample of images from one of the datasets/ folders, so you can eyeball that
labeling (classes + box positions/scaling) is correct BEFORE spending time
training on it.

Works on either dataset layout produced by src/dataset_export/:
    - Pascal VOC   (mobilenet_ssd/, nanodet/)      <split>/JPEGImages + <split>/Annotations/*.xml
    - COCO         (yolox/, rtmdet/)               <split>2017/ + annotations/instances_<split>2017.json
The layout is auto-detected from what's on disk - no need to tell it which one.

Every setting is a plain variable in the "Local settings" block below - no
command line arguments. Run with:

    python tools\\preview_dataset_labels.py
"""

import json
import os
import random
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Local settings for this tool - edit as needed
# ---------------------------------------------------------------------------
# Which dataset folder to preview. Point this at any of the 4 dataset dirs
# from config.py - the VOC vs COCO layout is auto-detected.
DATASET_DIR = config.YOLOX_DATASET_DIR
# DATASET_DIR = config.MOBILENET_SSD_DATASET_DIR
# DATASET_DIR = config.NANODET_DATASET_DIR
# DATASET_DIR = config.RTMDET_DATASET_DIR

SPLIT = "train"  # "train" or "val"
NUM_SAMPLES = 12  # how many random images to preview (10-15 suggested)
RANDOM_SEED = None  # set an int for a reproducible sample, None = different each run

OUTPUT_DIR = os.path.join(DATASET_DIR, "_label_preview")
BOX_THICKNESS = 2
FONT_SCALE = 0.6

# One (B, G, R) color per class name, cycled/hashed if there are more
# classes than colors.
_PALETTE = [
    (56, 189, 248), (34, 197, 94), (250, 204, 21), (244, 63, 94),
    (168, 85, 247), (14, 165, 233), (249, 115, 22), (163, 230, 53),
]


def _color_for(class_name: str):
    return _PALETTE[hash(class_name) % len(_PALETTE)]


def _draw_and_save(image_path: str, boxes, out_path: str) -> None:
    """boxes: list of (class_name, xmin, ymin, xmax, ymax)."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[preview_dataset_labels] WARNING: could not read image {image_path}, skipping.")
        return

    for class_name, xmin, ymin, xmax, ymax in boxes:
        color = _color_for(class_name)
        p1, p2 = (int(round(xmin)), int(round(ymin))), (int(round(xmax)), int(round(ymax)))
        cv2.rectangle(image, p1, p2, color, BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        label_top = max(0, p1[1] - th - 6)
        cv2.rectangle(image, (p1[0], label_top), (p1[0] + tw + 4, label_top + th + 6), color, -1)
        cv2.putText(image, class_name, (p1[0] + 2, label_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)


def _detect_layout(dataset_dir: str, split: str) -> str:
    voc_ann_dir = os.path.join(dataset_dir, split, "Annotations")
    coco_json = os.path.join(dataset_dir, "annotations", f"instances_{split}2017.json")
    if os.path.isdir(voc_ann_dir):
        return "voc"
    if os.path.isfile(coco_json):
        return "coco"
    raise FileNotFoundError(
        f"Could not detect a VOC or COCO layout for split '{split}' under {dataset_dir}. "
        f"Checked: {voc_ann_dir} and {coco_json}."
    )


def _preview_voc(dataset_dir: str, split: str, n: int, seed) -> int:
    ann_dir = os.path.join(dataset_dir, split, "Annotations")
    img_dir = os.path.join(dataset_dir, split, "JPEGImages")

    xml_files = sorted(f for f in os.listdir(ann_dir) if f.lower().endswith(".xml"))
    if not xml_files:
        print(f"[preview_dataset_labels] no .xml annotations found in {ann_dir}")
        return 0

    rng = random.Random(seed)
    sample = rng.sample(xml_files, k=min(n, len(xml_files)))

    saved = 0
    for xml_name in sample:
        tree = ET.parse(os.path.join(ann_dir, xml_name))
        root = tree.getroot()
        image_file_name = root.findtext("filename")
        image_path = os.path.join(img_dir, image_file_name)

        boxes = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name")
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            boxes.append((class_name, xmin, ymin, xmax, ymax))

        stem = os.path.splitext(image_file_name)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}__{len(boxes)}boxes.jpg")
        _draw_and_save(image_path, boxes, out_path)
        saved += 1
        print(f"[preview_dataset_labels] {image_file_name}: {len(boxes)} box(es) -> {out_path}")

    return saved


def _preview_coco(dataset_dir: str, split: str, n: int, seed) -> int:
    img_dir = os.path.join(dataset_dir, f"{split}2017")
    json_path = os.path.join(dataset_dir, "annotations", f"instances_{split}2017.json")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if not coco["images"]:
        print(f"[preview_dataset_labels] no images found in {json_path}")
        return 0

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    boxes_by_image_id = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        class_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
        boxes_by_image_id.setdefault(ann["image_id"], []).append((class_name, x, y, x + w, y + h))

    rng = random.Random(seed)
    sample = rng.sample(coco["images"], k=min(n, len(coco["images"])))

    saved = 0
    for image_entry in sample:
        image_path = os.path.join(img_dir, image_entry["file_name"])
        boxes = boxes_by_image_id.get(image_entry["id"], [])

        stem = os.path.splitext(image_entry["file_name"])[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}__{len(boxes)}boxes.jpg")
        _draw_and_save(image_path, boxes, out_path)
        saved += 1
        print(f"[preview_dataset_labels] {image_entry['file_name']}: {len(boxes)} box(es) -> {out_path}")

    return saved


def main():
    layout = _detect_layout(DATASET_DIR, SPLIT)
    print(f"[preview_dataset_labels] dataset_dir={DATASET_DIR} split={SPLIT} layout={layout}")

    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if layout == "voc":
        saved = _preview_voc(DATASET_DIR, SPLIT, NUM_SAMPLES, RANDOM_SEED)
    else:
        saved = _preview_coco(DATASET_DIR, SPLIT, NUM_SAMPLES, RANDOM_SEED)

    print(f"[preview_dataset_labels] saved {saved} annotated preview image(s) to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
