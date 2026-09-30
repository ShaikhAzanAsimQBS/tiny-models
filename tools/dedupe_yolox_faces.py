"""
Standalone pass over an already-exported YOLOX (COCO) dataset: if a person
box contains more than one face box, keep one face and drop the extras.

Does not re-run labeling or recopy images - it only rewrites
annotations/instances_{train,val}2017.json in place. Safe to run more than
once (already-clean files stay clean).

Every setting is a plain variable below - no command line arguments. Run:

    python tools\\dedupe_yolox_faces.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from src.dataset_export.dedupe_faces import dedupe_coco_dataset_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Local settings for this tool - edit as needed
# ---------------------------------------------------------------------------
DATASET_DIR = config.YOLOX_DATASET_DIR
SPLITS = ("train", "val")


def main():
    print(f"[dedupe_yolox_faces] dataset_dir={DATASET_DIR}")
    removed = dedupe_coco_dataset_dir(DATASET_DIR, splits=SPLITS)
    print(f"[dedupe_yolox_faces] done. removed {removed} extra face box(es) total.")


if __name__ == "__main__":
    main()
