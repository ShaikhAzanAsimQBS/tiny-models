"""Two-stage auto-labeler used by pipeline.py:

    Stage A - config.PERSON_MODEL_PATH (yolo11x.pt, stock COCO-pretrained)
              runs on the FULL image and keeps only its "person" class
              detections.
    Stage B - config.FACE_MODEL_PATH (face11n.pt, fine-tuned, 1-class
              "face") runs on a padded CROP of every person box found in
              stage A, not on the full frame. This gives the face model a
              much higher effective resolution per-person to work with.

Bounding-box correctness across the crop boundary:
    Ultralytics' `model.predict()` already internally letterboxes whatever
    image/array it's given to `imgsz` for the forward pass, then rescales
    the resulting boxes back to that *same input array's* original pixel
    dimensions before returning them (this is exactly what stage A already
    relies on via `result.orig_shape`/`box.xyxy`). So when stage B is handed
    a person crop, the boxes it returns are already correctly scaled to
    that crop's own pixel dimensions - the ONLY extra transform needed here
    is a plain translation: add the crop's (x_offset, y_offset) within the
    original full image to every returned face box coordinate. No extra
    resize-ratio math is required (and doing so on top would double-scale
    the boxes) - this module only ever adds an offset to already-correctly-
    scaled crop-local coordinates.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

import config
from src.common.annotation_types import BoxAnnotation, ImageAnnotation


class PersonFaceAutoLabeler:
    def __init__(self,
                 person_model_path: str = None,
                 person_class_name: str = None,
                 person_coco_class_id: int = None,
                 person_confidence_threshold: float = None,
                 person_iou_threshold: float = None,
                 person_img_size: int = None,
                 person_device: str = None,
                 face_model_path: str = None,
                 face_class_name: str = None,
                 face_confidence_threshold: float = None,
                 face_iou_threshold: float = None,
                 face_img_size: int = None,
                 face_device: str = None,
                 face_crop_padding_ratio: float = None,
                 face_min_crop_side: int = None):
        self.person_class_name = person_class_name or config.PERSON_CLASS_NAME
        self.person_coco_class_id = (
            config.PERSON_COCO_CLASS_ID if person_coco_class_id is None else person_coco_class_id
        )
        self.person_conf = (
            config.PERSON_CONFIDENCE_THRESHOLD if person_confidence_threshold is None else person_confidence_threshold
        )
        self.person_iou = config.PERSON_IOU_THRESHOLD if person_iou_threshold is None else person_iou_threshold
        self.person_img_size = config.PERSON_IMG_SIZE if person_img_size is None else person_img_size
        self.person_device = config.PERSON_DEVICE if person_device is None else person_device

        self.face_class_name = face_class_name or config.FACE_CLASS_NAME
        self.face_conf = config.FACE_CONFIDENCE_THRESHOLD if face_confidence_threshold is None else face_confidence_threshold
        self.face_iou = config.FACE_IOU_THRESHOLD if face_iou_threshold is None else face_iou_threshold
        self.face_img_size = config.FACE_IMG_SIZE if face_img_size is None else face_img_size
        self.face_device = config.FACE_DEVICE if face_device is None else face_device
        self.face_crop_padding_ratio = (
            config.FACE_CROP_PADDING_RATIO if face_crop_padding_ratio is None else face_crop_padding_ratio
        )
        self.face_min_crop_side = config.FACE_MIN_CROP_SIDE if face_min_crop_side is None else face_min_crop_side

        person_model_path = person_model_path or config.PERSON_MODEL_PATH
        face_model_path = face_model_path or config.FACE_MODEL_PATH

        print(f"[person_face_labeler] loading person model from {person_model_path}")
        self.person_model = YOLO(person_model_path)
        print(f"[person_face_labeler] loading face model from {face_model_path}")
        self.face_model = YOLO(face_model_path)

        self._sanity_check_class_names()

    def _sanity_check_class_names(self) -> None:
        """Defensive check: don't silently mislabel a whole dataset because
        one of the two checkpoints has an unexpected class layout."""
        person_names = self.person_model.names  # {id: name}
        actual_person_name = person_names.get(self.person_coco_class_id)
        if actual_person_name is None or actual_person_name.lower() != self.person_class_name.lower():
            raise ValueError(
                f"person model class id {self.person_coco_class_id} is "
                f"'{actual_person_name}', not '{self.person_class_name}'. "
                f"person_model.names = {person_names}. Check config.PERSON_MODEL_PATH / "
                f"config.PERSON_COCO_CLASS_ID."
            )

        face_names = self.face_model.names
        if len(face_names) != 1 or list(face_names.values())[0].lower() != self.face_class_name.lower():
            print(f"[person_face_labeler] WARNING: expected the face model to have exactly "
                  f"1 class named '{self.face_class_name}', but it has {face_names}. "
                  f"Every detection from it will still be labeled '{self.face_class_name}' - "
                  f"double check config.FACE_MODEL_PATH if that's not what you intended.")

    @staticmethod
    def _padded_clamped_crop_box(xmin: float, ymin: float, xmax: float, ymax: float,
                                  img_w: int, img_h: int, padding_ratio: float) -> Tuple[int, int, int, int]:
        box_w, box_h = xmax - xmin, ymax - ymin
        pad_x, pad_y = box_w * padding_ratio, box_h * padding_ratio
        cx1 = max(0, int(round(xmin - pad_x)))
        cy1 = max(0, int(round(ymin - pad_y)))
        cx2 = min(img_w, int(round(xmax + pad_x)))
        cy2 = min(img_h, int(round(ymax + pad_y)))
        return cx1, cy1, cx2, cy2

    def label_images(self, image_paths: List[str],
                      person_batch_size: int = None,
                      face_batch_size: int = None) -> List[ImageAnnotation]:
        person_batch_size = config.PERSON_INFERENCE_BATCH_SIZE if person_batch_size is None else person_batch_size
        face_batch_size = config.FACE_INFERENCE_BATCH_SIZE if face_batch_size is None else face_batch_size

        person_dets_per_image: Dict[str, List[BoxAnnotation]] = {}
        image_sizes: Dict[str, Tuple[int, int]] = {}
        # (image_path, crop_array, offset_x, offset_y) for every person box
        # big enough to bother running the face model on.
        crop_jobs: List[Tuple[str, np.ndarray, float, float]] = []

        # ---- Stage A: batched person detection over full images ----------
        for start in range(0, len(image_paths), person_batch_size):
            batch_paths = image_paths[start:start + person_batch_size]
            results = self.person_model.predict(
                source=batch_paths,
                conf=self.person_conf,
                iou=self.person_iou,
                imgsz=self.person_img_size,
                device=self.person_device,
                classes=[self.person_coco_class_id],
                verbose=False,
            )

            for path, result in zip(batch_paths, results):
                height, width = result.orig_shape
                image_sizes[path] = (width, height)

                boxes: List[BoxAnnotation] = []
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = [float(v) for v in box.xyxy[0].tolist()]
                    boxes.append(BoxAnnotation(
                        class_id=0, class_name=self.person_class_name,
                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                        confidence=float(box.conf.item()),
                    ))
                person_dets_per_image[path] = boxes

                # result.orig_img is the exact BGR array Ultralytics already
                # decoded for this image - reuse it instead of re-reading
                # the file from disk just to crop it.
                orig_img = result.orig_img
                for box in boxes:
                    cx1, cy1, cx2, cy2 = self._padded_clamped_crop_box(
                        box.xmin, box.ymin, box.xmax, box.ymax, width, height, self.face_crop_padding_ratio,
                    )
                    if (cx2 - cx1) < self.face_min_crop_side or (cy2 - cy1) < self.face_min_crop_side:
                        continue
                    crop = orig_img[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    crop_jobs.append((path, crop, float(cx1), float(cy1)))

            print(f"[person_face_labeler] stage A (person): "
                  f"{min(start + person_batch_size, len(image_paths))}/{len(image_paths)} images")

        total_person_boxes = sum(len(b) for b in person_dets_per_image.values())
        print(f"[person_face_labeler] found {total_person_boxes} person box(es) across "
              f"{len(image_paths)} images -> {len(crop_jobs)} crop(s) queued for face detection")

        # ---- Stage B: batched face detection over every person crop -------
        face_dets_per_image: Dict[str, List[BoxAnnotation]] = defaultdict(list)
        for start in range(0, len(crop_jobs), face_batch_size):
            batch = crop_jobs[start:start + face_batch_size]
            crop_arrays = [item[1] for item in batch]
            results = self.face_model.predict(
                source=crop_arrays,
                conf=self.face_conf,
                iou=self.face_iou,
                imgsz=self.face_img_size,
                device=self.face_device,
                verbose=False,
            )

            for (path, _crop, off_x, off_y), result in zip(batch, results):
                for box in result.boxes:
                    fx1, fy1, fx2, fy2 = [float(v) for v in box.xyxy[0].tolist()]
                    # Ultralytics already rescaled (fx1,fy1,fx2,fy2) to this
                    # crop's own pixel dimensions - just translate by the
                    # crop's offset within the full image.
                    face_dets_per_image[path].append(BoxAnnotation(
                        class_id=1, class_name=self.face_class_name,
                        xmin=fx1 + off_x, ymin=fy1 + off_y,
                        xmax=fx2 + off_x, ymax=fy2 + off_y,
                        confidence=float(box.conf.item()),
                    ))

            if crop_jobs:
                print(f"[person_face_labeler] stage B (face): "
                      f"{min(start + face_batch_size, len(crop_jobs))}/{len(crop_jobs)} crops")

        total_face_boxes = sum(len(b) for b in face_dets_per_image.values())
        print(f"[person_face_labeler] found {total_face_boxes} face box(es)")

        # ---- Merge stage A + stage B results per image --------------------
        annotations: List[ImageAnnotation] = []
        for path in image_paths:
            if path not in image_sizes:
                continue
            width, height = image_sizes[path]
            boxes = list(person_dets_per_image.get(path, [])) + list(face_dets_per_image.get(path, []))
            if config.DROP_IMAGES_WITHOUT_DETECTIONS and not boxes:
                continue
            annotations.append(ImageAnnotation(image_path=path, width=width, height=height, boxes=boxes))

        return annotations
