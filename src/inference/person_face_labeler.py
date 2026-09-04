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

Memory / robustness, for large (tens of thousands of images) runs:
    - Stage A and stage B are interleaved PER BATCH (stage B runs on a
      batch's crops immediately after stage A produces them) instead of
      running stage A over the *entire* dataset first and only then
      starting stage B. That bounds how many decoded crop images can be
      resident in memory at once to ~face_batch_size, instead of
      accumulating one crop per person box across the whole dataset.
    - Passing a list of paths to Ultralytics' `.predict()` routes through
      its PIL-based loader (`autocast_list`), which fully decodes every
      image in that list up front. A single corrupt/anomalous file in a
      batch (truncated write, decompression bomb, etc.) can throw
      (observed in practice as a `MemoryError`) and would otherwise kill an
      hours-long run over tens of thousands of images. Both stages retry
      failed batches one item at a time and skip+log only the offending
      file(s), so one bad image no longer aborts the whole pipeline.
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

    def _predict_person_batch(self, paths: List[str]) -> List[Tuple[str, object]]:
        """Runs the person model on a batch of image paths. If the whole
        batch call fails (e.g. MemoryError from one corrupt/anomalous
        image), retries one image at a time and skips+logs only the
        offending file(s), instead of aborting the entire run."""
        try:
            results = self.person_model.predict(
                source=paths,
                conf=self.person_conf,
                iou=self.person_iou,
                imgsz=self.person_img_size,
                device=self.person_device,
                classes=[self.person_coco_class_id],
                verbose=False,
            )
            return list(zip(paths, results))
        except Exception as exc:
            print(f"[person_face_labeler] WARNING: person-model batch of {len(paths)} image(s) "
                  f"failed ({type(exc).__name__}: {exc}). Retrying one image at a time to "
                  f"isolate the bad file(s)...")
            ok_pairs: List[Tuple[str, object]] = []
            for path in paths:
                try:
                    result = self.person_model.predict(
                        source=[path],
                        conf=self.person_conf,
                        iou=self.person_iou,
                        imgsz=self.person_img_size,
                        device=self.person_device,
                        classes=[self.person_coco_class_id],
                        verbose=False,
                    )[0]
                    ok_pairs.append((path, result))
                except Exception as exc2:
                    print(f"[person_face_labeler] SKIPPING unreadable/corrupt image "
                          f"(person stage): {path} ({type(exc2).__name__}: {exc2})")
            return ok_pairs

    def _predict_face_batch(self, jobs: List[Tuple[str, np.ndarray, float, float]]
                             ) -> List[Tuple[str, np.ndarray, float, float, object]]:
        """jobs: list of (image_path, crop, off_x, off_y). Same
        batch-then-fallback-to-one-at-a-time resilience as
        _predict_person_batch, applied to the face model on crops."""
        crop_arrays = [job[1] for job in jobs]
        try:
            results = self.face_model.predict(
                source=crop_arrays,
                conf=self.face_conf,
                iou=self.face_iou,
                imgsz=self.face_img_size,
                device=self.face_device,
                verbose=False,
            )
            return [(j[0], j[1], j[2], j[3], r) for j, r in zip(jobs, results)]
        except Exception as exc:
            print(f"[person_face_labeler] WARNING: face-model batch of {len(jobs)} crop(s) "
                  f"failed ({type(exc).__name__}: {exc}). Retrying one crop at a time...")
            ok_jobs: List[Tuple[str, np.ndarray, float, float, object]] = []
            for job in jobs:
                try:
                    result = self.face_model.predict(
                        source=[job[1]],
                        conf=self.face_conf,
                        iou=self.face_iou,
                        imgsz=self.face_img_size,
                        device=self.face_device,
                        verbose=False,
                    )[0]
                    ok_jobs.append((job[0], job[1], job[2], job[3], result))
                except Exception as exc2:
                    print(f"[person_face_labeler] SKIPPING a person crop from {job[0]} "
                          f"(face stage): ({type(exc2).__name__}: {exc2})")
            return ok_jobs

    def label_images(self, image_paths: List[str],
                      person_batch_size: int = None,
                      face_batch_size: int = None) -> List[ImageAnnotation]:
        person_batch_size = config.PERSON_INFERENCE_BATCH_SIZE if person_batch_size is None else person_batch_size
        face_batch_size = config.FACE_INFERENCE_BATCH_SIZE if face_batch_size is None else face_batch_size

        person_dets_per_image: Dict[str, List[BoxAnnotation]] = {}
        face_dets_per_image: Dict[str, List[BoxAnnotation]] = defaultdict(list)
        image_sizes: Dict[str, Tuple[int, int]] = {}

        total_person_boxes = 0
        total_crops_queued = 0
        total_face_boxes = 0

        for start in range(0, len(image_paths), person_batch_size):
            batch_paths = image_paths[start:start + person_batch_size]

            # ---- Stage A: person detection for this batch -----------------
            path_result_pairs = self._predict_person_batch(batch_paths)

            # (image_path, crop_array, offset_x, offset_y) for every person
            # box in THIS batch only - never accumulated across the whole
            # dataset, to keep memory bounded regardless of dataset size.
            batch_crop_jobs: List[Tuple[str, np.ndarray, float, float]] = []

            for path, result in path_result_pairs:
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
                total_person_boxes += len(boxes)

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
                    batch_crop_jobs.append((path, crop, float(cx1), float(cy1)))

            total_crops_queued += len(batch_crop_jobs)

            # ---- Stage B: face detection on this batch's crops, right away
            # (not deferred to the end), so crops never pile up in memory.
            for fstart in range(0, len(batch_crop_jobs), face_batch_size):
                fbatch = batch_crop_jobs[fstart:fstart + face_batch_size]
                for path, _crop, off_x, off_y, result in self._predict_face_batch(fbatch):
                    for box in result.boxes:
                        fx1, fy1, fx2, fy2 = [float(v) for v in box.xyxy[0].tolist()]
                        # Ultralytics already rescaled (fx1,fy1,fx2,fy2) to
                        # this crop's own pixel dimensions - just translate
                        # by the crop's offset within the full image.
                        face_dets_per_image[path].append(BoxAnnotation(
                            class_id=1, class_name=self.face_class_name,
                            xmin=fx1 + off_x, ymin=fy1 + off_y,
                            xmax=fx2 + off_x, ymax=fy2 + off_y,
                            confidence=float(box.conf.item()),
                        ))
                        total_face_boxes += 1

            print(f"[person_face_labeler] {min(start + person_batch_size, len(image_paths))}/{len(image_paths)} "
                  f"images - person boxes so far: {total_person_boxes}, "
                  f"face crops sent so far: {total_crops_queued}, face boxes so far: {total_face_boxes}")

        print(f"[person_face_labeler] done: {total_person_boxes} person box(es), {total_face_boxes} face box(es) "
              f"across {len(image_paths)} images ({total_crops_queued} person crop(s) sent to the face model)")

        # ---- Merge stage A + stage B results per image --------------------
        annotations: List[ImageAnnotation] = []
        for path in image_paths:
            if path not in image_sizes:
                continue  # image failed to load in stage A and was skipped
            width, height = image_sizes[path]
            boxes = list(person_dets_per_image.get(path, [])) + list(face_dets_per_image.get(path, []))
            if config.DROP_IMAGES_WITHOUT_DETECTIONS and not boxes:
                continue
            annotations.append(ImageAnnotation(image_path=path, width=width, height=height, boxes=boxes))

        return annotations
