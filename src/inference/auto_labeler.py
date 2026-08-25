"""Runs the fine-tuned YOLO11n teacher model (general_product_detection.pt)
over a list of images and turns the results into ImageAnnotation objects
that the dataset exporters understand."""

from typing import List

from PIL import Image
from ultralytics import YOLO

import config
from src.common.annotation_types import BoxAnnotation, ImageAnnotation


class YoloAutoLabeler:
    def __init__(self,
                 model_path: str = None,
                 class_names: List[str] = None,
                 confidence_threshold: float = None,
                 iou_threshold: float = None,
                 img_size: int = None,
                 device: str = None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.class_names = class_names or config.YOLO_CLASS_NAMES
        self.confidence_threshold = (
            config.YOLO_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.iou_threshold = config.YOLO_IOU_THRESHOLD if iou_threshold is None else iou_threshold
        self.img_size = config.YOLO_IMG_SIZE if img_size is None else img_size
        self.device = config.YOLO_DEVICE if device is None else device

        print(f"[auto_labeler] Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)

    def label_images(self, image_paths: List[str],
                      batch_size: int = None) -> List[ImageAnnotation]:
        batch_size = config.YOLO_INFERENCE_BATCH_SIZE if batch_size is None else batch_size
        annotations: List[ImageAnnotation] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            results = self.model.predict(
                source=batch_paths,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )

            for path, result in zip(batch_paths, results):
                height, width = result.orig_shape
                boxes: List[BoxAnnotation] = []
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    xmin, ymin, xmax, ymax = [float(v) for v in box.xyxy[0].tolist()]
                    class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                    boxes.append(BoxAnnotation(
                        class_id=cls_id,
                        class_name=class_name,
                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                        confidence=conf,
                    ))

                if config.DROP_IMAGES_WITHOUT_DETECTIONS and not boxes:
                    continue

                annotations.append(ImageAnnotation(
                    image_path=path, width=width, height=height, boxes=boxes,
                ))

            print(f"[auto_labeler] labeled {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")

        return annotations


def get_image_size(image_path: str):
    with Image.open(image_path) as img:
        return img.width, img.height
