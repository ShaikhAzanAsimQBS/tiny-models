"""
Live inference with a fine-tuned MobileNet-SSD (torchvision SSDLite320-
MobileNetV3-Large) checkpoint on a video, shown in a pop-up window with a
live FPS / inference-speed overlay.

Every setting (input video, checkpoint path, confidence threshold, window
size, device) lives in config.py - no command line arguments. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" inference\\infer_mobilenet_ssd.py
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torchvision
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

import config
from src.inference.live_runner import run_live_inference


def build_model(num_classes_with_background: int):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (config.MOBILENET_SSD_INPUT_SIZE,) * 2)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes_with_background, norm_layer
    )
    return model


def main():
    device = torch.device(config.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[infer_mobilenet_ssd] device = {device}")

    # weights_only=False: safe here since this loads our own checkpoint;
    # PyTorch 2.6+ defaults weights_only to True, which can reject globals
    # (e.g. numpy scalars) present in the saved checkpoint dict.
    ckpt = torch.load(config.MOBILENET_SSD_INFER_CHECKPOINT, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", config.DATASET_CLASS_NAMES)

    model = build_model(len(class_names) + 1)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    score_thr = config.INFERENCE_CONFIDENCE_THRESHOLD

    @torch.no_grad()
    def predict_fn(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
        output = model([tensor])[0]

        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()  # 1-indexed, 0 = background

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score.item() < score_thr:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append((x1, y1, x2, y2, score.item(), int(label.item()) - 1))
        return detections

    save_path = (
        os.path.join(config.INFERENCE_OUTPUT_DIR, "mobilenet_ssd_output.mp4")
        if config.SAVE_INFERENCE else None
    )
    run_live_inference(
        video_path=config.MOBILENET_SSD_INFER_VIDEO_PATH,
        predict_fn=predict_fn,
        class_names=class_names,
        window_title="MobileNet-SSD",
        window_divisor=config.INFERENCE_WINDOW_DIVISOR,
        fps_smoothing=config.INFERENCE_FPS_SMOOTHING,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
