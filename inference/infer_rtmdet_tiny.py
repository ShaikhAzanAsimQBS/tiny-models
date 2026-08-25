"""
Live inference with a fine-tuned RTMDet-tiny checkpoint on a video, shown in
a pop-up window with a live FPS / inference-speed overlay.

Every setting lives in config.py - no command line arguments. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" inference\\infer_rtmdet_tiny.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

import config
from src.dataset_export.coco_dataset_torch import letterbox
from src.models.rtmdet.model import PIXEL_MEAN, PIXEL_STD, build_rtmdet
from src.inference.live_runner import run_live_inference


def main():
    device = torch.device(config.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[infer_rtmdet_tiny] device = {device}")

    ckpt = torch.load(config.RTMDET_INFER_CHECKPOINT, map_location="cpu", weights_only=False)
    variant = ckpt.get("variant", config.RTMDET_VARIANT)
    class_names = ckpt.get("class_names", config.DATASET_CLASS_NAMES)
    input_size = tuple(ckpt.get("input_size", config.RTMDET_INPUT_SIZE))

    model = build_rtmdet(variant, num_classes=len(class_names))
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    mean = np.array(PIXEL_MEAN, dtype=np.float32)
    std = np.array(PIXEL_STD, dtype=np.float32)
    score_thr = config.INFERENCE_CONFIDENCE_THRESHOLD
    iou_thr = config.INFERENCE_NMS_IOU_THRESHOLD

    @torch.no_grad()
    def predict_fn(frame_bgr):
        canvas, _, scale = letterbox(frame_bgr, np.zeros((0, 4), dtype=np.float32), input_size)
        normed = (canvas.astype(np.float32) - mean) / std
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        boxes, scores, labels = model.predict(tensor, score_thr=score_thr, iou_thr=iou_thr)
        boxes = boxes.cpu() / scale  # undo letterbox scaling -> original frame pixel coords
        scores = scores.cpu()
        labels = labels.cpu()

        return [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(l))
            for b, s, l in zip(boxes, scores, labels)
        ]

    save_path = (
        os.path.join(config.INFERENCE_OUTPUT_DIR, "rtmdet_tiny_output.mp4")
        if config.SAVE_INFERENCE else None
    )
    run_live_inference(
        video_path=config.RTMDET_INFER_VIDEO_PATH,
        predict_fn=predict_fn,
        class_names=class_names,
        window_title="RTMDet-tiny",
        window_divisor=config.INFERENCE_WINDOW_DIVISOR,
        fps_smoothing=config.INFERENCE_FPS_SMOOTHING,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
