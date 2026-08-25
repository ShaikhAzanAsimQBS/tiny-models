"""
Live inference with a fine-tuned YOLOX-Tiny checkpoint on a video, shown in
a pop-up window with a live FPS / inference-speed overlay.

Reuses the exact same Exp-building logic as training/train_yolox_tiny.py so
the model architecture always matches what was trained, plus YOLOX's own
official preprocessing (`ValTransform`) and postprocessing (`postprocess`,
NMS) utilities from the vendored repo.

Every setting lives in config.py - no command line arguments. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" inference\\infer_yolox_tiny.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

sys.path.insert(0, config.YOLOX_REPO_DIR)

import torch  # noqa: E402

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

from training.train_yolox_tiny import build_exp  # noqa: E402
from src.inference.live_runner import run_live_inference  # noqa: E402


def main():
    device = torch.device(config.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[infer_yolox_tiny] device = {device}")

    exp = build_exp()
    model = exp.get_model()
    # weights_only=False: PyTorch 2.6+ defaults to True, which rejects the
    # numpy scalar globals (e.g. best_ap) that YOLOX's own trainer pickles
    # into its checkpoints. Safe here since this loads our own checkpoint.
    ckpt = torch.load(config.YOLOX_INFER_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    class_names = config.DATASET_CLASS_NAMES
    num_classes = len(class_names)
    test_size = exp.test_size
    preproc = ValTransform(legacy=False)

    score_thr = config.INFERENCE_CONFIDENCE_THRESHOLD
    nms_thr = config.INFERENCE_NMS_IOU_THRESHOLD

    @torch.no_grad()
    def predict_fn(frame_bgr):
        ratio = min(test_size[0] / frame_bgr.shape[0], test_size[1] / frame_bgr.shape[1])
        img, _ = preproc(frame_bgr, None, test_size)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

        outputs = model(tensor)
        outputs = postprocess(outputs, num_classes, score_thr, nms_thr, class_agnostic=True)[0]
        if outputs is None:
            return []

        outputs = outputs.cpu()
        boxes = outputs[:, :4] / ratio
        obj_conf = outputs[:, 4]
        class_conf = outputs[:, 5]
        labels = outputs[:, 6].long()
        scores = obj_conf * class_conf

        return [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(l))
            for b, s, l in zip(boxes, scores, labels)
        ]

    save_path = (
        os.path.join(config.INFERENCE_OUTPUT_DIR, "yolox_tiny_output.mp4")
        if config.SAVE_INFERENCE else None
    )
    run_live_inference(
        video_path=config.YOLOX_INFER_VIDEO_PATH,
        predict_fn=predict_fn,
        class_names=class_names,
        window_title="YOLOX-Tiny",
        window_divisor=config.INFERENCE_WINDOW_DIVISOR,
        fps_smoothing=config.INFERENCE_FPS_SMOOTHING,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
