"""
Live inference with a fine-tuned NanoDet-Plus checkpoint on a video, shown
in a pop-up window with a live FPS / inference-speed overlay.

Reuses the exact same generated-config logic as training/train_nanodet.py
so the model architecture always matches what was trained, plus NanoDet's
own official preprocessing (`Pipeline`) and decode (`head.post_process`)
from the vendored repo.

Every setting lives in config.py - no command line arguments. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" inference\\infer_nanodet.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

sys.path.insert(0, config.NANODET_REPO_DIR)

import torch  # noqa: E402

from nanodet.data.batch_process import stack_batch_img  # noqa: E402
from nanodet.data.collate import naive_collate  # noqa: E402
from nanodet.data.transform import Pipeline  # noqa: E402
from nanodet.model.arch import build_model  # noqa: E402
from nanodet.util import cfg, load_config  # noqa: E402
from nanodet.util.check_point import load_model_weight  # noqa: E402

from training.train_nanodet import build_generated_config_path  # noqa: E402
from src.inference.live_runner import run_live_inference  # noqa: E402


class _PrintLogger:
    def log(self, msg):
        print(f"[infer_nanodet] {msg}")


def main():
    device = torch.device(config.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[infer_nanodet] device = {device}")

    yaml_path = build_generated_config_path()
    load_config(cfg, yaml_path)

    model = build_model(cfg.model)
    # weights_only=False: safe here since this loads our own checkpoint;
    # PyTorch 2.6+ defaults weights_only to True, which can reject globals
    # (e.g. numpy scalars) present in NanoDet-style checkpoints.
    ckpt = torch.load(config.NANODET_INFER_CHECKPOINT, map_location="cpu", weights_only=False)
    load_model_weight(model, ckpt, _PrintLogger())
    model.to(device).eval()

    class_names = list(cfg.class_names)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    input_size = cfg.data.val.input_size
    score_thr = config.INFERENCE_CONFIDENCE_THRESHOLD

    @torch.no_grad()
    def predict_fn(frame_bgr):
        h, w = frame_bgr.shape[:2]
        img_info = {"id": 0, "file_name": None, "height": h, "width": w}
        meta = dict(img_info=img_info, raw_img=frame_bgr, img=frame_bgr)
        meta = pipeline(None, meta, input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        preds = model(meta["img"])
        results = model.head.post_process(preds, meta)
        dets = results[0]  # {label_id: [[x1, y1, x2, y2, score], ...]}

        detections = []
        for label_id, boxes in dets.items():
            for box in boxes:
                x1, y1, x2, y2, score = box
                if score < score_thr:
                    continue
                detections.append((float(x1), float(y1), float(x2), float(y2), float(score), int(label_id)))
        return detections

    save_path = (
        os.path.join(config.INFERENCE_OUTPUT_DIR, "nanodet_output.mp4")
        if config.SAVE_INFERENCE else None
    )
    run_live_inference(
        video_path=config.NANODET_INFER_VIDEO_PATH,
        predict_fn=predict_fn,
        class_names=class_names,
        window_title="NanoDet-Plus",
        window_divisor=config.INFERENCE_WINDOW_DIVISOR,
        fps_smoothing=config.INFERENCE_FPS_SMOOTHING,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
