"""Shared live-video inference loop used by all 4 inference/infer_*.py
scripts: opens a video, calls a model-specific `predict_fn(frame_bgr)` on
every frame, draws the returned boxes, overlays an exponentially-smoothed
FPS + per-frame inference time readout, and shows everything in a pop-up
window downscaled to (frame_w // divisor, frame_h // divisor) so a normal
video resolution comfortably fits a laptop screen.

Press 'q' or Esc to stop early.
"""

import os
import time
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

# One (B, G, R) color per class index, cycled if there are more classes.
_PALETTE = [
    (56, 189, 248), (34, 197, 94), (250, 204, 21), (244, 63, 94),
    (168, 85, 247), (14, 165, 233), (249, 115, 22), (163, 230, 53),
]

Detection = Tuple[float, float, float, float, float, int]  # x1, y1, x2, y2, score, label_idx


def _draw_detections(frame, detections: Sequence[Detection], class_names: List[str]):
    for x1, y1, x2, y2, score, label_idx in detections:
        color = _PALETTE[int(label_idx) % len(_PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        name = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)
        label_text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, label_text, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def run_live_inference(
    video_path: str,
    predict_fn: Callable[[any], List[Detection]],
    class_names: List[str],
    window_title: str,
    window_divisor: int = 2,
    fps_smoothing: float = 0.9,
    save_path: Optional[str] = None,
):
    """
    Args:
        video_path: path to a video file (or an int camera index).
        predict_fn: fn(frame_bgr: np.ndarray) -> list of
            (x1, y1, x2, y2, score, label_idx) in the *original* frame's
            pixel coordinates.
        class_names: ordered class names for label text.
        window_title: pop-up window title.
        window_divisor: the display window is frame_w // divisor by
            frame_h // divisor.
        fps_smoothing: EMA factor (0-1, higher = smoother/slower to react)
            for the on-screen FPS readout.
        save_path: if given, the full-resolution annotated frame (boxes +
            FPS overlay, *before* the pop-up-window downscale) is written to
            this video file as it plays. If None, nothing is saved to disk.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = src_fps if src_fps and src_fps > 1e-2 else 30.0
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if out_w and out_h:
            writer = cv2.VideoWriter(save_path, fourcc, out_fps, (out_w, out_h))
        # else: defer creation until the first frame is read, since some
        # backends report 0 for CAP_PROP_FRAME_WIDTH/HEIGHT up front.
        print(f"[live_runner] '{window_title}' - saving annotated output to: {save_path}")

    ema_fps = None
    total_frames, total_time, start_time = 0, 0.0, time.time()

    print(f"[live_runner] '{window_title}' - press 'q' or Esc to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            detections = predict_fn(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            instant_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            ema_fps = instant_fps if ema_fps is None else (
                fps_smoothing * ema_fps + (1 - fps_smoothing) * instant_fps
            )

            _draw_detections(frame, detections, class_names)

            h, w = frame.shape[:2]
            overlay = f"FPS: {ema_fps:5.1f}  |  inference: {infer_ms:6.2f} ms  |  detections: {len(detections)}"
            cv2.putText(frame, overlay, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if save_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, out_fps, (w, h))
                writer.write(frame)

            display_w = max(1, w // window_divisor)
            display_h = max(1, h // window_divisor)
            display_frame = cv2.resize(frame, (display_w, display_h))
            cv2.imshow(window_title, display_frame)

            total_frames += 1
            total_time += infer_ms / 1000.0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    wall_clock = time.time() - start_time
    avg_fps_wall = total_frames / wall_clock if wall_clock > 0 else 0.0
    avg_fps_infer_only = total_frames / total_time if total_time > 0 else 0.0
    print(f"[live_runner] '{window_title}' done - {total_frames} frames in {wall_clock:.1f}s "
          f"(avg {avg_fps_wall:.1f} FPS wall-clock, {avg_fps_infer_only:.1f} FPS inference-only, "
          f"avg inference time {1000.0 * total_time / max(1, total_frames):.2f} ms/frame)")
