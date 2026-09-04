"""
Standalone RTSP live-viewer + on-demand recorder.

Connects to an RTSP camera stream and shows it in a live pop-up window. You
can toggle recording on and off as many times as you like while it's
running:

    - Press 's' -> start a new recording (saved to a timestamped .mp4 file)
    - Press 'c' -> stop the current recording
    - Press 'q' or Esc -> quit (stops any active recording first)

This file is fully self-contained (no dependency on config.py or the rest of
the dataset_creator project) - every setting is a plain variable below, no
command line arguments needed.

Requirements: opencv-python (cv2). Run with:

    python rtsp_recorder.py

NOTE: this file has the camera's RTSP credentials hardcoded below in plain
text - avoid committing it to a shared/public git repo as-is.
"""

import os
import time
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# Settings - edit these as needed
# ---------------------------------------------------------------------------
RTSP_URL = "rtsp://qbs.cctv:$$ccqbs%402031@185.84.166.74:554/Streaming/Channels/1501"

# Where recorded clips are saved. Each recording session (s -> c) is written
# to its own timestamped file: recording_YYYYMMDD_HHMMSS.mp4
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# Force RTSP-over-TCP (via OpenCV's FFmpeg backend) instead of the default
# UDP transport. TCP is much more reliable over WiFi/the internet and avoids
# the corrupted/green/frozen frames that UDP packet loss causes.
FORCE_RTSP_TCP = True

# Fallback FPS used for the *saved recording file* if the camera doesn't
# report a valid FPS (very common with RTSP streams - cap.get(CAP_PROP_FPS)
# often returns 0 or a bogus value).
DEFAULT_RECORD_FPS = 20.0

# fourcc codec for the saved .mp4 files.
VIDEO_CODEC = "mp4v"

# The live pop-up window is shown at (frame_w // divisor, frame_h // divisor)
# so a normal camera resolution fits comfortably on screen. Set to 1 to show
# the stream at full resolution.
WINDOW_DIVISOR = 2

WINDOW_TITLE = "RTSP Live View  [s: start rec | c: stop rec | q: quit]"

# If a frame read fails (transient network hiccup, camera reboot, etc.), the
# stream is reopened and retried this many times before giving up.
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY_SEC = 2.0


def _redact_url(url: str) -> str:
    """Hides the password portion of an rtsp://user:pass@host URL, for safe printing/logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host_part = rest.split("@", 1)
    user = creds.split(":", 1)[0] if ":" in creds else creds
    return f"{scheme}://{user}:****@{host_part}"


def _open_capture(url: str) -> cv2.VideoCapture:
    if FORCE_RTSP_TCP:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[rtsp_recorder] connecting to {_redact_url(RTSP_URL)} ...")

    cap = _open_capture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the RTSP stream. Check the URL/credentials, "
            "network connectivity/firewall, and that the camera/channel is online."
        )
    print("[rtsp_recorder] connected. Live window opening - "
          "press 's' to start recording, 'c' to stop, 'q' or Esc to quit.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = None
    record_start_time = None
    recording_path = None
    reconnect_attempts = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[rtsp_recorder] lost frame from stream, attempting to reconnect...")
                cap.release()
                reconnect_attempts += 1
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    print("[rtsp_recorder] giving up after too many failed reconnect attempts.")
                    break
                time.sleep(RECONNECT_DELAY_SEC)
                cap = _open_capture(RTSP_URL)
                continue
            reconnect_attempts = 0

            # Recording always writes the raw, full-resolution frame -
            # only the displayed copy gets the overlay + downscale.
            if writer is not None:
                writer.write(frame)

            display_frame = frame.copy()
            if writer is not None:
                elapsed = time.time() - record_start_time
                cv2.circle(display_frame, (24, 24), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC {elapsed:6.1f}s", (44, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "not recording", (24, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(display_frame, "s: start rec | c: stop rec | q: quit",
                        (24, display_frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            h, w = display_frame.shape[:2]
            show_w, show_h = max(1, w // WINDOW_DIVISOR), max(1, h // WINDOW_DIVISOR)
            cv2.imshow(WINDOW_TITLE, cv2.resize(display_frame, (show_w, show_h)))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if writer is None:
                    h0, w0 = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 1e-2 or fps > 120:
                        fps = DEFAULT_RECORD_FPS
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.mp4")
                    writer = cv2.VideoWriter(recording_path, fourcc, fps, (w0, h0))
                    record_start_time = time.time()
                    print(f"[rtsp_recorder] recording STARTED -> {recording_path}")
                else:
                    print("[rtsp_recorder] already recording, ignoring 's'.")

            elif key == ord("c"):
                if writer is not None:
                    writer.release()
                    duration = time.time() - record_start_time
                    print(f"[rtsp_recorder] recording STOPPED -> {recording_path} ({duration:.1f}s)")
                    writer, record_start_time, recording_path = None, None, None
                else:
                    print("[rtsp_recorder] not currently recording, ignoring 'c'.")

            elif key == ord("q") or key == 27:  # 'q' or Esc
                print("[rtsp_recorder] quitting...")
                break

    finally:
        if writer is not None:
            writer.release()
            print(f"[rtsp_recorder] recording stopped (on exit) -> {recording_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
