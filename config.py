"""
Central configuration for the dataset_creator pipeline.

Design rule for this whole project: NOTHING is passed on the command line /
via argparse. Every value that could reasonably be a CLI flag is a plain
Python variable instead, defined here (individual scripts also keep a small
"local overrides" block at the top for values that only make sense for that
script, e.g. training hyperparameters). To change behaviour, just edit the
variables below and re-run the relevant script with the target venv:

    C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe pipeline.py
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
# venv that already has a CUDA-enabled torch/torchvision installed. All
# scripts in this project are meant to be launched with this interpreter.
VENV_PYTHON = r"C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe"

# ---------------------------------------------------------------------------
# Stage 1 - Input source: a single video OR a folder of images/videos
# ---------------------------------------------------------------------------
# Point this at either:
#   - a single video file, e.g. r"C:\data\shelf_walkthrough.mp4"
#   - a folder containing any mix of images and videos (searched recursively),
#     e.g. r"C:\data\raw_media"
INPUT_PATH = r"C:\Users\QBS PC\PycharmProjects\dataset_creator\input_vid\vid1.mp4"

# When a video is processed (INPUT_PATH itself, or a video found while
# walking a folder), only every Nth decoded frame is kept/labeled.
# 1 = keep every frame, 5 = keep 1 out of every 5 frames, etc.
FRAME_SKIP = 10

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Frames pulled out of videos are written here as .jpg files before labeling
# (images found directly on disk are read/labeled in place and are not copied
# here).
EXTRACTED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
FRAME_JPEG_QUALITY = 95  # 0-100

# ---------------------------------------------------------------------------
# Stage 2 - Auto-labeling with the fine-tuned YOLO11n teacher model
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "general_product_detection.pt")
YOLO_CLASS_NAMES = ["product"]  # the single class the teacher model detects

YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMG_SIZE = 640
YOLO_DEVICE = "cuda:0"  # "cuda:0", "cuda:1", "cpu", ...
YOLO_INFERENCE_BATCH_SIZE = 16

# If True, images/frames with zero detections are discarded and never make
# it into any of the 3 output datasets.
DROP_IMAGES_WITHOUT_DETECTIONS = True

# ---------------------------------------------------------------------------
# Stage 3 - Dataset build (auto-labeled detections -> 3 training datasets)
# ---------------------------------------------------------------------------
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

TRAIN_VAL_SPLIT_RATIO = 0.9  # fraction of images assigned to the train split
SPLIT_RANDOM_SEED = 42

DATASET_CLASS_NAMES = ["product"]  # must match YOLO_CLASS_NAMES

# MobileNet-SSD dataset - Pascal VOC layout:
#   JPEGImages/*.jpg, Annotations/*.xml, ImageSets/Main/{train,val}.txt
MOBILENET_SSD_DATASET_DIR = os.path.join(DATASETS_ROOT, "mobilenet_ssd")

# NanoDet dataset - same Pascal VOC layout (NanoDet's built-in XMLDataset
# reads this format natively), kept in its own folder.
NANODET_DATASET_DIR = os.path.join(DATASETS_ROOT, "nanodet")

# YOLOX dataset - COCO layout:
#   train2017/*.jpg, val2017/*.jpg, annotations/instances_{train,val}2017.json
YOLOX_DATASET_DIR = os.path.join(DATASETS_ROOT, "yolox")

# RTMDet dataset - COCO layout (identical structure to YOLOX's, this is the
# exact format mmdetection's CocoDataset expects):
#   train2017/*.jpg, val2017/*.jpg, annotations/instances_{train,val}2017.json
RTMDET_DATASET_DIR = os.path.join(DATASETS_ROOT, "rtmdet")

# ---------------------------------------------------------------------------
# Stage 4 - Training (shared)
# ---------------------------------------------------------------------------
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
TRAIN_DEVICE = "cuda:0"
TRAIN_NUM_WORKERS = 4
TRAIN_RANDOM_SEED = 42

# ---------------- MobileNet-SSD (torchvision SSDLite320-MobileNetV3-Large) -
MOBILENET_SSD_OUTPUT_DIR = os.path.join(RUNS_ROOT, "mobilenet_ssd")
MOBILENET_SSD_EPOCHS = 50
MOBILENET_SSD_BATCH_SIZE = 8
MOBILENET_SSD_LEARNING_RATE = 0.001
MOBILENET_SSD_WEIGHT_DECAY = 0.0005
MOBILENET_SSD_MOMENTUM = 0.9
MOBILENET_SSD_INPUT_SIZE = 320
MOBILENET_SSD_PRETRAINED_BACKBONE = True  # start from COCO-pretrained torchvision weights
MOBILENET_SSD_RESUME_CHECKPOINT = ""  # path to a .pt to resume from, "" = none
MOBILENET_SSD_CHECKPOINT_INTERVAL = 5  # save every N epochs
MOBILENET_SSD_SCORE_THRESH = 0.35  # inference score threshold baked into the exported model

# ---------------- YOLOX-Tiny (vendored external/YOLOX) ---------------------
YOLOX_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "YOLOX")
YOLOX_EXPERIMENT_NAME = "yolox_tiny_product"
YOLOX_OUTPUT_DIR = os.path.join(RUNS_ROOT, "yolox_tiny")
YOLOX_MAX_EPOCH = 45
YOLOX_BATCH_SIZE = 16
YOLOX_INPUT_SIZE = (416, 416)  # (height, width), must be multiples of 32
YOLOX_TEST_SIZE = (416, 416)
YOLOX_WARMUP_EPOCHS = 5
YOLOX_NO_AUG_EPOCHS = 15
YOLOX_BASIC_LR_PER_IMG = 0.01 / 64.0
YOLOX_EVAL_INTERVAL = 10
YOLOX_PRINT_INTERVAL = 10
YOLOX_DATA_NUM_WORKERS = 4
# Official COCO-pretrained YOLOX-Tiny weights, used as the fine-tuning start point.
YOLOX_PRETRAINED_CKPT = os.path.join(PROJECT_ROOT, "pretrained", "yolox_tiny.pth")
YOLOX_PRETRAINED_CKPT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
YOLOX_AUTO_DOWNLOAD_PRETRAINED = True
YOLOX_FP16 = False
YOLOX_RESUME = False
YOLOX_OCCUPY_GPU = False
YOLOX_CACHE_IMGS = None  # None, "ram" or "disk"

# ---------------- NanoDet-Plus-m (vendored external/nanodet) ---------------
NANODET_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "nanodet")
NANODET_OUTPUT_DIR = os.path.join(RUNS_ROOT, "nanodet")
NANODET_INPUT_SIZE = (320, 320)  # (width, height)
NANODET_MODEL_SIZE = "1.0x"  # ShuffleNetV2 width multiplier: 0.5x / 1.0x / 1.5x / 2.0x
NANODET_EPOCHS = 100
NANODET_BATCH_SIZE = 16
NANODET_LEARNING_RATE = 0.001
NANODET_WEIGHT_DECAY = 0.05
NANODET_WARMUP_STEPS = 300
NANODET_WARMUP_RATIO = 0.0001
NANODET_DETACH_EPOCH = 10
# Path to an official NanoDet-Plus .ckpt to fine-tune from. "" = train from scratch.
NANODET_PRETRAINED_CKPT = ""
NANODET_CHECKPOINT_INTERVAL = 5  # save every N epochs
NANODET_LOG_INTERVAL = 10

# ---------------- RTMDet-tiny (pure-PyTorch re-implementation, see ----------
# ---------------- src/models/rtmdet/) ---------------------------------------
# RTMDet-tiny is the smallest/fastest model in the official RTMDet family
# (4.8M params vs 8.9M/24.7M/52.3M/94.9M for s/m/l/x), so it's the variant
# used here. `mmcv`/`mmdet` (the official framework RTMDet ships in) fail to
# even build against this venv's very new torch build (mmcv ships prebuilt
# CUDA extensions pinned to specific torch versions), so the CSPNeXt
# backbone + CSPNeXtPAFPN neck + RTMDetSepBNHead are re-implemented in plain
# PyTorch in src/models/rtmdet/. The official COCO-pretrained checkpoint
# loads into this re-implementation with an exact, 1:1 key/shape match
# (verified: all 476/476 non-buffer tensors load cleanly).
RTMDET_VARIANT = "tiny"  # "tiny" (4.8M, fastest) / "s" / "l" also implemented
RTMDET_OUTPUT_DIR = os.path.join(RUNS_ROOT, "rtmdet_tiny")
RTMDET_INPUT_SIZE = (640, 640)  # (height, width), official RTMDet training/eval resolution
RTMDET_EPOCHS = 100
RTMDET_BATCH_SIZE = 16
RTMDET_LEARNING_RATE = 0.001
RTMDET_WEIGHT_DECAY = 0.05
RTMDET_WARMUP_STEPS = 300
RTMDET_WARMUP_RATIO = 0.0001
# Dynamic Soft Label Assigner (the same assigner RTMDet borrows from
# NanoDet-Plus, vendored at external/nanodet) top-k candidates per gt box.
RTMDET_ASSIGNER_TOPK = 13
RTMDET_QFL_BETA = 2.0
RTMDET_QFL_LOSS_WEIGHT = 1.0
RTMDET_GIOU_LOSS_WEIGHT = 2.0
# Official COCO-pretrained RTMDet-tiny weights, used as the fine-tuning start point.
RTMDET_PRETRAINED_CKPT = os.path.join(PROJECT_ROOT, "pretrained", "rtmdet_tiny_8xb32-300e_coco.pth")
RTMDET_PRETRAINED_CKPT_URL = ("https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
                               "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")
RTMDET_AUTO_DOWNLOAD_PRETRAINED = True
RTMDET_CHECKPOINT_INTERVAL = 5  # save every N epochs
RTMDET_LOG_INTERVAL = 10

# ---------------------------------------------------------------------------
# Stage 5 - Live inference (video -> pop-up window with FPS overlay)
# ---------------------------------------------------------------------------
# Shared across all 4 inference/infer_*.py scripts.
INFERENCE_CONFIDENCE_THRESHOLD = 0.3
INFERENCE_NMS_IOU_THRESHOLD = 0.1
INFERENCE_DEVICE = "cuda:0"  # "cuda:0", "cpu", ...
# The pop-up window shows the frame downscaled to (frame_w // this, frame_h // this)
# so a normal camera/video resolution fits comfortably on a laptop screen.
INFERENCE_WINDOW_DIVISOR = 2
INFERENCE_FPS_SMOOTHING = 0.9  # exponential moving average factor for the FPS readout, 0-1

# If True, each inference/infer_*.py script writes its annotated (full
# resolution, boxes + FPS overlay drawn) output video to disk - in addition
# to showing the live pop-up window - named after the model, e.g.
# "mobilenet_ssd_output.mp4". If False, nothing is written to disk; only the
# live pop-up window is shown.
SAVE_INFERENCE = True
INFERENCE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "inference_output")

# Each model gets its own input video + checkpoint pair so you can point
# them at different clips/checkpoints and compare side by side.
MOBILENET_SSD_INFER_VIDEO_PATH = INPUT_PATH
MOBILENET_SSD_INFER_CHECKPOINT = os.path.join(MOBILENET_SSD_OUTPUT_DIR, "mobilenet_ssd_final.pt")

YOLOX_INFER_VIDEO_PATH = INPUT_PATH
YOLOX_INFER_CHECKPOINT = os.path.join(YOLOX_OUTPUT_DIR, YOLOX_EXPERIMENT_NAME, "best_ckpt.pth")

NANODET_INFER_VIDEO_PATH = INPUT_PATH
NANODET_INFER_CHECKPOINT = os.path.join(NANODET_OUTPUT_DIR, "nanodet_final.pth")

RTMDET_INFER_VIDEO_PATH = INPUT_PATH
RTMDET_INFER_CHECKPOINT = os.path.join(RTMDET_OUTPUT_DIR, "rtmdet_tiny_epoch50.pth")
