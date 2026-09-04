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
# Point this at any of:
#   - a single video file, e.g. r"C:\data\shelf_walkthrough.mp4"
#   - a remote video URL (http:// / https:// / rtsp:// / rtmp://), e.g. a
#     direct S3/HTTP link to an .mp4 - streamed and decoded on the fly with
#     cv2.VideoCapture/ffmpeg, it is NEVER downloaded to disk first
#   - a folder containing any mix of images and videos (searched recursively),
#     e.g. r"C:\data\raw_media"
INPUT_PATH = r"C:\Users\QBS PC\PycharmProjects\dataset_creator\extracted_frames"

# When a video is processed (INPUT_PATH itself, or a video found while
# walking a folder), only every Nth decoded frame is kept/labeled.
# 1 = keep every frame, 5 = keep 1 out of every 5 frames, etc.
FRAME_SKIP = 15

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Frames pulled out of videos are written here as .jpg files before labeling
# (images found directly on disk are read/labeled in place and are not copied
# here).
EXTRACTED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
FRAME_JPEG_QUALITY = 95  # 0-100

# ---------------------------------------------------------------------------
# Stage 2 - Auto-labeling: two-stage "person" -> "face" detector
# ---------------------------------------------------------------------------
# Stage 2a: yolo11x.pt is the stock, COCO-pretrained Ultralytics YOLO11x
# checkpoint (NOT fine-tuned for this project) - it's only used for its
# "person" class (id 0 in the standard 80-class COCO taxonomy) to find every
# person in the frame.
PERSON_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo11x.pt")
PERSON_CLASS_NAME = "person"
PERSON_COCO_CLASS_ID = 0  # "person" in the standard 80-class COCO taxonomy
PERSON_CONFIDENCE_THRESHOLD = 0.25
PERSON_IOU_THRESHOLD = 0.45
PERSON_IMG_SIZE = 640
PERSON_DEVICE = "cuda:0"  # "cuda:0", "cuda:1", "cpu", ...
PERSON_INFERENCE_BATCH_SIZE = 16

# Stage 2b: face11n.pt is a fine-tuned YOLO11n checkpoint with a single
# "face" class. It does NOT run on the full frame - it runs on a padded crop
# of every person box found in stage 2a instead, which gives it a much
# higher effective resolution to work with per-face (especially important
# for people who are small/far-away in the full frame). Face boxes are then
# translated from crop-local pixel coordinates back into full-frame pixel
# coordinates (crop offset + Ultralytics' own internal letterbox-rescale,
# which already returns boxes in the crop's own original pixel space).
FACE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face11n.pt")
FACE_CLASS_NAME = "face"
FACE_CONFIDENCE_THRESHOLD = 0.25
FACE_IOU_THRESHOLD = 0.45
FACE_IMG_SIZE = 640
FACE_DEVICE = "cuda:0"
FACE_INFERENCE_BATCH_SIZE = 16
# Expands every person box by this fraction of its own width/height on each
# side (clamped to the image bounds) before cropping for face detection, so
# a face near the edge of a slightly-too-tight person box doesn't get cut off.
FACE_CROP_PADDING_RATIO = 0.15
# Person crops with either side smaller than this many pixels are skipped
# for face detection entirely (too small to contain a resolvable face, and
# not worth the extra inference call).
FACE_MIN_CROP_SIDE = 20

# If True, images/frames with zero detections (no person AND no face boxes)
# are discarded and never make it into any of the output datasets.
DROP_IMAGES_WITHOUT_DETECTIONS = True

# ---------------------------------------------------------------------------
# Stage 3 - Dataset build (auto-labeled detections -> training datasets)
# ---------------------------------------------------------------------------
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

TRAIN_VAL_SPLIT_RATIO = 0.75  # fraction of images assigned to the train split
SPLIT_RANDOM_SEED = 42
# "sequential" (recommended, default): splits WITHIN each source folder
# (each video's own subfolder under extracted_frames/, or a plain images
# folder) by path order, taking the first TRAIN_VAL_SPLIT_RATIO fraction as
# train and the rest as val. Consecutive video frames are near-duplicates
# of each other, so a plain "random" shuffle-then-slice scatters
# near-identical frames across BOTH splits - the model then gets validated
# on images almost pixel-identical to ones it trained on, which inflates
# val mAP and HIDES overfitting instead of catching it. "sequential" avoids
# that leak (only the few frames right at each group's split boundary are
# similar across train/val) while still letting every video/folder
# contribute to both splits. Use "random" only if every image is an
# independent photo (no video frames involved).
TRAIN_VAL_SPLIT_STRATEGY = "sequential"  # "sequential" or "random"

# class_id 0 = person, class_id 1 = face. Order also defines the COCO
# category ids written out for the YOLOX/RTMDet exports.
DATASET_CLASS_NAMES = ["person", "face"]

# Toggle which of the 4 dataset formats pipeline.py actually builds. Set any
# of these to False to skip that export entirely - e.g. if you only want to
# (re)train one or two of the 4 models right now, there's no need to also
# rebuild the datasets for the other ones.
CREATE_DATASET_MOBILENET_SSD = False
CREATE_DATASET_NANODET = False
CREATE_DATASET_YOLOX = True
CREATE_DATASET_RTMDET = False

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
YOLOX_EXPERIMENT_NAME = "yolox_tiny_person_face"
YOLOX_OUTPUT_DIR = os.path.join(RUNS_ROOT, "yolox_tiny")
# Megvii's own "train on custom data" guidance: if training overfits early,
# reduce max_epoch (and/or basic_lr_per_img / min_lr_ratio below) rather
# than fighting it with the schedule alone. 50 is already on the low end of
# their recommended 50-150 range for large custom datasets (ours is "large"
# in raw instance count, see the anti-overfitting note further below) -
# watch YOLOX_EVAL_INTERVAL's val AP in the training log / TensorBoard
# (`tensorboard --logdir runs/yolox_tiny`) and lower this if val AP peaks
# and then drops while train loss keeps falling (the textbook overfitting
# signature) well before epoch 50.
YOLOX_MAX_EPOCH = 50
YOLOX_BATCH_SIZE = 16
YOLOX_INPUT_SIZE = (416, 416)  # (height, width), must be multiples of 32
YOLOX_TEST_SIZE = (416, 416)
YOLOX_WARMUP_EPOCHS = 5
# Epochs use to close Mosaic/MixUp and switch on L1 loss (official YOLOX
# default and recommendation for custom datasets - see anti-overfitting
# block below). Do NOT set this much lower than ~10-15: the YOLOX authors
# found 5 measurably underperforms 10/15 on custom data, since it doesn't
# leave the detector enough iterations to converge on the true (non-mosaic)
# image distribution before training ends.
YOLOX_NO_AUG_EPOCHS = 15
YOLOX_BASIC_LR_PER_IMG = 0.01 / 64.0
# Evaluated every N epochs; best_ckpt.pth (highest val AP so far) is
# updated on every eval, and IS what YOLOX_INFER_CHECKPOINT points at below
# - i.e. this doubles as your "early stopping" checkpoint selection without
# needing to babysit/kill the run manually. A smaller interval than the
# official default (10) gives finer-grained monitoring/selection, cheap
# relative to how long training itself takes on a dataset this size.
YOLOX_EVAL_INTERVAL = 5
YOLOX_PRINT_INTERVAL = 10
YOLOX_DATA_NUM_WORKERS = 4
# Official COCO-pretrained YOLOX-Tiny weights, used as the fine-tuning start point.
YOLOX_PRETRAINED_CKPT = os.path.join(PROJECT_ROOT, "pretrained", "yolox_tiny.pth")
YOLOX_PRETRAINED_CKPT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
YOLOX_AUTO_DOWNLOAD_PRETRAINED = True
YOLOX_FP16 = False
# If True, continues an interrupted run instead of starting fresh from the
# pretrained COCO checkpoint: restores model + optimizer + epoch/best-AP
# state from this experiment's own "latest_ckpt.pth" (auto-saved after every
# epoch under YOLOX_OUTPUT_DIR/YOLOX_EXPERIMENT_NAME) and continues training
# up to YOLOX_MAX_EPOCH. Set back to False once you want to start a new run
# from the pretrained weights again.
YOLOX_RESUME = True
YOLOX_OCCUPY_GPU = False
YOLOX_CACHE_IMGS = None  # None, "ram" or "disk"

# ---- YOLOX-Tiny anti-overfitting / augmentation tuning ---------------------
# Sources: Megvii's official "train_custom_data.md", the YOLOX no_aug_epochs
# design discussion (github.com/Megvii-BaseDetection/YOLOX/issues/555), and
# MMYOLO's "training testing tricks" doc. Their consistent guidance:
#   - Small/tiny models -> WEAKEN geometric augmentation (degrees/
#     translate/shear/mosaic_scale) relative to the base (yolox-s/m/l/x)
#     defaults, and keep mixup OFF (official yolox_tiny.py already does
#     this - it's why our defaults below match it rather than yolox_base.py).
#   - Large datasets -> lighter augmentation and fewer epochs than the
#     300-epoch COCO-from-scratch recipe; small datasets -> stronger aug.
#   - If you still see val AP peak then decline while train loss keeps
#     dropping, reduce YOLOX_MAX_EPOCH / YOLOX_BASIC_LR_PER_IMG /
#     YOLOX_MIN_LR_RATIO next, per Megvii's own advice, rather than pushing
#     augmentation to extremes.
# Our dataset is large in raw instance count (see the person/face count
# discussion in chat) but many frames come from continuous video, i.e. lots
# of near-duplicate frames rather than truly independent scenes - that's an
# overfitting risk *to the specific background/scene*, not a "too little
# data" risk, and it's the main reason TRAIN_VAL_SPLIT_STRATEGY above is set
# to "sequential" rather than "random" (a random split would let
# near-duplicate frames leak between train/val and mask exactly this).
YOLOX_MOSAIC_PROB = 1.0           # kept on - auto-disabled for the final YOLOX_NO_AUG_EPOCHS anyway
YOLOX_MIXUP_PROB = 1.0            # only used if YOLOX_ENABLE_MIXUP is True
YOLOX_ENABLE_MIXUP = False        # matches official yolox_tiny.py - mixup is too aggressive for a tiny model
YOLOX_HSV_PROB = 1.0              # color-jitter aug; cheap and safe, doesn't distort box geometry
YOLOX_FLIP_PROB = 0.5
YOLOX_DEGREES = 5.0               # weaker than yolox_base.py's 10.0 default (tiny model -> weaker geometric aug)
YOLOX_TRANSLATE = 0.05            # weaker than yolox_base.py's 0.1 default
YOLOX_SHEAR = 1.0                 # weaker than yolox_base.py's 2.0 default
YOLOX_MOSAIC_SCALE = (0.5, 1.5)   # matches official yolox_tiny.py (yolox_base.py default is the wider (0.1, 2))
YOLOX_MIXUP_SCALE = (0.5, 1.5)
YOLOX_WEIGHT_DECAY = 5e-4         # official default - L2 regularization on conv/linear weights
YOLOX_MOMENTUM = 0.9
YOLOX_MIN_LR_RATIO = 0.05         # cosine LR floor, as a fraction of the peak LR
YOLOX_EMA = True                  # exponential moving average of weights - smooths noisy updates, reduces overfitting

# ---------------- NanoDet-Plus-m (vendored external/nanodet) ---------------
NANODET_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "nanodet")
NANODET_OUTPUT_DIR = os.path.join(RUNS_ROOT, "nanodet")
NANODET_INPUT_SIZE = (320, 320)  # (width, height)
NANODET_MODEL_SIZE = "1.0x"  # ShuffleNetV2 width multiplier: 0.5x / 1.0x / 1.5x / 2.0x
NANODET_EPOCHS = 50
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
RTMDET_EPOCHS = 50
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
