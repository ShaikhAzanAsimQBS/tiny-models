# dataset_creator

Turns raw video/images into training data for four lightweight detectors,
using a fine-tuned YOLO11n model as an auto-labeling "teacher":

```
video or folder of images/videos
        |
        v
  [pipeline.py]  --(YOLO11n: general_product_detection.pt)-->  per-image "product" boxes
        |
        +--> datasets/mobilenet_ssd/  (Pascal VOC layout)
        +--> datasets/nanodet/        (Pascal VOC layout)
        +--> datasets/yolox/          (COCO layout)
        +--> datasets/rtmdet/         (COCO layout)
        |
        v
  training/train_mobilenet_ssd.py  -> runs/mobilenet_ssd/
  training/train_yolox_tiny.py     -> runs/yolox_tiny/
  training/train_nanodet.py        -> runs/nanodet/
  training/train_rtmdet_tiny.py    -> runs/rtmdet_tiny/
        |
        v
  inference/infer_mobilenet_ssd.py  -- live video window + FPS overlay
  inference/infer_yolox_tiny.py
  inference/infer_nanodet.py
  inference/infer_rtmdet_tiny.py
```

**No argparse / CLI flags anywhere.** Every configurable value (paths,
thresholds, hyperparameters, epochs, batch sizes, ...) is a plain Python
variable in [`config.py`](config.py). Edit that file, then just run a script.

## 0. Environment

Everything runs with the existing CUDA-enabled venv:

```
C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe
```

That venv already had `torch`/`torchvision` (CUDA build), `ultralytics`, and
`opencv-python`. A few extra packages needed by the vendored YOLOX/NanoDet
training code were installed into it (see [`requirements.txt`](requirements.txt)):

```powershell
"C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## 1. Put the teacher model in place

Copy your fine-tuned single-class ("product") YOLO11n weights to:

```
models/general_product_detection.pt
```

(`config.YOLO_MODEL_PATH` points here by default.)

## 2. Configure `config.py`

Key variables (there are more, all grouped and commented in the file):

| Variable | Meaning |
|---|---|
| `INPUT_PATH` | A single video file, **or** a folder containing any mix of images/videos (searched recursively) |
| `FRAME_SKIP` | Keep 1 out of every N decoded frames from videos (1 = every frame) |
| `YOLO_MODEL_PATH` | Path to `general_product_detection.pt` |
| `YOLO_CONFIDENCE_THRESHOLD`, `YOLO_IOU_THRESHOLD`, `YOLO_IMG_SIZE`, `YOLO_DEVICE` | Auto-labeling inference settings |
| `TRAIN_VAL_SPLIT_RATIO` | Fraction of auto-labeled images used for training vs. validation |
| `MOBILENET_SSD_DATASET_DIR` / `NANODET_DATASET_DIR` / `YOLOX_DATASET_DIR` / `RTMDET_DATASET_DIR` | Where each of the 4 datasets is written |
| `MOBILENET_SSD_*`, `YOLOX_*`, `NANODET_*`, `RTMDET_*` | Per-model training hyperparameters (epochs, batch size, lr, input size, ...) |
| `INFERENCE_*` and `<MODEL>_INFER_VIDEO_PATH` / `<MODEL>_INFER_CHECKPOINT` | Live-inference video/checkpoint per model, confidence/NMS thresholds, pop-up window scale |

## 3. Build the 4 datasets

```powershell
"C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe" pipeline.py
```

This will:
1. Resolve `INPUT_PATH` (video or folder) into a flat list of images, decoding
   every `FRAME_SKIP`-th frame from any videos into `extracted_frames/`.
2. Run `general_product_detection.pt` over every image (batched, on
   `YOLO_DEVICE`) to get "product" bounding boxes.
3. Export the results into 4 ready-to-train dataset folders under `datasets/`.

Images with zero detections are dropped by default
(`DROP_IMAGES_WITHOUT_DETECTIONS`).

### Dataset layouts produced

**MobileNet-SSD** and **NanoDet** (`datasets/mobilenet_ssd/`, `datasets/nanodet/`) — Pascal VOC:

```
<dataset>/train/JPEGImages/*.jpg
<dataset>/train/Annotations/*.xml
<dataset>/val/JPEGImages/*.jpg
<dataset>/val/Annotations/*.xml
```

**YOLOX** and **RTMDet** (`datasets/yolox/`, `datasets/rtmdet/`) — COCO
(this is the exact layout mmdetection's own `CocoDataset` expects, so it's
reused verbatim for RTMDet too):

```
<dataset>/train2017/*.jpg
<dataset>/val2017/*.jpg
<dataset>/annotations/instances_train2017.json
<dataset>/annotations/instances_val2017.json
```

## 4. Fine-tune the 4 models

Each script is fully self-contained and reads its hyperparameters from
`config.py`.

```powershell
$py = "C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe"

& $py training\train_mobilenet_ssd.py
& $py training\train_yolox_tiny.py
& $py training\train_nanodet.py
& $py training\train_rtmdet_tiny.py
```

### MobileNet-SSD — `training/train_mobilenet_ssd.py`
Uses `torchvision.models.detection.ssdlite320_mobilenet_v3_large` (the
actively-maintained, modern implementation of the MobileNet-SSD family),
starting from COCO-pretrained weights with the classification head swapped
out for our 1 class + background. Pure `torch`/`torchvision`, no vendored
repo needed. Checkpoints saved to `runs/mobilenet_ssd/`.

### YOLOX-Tiny — `training/train_yolox_tiny.py`
Drives the official [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
code, vendored (and lightly patched — see below) under `external/YOLOX/`.
The script builds the `yolox_tiny` `Exp` object and its `Trainer` directly in
Python (bypassing YOLOX's argparse-based `tools/train.py`). Automatically
downloads the official COCO-pretrained `yolox_tiny.pth` on first run
(`YOLOX_AUTO_DOWNLOAD_PRETRAINED`) as the fine-tuning start point. **Requires
a CUDA GPU** (YOLOX's `Trainer` is GPU-only). Checkpoints saved to
`runs/yolox_tiny/yolox_tiny_product/`.

### NanoDet-Plus — `training/train_nanodet.py`
Drives the official [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)
model/dataset code, vendored under `external/nanodet/`. NanoDet's own
training entry point is built on PyTorch-Lightning `<2.0`, which conflicts
with the very new `torch` build in this venv — so instead this script
builds the real NanoDet-Plus-m model (`ShuffleNetV2` backbone + `GhostPAN`
neck + `NanoDetPlusHead`, ImageNet-pretrained backbone auto-downloaded) and
its native `XMLDataset` loader directly from the vendored package, then runs
a small hand-written training loop that mirrors NanoDet's own loss call and
linear-warmup + cosine LR schedule. Checkpoints saved to `runs/nanodet/`.

Optionally set `config.NANODET_PRETRAINED_CKPT` to an official NanoDet-Plus
checkpoint (e.g. from the
[NanoDet model zoo](https://github.com/RangiLyu/nanodet#model-zoo)) to
fine-tune the whole network instead of just the backbone.

### RTMDet-tiny — `training/train_rtmdet_tiny.py`

RTMDet officially ships as part of
[open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection), on top
of `mmcv`/`mmengine`. `mmcv` ships prebuilt CUDA extensions for a handful of
pinned `torch` versions only, and **fails to even build from source** against
the very new `torch` build in this venv - so instead of vendoring
mmdetection, RTMDet's architecture (`CSPNeXt` backbone + `CSPNeXtPAFPN` neck
+ `RTMDetSepBNHead`) is re-implemented in plain PyTorch at
`src/models/rtmdet/`. Its submodule/attribute names deliberately match
mmdetection's own naming, so the official COCO-pretrained checkpoint loads
in with an exact key/shape match (verified: all 476/476 real weight tensors
load with `strict=False` only skipping the two non-parameter
`data_preprocessor` buffers) - and produced correct, sane detections on a
real test image before this was trusted for training.

**RTMDet-tiny was chosen because it's the smallest/fastest model in the
official RTMDet family** (params, by variant):

| tiny | s | m | l | x |
|---|---|---|---|---|
| **4.8M** | 8.9M | 24.7M | 52.3M | 94.9M |

Training reuses the *exact same* Dynamic Soft Label Assigner + Quality Focal
Loss + GIoU Loss that RTMDet's official recipe uses - not reimplemented from
scratch, but imported straight from the vendored NanoDet repo
(`external/nanodet/nanodet/model/head/assigner/dsl_assigner.py`,
`.../loss/gfocal_loss.py`, `.../loss/iou_loss.py`), since RTMDet's paper
borrowed this exact assigner from NanoDet-Plus. Automatically downloads the
official COCO-pretrained `rtmdet_tiny_8xb32-300e_coco.pth`
(`RTMDET_AUTO_DOWNLOAD_PRETRAINED`) as the fine-tuning start point.
Checkpoints saved to `runs/rtmdet_tiny/`.

Known simplification vs. the official recipe: training here uses simple
letterbox-resize + random-horizontal-flip augmentation rather than the
official heavy Mosaic/MixUp/random-resize-crop pipeline - functionally
correct and enough for fine-tuning a single class, just less aggressively
augmented than the from-scratch 300-epoch COCO recipe.

## 5. Live inference (pop-up window + FPS)

Each of the 4 models has a matching `inference/infer_*.py` script: point it
at a checkpoint + video in `config.py`, run it, and a pop-up window shows
live detections with an FPS / inference-time overlay. The window is shown
at `frame_w // INFERENCE_WINDOW_DIVISOR` x `frame_h // INFERENCE_WINDOW_DIVISOR`
(default: half size) so a normal video resolution fits a laptop screen.
Press `q` or `Esc` to stop; final average FPS is printed to the console.

```powershell
& $py inference\infer_mobilenet_ssd.py
& $py inference\infer_yolox_tiny.py
& $py inference\infer_nanodet.py
& $py inference\infer_rtmdet_tiny.py
```

Set `MOBILENET_SSD_INFER_VIDEO_PATH`/`_INFER_CHECKPOINT` (and the equivalent
`YOLOX_*`, `NANODET_*`, `RTMDET_*` pairs) in `config.py` to point each script
at whichever video/checkpoint you want - they default to `INPUT_PATH` and
each model's own `runs/.../*_final.*` output path. `INFERENCE_CONFIDENCE_THRESHOLD`,
`INFERENCE_NMS_IOU_THRESHOLD` and `INFERENCE_DEVICE` are shared across all 4.

## Project layout

```
config.py                        # every configurable value lives here
pipeline.py                      # stage 1-3 entry point (labeling + dataset export)
requirements.txt
models/
    general_product_detection.pt # <- put your fine-tuned YOLO11n here
pretrained/                      # generated: auto-downloaded COCO-pretrained YOLOX/RTMDet weights
src/
    common/
        annotation_types.py      # BoxAnnotation / ImageAnnotation
        media_utils.py           # resolves INPUT_PATH into images/videos
        detection_transforms.py  # ToTensor / RandomHorizontalFlip for torchvision detection
    inference/
        frame_extractor.py       # video -> every-Nth-frame .jpg
        auto_labeler.py          # runs YOLO11n, returns ImageAnnotation list
        live_runner.py           # shared live-video display loop (FPS overlay, window resize, box drawing)
    dataset_export/
        split.py                 # deterministic train/val split
        voc_export.py            # -> Pascal VOC dataset (MobileNet-SSD, NanoDet)
        coco_export.py           # -> COCO dataset (YOLOX, RTMDet)
        voc_dataset_torch.py     # torch Dataset reading the VOC layout (used by MobileNet-SSD training)
        coco_dataset_torch.py    # torch Dataset reading the COCO layout (used by RTMDet training)
    models/
        rtmdet/                  # plain-PyTorch RTMDet re-implementation (see training section above)
            blocks.py            # ConvModule / CSPLayer / CSPNeXtBlock / etc. (mmcv-free)
            cspnext.py            # CSPNeXt backbone
            cspnext_pafpn.py      # CSPNeXtPAFPN neck
            head.py                # RTMDetSepBNHead (forward + decode + NMS)
            points.py              # point-prior generation, distance<->bbox coding
            loss.py                # QFL + GIoU + Dynamic Soft Label Assigner glue (training only)
            model.py               # full RTMDet model, presets, official-checkpoint loader
training/
    train_mobilenet_ssd.py
    train_yolox_tiny.py
    train_nanodet.py
    train_rtmdet_tiny.py
inference/
    infer_mobilenet_ssd.py
    infer_yolox_tiny.py
    infer_nanodet.py
    infer_rtmdet_tiny.py
external/
    YOLOX/                       # vendored official repo (patched-free; used via its Python API)
    nanodet/                     # vendored official repo (1-line patch: torch._six removed in torch 2.x)
extracted_frames/                # generated: video frames pulled by pipeline.py
datasets/                        # generated: the 4 training datasets
runs/                             # generated: training checkpoints
```

## Notes on the vendored repos

- `external/YOLOX` and `external/nanodet` are vendored (not pip installed)
  because neither ships a `pip install`-able package that supports this
  venv's very new `torch` build well, and because we need their internal
  `Exp`/model/dataset Python APIs rather than their CLI scripts.
- `external/nanodet/nanodet/data/collate.py` has one patched line:
  `from torch._six import string_classes` (removed in torch 2.0+) was
  replaced with `string_classes = str`.
- Both repos were smoke-tested end-to-end (a few real training steps on a
  tiny synthetic dataset, on GPU) while building this project.
- `external/nanodet`'s loss/assigner modules
  (`DynamicSoftLabelAssigner`, `QualityFocalLoss`, `GIoULoss`) are also
  reused directly by `training/train_rtmdet_tiny.py` - RTMDet's official
  training recipe uses algorithmically identical components (RTMDet's paper
  borrowed the assigner from NanoDet-Plus), so there was no need to
  re-derive them.
- **RTMDet is not vendored** - `mmdetection` (RTMDet's home) depends on
  `mmcv`'s compiled CUDA extensions, which only ship prebuilt wheels for a
  handful of pinned `torch` versions and fail to even build from source
  against this venv's very new `torch`. `src/models/rtmdet/` is a from-scratch,
  plain-PyTorch port of RTMDet-tiny's architecture instead (see the RTMDet
  training section above for how it was validated against the official
  checkpoint).
