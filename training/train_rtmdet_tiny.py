"""
Fine-tunes RTMDet-tiny (the smallest/fastest model in the RTMDet family -
4.8M params, see src/models/rtmdet/) on the COCO-format dataset produced by
pipeline.py at config.RTMDET_DATASET_DIR.

RTMDet officially ships as part of mmdetection, which depends on `mmcv`'s
compiled CUDA extensions. `mmcv` only publishes prebuilt wheels for a
handful of pinned torch versions and fails to even build from source against
the very new torch build in this venv, so the model (CSPNeXt backbone +
CSPNeXtPAFPN neck + RTMDetSepBNHead) is a plain-PyTorch re-implementation
(src/models/rtmdet/) that loads the official COCO-pretrained checkpoint with
an exact key/shape match. Training here uses the same recipe RTMDet's paper
describes - Dynamic Soft Label Assignment + Quality Focal Loss + GIoU Loss -
reusing the vendored NanoDet implementations of those (NanoDet-Plus is where
this assigner originally came from; RTMDet borrowed it).

No command line arguments - every knob lives in config.py. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" training\\train_rtmdet_tiny.py
"""

import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

# The vendored NanoDet repo is only used here for its loss/assigner modules
# (QualityFocalLoss, GIoULoss, DynamicSoftLabelAssigner), which RTMDet's
# training recipe borrows verbatim.
sys.path.insert(0, config.NANODET_REPO_DIR)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nanodet.model.head.assigner.dsl_assigner import DynamicSoftLabelAssigner  # noqa: E402
from nanodet.model.loss.gfocal_loss import QualityFocalLoss  # noqa: E402
from nanodet.model.loss.iou_loss import GIoULoss  # noqa: E402

from src.dataset_export.coco_dataset_torch import CocoDetectionDataset, collate_fn  # noqa: E402
from src.models.rtmdet.model import PIXEL_MEAN, PIXEL_STD, build_rtmdet, load_pretrained  # noqa: E402
from src.models.rtmdet.loss import compute_rtmdet_loss  # noqa: E402


def ensure_pretrained_checkpoint() -> str:
    ckpt_path = config.RTMDET_PRETRAINED_CKPT
    if os.path.exists(ckpt_path):
        return ckpt_path
    if not config.RTMDET_AUTO_DOWNLOAD_PRETRAINED:
        return None
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    print(f"[train_rtmdet_tiny] downloading pretrained checkpoint from {config.RTMDET_PRETRAINED_CKPT_URL}")
    try:
        urllib.request.urlretrieve(config.RTMDET_PRETRAINED_CKPT_URL, ckpt_path)
        return ckpt_path
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[train_rtmdet_tiny] WARNING: could not download pretrained checkpoint ({exc}). "
              f"Training will start from random weights.")
        return None


def warmup_lr_factor(global_step: int, warmup_steps: int, warmup_ratio: float) -> float:
    return 1 - (1 - global_step / warmup_steps) * (1 - warmup_ratio)


def main():
    torch.manual_seed(config.TRAIN_RANDOM_SEED)
    device = torch.device(config.TRAIN_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_rtmdet_tiny] device = {device}")

    class_names = config.DATASET_CLASS_NAMES
    num_classes = len(class_names)

    train_dataset = CocoDetectionDataset(
        config.RTMDET_DATASET_DIR, "train", input_size=config.RTMDET_INPUT_SIZE, train=True,
        pixel_mean=PIXEL_MEAN, pixel_std=PIXEL_STD,
    )
    val_dataset = CocoDetectionDataset(
        config.RTMDET_DATASET_DIR, "val", input_size=config.RTMDET_INPUT_SIZE, train=False,
        pixel_mean=PIXEL_MEAN, pixel_std=PIXEL_STD,
    )
    print(f"[train_rtmdet_tiny] train images: {len(train_dataset)}, val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.RTMDET_BATCH_SIZE, shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.RTMDET_BATCH_SIZE, shuffle=False,
        num_workers=config.TRAIN_NUM_WORKERS, collate_fn=collate_fn, drop_last=False,
    )

    model = build_rtmdet(config.RTMDET_VARIANT, num_classes=num_classes).to(device)

    pretrained_ckpt = ensure_pretrained_checkpoint()
    if pretrained_ckpt:
        load_pretrained(model, pretrained_ckpt)

    assigner = DynamicSoftLabelAssigner(topk=config.RTMDET_ASSIGNER_TOPK)
    qfl_loss_fn = QualityFocalLoss(beta=config.RTMDET_QFL_BETA, loss_weight=config.RTMDET_QFL_LOSS_WEIGHT)
    giou_loss_fn = GIoULoss(loss_weight=config.RTMDET_GIOU_LOSS_WEIGHT)

    # AdamW + no weight decay on norm/bias, matching RTMDet's official recipe.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.RTMDET_WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.RTMDET_LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.RTMDET_EPOCHS)
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    os.makedirs(config.RTMDET_OUTPUT_DIR, exist_ok=True)
    global_step = 0

    def save_checkpoint(path: str, epoch: int):
        torch.save({
            "model": model.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "variant": config.RTMDET_VARIANT,
            "input_size": config.RTMDET_INPUT_SIZE,
            "epoch": epoch,
        }, path)

    for epoch in range(1, config.RTMDET_EPOCHS + 1):
        model.train()
        running_cls, running_bbox, running_pos = 0.0, 0.0, 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            global_step += 1
            images = images.to(device)

            cls_scores, bbox_preds = model(images)
            loss_cls, loss_bbox, num_pos = compute_rtmdet_loss(
                cls_scores, bbox_preds, targets, num_classes, model.strides,
                assigner, qfl_loss_fn, giou_loss_fn,
            )
            loss = loss_cls + loss_bbox

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)

            if global_step <= config.RTMDET_WARMUP_STEPS:
                k = warmup_lr_factor(global_step, config.RTMDET_WARMUP_STEPS, config.RTMDET_WARMUP_RATIO)
                for pg, base_lr in zip(optimizer.param_groups, initial_lrs):
                    pg["lr"] = base_lr * k

            optimizer.step()

            running_cls += loss_cls.item()
            running_bbox += loss_bbox.item()
            running_pos += num_pos

            if step % config.RTMDET_LOG_INTERVAL == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[train_rtmdet_tiny] epoch {epoch}/{config.RTMDET_EPOCHS} step {step}/{len(train_loader)} "
                      f"lr={lr_now:.2e} loss_cls={loss_cls.item():.4f} loss_bbox={loss_bbox.item():.4f} "
                      f"pos={num_pos}")

        if global_step > config.RTMDET_WARMUP_STEPS:
            lr_scheduler.step()

        n_batches = max(1, len(train_loader))
        print(f"[train_rtmdet_tiny] epoch {epoch} done - avg_loss_cls={running_cls / n_batches:.4f} "
              f"avg_loss_bbox={running_bbox / n_batches:.4f} avg_pos_per_batch={running_pos / n_batches:.1f}")

        if len(val_dataset) > 0 and (epoch % config.RTMDET_CHECKPOINT_INTERVAL == 0 or epoch == config.RTMDET_EPOCHS):
            model.eval()
            val_cls, val_bbox = 0.0, 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    cls_scores, bbox_preds = model(images)
                    loss_cls, loss_bbox, _ = compute_rtmdet_loss(
                        cls_scores, bbox_preds, targets, num_classes, model.strides,
                        assigner, qfl_loss_fn, giou_loss_fn,
                    )
                    val_cls += loss_cls.item()
                    val_bbox += loss_bbox.item()
            n_val = max(1, len(val_loader))
            print(f"[train_rtmdet_tiny] epoch {epoch} val - loss_cls={val_cls / n_val:.4f} "
                  f"loss_bbox={val_bbox / n_val:.4f}")

        if epoch % config.RTMDET_CHECKPOINT_INTERVAL == 0 or epoch == config.RTMDET_EPOCHS:
            ckpt_path = os.path.join(config.RTMDET_OUTPUT_DIR, f"rtmdet_tiny_epoch{epoch}.pth")
            save_checkpoint(ckpt_path, epoch)
            print(f"[train_rtmdet_tiny] saved checkpoint -> {ckpt_path}")

    final_path = os.path.join(config.RTMDET_OUTPUT_DIR, "rtmdet_tiny_final.pth")
    save_checkpoint(final_path, config.RTMDET_EPOCHS)
    print(f"[train_rtmdet_tiny] training complete. Final weights -> {final_path}")


if __name__ == "__main__":
    main()
