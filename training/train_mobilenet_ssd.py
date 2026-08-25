"""
Fine-tunes MobileNet-SSD (torchvision's SSDLite320-MobileNetV3-Large, the
modern maintained implementation of the MobileNet-SSD family) on the VOC
dataset produced by pipeline.py at config.MOBILENET_SSD_DATASET_DIR.

No command line arguments - every knob lives in config.py. Run with the
CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" training\\train_mobilenet_ssd.py
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

import config
from src.common.detection_transforms import get_transform
from src.dataset_export.voc_dataset_torch import VocDetectionDataset, collate_fn


def build_model(num_classes_with_background: int, pretrained_backbone: bool):
    weights = "DEFAULT" if pretrained_backbone else None
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=weights,
        weights_backbone="DEFAULT" if pretrained_backbone else None,
    )

    in_channels = det_utils.retrieve_out_channels(model.backbone, (config.MOBILENET_SSD_INPUT_SIZE,) * 2)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes_with_background, norm_layer
    )
    return model


def evaluate_loss(model, data_loader, device):
    """Runs the model in "train mode" forward pass (which returns losses)
    over the val set without updating weights, just to report a val loss."""
    was_training = model.training
    model.train()
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss.item() for loss in loss_dict.values())
            num_batches += 1
    model.train(was_training)
    return total_loss / max(1, num_batches)


def main():
    torch.manual_seed(config.TRAIN_RANDOM_SEED)
    device = torch.device(config.TRAIN_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_mobilenet_ssd] device = {device}")

    class_names = config.DATASET_CLASS_NAMES
    num_classes_with_background = len(class_names) + 1

    train_dataset = VocDetectionDataset(
        config.MOBILENET_SSD_DATASET_DIR, "train", class_names, transforms=get_transform(train=True)
    )
    val_dataset = VocDetectionDataset(
        config.MOBILENET_SSD_DATASET_DIR, "val", class_names, transforms=get_transform(train=False)
    )
    print(f"[train_mobilenet_ssd] train images: {len(train_dataset)}, val images: {len(val_dataset)}")

    # drop_last=True avoids a stray batch of size 1 at the end of an epoch,
    # which would crash BatchNorm layers while the model is in train() mode
    # (train mode is required to get the SSD loss dict, including for eval).
    train_loader = DataLoader(
        train_dataset, batch_size=config.MOBILENET_SSD_BATCH_SIZE, shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.MOBILENET_SSD_BATCH_SIZE, shuffle=False,
        num_workers=config.TRAIN_NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )

    model = build_model(num_classes_with_background, config.MOBILENET_SSD_PRETRAINED_BACKBONE)

    if config.MOBILENET_SSD_RESUME_CHECKPOINT:
        # weights_only=False: safe here since this loads our own checkpoint;
        # PyTorch 2.6+ defaults weights_only to True, which can reject
        # globals (e.g. numpy scalars) present in the saved checkpoint dict.
        state = torch.load(config.MOBILENET_SSD_RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[train_mobilenet_ssd] resumed weights from {config.MOBILENET_SSD_RESUME_CHECKPOINT}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.MOBILENET_SSD_LEARNING_RATE,
        momentum=config.MOBILENET_SSD_MOMENTUM, weight_decay=config.MOBILENET_SSD_WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MOBILENET_SSD_EPOCHS)

    os.makedirs(config.MOBILENET_SSD_OUTPUT_DIR, exist_ok=True)

    for epoch in range(1, config.MOBILENET_SSD_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if step % 10 == 0:
                print(f"[train_mobilenet_ssd] epoch {epoch}/{config.MOBILENET_SSD_EPOCHS} "
                      f"step {step}/{len(train_loader)} loss {losses.item():.4f}")

        lr_scheduler.step()
        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate_loss(model, val_loader, device) if len(val_dataset) > 0 else float("nan")
        print(f"[train_mobilenet_ssd] epoch {epoch} done - avg_train_loss={avg_train_loss:.4f} "
              f"val_loss={val_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.6f}")

        if epoch % config.MOBILENET_SSD_CHECKPOINT_INTERVAL == 0 or epoch == config.MOBILENET_SSD_EPOCHS:
            ckpt_path = os.path.join(config.MOBILENET_SSD_OUTPUT_DIR, f"mobilenet_ssd_epoch{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "class_names": class_names,
            }, ckpt_path)
            print(f"[train_mobilenet_ssd] saved checkpoint -> {ckpt_path}")

    final_path = os.path.join(config.MOBILENET_SSD_OUTPUT_DIR, "mobilenet_ssd_final.pt")
    torch.save({"model": model.state_dict(), "class_names": class_names}, final_path)
    print(f"[train_mobilenet_ssd] training complete. Final weights -> {final_path}")


if __name__ == "__main__":
    main()
