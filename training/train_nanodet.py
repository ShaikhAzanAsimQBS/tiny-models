"""
Fine-tunes NanoDet-Plus-m (using the vendored official repo at
external/nanodet) on the Pascal-VOC-format dataset produced by pipeline.py
at config.NANODET_DATASET_DIR.

NanoDet's own tools/train.py drives training through PyTorch-Lightning +
argparse. To keep this project argparse-free (and to sidestep NanoDet's
pytorch-lightning<2.0 pin, which conflicts with the very new torch build in
this venv), this script instead builds the real NanoDet-Plus model/dataset
objects straight from the vendored repo's Python API and drives a small
hand-written training loop that mirrors what nanodet.trainer.task.py does
(same loss call, same LR warmup formula, same checkpoint format).

Run with the CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" training\\train_nanodet.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

# The vendored NanoDet repo must be importable as the `nanodet` package.
sys.path.insert(0, config.NANODET_REPO_DIR)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nanodet.data.batch_process import stack_batch_img  # noqa: E402
from nanodet.data.collate import naive_collate  # noqa: E402
from nanodet.data.dataset import build_dataset  # noqa: E402
from nanodet.model.arch import build_model  # noqa: E402
from nanodet.optim import build_optimizer  # noqa: E402
from nanodet.util import cfg, load_config, mkdir  # noqa: E402
from nanodet.util.check_point import load_model_weight, save_model  # noqa: E402


CONFIG_TEMPLATE = """
save_dir: {save_dir}
model:
  weight_averager:
    name: ExpMovingAverager
    decay: 0.9998
  arch:
    name: NanoDetPlus
    detach_epoch: {detach_epoch}
    backbone:
      name: ShuffleNetV2
      model_size: {model_size}
      out_stages: [2,3,4]
      activation: LeakyReLU
      pretrain: True
    fpn:
      name: GhostPAN
      in_channels: {fpn_in_channels}
      out_channels: 96
      kernel_size: 5
      num_extra_level: 1
      use_depthwise: True
      activation: LeakyReLU
    head:
      name: NanoDetPlusHead
      num_classes: {num_classes}
      input_channel: 96
      feat_channels: 96
      stacked_convs: 2
      kernel_size: 5
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7
      norm_cfg:
        type: BN
      loss:
        loss_qfl:
          name: QualityFocalLoss
          use_sigmoid: True
          beta: 2.0
          loss_weight: 1.0
        loss_dfl:
          name: DistributionFocalLoss
          loss_weight: 0.25
        loss_bbox:
          name: GIoULoss
          loss_weight: 2.0
    aux_head:
      name: SimpleConvHead
      num_classes: {num_classes}
      input_channel: 192
      feat_channels: 192
      stacked_convs: 4
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7

class_names: {class_names}
data:
  train:
    name: XMLDataset
    class_names: {class_names}
    img_path: {train_img_path}
    ann_path: {train_ann_path}
    input_size: [{input_w},{input_h}]
    keep_ratio: True
    pipeline:
      perspective: 0.0
      scale: [0.6, 1.4]
      stretch: [[1, 1], [1, 1]]
      rotation: 0
      shear: 0
      translate: 0.2
      flip: 0.5
      brightness: 0.2
      contrast: [0.8, 1.2]
      saturation: [0.8, 1.2]
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
  val:
    name: XMLDataset
    class_names: {class_names}
    img_path: {val_img_path}
    ann_path: {val_ann_path}
    input_size: [{input_w},{input_h}]
    keep_ratio: True
    pipeline:
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
device:
  gpu_ids: [0]
  workers_per_gpu: {workers}
  batchsize_per_gpu: {batch_size}
schedule:
  optimizer:
    name: AdamW
    lr: {lr}
    weight_decay: {weight_decay}
  warmup:
    name: linear
    steps: {warmup_steps}
    ratio: {warmup_ratio}
  total_epochs: {total_epochs}
  lr_schedule:
    name: CosineAnnealingLR
    T_max: {total_epochs}
    eta_min: 0.00005
  val_intervals: {checkpoint_interval}
grad_clip: 35
evaluator:
  name: CocoDetectionEvaluator
  save_key: mAP
log:
  interval: {log_interval}
"""


def build_generated_config_path() -> str:
    os.makedirs(config.NANODET_OUTPUT_DIR, exist_ok=True)

    model_size_map = {"0.5x": [48, 96, 192], "1.0x": [116, 232, 464],
                       "1.5x": [176, 352, 704], "2.0x": [244, 488, 976]}
    fpn_in_channels = model_size_map[config.NANODET_MODEL_SIZE]

    yaml_text = CONFIG_TEMPLATE.format(
        save_dir=config.NANODET_OUTPUT_DIR.replace("\\", "/"),
        detach_epoch=config.NANODET_DETACH_EPOCH,
        model_size=config.NANODET_MODEL_SIZE,
        fpn_in_channels=fpn_in_channels,
        num_classes=len(config.DATASET_CLASS_NAMES),
        class_names=list(config.DATASET_CLASS_NAMES),
        train_img_path=os.path.join(config.NANODET_DATASET_DIR, "train", "JPEGImages").replace("\\", "/"),
        train_ann_path=os.path.join(config.NANODET_DATASET_DIR, "train", "Annotations").replace("\\", "/"),
        val_img_path=os.path.join(config.NANODET_DATASET_DIR, "val", "JPEGImages").replace("\\", "/"),
        val_ann_path=os.path.join(config.NANODET_DATASET_DIR, "val", "Annotations").replace("\\", "/"),
        input_w=config.NANODET_INPUT_SIZE[0],
        input_h=config.NANODET_INPUT_SIZE[1],
        workers=config.TRAIN_NUM_WORKERS,
        batch_size=config.NANODET_BATCH_SIZE,
        lr=config.NANODET_LEARNING_RATE,
        weight_decay=config.NANODET_WEIGHT_DECAY,
        warmup_steps=config.NANODET_WARMUP_STEPS,
        warmup_ratio=config.NANODET_WARMUP_RATIO,
        total_epochs=config.NANODET_EPOCHS,
        checkpoint_interval=config.NANODET_CHECKPOINT_INTERVAL,
        log_interval=config.NANODET_LOG_INTERVAL,
    )

    yaml_path = os.path.join(config.NANODET_OUTPUT_DIR, "nanodet_generated_config.yml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return yaml_path


def preprocess_batch(batch, device):
    batch_imgs = batch["img"]
    if isinstance(batch_imgs, list):
        batch_imgs = [img.to(device) for img in batch_imgs]
        batch["img"] = stack_batch_img(batch_imgs, divisible=32)
    else:
        batch["img"] = batch_imgs.to(device)
    return batch


def warmup_lr_factor(global_step: int, warmup_steps: int, warmup_ratio: float) -> float:
    # Linear warmup, identical formula to nanodet.trainer.task.TrainingTask.optimizer_step.
    return 1 - (1 - global_step / warmup_steps) * (1 - warmup_ratio)


def main():
    torch.manual_seed(config.TRAIN_RANDOM_SEED)
    device = torch.device(config.TRAIN_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_nanodet] device = {device}")

    yaml_path = build_generated_config_path()
    load_config(cfg, yaml_path)
    print(f"[train_nanodet] generated config -> {yaml_path}")

    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")
    print(f"[train_nanodet] train images: {len(train_dataset)}, val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.device.batchsize_per_gpu, shuffle=True,
        num_workers=cfg.device.workers_per_gpu, pin_memory=True,
        collate_fn=naive_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.device.batchsize_per_gpu, shuffle=False,
        num_workers=cfg.device.workers_per_gpu, pin_memory=True,
        collate_fn=naive_collate, drop_last=False,
    )

    model = build_model(cfg.model).to(device)

    if config.NANODET_PRETRAINED_CKPT and os.path.exists(config.NANODET_PRETRAINED_CKPT):
        class _PrintLogger:
            def log(self, msg):
                print(f"[train_nanodet] {msg}")

        # weights_only=False: safe here since this is the official NanoDet
        # pretrained checkpoint we intentionally downloaded ourselves;
        # PyTorch 2.6+ defaults weights_only to True, which can reject
        # globals (e.g. numpy scalars) present in the checkpoint.
        ckpt = torch.load(config.NANODET_PRETRAINED_CKPT, map_location="cpu", weights_only=False)
        load_model_weight(model, ckpt, _PrintLogger())
        print(f"[train_nanodet] loaded pretrained weights from {config.NANODET_PRETRAINED_CKPT}")

    optimizer = build_optimizer(model, cfg.schedule.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.schedule.total_epochs, eta_min=cfg.schedule.lr_schedule.eta_min
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    mkdir(0, cfg.save_dir)
    global_step = 0

    for epoch in range(cfg.schedule.total_epochs):
        model.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = preprocess_batch(batch, device)

            _, loss, loss_states = model.forward_train(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 35))

            if global_step <= cfg.schedule.warmup.steps:
                k = warmup_lr_factor(global_step, cfg.schedule.warmup.steps, cfg.schedule.warmup.ratio)
                for pg, base_lr in zip(optimizer.param_groups, initial_lrs):
                    pg["lr"] = base_lr * k

            optimizer.step()
            running_loss += loss.item()

            if step % cfg.log.interval == 0:
                loss_str = ", ".join(f"{k}:{v.mean().item():.4f}" for k, v in loss_states.items())
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[train_nanodet] epoch {epoch + 1}/{cfg.schedule.total_epochs} "
                      f"step {step}/{len(train_loader)} lr={lr_now:.2e} loss={loss.item():.4f} ({loss_str})")

        if global_step > cfg.schedule.warmup.steps:
            lr_scheduler.step()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"[train_nanodet] epoch {epoch + 1} done - avg_train_loss={avg_train_loss:.4f}")

        if (epoch + 1) % config.NANODET_CHECKPOINT_INTERVAL == 0 or (epoch + 1) == cfg.schedule.total_epochs:
            ckpt_path = os.path.join(cfg.save_dir, f"nanodet_epoch{epoch + 1}.pth")
            # save_model is @rank_filter-decorated: first positional arg is local_rank (0 = single process).
            save_model(0, model, ckpt_path, epoch + 1, global_step, optimizer=optimizer)
            print(f"[train_nanodet] saved checkpoint -> {ckpt_path}")

    final_path = os.path.join(cfg.save_dir, "nanodet_final.pth")
    save_model(0, model, final_path, cfg.schedule.total_epochs, global_step, optimizer=optimizer)
    print(f"[train_nanodet] training complete. Final weights -> {final_path}")


if __name__ == "__main__":
    main()
