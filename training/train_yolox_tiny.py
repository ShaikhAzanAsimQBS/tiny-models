"""
Fine-tunes YOLOX-Tiny (using the vendored official repo at external/YOLOX) on
the COCO-format dataset produced by pipeline.py at config.YOLOX_DATASET_DIR.

This drives YOLOX's own Exp/Trainer classes directly (single GPU, no
torch.distributed launch) instead of going through their tools/train.py CLI,
so no argparse/CLI flags are involved - every knob lives in config.py.

Run with the CUDA-enabled venv:

    "C:\\Users\\QBS PC\\PycharmProjects\\ais-handler-template\\.venv\\Scripts\\python.exe" training\\train_yolox_tiny.py
"""

import importlib.util
import os
import sys
import urllib.request
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

# The vendored YOLOX repo must be importable as the `yolox` package.
sys.path.insert(0, config.YOLOX_REPO_DIR)

import torch  # noqa: E402
from loguru import logger  # noqa: E402


def _load_tiny_exp_class():
    """exps/default/yolox_tiny.py is a standalone script in the YOLOX repo
    (not part of the installable `yolox` package), so it's loaded by path."""
    exp_path = os.path.join(config.YOLOX_REPO_DIR, "exps", "default", "yolox_tiny.py")
    spec = importlib.util.spec_from_file_location("yolox_tiny_exp", exp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp


def build_exp():
    TinyExp = _load_tiny_exp_class()
    exp = TinyExp()

    exp.num_classes = len(config.DATASET_CLASS_NAMES)
    exp.data_dir = config.YOLOX_DATASET_DIR
    exp.train_ann = "instances_train2017.json"
    exp.val_ann = "instances_val2017.json"
    exp.input_size = config.YOLOX_INPUT_SIZE
    exp.test_size = config.YOLOX_TEST_SIZE
    exp.max_epoch = config.YOLOX_MAX_EPOCH
    exp.warmup_epochs = config.YOLOX_WARMUP_EPOCHS
    exp.no_aug_epochs = config.YOLOX_NO_AUG_EPOCHS
    exp.basic_lr_per_img = config.YOLOX_BASIC_LR_PER_IMG
    exp.eval_interval = config.YOLOX_EVAL_INTERVAL
    exp.print_interval = config.YOLOX_PRINT_INTERVAL
    exp.data_num_workers = config.YOLOX_DATA_NUM_WORKERS
    exp.output_dir = config.YOLOX_OUTPUT_DIR
    exp.exp_name = config.YOLOX_EXPERIMENT_NAME
    return exp


def ensure_pretrained_checkpoint() -> str:
    """Downloads the official COCO-pretrained YOLOX-Tiny checkpoint (used as
    the fine-tuning start point) if it isn't already on disk."""
    if not config.YOLOX_AUTO_DOWNLOAD_PRETRAINED:
        return config.YOLOX_PRETRAINED_CKPT if os.path.exists(config.YOLOX_PRETRAINED_CKPT) else None

    ckpt_path = config.YOLOX_PRETRAINED_CKPT
    if os.path.exists(ckpt_path):
        return ckpt_path

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    print(f"[train_yolox_tiny] downloading pretrained checkpoint from {config.YOLOX_PRETRAINED_CKPT_URL}")
    try:
        urllib.request.urlretrieve(config.YOLOX_PRETRAINED_CKPT_URL, ckpt_path)
        return ckpt_path
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[train_yolox_tiny] WARNING: could not download pretrained checkpoint ({exc}). "
              f"Training will start from random weights.")
        return None


def resume_checkpoint_path(exp) -> str:
    """Path to the "latest_ckpt.pth" that YOLOX's Trainer auto-saves after
    every epoch (model + optimizer + start_epoch + best_ap), used to
    continue an interrupted run via YOLOX_RESUME=True."""
    return os.path.join(exp.output_dir, exp.exp_name, "latest_ckpt.pth")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "YOLOX's vendored Trainer requires a CUDA GPU. "
            "Make sure you're running with the CUDA-enabled venv."
        )

    from yolox.utils import configure_nccl, configure_omp

    exp = build_exp()
    exp.seed = config.TRAIN_RANDOM_SEED

    if config.YOLOX_RESUME:
        # Continue a previously interrupted run: load model + optimizer +
        # epoch/best-AP state from this experiment's own latest checkpoint,
        # NOT the official pretrained weights (those don't contain
        # optimizer/epoch state and would restart from epoch 0).
        ckpt_path = resume_checkpoint_path(exp)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"YOLOX_RESUME is True but no checkpoint was found at {ckpt_path}. "
                f"Run a first training pass (with YOLOX_RESUME=False) before resuming."
            )
        logger.info(f"[train_yolox_tiny] YOLOX_RESUME=True -> resuming from {ckpt_path}")
    else:
        ckpt_path = ensure_pretrained_checkpoint()

    args = SimpleNamespace(
        experiment_name=exp.exp_name,
        name=None,
        batch_size=config.YOLOX_BATCH_SIZE,
        devices=1,
        fp16=config.YOLOX_FP16,
        cache=config.YOLOX_CACHE_IMGS,
        occupy=config.YOLOX_OCCUPY_GPU,
        logger="tensorboard",
        resume=config.YOLOX_RESUME,
        ckpt=ckpt_path,
        start_epoch=None,
        dist_backend="nccl",
        dist_url=None,
        num_machines=1,
        machine_rank=0,
    )

    if exp.seed is not None:
        import random
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)

    configure_nccl()
    configure_omp()
    torch.backends.cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()

    logger.info(f"[train_yolox_tiny] done. Checkpoints saved under "
                f"{os.path.join(exp.output_dir, args.experiment_name)}")


if __name__ == "__main__":
    main()
