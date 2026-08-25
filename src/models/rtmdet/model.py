"""Full RTMDet model (backbone + neck + head) and the small amount of glue
needed to load an official OpenMMLab-trained checkpoint into it.

Submodule attribute names (`backbone`, `neck`, `bbox_head`) intentionally
match mmdetection's own `SingleStageDetector`, so an official
`rtmdet_*_8xb32-300e_coco.pth` checkpoint's `state_dict` loads directly.
"""

import torch
import torch.nn as nn

from .cspnext import CSPNeXt
from .cspnext_pafpn import CSPNeXtPAFPN
from .head import RTMDetSepBNHead
from .points import get_mlvl_points_with_stride

# widen/deepen factors + channel counts taken verbatim from the official
# mmdetection configs (configs/rtmdet/rtmdet_{tiny,s,l}_8xb32-300e_coco.py).
PRESETS = {
    "tiny": dict(
        deepen_factor=0.167, widen_factor=0.375,
        neck_in_channels=[96, 192, 384], neck_out_channels=96, num_csp_blocks=1,
        head_in_channels=96, head_feat_channels=96, exp_on_reg=False,
        params_million=4.8,
        pretrained_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
                       "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    ),
    "s": dict(
        deepen_factor=0.33, widen_factor=0.5,
        neck_in_channels=[128, 256, 512], neck_out_channels=128, num_csp_blocks=1,
        head_in_channels=128, head_feat_channels=128, exp_on_reg=False,
        params_million=8.89,
        pretrained_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
                       "rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    ),
    "l": dict(
        deepen_factor=1.0, widen_factor=1.0,
        neck_in_channels=[256, 512, 1024], neck_out_channels=256, num_csp_blocks=3,
        head_in_channels=256, head_feat_channels=256, exp_on_reg=True,
        params_million=52.3,
        pretrained_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
                       "rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    ),
}

STRIDES = (8, 16, 32)

# RTMDet's own DetDataPreprocessor mean/std (BGR order - bgr_to_rgb=False in
# every official config, which conveniently matches cv2's default BGR frames).
PIXEL_MEAN = (103.53, 116.28, 123.675)
PIXEL_STD = (57.375, 57.12, 58.395)


class RTMDet(nn.Module):
    def __init__(self, num_classes: int, variant: str = "tiny"):
        super().__init__()
        if variant not in PRESETS:
            raise ValueError(f"Unknown RTMDet variant '{variant}'. Choose from {list(PRESETS)}.")
        cfg = PRESETS[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.strides = STRIDES

        self.backbone = CSPNeXt(deepen_factor=cfg["deepen_factor"], widen_factor=cfg["widen_factor"])
        self.neck = CSPNeXtPAFPN(
            cfg["neck_in_channels"], cfg["neck_out_channels"], num_csp_blocks=cfg["num_csp_blocks"],
        )
        self.bbox_head = RTMDetSepBNHead(
            num_classes=num_classes, in_channels=cfg["head_in_channels"],
            feat_channels=cfg["head_feat_channels"], strides=STRIDES, exp_on_reg=cfg["exp_on_reg"],
        )

    def extract_feats(self, x):
        return self.neck(self.backbone(x))

    def forward(self, x):
        """Raw (cls_scores, bbox_preds) lists (one tensor per FPN level)."""
        return self.bbox_head(self.extract_feats(x))

    @torch.no_grad()
    def predict(self, x, score_thr=0.3, iou_thr=0.65, max_per_img=100, nms_pre=1000):
        """x: (1, 3, H, W) preprocessed tensor -> (boxes_xyxy, scores, labels)."""
        cls_scores, bbox_preds = self.forward(x)
        img_h, img_w = x.shape[-2:]
        return self.bbox_head.get_bboxes(
            cls_scores, bbox_preds, (img_h, img_w),
            score_thr=score_thr, iou_thr=iou_thr, max_per_img=max_per_img, nms_pre=nms_pre,
        )

    def get_mlvl_priors(self, featmap_sizes, device):
        return get_mlvl_points_with_stride(featmap_sizes, self.strides, device)


def build_rtmdet(variant: str, num_classes: int) -> RTMDet:
    return RTMDet(num_classes=num_classes, variant=variant)


def load_pretrained(model: RTMDet, checkpoint_path: str, verbose: bool = True) -> RTMDet:
    """Loads an official mmdetection RTMDet checkpoint's `state_dict`.
    Tensors whose shape doesn't match (namely the head's final classification
    conv, whose channel count depends on num_classes) are skipped so the rest
    of the network - the entire COCO-pretrained backbone/neck plus the head's
    shared feature convs - still transfers over for fine-tuning."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_")}

    own_state = model.state_dict()
    to_load, skipped = {}, []
    for k, v in state_dict.items():
        if k in own_state and own_state[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)
    missing = sorted(set(own_state.keys()) - set(to_load.keys()))

    model.load_state_dict(to_load, strict=False)
    if verbose:
        print(f"[rtmdet] loaded {len(to_load)}/{len(own_state)} tensors from {checkpoint_path}")
        if skipped:
            preview = skipped[:5]
            print(f"[rtmdet] skipped {len(skipped)} checkpoint tensor(s) (shape mismatch, e.g. num_classes "
                  f"head convs - expected when fine-tuning on a different class count): {preview}"
                  f"{' ...' if len(skipped) > 5 else ''}")
        if missing:
            preview = missing[:5]
            print(f"[rtmdet] {len(missing)} model tensor(s) left at random init: {preview}"
                  f"{' ...' if len(missing) > 5 else ''}")
    return model
