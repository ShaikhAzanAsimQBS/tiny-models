"""RTMDetSepBNHead (mmdet.models.dense_heads.rtmdet_head.RTMDetSepBNHead),
re-implemented in plain PyTorch: "separated BN, shared conv" head - the same
3x3 conv weights are reused at every FPN level, but each level keeps its own
BatchNorm statistics. See blocks.py for why this is hand-rolled instead of
imported from mmdet."""

from typing import List

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from .blocks import ConvModule
from .points import distance2bbox, get_mlvl_points_with_stride


class RTMDetSepBNHead(nn.Module):
    def __init__(self, num_classes, in_channels, feat_channels, stacked_convs=2,
                 strides=(8, 16, 32), exp_on_reg=False, pred_kernel_size=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = list(strides)
        self.exp_on_reg = exp_on_reg
        self.pred_kernel_size = pred_kernel_size
        self.cls_out_channels = num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        pred_pad = pred_kernel_size // 2
        for _ in self.strides:
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(ConvModule(chn, feat_channels, 3, stride=1, padding=1))
                reg_convs.append(ConvModule(chn, feat_channels, 3, stride=1, padding=1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.rtm_cls.append(nn.Conv2d(feat_channels, num_classes, pred_kernel_size, padding=pred_pad))
            self.rtm_reg.append(nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pred_pad))

        # "share_conv=True": the conv weights (but not the per-level BatchNorm)
        # are tied across levels - matches the official checkpoint layout.
        for n in range(len(self.strides)):
            for i in range(stacked_convs):
                self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.strides)):
            cls_feat = x
            reg_feat = x
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return cls_scores, bbox_preds

    @torch.no_grad()
    def get_bboxes(self, cls_scores, bbox_preds, img_shape, score_thr=0.3,
                    iou_thr=0.65, max_per_img=100, nms_pre=1000):
        """Decode raw (cls_scores, bbox_preds) for a single image (batch=1)
        into final (boxes_xyxy, scores, labels), clipped to img_shape."""
        device = cls_scores[0].device
        featmap_sizes = [cs.shape[-2:] for cs in cls_scores]
        mlvl_points = get_mlvl_points_with_stride(featmap_sizes, self.strides, device)

        mlvl_bboxes, mlvl_scores = [], []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds, mlvl_points):
            # (1, C, H, W) -> (H*W, C)
            scores = cls_score[0].permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bboxes = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)

            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds]
                bboxes = bboxes[topk_inds]
                scores = scores[topk_inds]

            bboxes = distance2bbox(points, bboxes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes, dim=0)
        scores = torch.cat(mlvl_scores, dim=0)

        h, w = img_shape
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(min=0, max=w)
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(min=0, max=h)

        max_scores, labels = scores.max(dim=1)
        keep = max_scores > score_thr
        bboxes, max_scores, labels = bboxes[keep], max_scores[keep], labels[keep]

        if bboxes.numel() == 0:
            return bboxes, max_scores, labels

        keep_idx = batched_nms(bboxes, max_scores, labels, iou_thr)[:max_per_img]
        return bboxes[keep_idx], max_scores[keep_idx], labels[keep_idx]
