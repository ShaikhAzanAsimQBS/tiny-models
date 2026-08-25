"""RTMDet's training objective: Dynamic Soft Label Assigner (the exact same
assigner RTMDet's paper borrows from NanoDet-Plus, vendored at
external/nanodet) + Quality Focal Loss (classification) + GIoU Loss (box
regression) - reusing the vendored NanoDet loss/assigner implementations
directly rather than re-deriving them, since they are algorithmically
identical to mmdetection's `DynamicSoftLabelAssigner`/`QualityFocalLoss`/
`GIoULoss` used by the official RTMDet training recipe."""

import torch

from .points import distance2bbox, get_mlvl_points_with_stride


def compute_rtmdet_loss(cls_scores, bbox_preds, targets, num_classes, strides,
                         assigner, qfl_loss_fn, giou_loss_fn):
    """
    Args:
        cls_scores: list[(B, num_classes, H, W)] raw logits, one per FPN level.
        bbox_preds: list[(B, 4, H, W)] predicted (l, t, r, b) distances already
            scaled by stride (i.e. in input-image pixel units).
        targets: list of length B, each {"boxes": (N,4) xyxy pixel coords,
            "labels": (N,) 0-indexed class ids}.
        strides: tuple of ints, e.g. (8, 16, 32).
        assigner: nanodet.model.head.assigner.dsl_assigner.DynamicSoftLabelAssigner
        qfl_loss_fn: nanodet.model.loss.gfocal_loss.QualityFocalLoss instance
        giou_loss_fn: nanodet.model.loss.iou_loss.GIoULoss instance

    Returns:
        (loss_cls, loss_bbox, num_pos) - two scalar loss tensors + positive-sample count for logging.
    """
    device = cls_scores[0].device
    batch_size = cls_scores[0].shape[0]
    featmap_sizes = [cs.shape[-2:] for cs in cls_scores]

    mlvl_priors = get_mlvl_points_with_stride(featmap_sizes, strides, device)
    priors = torch.cat(mlvl_priors, dim=0)  # (N_total, 4) [x, y, stride_w, stride_h]
    num_priors = priors.size(0)

    flat_cls = torch.cat([
        cs.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes) for cs in cls_scores
    ], dim=1)
    flat_reg = torch.cat([
        bp.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bp in bbox_preds
    ], dim=1)
    decoded = distance2bbox(priors.unsqueeze(0), flat_reg)  # (B, N_total, 4) xyxy

    all_labels, all_label_weights, all_bbox_targets, all_assign_metrics = [], [], [], []
    for b in range(batch_size):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)

        labels = priors.new_full((num_priors,), num_classes, dtype=torch.long)
        label_weights = priors.new_zeros(num_priors)
        bbox_targets = torch.zeros_like(priors[:, :4])
        assign_metrics = priors.new_zeros(num_priors)

        if gt_boxes.numel() > 0:
            assign_result = assigner.assign(
                flat_cls[b].detach(), priors, decoded[b].detach(), gt_boxes, gt_labels,
            )
            pos_inds = (assign_result.gt_inds > 0).nonzero(as_tuple=False).squeeze(-1)
            neg_inds = (assign_result.gt_inds == 0).nonzero(as_tuple=False).squeeze(-1)
            if pos_inds.numel() > 0:
                pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
                bbox_targets[pos_inds] = gt_boxes[pos_gt_inds]
                labels[pos_inds] = gt_labels[pos_gt_inds]
                label_weights[pos_inds] = 1.0
                assign_metrics[pos_inds] = assign_result.max_overlaps[pos_inds]
            if neg_inds.numel() > 0:
                label_weights[neg_inds] = 1.0
        else:
            label_weights[:] = 1.0

        all_labels.append(labels)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_assign_metrics.append(assign_metrics)

    labels = torch.stack(all_labels, dim=0).reshape(-1)
    label_weights = torch.stack(all_label_weights, dim=0).reshape(-1)
    bbox_targets = torch.stack(all_bbox_targets, dim=0).reshape(-1, 4)
    assign_metrics = torch.stack(all_assign_metrics, dim=0).reshape(-1)

    flat_cls_all = flat_cls.reshape(-1, num_classes)
    decoded_all = decoded.reshape(-1, 4)

    pos_mask = (labels >= 0) & (labels < num_classes)
    num_pos = int(pos_mask.sum().item())
    avg_factor = max(float(assign_metrics.sum().item()), 1.0)

    loss_cls = qfl_loss_fn(flat_cls_all, (labels, assign_metrics), weight=label_weights, avg_factor=avg_factor)

    if num_pos > 0:
        loss_bbox = giou_loss_fn(
            decoded_all[pos_mask], bbox_targets[pos_mask],
            weight=assign_metrics[pos_mask], avg_factor=avg_factor,
        )
    else:
        loss_bbox = decoded_all.sum() * 0.0

    return loss_cls, loss_bbox, num_pos
