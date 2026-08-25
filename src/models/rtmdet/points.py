"""Point-prior generation and distance<->bbox coding for RTMDet's
anchor-free head. Equivalent to mmdet's MlvlPointGenerator(offset=0) +
DistancePointBBoxCoder, re-implemented directly since only a couple of
formulas are actually needed."""

import torch


def get_points_single(feat_h, feat_w, stride, device, dtype=torch.float32):
    """Grid of (x, y) pixel-space point priors for one feature level.
    offset=0 (RTMDet's setting) means points sit at (col*stride, row*stride),
    i.e. the top-left corner of each stride x stride cell - not its center.
    """
    shift_x = torch.arange(0, feat_w, device=device, dtype=dtype) * stride
    shift_y = torch.arange(0, feat_h, device=device, dtype=dtype) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (feat_h*feat_w, 2)


def get_mlvl_points_with_stride(featmap_sizes, strides, device, dtype=torch.float32):
    """Returns a list of (N_level, 4) tensors: [x, y, stride_w, stride_h]."""
    mlvl_points = []
    for (feat_h, feat_w), stride in zip(featmap_sizes, strides):
        points = get_points_single(feat_h, feat_w, stride, device, dtype)
        stride_tensor = points.new_full((points.size(0), 2), stride)
        mlvl_points.append(torch.cat([points, stride_tensor], dim=1))
    return mlvl_points


def distance2bbox(points, distance):
    """points: (..., 2 or 4) - only the first 2 columns (x, y) are used.
    distance: (..., 4) predicted (left, top, right, bottom) offsets, already
    in pixel units. Returns (..., 4) boxes in (x1, y1, x2, y2) format."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)
