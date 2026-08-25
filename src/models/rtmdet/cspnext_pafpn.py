"""CSPNeXtPAFPN neck used by RTMDet
(mmdet.models.necks.cspnext_pafpn.CSPNeXtPAFPN), re-implemented in plain
PyTorch. See blocks.py for why."""

import torch
import torch.nn as nn

from .blocks import ConvModule, CSPLayer


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_csp_blocks=3, expand_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(CSPLayer(
                in_channels[idx - 1] * 2, in_channels[idx - 1], num_blocks=num_csp_blocks,
                add_identity=False, expand_ratio=expand_ratio,
            ))

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[idx], in_channels[idx], 3, stride=2, padding=1))
            self.bottom_up_blocks.append(CSPLayer(
                in_channels[idx] * 2, in_channels[idx + 1], num_blocks=num_csp_blocks,
                add_identity=False, expand_ratio=expand_ratio,
            ))

        self.out_convs = nn.ModuleList([
            ConvModule(in_channels[i], out_channels, 3, padding=1) for i in range(len(in_channels))
        ])

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
