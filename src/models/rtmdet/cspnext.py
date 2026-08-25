"""CSPNeXt backbone used by RTMDet (mmdet.models.backbones.cspnext.CSPNeXt),
re-implemented in plain PyTorch. See blocks.py for why."""

import torch.nn as nn

from .blocks import ConvModule, CSPLayer, SPPBottleneck

# [in_channels, out_channels, num_blocks, add_identity, use_spp] at widen/deepen_factor=1.0
ARCH_SETTINGS_P5 = [
    [64, 128, 3, True, False],
    [128, 256, 6, True, False],
    [256, 512, 6, True, False],
    [512, 1024, 3, False, True],
]


class CSPNeXt(nn.Module):
    def __init__(self, deepen_factor=1.0, widen_factor=1.0, out_indices=(2, 3, 4),
                 expand_ratio=0.5, channel_attention=True, spp_kernel_sizes=(5, 9, 13)):
        super().__init__()
        arch_setting = ARCH_SETTINGS_P5
        self.out_indices = out_indices

        stem_out = int(arch_setting[0][0] * widen_factor)
        stem_mid = int(arch_setting[0][0] * widen_factor // 2)
        self.stem = nn.Sequential(
            ConvModule(3, stem_mid, 3, stride=2, padding=1),
            ConvModule(stem_mid, stem_mid, 3, stride=1, padding=1),
            ConvModule(stem_mid, stem_out, 3, stride=1, padding=1),
        )
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)

            stage = [ConvModule(in_channels, out_channels, 3, stride=2, padding=1)]
            if use_spp:
                stage.append(SPPBottleneck(out_channels, out_channels, kernel_sizes=spp_kernel_sizes))
            stage.append(CSPLayer(
                out_channels, out_channels, expand_ratio=expand_ratio, num_blocks=num_blocks,
                add_identity=add_identity, channel_attention=channel_attention,
            ))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
