"""
Pure-PyTorch re-implementation of the small set of mmcv/mmdetection building
blocks RTMDet is built from (ConvModule, DepthwiseSeparableConvModule,
ChannelAttention, SPPBottleneck, CSPNeXtBlock, CSPLayer).

`mmcv` (the OpenMMLab library these normally come from) ships compiled CUDA
extensions and only publishes wheels for a handful of pinned torch versions,
so it fails to even build against the very new torch build in this project's
venv. Reimplementing this small slice of it in plain torch/torchvision avoids
that dependency entirely.

Submodule names/attribute names below (`.conv`, `.bn`, `.activate`,
`.depthwise_conv`, `.pointwise_conv`, `.main_conv`, `.short_conv`,
`.final_conv`, `.blocks`, `.attention`, ...) intentionally match mmcv/mmdet's
own naming 1:1, so the official COCO-pretrained RTMDet-tiny checkpoint
(a plain `state_dict`) can be loaded into this implementation with a plain
`load_state_dict()` call - no key remapping needed.
"""

import torch
import torch.nn as nn

BN_MOMENTUM = 0.03
BN_EPS = 0.001


class ConvModule(nn.Module):
    """Conv2d -> BatchNorm2d -> activation, matching mmcv.cnn.ConvModule's
    default ("conv", "norm", "act") ordering and attribute names."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.activate = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise conv block + pointwise conv block, matching
    mmcv.cnn.bricks.DepthwiseSeparableConvModule's attribute names."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation-style channel attention used inside CSPNeXt's
    CSPLayer (mmdet.models.layers.se_layer.ChannelAttention)."""

    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling bottleneck used at the end of CSPNeXt's
    backbone (mmdet.models.backbones.csp_darknet.SPPBottleneck)."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPNeXtBlock(nn.Module):
    """Basic residual block used inside CSPNeXt's CSPLayer: a 3x3 ConvModule
    followed by a 5x5 depthwise-separable ConvModule."""

    def __init__(self, in_channels, out_channels, expansion=0.5,
                 add_identity=True, kernel_size=5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer used throughout CSPNeXt/CSPNeXtPAFPN."""

    def __init__(self, in_channels, out_channels, expand_ratio=0.5,
                 num_blocks=1, add_identity=True, channel_attention=False):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity)
            for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)
