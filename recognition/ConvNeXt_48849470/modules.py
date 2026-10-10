"""
Model components. Only torch / torch.nn are used in this file.
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    3x3 conv -> BatchNorm -> ReLU -> 2x2 max-pool (halves the spatial size).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)



class SimpleCNN(nn.Module):
    """
    Baseline CNN: four ConvBlocks, global average pooling, linear classifier.

    Input: (B, 1, 256, 256). Output: (B, num_classes) raw logits.
    """

    def __init__(self, in_channels=1, num_classes=2,
                 widths=(16, 32, 64, 128), dropout=0.2):
        super().__init__()
        blocks, prev = [], in_channels
        for w in widths:
            blocks.append(ConvBlock(prev, w))
            prev = w

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class BasicBlock(nn.Module):
    """
    Residual block from He et al. (2016), "Deep Residual Learning for Image
    Recognition": two 3x3 conv + BatchNorm layers whose output is added to a
    shortcut. The shortcut is the identity, or a 1x1 conv + BatchNorm when the
    block changes the channel count or downsamples.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        if stride != 1 or in_ch != out_ch:
            # projection shortcut so the two branches have matching shapes
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class SimpleResNet(nn.Module):
    """
    Small ResNet baseline: a stride-2 stem, four stages of BasicBlocks with
    stride-2 downsampling between stages, global average pooling and a linear
    head. With the default widths a 256x256 input becomes 128 (stem), then
    128, 64, 32 and 16 pixels across the four stages.
    """

    def __init__(self, in_channels=1, num_classes=2, widths=(16, 32, 64, 128),
                 blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        stages, in_ch = [], widths[0]
        for i, width in enumerate(widths):
            stride = 1 if i == 0 else 2  # first stage keeps the stem resolution
            for b in range(blocks_per_stage):
                stages.append(BasicBlock(in_ch, width, stride if b == 0 else 1))
                in_ch = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], num_classes)
        for m in self.modules():  # Kaiming init for the convolutions
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.head(self.pool(x).flatten(1))


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm over the channel dimension of an (N, C, H, W) tensor, as used by
    ConvNeXt in its stem and downsampling layers.
    """

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """
    Stochastic depth (Huang et al., 2016): during training, zero the residual
    branch for a random subset of samples and rescale the rest by 1/(1-p).
    The identity at evaluation time.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (Liu et al., 2022, "A ConvNet for the 2020s"): 7x7 depthwise
    conv, LayerNorm, inverted bottleneck (expand 4x, GELU, project back), layer
    scale, then a residual connection with stochastic depth. The two pointwise
    layers are Linear layers applied channels-last, as in the paper.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        # layer scale: a learnable per-channel gain that starts near zero, so
        # each block starts close to the identity
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)  # back to NCHW
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    """
    Reduced ConvNeXt for 2D brain MRI slices (trained from scratch): a 4x4
    stride-4 patchify stem, four stages of ConvNeXtBlocks with separate
    LayerNorm + 2x2 stride-2 downsampling layers between them, global average
    pooling, LayerNorm and a linear head. Depths and widths are far smaller than
    ConvNeXt-Tiny (3,3,9,3 / 96..768). For 256x256 input the stage resolutions
    are 64, 32, 16 and 8.
    """

    def __init__(self, in_channels=1, num_classes=2, depths=(2, 2, 4, 2),
                 dims=(32, 64, 128, 256), drop_path_rate=0.1,
                 layer_scale_init=1e-6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        # drop-path probability rises linearly with block depth
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        layers, k = [], 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            if i > 0:
                layers.append(nn.Sequential(
                    LayerNorm2d(dims[i - 1], eps=1e-6),
                    nn.Conv2d(dims[i - 1], dim, 2, stride=2),
                ))
            for _ in range(depth):
                layers.append(ConvNeXtBlock(dim, rates[k], layer_scale_init))
                k += 1
        self.stages = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        for m in self.modules():  # truncated-normal init as in the paper
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.head(self.norm(x.mean(dim=(2, 3))))