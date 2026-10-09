"""
Model components. Only torch / torch.nn are used in this file.
"""
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