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