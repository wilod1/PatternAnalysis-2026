
"""
Neural network architectures for ADNI 2D MRI classification.

Models:
    SimpleCNN  - compact convolutional baseline
    SimpleResNet - residual convolutional baseline
    ConvNeXt   - primary model, configured at ConvNeXt-Tiny scale

All models accept grayscale images and return raw class logits.
Class indices: NC = 0, AD = 1.
"""

import torch
from torch import nn


# ---------------------------------------------------------------------
# CNN baseline
# ---------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Convolution, batch normalisation, activation and downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """Four-stage CNN baseline with global average pooling."""

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        widths=(16, 32, 64, 128),
        dropout=0.2,
    ):
        super().__init__()

        blocks = []
        previous_channels = in_channels

        for width in widths:
            blocks.append(ConvBlock(previous_channels, width))
            previous_channels = width

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------
# ResNet baseline
# ---------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.act(out + identity)


class SimpleResNet(nn.Module):
    """Compact residual network retained as a comparison baseline."""

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        widths=(16, 32, 64, 128),
        blocks_per_stage=2,
    ):
        super().__init__()

        if len(widths) == 0 or blocks_per_stage < 1:
            raise ValueError(
                "widths must be non-empty and blocks_per_stage must be positive"
            )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, widths[0],
                kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = widths[0]

        for stage_index, width in enumerate(widths):
            stride = 1 if stage_index == 0 else 2

            for block_index in range(blocks_per_stage):
                block_stride = stride if block_index == 0 else 1

                stages.append(
                    BasicBlock(in_ch, width, stride=block_stride)
                )
                in_ch = width

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------
# ConvNeXt building blocks
# ---------------------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """Apply LayerNorm to channels of a channels-first feature map."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class DropPath(nn.Module):
    """
    Stochastic depth for a residual branch.

    During training, each sample independently keeps or drops the branch.
    At evaluation time, the branch is unchanged.
    """

    def __init__(self, probability=0.0):
        super().__init__()

        if not 0.0 <= probability < 1.0:
            raise ValueError("DropPath probability must be in [0, 1)")

        self.probability = float(probability)

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x

        keep_probability = 1.0 - self.probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt residual block.

    The feature transformation consists of:
        1. 7x7 depthwise convolution
        2. channel-wise LayerNorm
        3. 4x channel expansion
        4. GELU activation
        5. channel projection
        6. learnable layer scaling and stochastic depth
    """

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init=1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()

        return identity + self.drop_path(x)


# ---------------------------------------------------------------------
# Main ConvNeXt model
# ---------------------------------------------------------------------

class ConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny-scale model for grayscale brain MRI classification.

    Default configuration:
        Input:       (B, 1, 256, 256)
        Stage depths: (3, 3, 9, 3)
        Stage widths: (96, 192, 384, 768)
        Output:      (B, num_classes) raw logits

    The model is initialised from scratch. No pretrained weights are loaded.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.1,
        layer_scale_init=1e-6,
        dropout=0.2,
    ):
        super().__init__()

        if len(depths) != 4 or len(dims) != 4:
            raise ValueError(
                "ConvNeXt requires four stage depths and four stage widths"
            )

        if any(depth < 1 for depth in depths):
            raise ValueError("Every ConvNeXt stage must contain a block")

        if any(dim < 1 for dim in dims):
            raise ValueError("All ConvNeXt widths must be positive")

        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")

        # Patchifying stem: 256x256 -> 64x64 feature map.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        total_blocks = sum(depths)
        drop_rates = torch.linspace(
            0.0, drop_path_rate, total_blocks
        ).tolist()

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        block_index = 0

        for stage_index, (depth, dim) in enumerate(zip(depths, dims)):
            if stage_index == 0:
                downsample = nn.Identity()
            else:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[stage_index - 1], eps=1e-6),
                    nn.Conv2d(
                        dims[stage_index - 1],
                        dim,
                        kernel_size=2,
                        stride=2,
                    ),
                )

            self.downsample_layers.append(downsample)

            blocks = []

            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dim,
                        drop_path=drop_rates[block_index],
                        layer_scale_init=layer_scale_init,
                    )
                )
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes),
        )

        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module):
        """Use truncated-normal initialisation for convolution and linear layers."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.stem(x)

        for downsample, stage in zip(
            self.downsample_layers, self.stages
        ):
            x = downsample(x)
            x = stage(x)

        # Spatial average pooling: (B, C, H, W) -> (B, C).
        x = x.mean(dim=(2, 3))
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
