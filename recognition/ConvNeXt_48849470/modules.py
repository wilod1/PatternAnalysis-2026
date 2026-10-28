"""
Neural network architectures for ADNI 2D MRI classification.

Models:
    SimpleCNN  - compact convolutional baseline
    SimpleResNet - residual convolutional baseline
    ConvNeXt   - primary model, configured at ConvNeXt-Tiny scale

All models accept grayscale images and return raw class logits.
Class indices: NC = 0, AD = 1.

Parameter counts of the configurations used in the report:
    SimpleCNN 97,650 | SimpleResNet 700,146
    ConvNeXt-Mini 8,531,650 | ConvNeXt-Tiny 27,818,594

Shapes in the comments below are written (batch, channels, height, width).
Every model ends in global average pooling, so it accepts any input size; the
headline runs use 224 x 224 slices and the default recipe uses 256 x 256.

ConvNeXt follows Liu et al. (2022), "A ConvNet for the 2020s". It is built
here from scratch and trained from random initialisation (no pretrained
weights).
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
            # 3x3 convolution; padding=1 keeps the height and width unchanged.
            # No bias is needed because batch norm has its own learnable shift.
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # 2x2 max pooling halves the height and width
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        """(B, in_ch, H, W) -> (B, out_ch, H/2, W/2)."""
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

        # One ConvBlock per entry in `widths`; each block takes the previous
        # block's number of channels as its input.
        blocks = []
        previous_channels = in_channels

        for width in widths:
            blocks.append(ConvBlock(previous_channels, width))
            previous_channels = width

        self.features = nn.Sequential(*blocks)
        # Global average pooling: each channel's feature map becomes one number
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous_channels, num_classes),
        )

    def forward(self, x):
        """
        (B, 1, H, W) -> (B, num_classes) logits.

        With 224 x 224 input the four blocks give 112, 56, 28 and 14 pixels
        per side; pooling then reduces the 128 maps to a vector of 128.
        """
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

        # First convolution: may downsample (stride 2) and change the width
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        # Second convolution: keeps the size
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

        # The shortcut is added to the block output, so it must have the same
        # shape. If the block changes the size or width, a 1x1 convolution
        # (the "projection") reshapes the input; otherwise it is passed on as is.
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
        """(B, in_ch, H, W) -> (B, out_ch, H/stride, W/stride)."""
        identity = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # residual connection, then the activation (as in the original ResNet)
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

        # Stem: one 3x3 convolution with stride 2 halves the resolution first
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, widths[0],
                kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        # Stages: `blocks_per_stage` residual blocks per width. The first stage
        # keeps the resolution; every later stage halves it, and its first
        # block widens the channels (the rest of its blocks keep the shape).
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

        # He (Kaiming) initialisation for every convolution, as in ResNet
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(self, x):
        """
        (B, 1, H, W) -> (B, num_classes) logits.

        With 224 x 224 input: stem 112, stages 112, 56, 28 and 14 pixels per
        side, pooled to a vector of 128 channels.
        """
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
        """(B, C, H, W) -> (B, C, H, W), normalised over the C channels."""
        # nn.LayerNorm normalises the last dimension, so move the channels
        # there, normalise each pixel's channel vector, and move them back
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
        """Randomly zero the whole branch output for some samples in a batch."""
        # nothing to do when disabled or when predicting
        if self.probability == 0.0 or not self.training:
            return x

        keep_probability = 1.0 - self.probability
        # one 0/1 draw per sample, shared by all channels and pixels of it
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        # dividing by the keep probability keeps the expected value unchanged,
        # so no rescaling is needed at evaluation time
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

        # Depthwise convolution (groups=dim): every channel gets its own 7x7
        # filter, so wide spatial context is gathered cheaply. padding=3 keeps
        # the height and width unchanged.
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        # Inverted bottleneck: expand the channels 4x, apply GELU, project back.
        # The two Linear layers act on the channel axis of each pixel, which is
        # the same as 1x1 convolutions (the tensor is moved channels-last first).
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

        # Layer scale: one learnable factor per channel, starting at 1e-6 so
        # that every block begins close to the identity and training is stable.
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """(B, dim, H, W) -> (B, dim, H, W)."""
        identity = x

        x = self.dwconv(x)
        # (B, C, H, W) -> (B, H, W, C): channels last for LayerNorm and Linear
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        x = self.gamma * x
        # back to channels first
        x = x.permute(0, 3, 1, 2).contiguous()

        # residual connection; stochastic depth acts on the branch only
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

    Feature-map sizes for a 224 x 224 input (headline runs; a 256 x 256 input
    gives 64, 32, 16 and 8 pixels per side):
        stem             (B,  96, 56, 56)
        stage 1 output   (B,  96, 56, 56)
        stage 2 output   (B, 192, 28, 28)
        stage 3 output   (B, 384, 14, 14)
        stage 4 output   (B, 768,  7,  7)
        pooled           (B, 768)  ->  logits (B, num_classes)
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

        # Reject settings that would build a broken network
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

        # Patchifying stem: a 4x4 convolution with stride 4 cuts the image into
        # non-overlapping 4x4 patches, one feature vector each.
        # 256x256 -> 64x64 feature map (224x224 -> 56x56).
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # Stochastic depth rate rises linearly from 0 at the first block to
        # drop_path_rate at the last, so deeper blocks are dropped more often.
        total_blocks = sum(depths)
        drop_rates = torch.linspace(
            0.0, drop_path_rate, total_blocks
        ).tolist()

        # Four stages. Each has a downsampling layer in front of it and a stack
        # of ConvNeXt blocks after it.
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        block_index = 0  # counts blocks across all stages, to pick the drop rate

        for stage_index, (depth, dim) in enumerate(zip(depths, dims)):
            if stage_index == 0:
                # the stem has already reduced the resolution
                downsample = nn.Identity()
            else:
                # LayerNorm, then a 2x2 convolution with stride 2: halves the
                # resolution and moves from the previous width to this width
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

        # Classifier: LayerNorm on the pooled vector, dropout, then a linear
        # layer that produces one logit per class
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes),
        )

        # apply() calls the initialiser on every sub-module
        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module):
        """Use truncated-normal initialisation for convolution and linear layers."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """
        (B, 1, H, W) -> (B, dims[-1]) pooled feature vector, before the classifier.

        Kept separate from forward() so the features can be inspected on their own.
        """
        x = self.stem(x)

        # downsample, then run the blocks of that stage, four times in a row
        for downsample, stage in zip(
            self.downsample_layers, self.stages
        ):
            x = downsample(x)
            x = stage(x)

        # Spatial average pooling: (B, C, H, W) -> (B, C).
        x = x.mean(dim=(2, 3))
        return self.norm(x)

    def forward(self, x):
        """(B, 1, H, W) -> (B, num_classes) raw logits (NC = 0, AD = 1)."""
        x = self.forward_features(x)
        return self.head(x)


# ---------------------------------------------------------------------
# Size presets and checkpoint helpers
# ---------------------------------------------------------------------

# Named stage depths and widths. The report uses "tiny" (27.8 M parameters) and
# "mini" (8.5 M); "micro" is an extra, smaller preset.
CONVNEXT_VARIANTS = {
    "tiny": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
    "mini": {"depths": (2, 2, 6, 2), "dims": (64, 128, 256, 512)},
    "micro": {"depths": (2, 2, 4, 2), "dims": (48, 96, 192, 384)},
}


def build_convnext(variant="mini", **overrides):
    """Build a ConvNeXt from a named size preset plus keyword overrides."""
    if variant not in CONVNEXT_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; "
            f"choose from {list(CONVNEXT_VARIANTS)}"
        )

    # start from the preset; any keyword (e.g. dropout, drop_path_rate)
    # replaces the corresponding default of ConvNeXt
    config = dict(CONVNEXT_VARIANTS[variant])
    config.update(overrides)
    return ConvNeXt(**config)


def convnext_from_checkpoint(checkpoint):
    """
    Rebuild a ConvNeXt from a checkpoint written by train.py.

    The caller loads the weights:
        model.load_state_dict(checkpoint["model_state_dict"])
    """
    config = dict(checkpoint["model_config"])
    # depths and widths may come back as lists after saving; the model expects
    # tuples
    config["depths"] = tuple(config["depths"])
    config["dims"] = tuple(config["dims"])
    return ConvNeXt(**config)