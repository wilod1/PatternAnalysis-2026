"""
Shape and gradient tests for the model components.
"""
import os
import sys

import torch

sys.path.inser(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import ConvBlock, SimpleCNN


def test_convblock_halves_size():
    """
    A ConvBlock changes channels and halves height and width.
    """
    y = ConvBlock(1, 16)(torch.randn(2, 1, 256, 256))
    assert y.shape == (2, 16, 128, 128)


def test_simplecnn_output_shape():
    """
    SimpleCNN maps a batch of slices to one logit pair per slice.
    """
    model = SimpleCNN()
    assert model(torch.randn(2, 1, 256, 256)).shape == (2, 2)
    print("SimpleCNN parameters:", sum(p.numel() for p in model.parameters()))


def test_simplecnn_gradients_flow():
    """
    A backward pass gives every paramater a gradient.
    """
    model = SimpleCNN()
    model(torch.randn(2, 1, 256, 256)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)