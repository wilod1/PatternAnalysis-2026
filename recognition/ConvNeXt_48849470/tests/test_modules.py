"""
Shape and gradient tests for the model components.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import ConvBlock, SimpleCNN, BasicBlock, SimpleResNet, LayerNorm2d, DropPath, ConvNeXt, ConvNeXtBlock


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


def test_basic_block_shapes():
    x = torch.randn(2, 16, 32, 32)
    assert BasicBlock(16, 16)(x).shape == (2, 16, 32, 32)             # identity shortcut
    assert BasicBlock(16, 32, stride=2)(x).shape == (2, 32, 16, 16)   # projection shortcut


def test_resnet_output_and_gradients():
    model = SimpleResNet()
    out = model(torch.randn(4, 1, 256, 256))
    assert out.shape == (4, 2)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_resnet_other_sizes_and_eval_mode():
    model = SimpleResNet().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 64, 64)).shape == (1, 2)

def test_layernorm2d_normalises_channels():
    y = LayerNorm2d(8)(torch.randn(2, 8, 4, 4))
    assert y.mean(dim=1).abs().max() < 1e-5


def test_droppath_identity_in_eval_and_stochastic_in_train():
    dp, x = DropPath(0.5), torch.ones(64, 1, 2, 2)
    dp.eval()
    assert torch.equal(dp(x), x)
    dp.train()
    out = dp(x)
    dropped = out.flatten(1).sum(1) == 0
    assert dropped.any() and (~dropped).any()
    assert torch.allclose(out[~dropped], x[~dropped] * 2)  # rescaled by 1/(1-p)


def test_convnext_block_is_near_identity_at_init():
    block, x = ConvNeXtBlock(32).eval(), torch.randn(2, 32, 16, 16)
    y = block(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)  # layer scale 1e-6


def test_convnext_output_and_gradients():
    model = ConvNeXt(drop_path_rate=0.1)
    out = model(torch.randn(2, 1, 256, 256))
    assert out.shape == (2, 2)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_convnext_other_sizes_and_eval_mode():
    model = ConvNeXt().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 64, 64)).shape == (1, 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)