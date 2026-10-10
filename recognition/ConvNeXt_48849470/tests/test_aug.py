import pytest
import torch

from dataset import RandomGamma, build_transforms


def test_eval_transform_is_deterministic_and_resizes():
    x = torch.rand(1, 240, 256)
    tf = build_transforms(False, 0.3, 0.2, img_size=64)
    out = tf(x)
    assert out.shape == (1, 64, 64)
    assert torch.equal(out, tf(x))


@pytest.mark.parametrize("aug", ["mild", "strong"])
def test_train_transform_shape_finite_and_random(aug):
    x = torch.rand(1, 240, 256)
    tf = build_transforms(True, 0.3, 0.2, img_size=64, aug=aug)
    a, b = tf(x), tf(x)
    assert a.shape == (1, 64, 64) and torch.isfinite(a).all()
    assert not torch.equal(a, b)


def test_unknown_aug_raises():
    with pytest.raises(ValueError):
        build_transforms(True, 0.3, 0.2, aug="extreme")


def test_random_gamma():
    x = torch.rand(1, 8, 8)
    assert torch.allclose(RandomGamma(1.0, 1.0)(x), x)
    y = RandomGamma(0.5, 2.0)(x)
    assert y.min() >= 0 and y.max() <= 1