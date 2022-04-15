from typing import cast

import torch
from pytorch_advanced.gan_generation.util.model import Discriminator, Generator


def test_generator():
    g = Generator(z_dim=20, image_size=64)
    input_z = torch.rand(1, 20, 1, 1)
    fake_image = cast(torch.Tensor, g(input_z))
    assert fake_image.size() == torch.Size([1, 1, 64, 64])


def test_discriminator():
    d = Discriminator(image_size=64)
    image = torch.rand(1, 1, 64, 64)
    out = cast(torch.Tensor, d(image))
    assert out.size() == torch.Size([1, 1, 1, 1])
