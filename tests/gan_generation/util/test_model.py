from typing import cast
from zlib import Z_BEST_COMPRESSION

import torch
from pytorch_advanced.gan_generation.util.model import (
    Discriminator,
    Generator,
    SAGANGenerator,
    SelfAttention,
    weights_init,
)


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


def test_weights_init_conv():
    conv_weight_mean = 0.0
    conv_weight_std = 0.02
    conv_bias = 0.0

    conv = torch.nn.Conv2d(1, 64, 4)
    weights_init(conv, conv_weight_mean=conv_weight_mean, conv_weight_std=conv_weight_std, conv_bias=conv_bias)
    assert torch.mean(conv.weight.data).round(decimals=1) == torch.tensor(conv_weight_mean)
    assert torch.std(conv.weight.data).round(decimals=2) == torch.tensor(conv_weight_std)
    assert torch.equal(conv.bias.data, torch.full_like(conv.bias.data, conv_bias))


def test_weights_init_batchnorm():
    batchnorm_weight_mean = 0.0
    batchnorm_weight_std = 0.02
    batchnorm_bias = 0.0

    batchnorm = torch.nn.BatchNorm2d(64)
    weights_init(
        batchnorm,
        batchnorm_weight_mean=batchnorm_weight_mean,
        batchnorm_weight_std=batchnorm_weight_std,
        batchnorm_bias=batchnorm_bias,
    )
    assert torch.mean(batchnorm.weight.data).round(decimals=1) == torch.tensor(0.0)
    assert torch.std(batchnorm.weight.data).round(decimals=2) == torch.tensor(0.02)
    assert torch.equal(batchnorm.bias.data, torch.full_like(batchnorm.bias.data, batchnorm_bias))


def test_self_attention():
    batch_size = 4
    in_dim = 64  # in_channels
    image_size = 32

    self_attention = SelfAttention(in_dim)
    input_tensor = torch.rand(4, in_dim, image_size, image_size)
    out, attention_map = self_attention(input_tensor)
    assert out.size() == input_tensor.size()
    assert attention_map.size() == torch.Size([batch_size, image_size**2, image_size**2])


def test_sagan_generator():
    batch_size = 64
    z_dim = 20
    image_size = 64

    g = SAGANGenerator(z_dim, image_size)
    input_tensor = torch.rand(batch_size, z_dim, 1, 1)
    out, map_1, map_2 = g(input_tensor)
    assert out.size() == torch.Size([batch_size, 1, image_size, image_size])
    assert map_1.size() == torch.Size([batch_size, (image_size // 4) ** 2, (image_size // 4) ** 2])
    assert map_2.size() == torch.Size([batch_size, (image_size // 2) ** 2, (image_size // 2) ** 2])
