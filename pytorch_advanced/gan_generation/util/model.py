from typing import cast

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    last: nn.Sequential

    def __init__(self, z_dim: int = 20, image_size: int = 64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=image_size * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(num_features=image_size * 8),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=image_size * 8, out_channels=image_size * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=image_size * 4),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=image_size * 4, out_channels=image_size * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=image_size * 2),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=image_size * 2, out_channels=image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=image_size, out_channels=1, kernel_size=4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, z: Tensor):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return cast(Tensor, out)


class Discriminator(nn.Module):
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    last: nn.Conv2d

    def __init__(self, image_size: int = 64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=image_size, out_channels=image_size * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=image_size * 2, out_channels=image_size * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=image_size * 4, out_channels=image_size * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.last = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x: Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return cast(Tensor, out)


def weights_init(
    m,
    *,
    conv_weight_mean=0.0,
    conv_weight_std=0.02,
    conv_bias=0.0,
    batchnorm_weight_mean=1.0,
    batchnorm_weight_std=0.02,
    batchnorm_bias=0.0,
):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, conv_weight_mean, conv_weight_std)
        nn.init.constant_(m.bias.data, conv_bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, batchnorm_weight_mean, batchnorm_weight_std)
        nn.init.constant_(m.bias.data, batchnorm_bias)


class SelfAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))

        s = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(s)
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(x.size())
        out = x + self.gamma * o

        return out, attention_map


class SAGANGenerator(nn.Module):
    def __init__(self, z_dim: int, image_size: int = 64):
        super().__init__()

        self.layer1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(z_dim, image_size * 8, 4, 1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1)),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
        )
        self.self_attention_1 = SelfAttention(image_size * 2)
        self.layer4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
        )
        self.self_attention_2 = SelfAttention(image_size)
        self.last = nn.Sequential(nn.ConvTranspose2d(image_size, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map_1 = self.self_attention_1(out)
        out = self.layer4(out)
        out, attention_map_2 = self.self_attention_2(out)
        out = self.last(out)
        return out, attention_map_1, attention_map_2


class SAGANDiscriminator(nn.Module):
    def __init__(self, image_size: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(spectral_norm(nn.Conv2d(1, image_size, 4, 2, 1)), nn.LeakyReLU(0.1, True))
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(image_size, image_size * 2, 4, 2, 1)), nn.LeakyReLU(0.1, True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1)), nn.LeakyReLU(0.1, True)
        )
        self.self_attention_1 = SelfAttention(image_size * 4)
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1)), nn.LeakyReLU(0.1, True)
        )
        self.self_attention_2 = SelfAttention(image_size * 8)
        self.last = nn.Conv2d(image_size * 8, 1, 4, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map_1 = self.self_attention_1(out)
        out = self.layer4(out)
        out, attention_map_2 = self.self_attention_2(out)
        out = self.last(out)
        return out, attention_map_1, attention_map_2
