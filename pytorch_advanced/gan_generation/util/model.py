from typing import cast

from torch import Tensor, nn


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
