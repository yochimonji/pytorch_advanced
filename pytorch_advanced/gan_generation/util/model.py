from typing import cast

from torch import Tensor, nn


class Generator(nn.Module):
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
