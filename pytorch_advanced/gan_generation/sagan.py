import matplotlib.pyplot as plt
import torch
from torch import cuda
from torch.utils.data import DataLoader

from util.dataloader import GANImageDataset, ImageTransform, make_datapath_list
from util.model import SAGANDiscriminator, SAGANGenerator, weights_init
from util.train import train_sagan


def main():
    mean = (0.5,)
    std = (0.5,)
    batch_size = 64
    z_dim = 20
    epochs = 300
    torch.backends.cudnn.benchmark = True

    train_img_list = make_datapath_list("pytorch_advanced/gan_generation/data/img_78/")
    transform = ImageTransform(mean, std)
    train_dataset = GANImageDataset(train_img_list, transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    generator = SAGANGenerator(z_dim=20, image_size=64)
    discriminator = SAGANDiscriminator(image_size=64)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    print("ネットワーク初期化完了")

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(generator.parameters(), g_lr, betas=(beta1, beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), d_lr, betas=(beta1, beta2))

    g, d = train_sagan(
        generator, discriminator, train_dataloader, g_optimizer, d_optimizer, z_dim, epochs, device=device
    )

    fixed_z = torch.randn(batch_size, z_dim, 1, 1)
    fake_images, am1, am2 = g(fixed_z.to(device))
    real_images = next(iter(train_dataloader))

    _ = plt.figure(figsize=(15, 9))
    for i in range(0, 5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(real_images[i][0].cpu().detach().numpy(), "gray")

        plt.subplot(3, 5, 5 + i + 1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")

        plt.subplot(3, 5, 10 + i + 1)
        am = am1[i].view(16, 16, 16, 16)
        am = am[7][7]
        plt.imshow(am.cpu().detach().numpy(), "Reds")
    plt.show()


if __name__ == "__main__":
    main()
