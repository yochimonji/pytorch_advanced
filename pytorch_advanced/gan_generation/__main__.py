import torch
from torch.utils.data import DataLoader

from util.dataloader import GANImageDataset, ImageTransform, make_datapath_list
from util.model import Discriminator, Generator, weights_init


def run_train():
    mean = (0.5,)
    std = (0.5,)
    batch_size = 64

    train_img_list = make_datapath_list("pytorch_advanced/gan_generation/data/img_78/")
    transform = ImageTransform(mean, std)
    train_dataset = GANImageDataset(train_img_list, transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    generator = Generator(z_dim=20, image_size=64)
    discriminator = Discriminator(image_size=64)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    print("ネットワーク初期化完了")


if __name__ == "__main__":
    run_train()
