from typing import cast

import torch
from PIL import Image
from pytorch_advanced.gan_generation.util.dataloader import GANImageDataset, ImageTransform, make_datapath_list
from torch.utils.data import DataLoader


def test_make_datapath_list():
    file_list = make_datapath_list()
    assert (file_list[0] == "../data/img_78/img_7_0.jpg") and (file_list[-1] == "../data/img_78/img_8_199.jpg")


def test_image_transform():
    mean = (0.5,)
    std = (0.5,)
    img = Image.open("pytorch_advanced/gan_generation/data/img_78/img_7_0.jpg")
    transform = ImageTransform(mean, std)
    img_transformed = transform(img)
    assert img_transformed.size() == torch.Size([1, 64, 64])


def test_gan_image_Dataset():
    mean = (0.5,)
    std = (0.5,)
    batch_size = 64

    file_list = make_datapath_list("pytorch_advanced/gan_generation/data/img_78/")
    dataset = GANImageDataset(file_list, ImageTransform(mean, std))
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    img = cast(torch.Tensor, next(iter(dataloader)))
    assert img.size() == torch.Size([64, 1, 64, 64])
