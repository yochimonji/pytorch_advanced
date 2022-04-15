import warnings
from typing import cast

import torch
from PIL import Image
from torch.utils.data import Dataset

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from torchvision import transforms


def make_datapath_list(dir_path: str = "../data/img_78/"):
    train_img_list: list[str] = []
    for img_idx in range(200):
        img_path = dir_path + "img_7_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)
        img_path = dir_path + "img_8_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)
    return train_img_list


class ImageTransform:
    data_transforms: transforms.Compose

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]):
        self.data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, img: Image) -> torch.Tensor:
        return self.data_transforms(img)


class GANImageDataset(Dataset):
    file_list: list[str]
    transform: ImageTransform

    def __init__(self, file_list: list[str], transform: ImageTransform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return cast(torch.Tensor, img_transformed)
