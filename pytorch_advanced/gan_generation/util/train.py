import time
from typing import cast

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_dcgan_one_epoch(
    g: Module,
    d: Module,
    dataloader: DataLoader,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    criterion: Module,
    z_dim=20,
    device=torch.device("cpu"),
) -> tuple[float, float]:
    epoch_g_loss: float = 0.0
    epoch_d_loss: float = 0.0

    for images in tqdm(dataloader):
        # Discriminatorの学習
        images = cast(torch.Tensor, images)

        if images.size(0) == 1:
            continue

        images = images.to(device)

        mini_batch_size = images.size(0)
        label_real = torch.ones(mini_batch_size).to(device)
        label_fake = torch.zeros(mini_batch_size).to(device)

        d_out_real = d(images)
        d_out_real = cast(torch.Tensor, d_out_real)

        input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
        fake_images = g(input_z)
        d_out_fake = d(fake_images)
        d_out_fake = cast(torch.Tensor, d_out_fake)

        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Generatorの学習
        input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
        fake_images = g(input_z)
        d_out_fake = d(fake_images)
        d_out_fake = cast(torch.Tensor, d_out_fake)

        g_loss = criterion(d_out_fake.view(-1), label_real)

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_d_loss += cast(torch.Tensor, d_loss).cpu().item()
        epoch_g_loss += cast(torch.Tensor, g_loss).cpu().item()

    batch_size = dataloader.batch_size
    if batch_size:
        mean_g_loss = epoch_g_loss / batch_size
        mean_d_loss = epoch_d_loss / batch_size
    else:
        mean_g_loss = epoch_g_loss
        mean_d_loss = epoch_d_loss
    return mean_g_loss, mean_d_loss


def train_dcgan(
    g: Module,
    d: Module,
    dataloader: DataLoader,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    criterion: Module,
    z_dim=20,
    epochs=200,
    device=torch.device("cpu"),
) -> tuple[Module, Module]:
    g.to(device)
    d.to(device)
    g.train()
    d.train()

    for epoch in range(epochs):
        t_epoch_start = time.time()

        print("---------------")
        print(f"Epoch {epoch+1}/{epochs}")
        print("---------------")

        g_loss, d_loss = train_dcgan_one_epoch(g, d, dataloader, g_optimizer, d_optimizer, criterion, z_dim, device)

        t_epoch_finish = time.time()
        print("---------------")
        print(f"Epoch_G_Loss:{g_loss:.4f} ||Epoch_D_Loss:{d_loss:.4f} ||timer:{t_epoch_finish - t_epoch_start:.4f}")
    return g, d


def train_sagan_one_epoch(
    g: Module,
    d: Module,
    dataloader: DataLoader,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    z_dim=20,
    device=torch.device("cpu"),
) -> tuple[float, float]:
    epoch_g_loss: float = 0.0
    epoch_d_loss: float = 0.0

    for images in tqdm(dataloader):
        # Discriminatorの学習
        images = cast(torch.Tensor, images)

        if images.size(0) == 1:
            continue

        images = images.to(device)

        mini_batch_size = images.size(0)

        d_out_real, _, _ = d(images)
        d_out_real = cast(torch.Tensor, d_out_real)

        input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
        fake_images, _, _ = g(input_z)
        d_out_fake, _, _ = d(fake_images)
        d_out_fake = cast(torch.Tensor, d_out_fake)

        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        d_loss = d_loss_real + d_loss_fake

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Generatorの学習
        input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
        fake_images, _, _ = g(input_z)
        d_out_fake, _, _ = d(fake_images)
        d_out_fake = cast(torch.Tensor, d_out_fake)

        g_loss = -d_out_fake.mean()

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_d_loss += cast(torch.Tensor, d_loss).cpu().item()
        epoch_g_loss += cast(torch.Tensor, g_loss).cpu().item()

    batch_size = dataloader.batch_size
    if batch_size:
        mean_g_loss = epoch_g_loss / batch_size
        mean_d_loss = epoch_d_loss / batch_size
    else:
        mean_g_loss = epoch_g_loss
        mean_d_loss = epoch_d_loss
    return mean_g_loss, mean_d_loss


def train_sagan(
    g: Module,
    d: Module,
    dataloader: DataLoader,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    z_dim=20,
    epochs=200,
    device=torch.device("cpu"),
) -> tuple[Module, Module]:
    g.to(device)
    d.to(device)
    g.train()
    d.train()

    for epoch in range(epochs):
        t_epoch_start = time.time()

        print("---------------")
        print(f"Epoch {epoch+1}/{epochs}")
        print("---------------")

        g_loss, d_loss = train_sagan_one_epoch(g, d, dataloader, g_optimizer, d_optimizer, z_dim, device)

        t_epoch_finish = time.time()
        print("---------------")
        print(f"Epoch_G_Loss:{g_loss:.4f} ||Epoch_D_Loss:{d_loss:.4f} ||timer:{t_epoch_finish - t_epoch_start:.4f}")
    return g, d
