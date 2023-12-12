import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from unet import UNet
from model import DiffusionModel
from util import get_transforms, get_dataset, get_image_size, get_dataloader
import sys
from unet_power import UNet
import torch.nn.functional as F
from torch import nn

def run_latent_all_samples(unet, loss_fn, trainloader, diffusion_model, device, batch_size, record_timesteps):
    unet.eval()
    related_loss = { key:[] for key in record_timesteps }
    for timestep in tqdm(record_timesteps):
        for batch, _ in tqdm(trainloader, leave=False):
            t = torch.full((batch_size, ), timestep).to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device) 
            predicted_noise = unet(batch_noisy, t)
            loss = loss_fn(noise, predicted_noise)
            elementwise_loss = F.mse_loss(noise, predicted_noise, reduction='none')
            related_loss[timestep].extend(elementwise_loss.mean(dim=(1,2,3)).detach().cpu().tolist())
            # print(elementwise_loss.detach().cpu().tolist())
            # print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')

    torch.save(unet.record_latent_features, "weight/record_latent_features_power_cifar10_299.pt")
    torch.save(related_loss, "weight/record_latent_features_loss_power_cifar10_299.pt")

    print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')


def main():
    dataset_name = 'CIFAR10'
    batch_size = 100  # record each loss element, not mean
    timesteps = 1000
    device = 'cuda'

    image_size = get_image_size(dataset_name)
    transform, _ = get_transforms(image_size=image_size[1:])
    trainset, testset = get_dataset(dataset_name, transform)
    # trainloader, testloader = get_dataloader(trainset, testset, batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    record_timesteps = np.linspace(0, 1000, 10, endpoint=False).astype(int)
    unet = UNet(T=timesteps, ch=32, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=2, dropout=0.1, in_ch=image_size[0]).to('cuda')
    unet.load_state_dict(torch.load('weight/parameters_power.pkl'))
    diffusion_model = DiffusionModel(timesteps=timesteps)
    loss_fn = torch.nn.MSELoss()
    unet._record_latent_features(record_timesteps=record_timesteps)
    run_latent_all_samples(unet, loss_fn, trainloader, diffusion_model, device, batch_size, record_timesteps)


if __name__ == '__main__':
    main()