import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from model import DiffusionModel
from util import get_transforms, get_dataset, get_image_size, get_dataloader
import sys


def run_latent_all_samples(unet, loss_fn, trainloader, diffusion_model, device, batch_size, record_timesteps):
    unet.eval()
    related_loss = { key:[] for key in record_timesteps }
    for timestep in record_timesteps:
        for batch, _ in trainloader:
            t = torch.full((batch_size, ), timestep).to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device) 
            predicted_noise = unet(batch_noisy, t)
            loss = loss_fn(noise, predicted_noise)
            related_loss[timestep].append(loss.item())
            print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')

    torch.save(unet.record_latent_features, "weight/record_latent_features.pt")
    torch.save(related_loss, "weight/record_latent_features_loss.pt")

    print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')


def main():
    dataset_name = 'MNIST'
    batch_size = 1  # record each loss element, not mean
    timesteps = 300

    image_size = get_image_size(dataset_name)
    transform, _ = get_transforms(image_size=image_size)
    trainset, testset = get_dataset(dataset_name, transform)
    # trainloader, testloader = get_dataloader(trainset, testset, batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    record_timesteps = (0,1,5,10,25,50,100,200,299)
    unet = UNet(input_channels=1, output_channels=1, record_latent=True).to('cuda')
    diffusion_model = DiffusionModel(timesteps=timesteps)
    loss_fn = torch.nn.MSELoss()
    unet._record_latent_features(record_timesteps=record_timesteps)
    run_latent_all_samples(unet, loss_fn, trainloader, diffusion_model, 'cuda', batch_size, record_timesteps)


if __name__ == '__main__':
    main()