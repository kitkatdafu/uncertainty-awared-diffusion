import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from model import DiffusionModel
from util import get_transforms
from visdom import Visdom


def infer(unet, diffusion_model, device, viz, T, reverse_transform, n):

    unet.train()
    sd = []
    with torch.no_grad():
        for _ in range(n):
            samples = []
            image = torch.randn((1, 1, 32, 32)).to(device)
            for i in reversed(range(diffusion_model.timesteps)):
                samples_at_step = []
                for _ in range(T):
                    image = diffusion_model.backward(image, torch.full((1, ), i, dtype=torch.long, device=device), unet)
                    samples_at_step.append(image)
                samples_at_step = torch.cat(samples_at_step, dim=0)
                mean_sample = samples_at_step.mean(dim=0)
                sd_sample = samples_at_step.std(dim=0).mean()
                if i % 50 == 0:
                    samples.append(reverse_transform(mean_sample).cpu())
            viz.images(samples)
            sd.append(sd_sample.cpu().item())
    viz.histogram(X=sd)    


def main():
    device = 'cuda'
    TIMESTEPS = 300
    IMAGE_SIZE = (28, 28)
    N = 2
    T = 2

    viz = Visdom()

    _, reverse_transform = get_transforms(image_size=IMAGE_SIZE)

    unet = UNet(input_channels=1, output_channels=1).to(device)
    unet.load_state_dict(torch.load('weight/parameters.pkl'))
    diffusion_model = DiffusionModel(timesteps=TIMESTEPS)

    infer(unet, diffusion_model, device, viz, T, reverse_transform, N)

    


if __name__ == '__main__':
    main()