import torch
import numpy as np
from tqdm import tqdm
from unet import UNet
from model import DiffusionModel
from util import get_transforms, get_ramya
from visdom import Visdom


def main():
    device = 'cuda'
    image_size = (32, 32) 
    NO_EPOCHS = 1000
    LR = 0.001
    BATCH_SIZE = 256
    TIMESTEPS = 2000
    T = 5

    viz = Visdom()

    transform, reverse_transform = get_transforms(image_size)
    pil_image = get_ramya()
    torch_image = transform(pil_image).to(device)
    diffusion_model = DiffusionModel(timesteps=TIMESTEPS)
    unet = UNet().to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()

    win = None
    unet.train()
    for epoch in tqdm(range(NO_EPOCHS)):
        batch = torch.stack([torch_image] * BATCH_SIZE)
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE, )).long().to(device)
        noisy_image, gt_noise = diffusion_model.forward(batch, t, device)
        predicted_noise = unet(noisy_image, t)

        optimizer.zero_grad()
        loss = loss_fn(predicted_noise, gt_noise)
        loss.backward()
        optimizer.step()

        if not win:
            win = viz.line(X=[epoch], Y=[loss.item()], name='[Ramya] Training Loss')
        else:
            viz.line(X=[epoch], Y=[loss.item()], win=win, update='append')

    unet.eval()
    samples = []
    image = torch.randn((1, 3, 32, 32)).to(device)
    for i in reversed(range(diffusion_model.timesteps)):
        samples_at_step = []
        for _ in range(T):
            image = diffusion_model.backward(image, torch.full((1, ), i, dtype=torch.long, device=device), unet)
            samples_at_step.append(image)
        samples_at_step = torch.cat(samples_at_step, dim=0)
        mean_sample = samples_at_step.mean(dim=0)
        print(mean_sample.std(dim=0).mean())
        if i % 500 == 0:
            samples.append(reverse_transform(mean_sample).cpu())
    viz.images(samples)
    

if __name__ == '__main__':
    main()