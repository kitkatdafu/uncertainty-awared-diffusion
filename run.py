import wandb
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from model import DiffusionModel
from util import get_transforms, get_dataset, get_image_size, get_dataloader


def train(no_epochs, unet, optimizer, loss_fn, diffusion_model, trainloader, testloader, device, batch_size):

    for _ in tqdm(range(no_epochs)):
        unet.train()
        batch_loss = []
        for batch, _ in trainloader:
            t = torch.randint(0, diffusion_model.timesteps, (batch_size,)).long().to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device) 
            predicted_noise = unet(batch_noisy, t)
            optimizer.zero_grad()
            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()
        training_loss = np.mean(batch_loss)

        unet.eval()
        batch_loss = []
        for batch, _ in testloader:
            t = torch.randint(0, diffusion_model.timesteps, (batch_size,)).long().to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device) 
            predicted_noise = unet(batch_noisy, t)
            loss = loss_fn(noise, predicted_noise)
            batch_loss.append(loss.item())
        testing_loss = np.mean(batch_loss)

        wandb.log({'Training Loss': training_loss, 'Testing Loss': testing_loss})

        torch.save(unet.state_dict(), f"weight/parameters.pkl")


def cla():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps'])
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST'])
    return parser.parse_args()


def main():
    args = cla()
    image_size = get_image_size(args.dataset)
    transform, _ = get_transforms(image_size=image_size)
    trainset, testset = get_dataset(args.dataset, transform)
    trainloader, testloader = get_dataloader(trainset, testset, args.batch_size)

    wandb.init(
        project='uncertainty-awared-diffusion',
        config=args
    )
    
    unet = UNet(input_channels=1, output_channels=1).to(args.device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    diffusion_model = DiffusionModel(timesteps=args.timesteps)

    train(args.no_epochs, unet, optimizer, loss_fn, diffusion_model, trainloader, testloader, args.device, args.batch_size)

    wandb.finish()


if __name__ == '__main__':
    main()