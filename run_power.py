import wandb
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


def train(no_epochs, unet, optimizer, loss_fn, diffusion_model, trainloader, testloader, device, batch_size, scheduler, reverse_transform, image_size):

    iter = len(trainloader)
    for epoch in tqdm(range(no_epochs)):
        if True:
            unet.train()
            batch_loss = []
            for i, (batch, _) in enumerate(trainloader):
                t = torch.randint(0, diffusion_model.timesteps, (batch_size,)).long().to(device)
                batch = batch.to(device)
                batch_noisy, noise = diffusion_model.forward(batch, t, device) 
                predicted_noise = unet(batch_noisy, t)
                optimizer.zero_grad()
                loss = loss_fn(noise, predicted_noise)
                loss.backward()
                batch_loss.append(loss.item())
                optimizer.step()
                scheduler.step(epoch + i / iter)
            training_loss = np.mean(batch_loss)

            unet.eval()
            batch_loss = []
            with torch.no_grad():
                for batch, _ in testloader:
                    t = torch.randint(0, diffusion_model.timesteps, (batch_size,)).long().to(device)
                    batch = batch.to(device)
                    batch_noisy, noise = diffusion_model.forward(batch, t, device) 
                    predicted_noise = unet(batch_noisy, t)
                    loss = loss_fn(noise, predicted_noise)
                    batch_loss.append(loss.item())
            testing_loss = np.mean(batch_loss)

            wandb.log({'Training Loss': training_loss, 'Testing Loss': testing_loss})

        if epoch % 10 == 0:
            with torch.no_grad():
                image = torch.randn(16, *image_size).to(device)
                for i in reversed(range(diffusion_model.timesteps)):
                    image = diffusion_model.backward(image, torch.full((1, ), i, dtype=torch.long, device=device), unet)
                image = torch.permute(reverse_transform(image).to(torch.uint8), (0, 2, 3, 1)).cpu().numpy()
                big_image = np.zeros((image_size[1] * 4, image_size[2] * 4, 3), dtype=np.uint8)
                for row in range(4):
                    for col in range(4):
                        _id = row * 4 + col
                        big_image[row * image_size[1]:((row + 1)  * image_size[1]), (col * image_size[2]): ((col + 1) * image_size[2])] = image[_id]

                big_image = wandb.Image(big_image, caption=f"Generate image {i}, epoch {epoch}")
                wandb.log({f"generated images": big_image})

        torch.save(unet.state_dict(), f"weight/parameters_power.pkl")


def run_latent_all_samples(unet, trainloader, diffusion_model, device, batch_size, record_timesteps):
    unet.eval()
    loss_fn = torch.nn.MSELoss()
    for timestep in record_timesteps:
        for batch, _ in trainloader:
            t = torch.full((batch_size, ), timestep).to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device) 
            predicted_noise = unet(batch_noisy, t)
            loss = loss_fn(noise, predicted_noise)
            print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')
        ...
    torch.save(unet.record_latent_features, "weight/record_latent_features_power.pt")
    print(f'length of record_latent_features: {np.sum([len(unet.record_latent_features[key]) for key in unet.record_latent_features.keys()])}')


def cla():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps'])
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no_epochs', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    return parser.parse_args()


def main():
    args = cla()
    image_size = get_image_size(args.dataset)
    transform, reverse_transform = get_transforms(image_size=image_size[1:])
    trainset, testset = get_dataset(args.dataset, transform)
    trainloader, testloader = get_dataloader(trainset, testset, args.batch_size)

    wandb.init(
        project='uncertainty-awared-diffusion',
        config=args
    )
    
    # unet = UNet(input_channels=1, output_channels=1).to(args.device)
    unet = UNet(T=args.timesteps, ch=32, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=2, 
                 dropout=0.1, in_ch=image_size[0]).to(args.device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, T_mult=2)
    loss_fn = torch.nn.MSELoss()
    diffusion_model = DiffusionModel(timesteps=args.timesteps)

    train(args.no_epochs, unet, optimizer, loss_fn, diffusion_model, trainloader, testloader, args.device, args.batch_size, scheduler, reverse_transform, image_size)

    wandb.finish()


if __name__ == '__main__':
    main()