import torch
import pickle
import argparse
from collections import defaultdict
from util import get_image_size
from tqdm import tqdm
from unet import UNet
from model import DiffusionModel
from util import get_transforms


def infer(unet, diffusion_model, device, T, reverse_transform, n):


    _t = None
    i = None
    

    bottleneck = defaultdict(list)

    unet.eval()
    unet.bottleneck.register_forward_hook(lambda x, y, output: bottleneck[_t, i].append(output.cpu().numpy()))

    samples = defaultdict(list)

    with torch.no_grad():
        image = torch.randn((n, 1, 28, 28)).to(device)

        for _t in range(T):
            torch.manual_seed(_t)
            for i in reversed(range(diffusion_model.timesteps)):
                if i >= diffusion_model.timesteps - 0.1 * diffusion_model.timesteps:
                    for dropout in unet.dropouts:
                        dropout.train() 
                else:
                    for dropout in unet.dropouts:
                        dropout.eval() 
                image = diffusion_model.backward(image, torch.full((1, ), i, dtype=torch.long, device=device), unet) + 0.001
                samples[i].append(image.cpu().numpy())
    
    with open('pickels/results.pkl', 'wb') as f:
        pickle.dump(samples, f)

    with open('pickels/bottlenecks.pkl', 'wb') as f:
        pickle.dump(bottleneck, f)


            
def cla():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST'])
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--N', type=int, default=1, help='number of images generated')
    parser.add_argument('--T', type=int, default=10, help='number of resamples')
    return parser.parse_args()



def main():
    torch.manual_seed(1)
    args = cla()
    image_size = get_image_size(args.dataset)

    _, reverse_transform = get_transforms(image_size=image_size)

    unet = UNet(input_channels=1, output_channels=1).to(args.device)
    unet.load_state_dict(torch.load('weights/parameters.pkl'))
    diffusion_model = DiffusionModel(timesteps=args.timesteps)

    infer(unet, diffusion_model, args.device, args.T, reverse_transform, args.N)

    


if __name__ == '__main__':
    main()