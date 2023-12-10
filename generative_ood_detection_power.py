import wandb
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
# from unet import UNet
from model import DiffusionModel
from util import get_transforms, get_dataset, get_image_size, get_dataloader
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import pickle
from unet_power import UNet


def flatten_features(record_latent_Features):
    for key in record_latent_Features.keys():
        for i in range(len(record_latent_Features[key])):
            record_latent_Features[key][i] = record_latent_Features[key][i].flatten()


class pca_ood_detector:
    def __init__(self, record_latent_features):
        
        self.record_latent_features = record_latent_features
        # self.timesteps = self.record_latent_features.keys()
        self.timesteps = [key for key in self.record_latent_features]
        flatten_features(self.record_latent_features)

    def pca_analyze(self):

        self.pca_models = { key:None for key in self.record_latent_features.keys() }
        self.pca_projected_latents = { key:None for key in self.record_latent_features.keys() }
        tqmdr = tqdm(self.record_latent_features.keys(), desc='pca_analysis')
        
        for key in tqmdr:
            pca_model = PCA(n_components='mle')
            res = pca_model.fit_transform(self.record_latent_features[key])
            self.pca_models[key] = pca_model
            self.pca_projected_latents[key] = res

    def calc_avg_distance(self):    # setting threshold

        pca_projected_latents_dist = {}
        pca_projected_latents_stats = {}

        tqdmr = tqdm(self.pca_projected_latents.keys(), desc='calculate the stats of latent features for training samples')

        for key  in tqdmr:

            rand_ids = np.random.choice(len(self.pca_projected_latents[key]), size=10000, replace=False)
            vectors = np.array(self.pca_projected_latents[key])[rand_ids,:]
            
            # Compute squared distances
            dot_product = np.dot(vectors, vectors.T)
            squared_norms = np.sum(vectors**2, axis=1, keepdims=True)
            squared_distances = squared_norms + squared_norms.T - 2 * dot_product
            
            # Ensure distances are non-negative
            squared_distances = np.maximum(squared_distances, 0)
            
            # Take the square root to get L2 distance
            distances = np.sqrt(squared_distances)

            pca_projected_latents_dist[key] = distances
            pca_projected_latents_stats[key] = (np.mean(distances), np.std(distances))

        self.pca_projected_latents_dist = pca_projected_latents_dist
        self.pca_projected_latents_stats = pca_projected_latents_stats

    def set_threshold(self, sigma_threshold=5): # mean+3*std

        self.thresholds = { key:self.pca_projected_latents_stats[key][0]+sigma_threshold*self.pca_projected_latents_stats[key][1] 
                          for key in self.pca_projected_latents_stats.keys() }

    def detect_l2_distance_ood(self, latent, timestep):
        assert timestep in self.timesteps 
        projected_latent = self.pca_models[timestep].transform(latent)
        l2_distances = np.linalg.norm(self.pca_projected_latents[timestep] - projected_latent, axis=1)
        min_distance = np.min(l2_distances)
        if min_distance > self.thresholds[timestep]:
            print('ood sample!')
            return True, min_distance
        else:
            print('id sample!')
            return False, min_distance


def infer_ood(unet, diffusion_model, ood_detector, device, reverse_transform, n_imgs, image_size):

    detect_timesteps = np.linspace(0, 2000, 20, endpoint=False).astype(int)
    unet.eval()
    unet._start_ood_detection(ood_detector, detect_timesteps)
    
    infer_samples = []
    with torch.no_grad():
        for _ in trange(n_imgs):
            samples = []
            image = torch.randn((1, *(image_size))).to(device) * 2
            for i in reversed(range(diffusion_model.timesteps)):
                image = diffusion_model.backward(image, torch.full((1, ), i, dtype=torch.long, device=device), unet)
                if i % 50 == 0:
                    samples.append(reverse_transform(image).cpu())
                if i in detect_timesteps:
                    print(unet.ood_detect_res)
            infer_samples.append(samples)

    return infer_samples

def cla():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_imgs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps'])
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--thd', type=int, default=3)
    parser.add_argument('--record_latent_features', type=str)
    return parser.parse_args()

def main():

    torch.manual_seed(1)
    args = cla()

    image_size = get_image_size(args.dataset)
    _, reverse_transform = get_transforms(image_size=image_size[1:])

    unet = UNet(T=args.timesteps, ch=32, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=2, 
                 dropout=0.1, in_ch=image_size[0]).to(args.device)
    unet.load_state_dict(torch.load('weight/parameters_power.pkl'))
    diffusion_model = DiffusionModel(timesteps=args.timesteps)

    record_latent_features = torch.load(f'./weight/record_latent_features_power_{args.dataset.lower()}_{args.record_latent_features}.pt')

    detector = None
    if os.path.exists(f'./results/stored_detector_power_{args.dataset.lower()}_{args.record_latent_features}_{args.thd}.pkl'):
        with open(f'./results/stored_detector_power_{args.dataset.lower()}_{args.record_latent_features}_{args.thd}.pkl', 'rb') as file:
            detector = pickle.load(file)
    else:
        detector = pca_ood_detector(record_latent_features)
        detector.pca_analyze()
        detector.calc_avg_distance()
        detector.set_threshold(args.thd)
        with open(f'./results/stored_detector_power_{args.dataset.lower()}_{args.record_latent_features}_{args.thd}.pkl', 'wb') as file:
            pickle.dump(detector, file)
    
    infer_samples = infer_ood(unet, diffusion_model, detector, args.device, reverse_transform,args.n_imgs, image_size)
    ood_detect_result = unet.ood_detect_res

    torch.save((infer_samples, ood_detect_result), f'./results/test_ood_detection_power_{args.dataset.lower()}_{args.record_latent_features}_{args.thd}.pt')

if __name__ == '__main__':
    main()