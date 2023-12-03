import numpy as np
import torch
import math
import os

class unet_ood_detect:
    def __init__(self, record_latent_features, threshold):
        # record_latent_features: (record_timesteps, latent_features.shape)

        self.record_latent_features = record_latent_features
        self.threshold = threshold
    
    def calc_dis_each_timestep(self):
        # determine the threshold
        ...

    def calc_nearest_distance(self, latent, timestep):
        ...