import torch
import torchvision
import numpy as np
import urllib.request
from PIL import Image
from torchvision import transforms


def get_transforms(image_size):

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    )
    reverse_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: (x + 1) / 2),
            # transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            transforms.Lambda(lambda x: x * 255.0),
            # transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
            # transforms.ToPILImage(),
        ]
    )

    return transform, reverse_transform


def get_ramya():
    url = "https://ramyakv.github.io/RamyaVinayak2.jpg"
    filepath = "image.jpg"
    urllib.request.urlretrieve(url, filepath)
    image = Image.open(filepath)
    return image

    
def get_image_size(name):
    if name == 'MNIST':
        return (28, 28)
    else:
        raise ValueError('No such dataset')
    

def get_dataset(name, transform):
    if name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    else:
        raise ValueError('No such dataset')
    return trainset, testset

    
def get_dataloader(trainset, testset, batch_size):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    return trainloader, testloader