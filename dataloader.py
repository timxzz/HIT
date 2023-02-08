import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, SVHN

import torchvision.utils

# transforms.ToTensor(): 
#
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if 
# the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, 
# RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
#
# transforms.Resize():
# will resample the image

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

binMNIST_transform=torchvision.transforms.Compose([
        transforms.ToTensor(),
        lambda x: torch.round(x),
    ])

ImageNet_transform=torchvision.transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
    ])

svhn_transform = transforms.Compose([
    transforms.ToTensor()
])

def load_data(dataset_name, batch_size, train_transform=None):

    if dataset_name == "CIFAR10":
        train_transform = img_transform if train_transform is None else train_transform
        train_dataset = CIFAR10(root='../data/CIFAR10', train=True, download=True, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CIFAR10(root='../data/CIFAR10', train=False, download=True, transform=img_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "MNIST":
        train_transform = img_transform if train_transform is None else train_transform
        train_dataset = MNIST(root='../data/MNIST', download=True, train=True, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MNIST(root='../data/MNIST', download=True, train=False, transform=img_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "BinaryMNIST":
        train_transform = binMNIST_transform if train_transform is None else train_transform
        train_dataset = MNIST(root='../data/MNIST', download=True, train=True, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MNIST(root='../data/MNIST', download=True, train=False, transform=binMNIST_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "SVHN":
        train_transform = svhn_transform if train_transform is None else train_transform
        train_dataset = SVHN(root='../data/SVHN', split='train', download=True, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = SVHN(root='../data/SVHN', split='test', download=True, transform=svhn_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "ImageNet":
        train_transform = ImageNet_transform if train_transform is None else train_transform
        train_dataset = ImageFolder(root='../data/ImageNet/train', transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = ImageFolder(root='../data/ImageNet/val', transform=ImageNet_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader