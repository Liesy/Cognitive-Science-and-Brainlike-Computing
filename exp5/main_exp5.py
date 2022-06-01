import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2 as cv
import numpy as np
from argparse import ArgumentParser
from os import path
from typing import Tuple

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data() -> Tuple[DataLoader, DataLoader]:
    im_transform = transforms.Compose([
        transforms.Totensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])
    # Dataset
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   transform=im_transform,
                                   download=True)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  transform=im_transform)
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    return train_loader, test_loader


def main():
    ...


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str, default='./')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epoch', type=int, default=1000)
    args = parser.parse_args()

    main()
