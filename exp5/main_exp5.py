import logging
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hmax import HMax

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data() -> Tuple[DataLoader, DataLoader]:
    im_transform = transforms.Compose([
        transforms.ToTensor(),
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
    train_dataloader, test_dataloader = load_data()  # img:(64,1,28,28)  label:(64)
    model = HMax()

    it = iter(train_dataloader)
    sample, label = it.__next__()

    logging.info(f"the sample's shape is {sample.shape}")

    output = model(sample)
    logging.info(f"the output's shape is {output.shape}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str, default='../')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epoch', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    main()
