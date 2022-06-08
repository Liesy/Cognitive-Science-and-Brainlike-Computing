import logging
from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn import MyCNN

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


def vis(train_loss, test_loss, train_epochs_loss, test_epochs_loss, acc) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(train_loss[1:], '-o', label="train loss")
    plt.plot(test_loss[1:], '-o', label="test loss")
    plt.title("train/test loss for all")
    plt.subplot(132)
    plt.plot(train_epochs_loss[1:], '-o', label="train loss")
    plt.plot(test_epochs_loss[1:], '-o', label="test loss")
    plt.title("train/test loss for epochs")
    plt.subplot(133)
    plt.plot(acc)
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()


def main():
    train_dataloader, test_dataloader = load_data()

    model = MyCNN(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loss, test_loss = [], []
    train_epochs_loss, test_epochs_loss = [], []
    acc = []
    for epoch in range(args.epoch):
        # train
        logging.info("start training ...")
        model.train()
        train_epoch_loss = []
        for idx, (x, y) in enumerate(tqdm(train_dataloader), 1):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            if idx % 1000 == 1:
                logging.info(f"epoch={epoch}/{args.epoch}, "
                             f"{idx}/{len(train_dataloader)} of train, "
                             f"loss={loss.item():.4f}")
        train_epochs_loss.append(np.average(train_epoch_loss))

        # test
        logging.info("start testing ...")
        model.eval()
        test_epoch_loss = []
        for idx, (x, y) in enumerate(tqdm(test_dataloader), 1):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            pred_y = torch.max(output, 1)[1].data.squeeze()
            accuracy = np.sum(pred_y == y) / y.size(0)

            loss = criterion(output, y)
            test_epoch_loss.append(loss.item())
            test_loss.append(loss.item())
            acc.append(accuracy)

            if idx % 1000 == 1:
                logging.info(f"epoch={epoch}/{args.epoch}, "
                             f"{idx}/{len(test_dataloader)} of test, "
                             f"loss={loss.item():.4f}, "
                             f"accuracy={accuracy:.2f}")
        test_epochs_loss.append(np.average(test_epoch_loss))

        # adjust learning rate
        if epoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        logging.info(f"adjust learning rate to {optimizer.state_dict()['param_groups'][0]['lr']}")

    if args.vis:
        vis(train_loss, test_loss, train_epochs_loss, test_epochs_loss, acc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str, default='../')
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-vis', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    main()
