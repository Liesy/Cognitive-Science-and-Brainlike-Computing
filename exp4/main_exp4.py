import logging
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from lstm import MyLSTM

seed = 42
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_one_data():
    x_int = np.random.randint(0, 128, (2, 1))
    y_int = np.expand_dims(x_int.sum(), axis=0)
    x_y = np.unpackbits(np.concatenate([x_int, [y_int]]).astype(np.uint8), axis=1).astype(np.float32)
    # 注意dtype，其他类型会触发GPU错误
    x_bi = torch.from_numpy(x_y[:2]).to(device)  # (2,8)
    y_bi = torch.from_numpy(x_y[-1:]).to(device)  # (1,8)
    return torch.from_numpy(x_int).to(device), torch.from_numpy(y_int).to(device), x_bi, y_bi


def create_train_data(capacity: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    train_data = list()
    for i in range(capacity):
        _, _, data_x, data_y = create_one_data()
        train_data.append((data_x, data_y))
    logging.info(f"{len(train_data)} train data created.")
    return train_data


def main():
    model = MyLSTM(input_size=8, hidden_size=16, output_size=8, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_loss = []

    for epoch in range(args.epoch):
        model.train()
        train_data = create_train_data(1000)
        for idx, (data_x, data_y) in enumerate(train_data):
            data_x = torch.unsqueeze(data_x, dim=0)  # (1,2,8)
            data_y = torch.unsqueeze(data_y, dim=0)  # (1,1,8)

            optimizer.zero_grad()

            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if idx % 100 == 0:
                info = f"epoch={epoch + 1}/{args.epoch}, {idx}/{len(train_data)} of train, loss={loss.item():.2f}"
                # logging.info(info)
                print(info)

    # torch.save(model.state_dict(), 'model_checkpoint.pth')

    model.eval()
    for i in range(5):
        x_int, y_int, test_x, test_y = create_one_data()
        pred = model(torch.unsqueeze(test_x, dim=0)).squeeze(dim=0)

        x_int = x_int.detach().cpu().numpy().astype(np.int).tolist()
        y_int = y_int.detach().cpu().numpy().astype(np.int).tolist()
        test_x = test_x.detach().cpu().numpy().astype(np.int).tolist()
        test_y = test_y.detach().cpu().numpy().astype(np.int).tolist()

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = torch.squeeze(pred, dim=0).detach().cpu().numpy().astype(np.int).tolist()

        print(f"\na:{x_int[0]}\n{test_x[0]}\nb:{x_int[1]}\n{test_x[1]}")
        print(f"a+b  pred:{pred}")
        print(f"a+b truth:{test_y[0]} {y_int[0]}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.07)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    main()
