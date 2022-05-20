from create_matrix import create_number_matrix
from hebb import Hebb
import numpy as np
from typing import Dict
from collections import defaultdict
from argparse import ArgumentParser
from scipy.io import loadmat
from os import path
from random import randint


def load_data(mat_path: str) -> Dict[int, np.ndarray]:
    file = path.join(mat_path, "number_metrix.mat")
    if not path.exists(file):
        create_number_matrix(mat_path)
    mat = loadmat(file)
    data = defaultdict(np.ndarray)
    for i in range(3):
        data[i] = mat[str(i)]
    return data


def display_matrix(number: np.ndarray) -> None:  # 绘图
    number = np.clip(number, 0, 1)
    number = number.reshape((6, 5))
    for row in number.tolist():
        print('| ' + ' '.join(' *'[val] for val in row))


def add_noise(number: np.ndarray) -> np.ndarray:  # 加噪函数，在记忆样本的基础上增加30%的噪声
    for x in range(0, 30):
        if randint(0, 10) > 8:
            number[0, x] = -1
        if randint(0, 10) < 2:
            number[0, x] = 1
    return number


def main():
    numbers = load_data(args.mat_path)

    train_data = np.concatenate([np.expand_dims(x, axis=0) for _, x in numbers.items()], axis=0)  # (3,1,30)
    network = Hebb()
    network.train(train_data)

    for idx, test_data in numbers.items():
        print(f"origin number {idx}:")
        display_matrix(test_data)

        test_data = add_noise(test_data)
        print(f"number {idx} after adding noise:")
        display_matrix(test_data)

        pred = network.predict(test_data)
        print(f"predicted by Hebb for number {idx} with noise:")
        display_matrix(pred)

        print()


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--mat_path", type=str, default="./")
    args = parse.parse_args()
    main()
