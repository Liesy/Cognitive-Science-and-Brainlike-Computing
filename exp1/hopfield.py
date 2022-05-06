from create_matrix import create_number_matrix
import numpy as np
from typing import Dict
from collections import defaultdict
from neupy.algorithms import DiscreteHopfieldNetwork
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
    for i in range(10):
        data[i] = mat[str(i)]
    return data


def fit_hopfield(data: np.ndarray) -> DiscreteHopfieldNetwork:
    dhnet = DiscreteHopfieldNetwork(mode="sync")
    dhnet.train(data)
    return dhnet


def test_hopefield(hopfield_network: DiscreteHopfieldNetwork, data: np.ndarray, idx: int) -> None:
    def display_matrix(number):  # 绘图
        number = number.reshape((6, 5))
        for row in number.tolist():
            print('| ' + ' '.join(' *'[val] for val in row))

    def add_noise(number):  # 加噪函数，在记忆样本的基础上增加30%的噪声
        for x in range(0, 30):
            if randint(0, 10) > 8:
                number[0, x] = randint(0, 1)
        return number

    print(f"origin number {idx}:")
    display_matrix(data)
    data_with_noise = add_noise(data)
    print(f"number {idx} afering adding noise:")
    display_matrix(data_with_noise)
    pred = hopfield_network.predict(data_with_noise)
    print(f"predicted by Hopfield Network for number {idx} with noise:")
    display_matrix(pred)
    print()


def main():
    numbers = load_data(args.mat_path)
    train_data = np.concatenate([numbers[i] for i in range(5)], axis=0)
    dhnet = fit_hopfield(train_data)
    for idx, test_data in numbers.items():
        test_hopefield(dhnet, test_data, idx)


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--mat_path", type=str, default="./")
    args = parse.parse_args()
    main()
