from argparse import ArgumentParser

import numpy as np

from MSEloss import mean_square_error

seed = 42
np.random.seed(seed)


def main():
    X = [[0.35, 0.9]]  # N=1,D=2
    y = [[0.5]]  # C=1

    W = np.random.randn(2, 1)  # D=2,C=1
    origin_ret = np.dot(X, W)
    print(f"result without bp is {origin_ret.item():.4f}")

    for epoch in range(1, args.epoch + 1):
        loss, dW = mean_square_error(W, X, y)
        W -= args.lr * dW
        pred = np.dot(X, W)
        err = y - pred
        print(f"epoch={epoch:>2d}/{args.epoch}, "
              f"loss={loss:.4f}, "
              f"result after bp is {pred.item():.4f}, "
              f"err is {err.item():.4f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.2)
    args = parser.parse_args()

    main()
