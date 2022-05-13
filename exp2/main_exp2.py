import numpy as np

from mlp import MLP


def main():
    network = MLP()
    x = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.int)
    y = np.array([1, 0, 0, 0], dtype=np.int)
    network.train(x, y, 0.01, 20)
    y_pred = network.predict([1, 1])
    print(f"the prediction of [1,1] is {y_pred}")


if __name__ == '__main__':
    main()
