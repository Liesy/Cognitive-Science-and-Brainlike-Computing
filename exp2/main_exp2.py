import numpy as np

from mlp import MLP


def main():
    network = MLP()

    x_and = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.int)
    y_and = np.array([1, 0, 0, 0], dtype=np.int)
    network.train(x_and, y_and, 0.01, 100)
    pred_and = network.predict([1, 1])
    print(f"the prediction of [1,1] with operation & is {pred_and}")

    x_or = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.int)
    y_or = np.array([1, 0, 1, 1], dtype=np.int)
    network.train(x_or, y_or, 0.01, 100)
    pred_or = network.predict([1, 1])
    print(f"the prediction of [1,1] with operation | is {pred_or}")

    x_not = np.array([[1], [0]], dtype=np.int)
    y_not = np.array([0, 1], dtype=np.int)
    network.train(x_not, y_not, 0.01, 100)
    pred_not = network.predict([1])
    print(f"the prediction of [1] with operation ~ is {pred_not}")


if __name__ == '__main__':
    main()
