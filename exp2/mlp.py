import numpy as np


class MLP:
    def __init__(self):
        self.__W = None
        self.__b = None
        self.__output_layer = lambda x: 0 if x < 0 else 1

    def __reset(self, x: np.ndarray):
        self.__W = np.random.rand(x.__len__())
        self.__b = 0.0

    def __update(self, W, b):
        self.__W = W
        self.__b = b

    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float, iter: int) -> None:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.__len__() == y.__len__(), "x, y must have the same length."
        self.__reset(x[0])
        W, b = self.__W, self.__b
        for _ in range(iter):
            for x_data, label in zip(x, y):
                ret = np.sum(x_data * W + b)
                y_pred = self.__output_layer(ret)

                W += learning_rate * (label - y_pred) * x_data
                b += learning_rate * (label - y_pred)
        self.__update(W, b)

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        ret = np.sum(x * self.__W + self.__b)
        pred = self.__output_layer(ret)
        return pred
