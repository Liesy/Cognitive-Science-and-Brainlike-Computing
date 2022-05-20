import numpy as np
from typing import List, Union


class Hebb:

    def __init__(self):
        self.__W = np.zeros((30, 30))

    def __reset(self):
        self.__W = np.zeros((30, 30))

    def __update(self, W):
        self.__W = W

    def train(self, x: Union[List, np.ndarray], iteration: int = 600, learning_rate: float = 0.2) -> None:
        self.__reset()
        x = x if isinstance(x, np.ndarray) else np.array(x, dtype=np.int16)
        W = self.__W
        for _ in range(iteration):
            for vec in x:
                W += learning_rate * np.dot(vec.T, vec)
        self.__update(W)

    def predict(self, x: Union[List, np.ndarray]) -> np.ndarray:
        x = x if isinstance(x, np.ndarray) else np.array(x)
        pred = np.dot(self.__W, x.T)
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred.astype(np.int16)
