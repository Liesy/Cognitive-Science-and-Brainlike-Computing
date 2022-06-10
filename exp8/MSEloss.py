from typing import Tuple, Union, Optional

import numpy as np

seed = 42
np.random.seed(seed)


def mean_square_error(W: Union[list, np.ndarray],
                      X: Union[list, np.ndarray],
                      y: Union[list, np.ndarray],
                      reg: Optional[float] = None) -> Tuple[float, np.ndarray]:
    """
    Calculate the MSE loss and return the derivative of W

    :param W: weight matrix with shape (D, C)
    :param X: input data matrix with shape (N,D)
    :param y: ground truth label with shape (N,)
    :param reg: regularization parameter.
    :return: MSE loss (loss) and derivative of W (dW) as the shape of W.
    """
    W = W if isinstance(W, np.ndarray) else np.array(W, dtype=np.float32)
    X = X if isinstance(X, np.ndarray) else np.array(X, dtype=np.float32)
    y = y if isinstance(y, np.ndarray) else np.array(y, dtype=np.float32)

    assert W.shape[0] == X.shape[1], "dim 0 of W and dim 1 of X must have the same size."
    assert X.shape[0] == y.shape[0], "X and y must have the same size in dimension 0."

    N, D = X.shape
    _, C = W.shape
    pred = np.dot(X, W)  # (N,D)*(D,C)->(N,C)
    loss = np.sum(np.square(pred - y)) / N
    if reg is not None:
        loss += reg * np.sum(np.square(W))

    dL = 2 * (pred - y) / N  # (N,C)
    dW = np.dot(X.T, dL)  # (D,N)*(N,C)->(D,C)

    return loss, dW
