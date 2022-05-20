import numpy as np
from scipy.io import savemat
from os import path


def create_number_matrix(save_path: str = "./") -> None:
    mdict = {
        '0': np.array([-1, 1, 1, 1, -1,
                       1, -1, -1, -1, 1,
                       1, -1, -1, -1, 1,
                       1, -1, -1, -1, 1,
                       1, -1, -1, -1, 1,
                       -1, 1, 1, 1, -1], dtype=np.int16),

        '1': np.array([-1, 1, 1, -1, -1,
                       -1, -1, 1, -1, -1,
                       -1, -1, 1, -1, -1,
                       -1, -1, 1, -1, -1,
                       -1, -1, 1, -1, -1,
                       -1, -1, 1, -1, -1], dtype=np.int16),

        '2': np.array([1, 1, 1, -1, -1,
                       -1, -1, -1, 1, -1,
                       -1, -1, -1, 1, -1,
                       -1, 1, 1, -1, -1,
                       1, -1, -1, -1, -1,
                       1, 1, 1, 1, 1], dtype=np.int16)
    }
    savemat(file_name=path.join(save_path, "number_metrix.mat"), mdict=mdict)
