from math import inf

import numpy as np


class ProblemInstance:
    def __init__(self, min_soaking: np.ndarray,
                 max_soaking: np.ndarray,
                 full_move: np.ndarray,
                 empty_move: np.ndarray):
        self.a = np.concatenate((np.array([0]), min_soaking))
        self.max_soaking = np.concatenate((np.array([inf]), max_soaking))
        self.d = full_move
        self.c = empty_move.reshape(int(np.sqrt(empty_move.size)), -1)
        # This is necessary as the original array has the first row as the last + could be a flat array
        self.c[0, :], self.c[-1, :] = self.c[-1, :], self.c[0, :]
        if self.c[0, 0] != 0:
            raise Exception("First index of empty moves is not 0, the array is therefore wrong.")
