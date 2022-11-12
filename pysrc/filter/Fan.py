import numpy as np
from pysrc.filter.Base import WeightCS


class Fan(WeightCS):
    def __init__(self, nmax, smooth_r1, smooth_r2):
        super(Fan, self).__init__(nmax)

        self.r1 = smooth_r1 * 1000
        self.r2 = smooth_r2 * 1000

        self.weight_matrix = self._getWeightMat()

    def _getWeightMat(self):
        matrix = np.ones((self.nmax + 1, self.nmax + 1))

        w1 = np.zeros(self.nmax + 1)
        b1 = np.log(2) / (1 - np.cos(self.r1 / self.radius_earth))
        w1[0] = 1
        w1[1] = (1 + np.exp(-2 * b1)) / (1 - np.exp(-2 * b1)) - 1 / b1
        for i in range(1, self.nmax):
            w1[i + 1] = -(w1[i] * (2 * i + 1)) / b1 + w1[i - 1]

        w2 = np.zeros(self.nmax + 1)
        b2 = np.log(2) / (1 - np.cos(self.r2 / self.radius_earth))
        w2[0] = 1
        w2[1] = (1 + np.exp(-2 * b2)) / (1 - np.exp(-2 * b2)) - 1 / b2
        for i in range(1, self.nmax):
            w2[i + 1] = -(w2[i] * (2 * i + 1)) / b2 + w2[i - 1]

        for n in range(self.nmax + 1):
            for m in range(n + 1):
                matrix[n][m] = w1[n] * w2[m]

        return matrix
