import numpy as np
from pysrc.filter.Base import WeightCS


class IsotropyGaussian(WeightCS):
    def __init__(self, nmax, smooth_radius):
        super(IsotropyGaussian, self).__init__(nmax)

        self.smooth_radius = smooth_radius * 1000

        self.weight_matrix = self._getWeightMat()

    def _getWeightMat(self):
        w = np.zeros(self.nmax + 1)
        b = np.log(2) / (1 - np.cos(self.smooth_radius / self.radius_earth))
        w[0] = 1
        w[1] = (1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1 / b
        for i in range(1, self.nmax):
            w[i + 1] = -(w[i] * (2 * i + 1)) / b + w[i - 1]

        return np.array([[w[n] for i in range(self.nmax + 1)] for n in range(len(w))])
