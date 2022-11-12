from abc import ABC, abstractmethod

import numpy as np
from pysrc.preference.Constants import GeoConstants


class SHCFilter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_to(self, Cqlm, Sqlm):
        pass


class WeightCS(SHCFilter):
    def __init__(self, nmax):
        super(WeightCS, self).__init__()
        self.nmax = nmax
        self.radius_earth = GeoConstants.radius_earth
        self.weight_matrix = None

    def apply_to(self, Cqlm, Sqlm=None):
        assert np.shape(Cqlm[0]) == np.shape(self.weight_matrix)
        Cqlm *= self.weight_matrix

        if Sqlm is not None:
            assert np.shape(Cqlm[0]) == np.shape(self.weight_matrix)
            Sqlm *= self.weight_matrix

            return Cqlm, Sqlm

        else:
            return Cqlm
