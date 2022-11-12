import time

import numpy as np

from pysrc.auxiliary.GeoMathKit import GeoMathKit


class Harmonic:
    """
    Harmonic analysis and synthesis: Ordinary 2D integration for computing Spherical Harmonic coefficients
    """

    def __init__(self, lat, lon, Pilm, Nmax: int, option=0):
        """

        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lon: If option=0, unit[rad]; else unit[degree]
        :param Pilm: Associative Legendre polynomials, indexes stand for (co-lat[rad], degree l, order m)
        :param Nmax:
        :param option:
        """
        if option != 0:
            self.lat, self.lon = GeoMathKit.getCoLatLoninRad(lat, lon)

        self.nmax = Nmax

        self.nlat, self.nlon = len(lat), len(lon)
        self.PnmMatrix = Pilm

        m = np.arange(Nmax + 1)
        self.g = m[:, None] @ self.lon[None, :]

        self.factor1 = np.ones((self.nlat, Nmax + 1))
        self.factor1[:, 0] += 1
        self.factor1 = 1 / (self.factor1 * self.nlon)

        self.factor2 = np.ones((Nmax + 1, Nmax + 1))
        self.factor2[:, 0] += 1
        self.factor2 *= np.pi / (2 * self.nlat)
        pass

    def analysis(self, Gqij: np.ndarray):
        g = self.g.T
        co = np.cos(g)
        so = np.sin(g)

        Am = np.einsum('pij,jm->pim', Gqij, co, optimize='greedy') * self.factor1
        Bm = np.einsum('pij,jm->pim', Gqij, so, optimize='greedy') * self.factor1

        Cnms = np.einsum('pim,ilm,i->plm', Am, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2
        Snms = np.einsum('pim,ilm,i->plm', Bm, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2
        return Cnms, Snms

    def analysis3d(self, Inners: np.ndarray):
        assert np.shape(Inners)[0] >= self.nmax

        Cqlm, Sqlm = self.analysis(Inners)

        Clm = np.zeros((self.nmax + 1, self.nmax + 1))
        Slm = np.zeros((self.nmax + 1, self.nmax + 1))

        for l in range(self.nmax + 1):
            Clm[l, :l + 1] = Cqlm[l, l, :l + 1]
            Slm[l, :l + 1] = Sqlm[l, l, :l + 1]

        return Clm, Slm

    def analysis3d_new(self, Inners: np.ndarray):
        """

        :param Inners: 4-d array(qpij), d1(q) for the amount of data(such as months), d2(p) for the layers(>=l),
        d3(i) and d4(j) for the grid index, i.e. co-latitude index and longitude index.
        :return: 3-d array(plm)
        """
        assert np.shape(Inners)[1] >= self.nmax

        # Cqlm, Sqlm = self.analysis(Inners)
        g = self.g.T
        co = np.cos(g)
        so = np.sin(g)

        Am = np.einsum('qpij,jm->qpim', Inners, co, optimize='greedy') * self.factor1
        Bm = np.einsum('qpij,jm->qpim', Inners, so, optimize='greedy') * self.factor1

        Cqplm = np.einsum('qpim,ilm,i->qplm', Am, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2
        Sqplm = np.einsum('qpim,ilm,i->qplm', Bm, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2

        Cqlm = np.zeros((np.shape(Inners)[0], self.nmax + 1, self.nmax + 1))
        Sqlm = np.zeros((np.shape(Inners)[0], self.nmax + 1, self.nmax + 1))

        for l in range(self.nmax + 1):
            Cqlm[:, l, :l + 1] = Cqplm[:, l, l, :l + 1]
            Sqlm[:, l, :l + 1] = Sqplm[:, l, l, :l + 1]

        return Cqlm, Sqlm

    def synthesis(self, Cnms: iter, Snms: iter):
        assert len(Cnms) == len(Snms)
        Cnms = np.array(Cnms)
        Snms = np.array(Snms)

        Am = np.einsum('ijk,ljk->ilk', Cnms, self.PnmMatrix)
        Bm = np.einsum('ijk,ljk->ilk', Snms, self.PnmMatrix)

        co = np.cos(self.g)
        so = np.sin(self.g)

        Fout = Am @ co + Bm @ so

        return Fout
