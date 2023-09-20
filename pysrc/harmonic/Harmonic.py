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

        single_data_flag = False
        if Gqij.ndim == 2:
            single_data_flag = True
            Gqij = GeoMathKit.getCSGridin3d(Gqij)

        g = self.g.T
        co = np.cos(g)
        so = np.sin(g)

        Am = np.einsum('pij,jm->pim', Gqij, co, optimize='greedy') * self.factor1
        Bm = np.einsum('pij,jm->pim', Gqij, so, optimize='greedy') * self.factor1

        cqlm = np.einsum('pim,ilm,i->plm', Am, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2
        sqlm = np.einsum('pim,ilm,i->plm', Bm, self.PnmMatrix, np.sin(self.lat), optimize='greedy') * self.factor2

        if single_data_flag:
            cqlm, sqlm = cqlm[0], sqlm[0]
        return cqlm, sqlm

    def synthesis(self, cqlm: np.ndarray, sqlm: np.ndarray):
        assert len(cqlm) == len(sqlm)

        single_data_flag = False
        if cqlm.ndim == 2:
            single_data_flag = True
            cqlm, sqlm = GeoMathKit.getCSGridin3d(cqlm, sqlm)

        Am = np.einsum('ijk,ljk->ilk', cqlm, self.PnmMatrix)
        Bm = np.einsum('ijk,ljk->ilk', sqlm, self.PnmMatrix)

        co = np.cos(self.g)
        so = np.sin(self.g)

        Fout = Am @ co + Bm @ so

        if single_data_flag:
            Fout = Fout[0]
        return Fout
