import gzip
import numpy as np


class GeoMathKit:
    @staticmethod
    def getCoLatLoninRad(lat, lon):
        '''
        :param lat: geophysical coordinate in degree
        :param lon: geophysical coordinate in degree
        :return: Co-latitude and longitude in rad
        '''

        theta = (90. - lat) / 180. * np.pi
        phi = lon / 180. * np.pi

        return theta, phi

    @staticmethod
    def getLegendre(lat, Nmax: int, option=0):
        """
        get legendre function up to degree/order Nmax in Lat.
        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param Nmax:
        :param option:
        :return: 3d-ndarray, indexes stand for (co-lat[rad], degree l, order m)
        """

        if option != 0:
            lat = (90. - lat) / 180. * np.pi

        if type(lat) is np.ndarray:
            Nsize = np.size(lat)
        else:
            Nsize = 1

        Pnm = np.zeros((Nsize, Nmax + 1, Nmax + 1))
        Pnm[:, 0, 0] = 1
        Pnm[:, 1, 1] = np.sqrt(3) * np.sin(lat)

        '''For the diagonal element'''
        for n in range(2, Nmax + 1):
            Pnm[:, n, n] = np.sqrt((2 * n + 1) / (2 * n)) * np.sin(lat) * Pnm[:, n - 1, n - 1]

        for n in range(1, Nmax + 1):
            Pnm[:, n, n - 1] = np.sqrt(2 * n + 1) * np.cos(lat) * Pnm[:, n - 1, n - 1]

        for n in range(2, Nmax + 1):
            for m in range(n - 2, -1, -1):
                Pnm[:, n, m] = \
                    np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (2 * n - 1)) \
                    * np.cos(lat) * Pnm[:, n - 1, m] \
                    - np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (n - m - 1) * (n + m - 1) / (2 * n - 3)) \
                    * Pnm[:, n - 2, m]

        return Pnm
