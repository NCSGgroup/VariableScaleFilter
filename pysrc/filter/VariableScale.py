import numpy as np

from pysrc.filter.Base import SHCFilter
from pysrc.harmonic.Harmonic import Harmonic
from pysrc.preference.Constants import GeoConstants
from pysrc.preference.Enumclasses import VaryRadiusWay


def getPsi(sp, lat, lon):
    """
    lat, lon in unit[rad].
    """
    Psi = np.zeros((int(180 / sp), int(360 / sp), 2))

    colat = np.pi / 2 - lat

    colat_pie = np.pi / 2 - np.radians(np.arange(-89.5, 90.5, sp))
    delta_lat = colat_pie - colat

    lon_pie = np.radians(np.arange(-180, 180, sp))
    delta_lon = lon - lon_pie

    colat_pie, lon_pie = np.meshgrid(colat_pie, lon_pie)

    delta_lat, delta_lon = np.meshgrid(delta_lat, delta_lon)

    Psi[:, :, 0] = (np.sin(delta_lon / 2) * np.sin((colat_pie + colat) / 2)).T
    Psi[:, :, 1] = (np.sin(delta_lat / 2)).T

    return Psi


class VariableScale():
    """
    This class is to smooth grids by applying a spatial convolution with a variable-scale anisotropy Gaussian kernel.
    """

    def __init__(self, r_min, r_max=None, sigma=None, harmonic: Harmonic = None,
                 vary_radius_mode: VaryRadiusWay = VaryRadiusWay.sin2):

        if r_max is None:
            r_max = r_min

        if sigma is None:
            sigma = np.mat([[1, 0], [0, 1]])

        self.r_min = r_min * 1000
        self.r_max = r_max * 1000
        self.sigma = sigma

        self.vary_way = vary_radius_mode

        self.harmonic = harmonic

        self.radius_e = GeoConstants.radius_earth

    def get_kernel_at_one_point(self, sp, lat, lon):
        """

        :param sp:
        :param lat: [rad]
        :param lon: [rad]
        :return:
        """

        Psi = getPsi(sp, lat, lon)

        if self.vary_way == VaryRadiusWay.sin2:
            r_lat = (self.r_max - self.r_min) * (np.sin(np.pi / 2 - lat)) ** 2 + self.r_min
        else:
            r_lat = (self.r_max - self.r_min) * np.sin(np.pi / 2 - lat) + self.r_min

        alpha_0 = r_lat / self.radius_e

        a = np.log(2) / (1 - np.cos(alpha_0))

        PsiT_SigmaI_Psi = np.einsum('ijl,lm,ijm->ij', Psi, np.linalg.inv(self.sigma), Psi)

        weight = a * np.exp(-a * (2 * PsiT_SigmaI_Psi)) / (
                (1 - np.exp(-2 * a)) * 2 * np.pi * np.sqrt(np.linalg.det(self.sigma)))

        return weight

    def apply_to(self, *params, option=0):
        """

        :param params: SHC Cqlm and Sqlm if option=0, else grids.
        :param option:
        :return:
        """
        params = np.array(params)
        if option == 0:
            assert len(params) == 2
            assert self.harmonic is not None

            Gqij = self.harmonic.synthesis(*params)

        else:
            assert len(params) == 1
            Gqij = params[0]

        grid_space = 180 / np.shape(Gqij)[1]

        lat = np.radians(np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space))
        lon = np.radians(np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space))

        theta = np.radians(180 - np.arange(np.shape(Gqij)[1]) * grid_space)
        d_sigma = np.radians(grid_space) ** 2 * np.sin(theta)

        length_of_lat = len(lat)
        length_of_lon = len(lon)
        Wipq = np.zeros((length_of_lat, length_of_lat, length_of_lon))

        Gqij_filtered = np.zeros_like(Gqij)

        for i in range(length_of_lat):
            this_lat = lat[i]
            Wpq = self.get_kernel_at_one_point(grid_space, this_lat, lon[0])

            Wipq[i] = Wpq

        for j in range(length_of_lon):
            print('\rvs filtering {}/{}'.format(j + 1, length_of_lon), end='\t')

            Gqij_pie = np.zeros_like(Gqij)
            Gqij_pie[:, :, :length_of_lon - j], Gqij_pie[:, :, length_of_lon - j:] = Gqij[:, :, j:], Gqij[:, :, :j]
            Gqij_filtered[:, :, j] = np.einsum('rij,qij->qr', Wipq, Gqij_pie) * d_sigma
        print('done!')

        if option == 0:
            return self.harmonic.analysis(Gqij_filtered)

        else:
            return Gqij_filtered
