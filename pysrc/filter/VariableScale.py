from inspect import isfunction
import numpy as np

from auxiliary.GeoMathKit import GeoMathKit
from harmonic.Harmonic import Harmonic
from preference.Constants import GeoConstants


class VariableScale:
    """
    This class is to smooth grids by applying a spatial convolution with a variable-scale non-isotropy Gaussian kernel.
    """

    def __init__(self, r_function, sigma=None, harmonic: Harmonic = None):
        """

        :param r_function: a function to explain the relationship between smoothing radius [m] and latitude [degree], that is, input a parameter latitude and return the corresponding smoothing radius.
        :param sigma: relative standard deviation of Gaussian distribution in the direction of latitude (that in the longitude is supposed to be 1), generally between 0 and 1.
        :param harmonic: harmonic tool define in ../harmonic/Harmonic.py, necessary if smoothed object in the spherical harmonic form.
        """
        assert isfunction(r_function)

        if sigma is None:
            sigma = 1

        self.r_function = r_function
        self.SigmaMat = np.diag((1, sigma ** 2))

        self.harmonic = harmonic

        self.radius_e = GeoConstants.radius_earth

    def get_kernel_at_one_point(self, sp, lat, lon):
        """

        :param sp: grid space, unit [degree]
        :param lat: unit [rad]
        :param lon: unit [rad]
        :return:
        """

        Psi = self.getPsi(sp, lat, lon)

        r_lat = self.r_function(lat)

        alpha_0 = r_lat / self.radius_e

        a = np.log(2) / (1 - np.cos(alpha_0))

        PsiT_SigmaI_Psi = np.einsum('ijl,lm,ijm->ij', Psi, np.linalg.inv(self.SigmaMat), Psi)

        weight = a * np.exp(-a * (2 * PsiT_SigmaI_Psi)) / (
                (1 - np.exp(-2 * a)) * 2 * np.pi * np.sqrt(np.linalg.det(self.SigmaMat)))

        return weight

    def apply_to(self, *params: np.ndarray, option=0):
        """

        :param params: SHC Cqlm and Sqlm if option=0, else grids.
        :param option:
        :return:
        """
        if option == 0:
            assert len(params) == 2
            assert self.harmonic is not None

            Gqij = self.harmonic.synthesis(*params)

        else:
            assert len(params) == 1
            Gqij = params[0]

        single_data_flag = False
        if Gqij.ndim == 2:
            single_data_flag = True
            Gqij = GeoMathKit.getCSGridin3d(Gqij)

        grid_space = 180 / np.shape(Gqij)[1]

        lat = np.radians(np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space))
        lon = np.radians(np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space))

        theta = np.radians(180 - np.arange(np.shape(Gqij)[1]) * grid_space - grid_space / 2)
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
            print('\rfiltering {}/{}'.format(j + 1, length_of_lon), end='\t')

            Gqij_pie = np.zeros_like(Gqij)
            Gqij_pie[:, :, :length_of_lon - j], Gqij_pie[:, :, length_of_lon - j:] = Gqij[:, :, j:], Gqij[:, :, :j]
            Gqij_filtered[:, :, j] = np.einsum('rij,qij->qr', Wipq, Gqij_pie) * d_sigma
        print('done!')

        if single_data_flag:
            Gqij_filtered = Gqij_filtered[0]

        if option == 0:
            return self.harmonic.analysis(Gqij_filtered)

        else:
            return Gqij_filtered

    @staticmethod
    def getPsi(sp, lat, lon):
        """
        lat, lon in unit[rad].
        """
        Psi = np.zeros((int(180 / sp), int(360 / sp), 2))

        colat = np.pi / 2 - lat

        colat_pie = np.pi / 2 - np.radians(np.arange(-90 + sp / 2, 90 + sp / 2, sp))
        delta_lat = colat_pie - colat

        lon_pie = np.radians(np.arange(-180 + sp / 2, 180 + sp / 2, sp))
        delta_lon = lon - lon_pie

        colat_pie, lon_pie = np.meshgrid(colat_pie, lon_pie)

        delta_lat, delta_lon = np.meshgrid(delta_lat, delta_lon)

        # Psi[:, :, 0] = (np.sin(delta_lon / 2) * np.sin((colat_pie + colat) / 2)).T
        Psi[:, :, 0] = (np.sin(delta_lon / 2) * np.sqrt(np.sin(colat_pie) * np.sin(colat))).T
        Psi[:, :, 1] = (np.sin(delta_lat / 2)).T

        return Psi
