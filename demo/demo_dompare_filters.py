import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from pysrc.filter.VariableScale import VariableScale, VaryRadiusWay
from pysrc.filter.Gaussian import IsotropyGaussian
from pysrc.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.GeoMathKit import GeoMathKit

config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)


def demo():
    """an example for using variable-scale filter and comparing with Gaussian one"""

    '''define filter parameters'''
    nmax = 60
    grid_space = 1

    gs_r1 = 200
    gs_r2 = 500

    vs_r_min = 200
    vs_r_max = 500
    vs_sigma = np.array([
        [1, 0],
        [0, 0.5]
    ])

    '''load EWHA file'''
    '''
    file EWHA_200501.npy is calculated using GRACE Level-2 GSM (CSR Release 06), 
    with degree1/c20/c30 replaced by TN-13 and TN-14 files,
    deducting a long-term average field over the period from 2005 to 2015,
    and then convert geoid anomaly into equivalent water height anomaly (EWHA).
    '''
    shc_ewha = np.load('../data/GRACE_GSM/EWHA_200501.npy')
    Clm, Slm = shc_ewha[0] * 100, shc_ewha[1] * 100  # unit [cm]

    '''define harmonic tool'''
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
    Pilm = GeoMathKit.getLegendre(lat, nmax, option=1)
    har = Harmonic(lat, lon, Pilm, nmax, option=1)

    '''define filterer'''
    gs_filter1 = IsotropyGaussian(nmax, gs_r1)

    gs_filter2 = IsotropyGaussian(nmax, gs_r2)

    vs_filter = VariableScale(vs_r_min, vs_r_max, vs_sigma, harmonic=har, vary_radius_mode=VaryRadiusWay.sin)

    '''apply filtering to Clm, Clm'''
    CS_gs1 = gs_filter1.apply_to([Clm], [Slm])

    CS_gs2 = gs_filter2.apply_to([Clm], [Slm])

    CS_vs = vs_filter.apply_to([Clm], [Slm])

    '''harmonic synthesis'''
    grid_no_filtering = har.synthesis([Clm], [Slm])[0]
    grid_gs1 = har.synthesis(*CS_gs1)[0]
    grid_gs2 = har.synthesis(*CS_gs2)[0]
    grid_vs = har.synthesis(*CS_vs)[0]

    grid_list = [grid_no_filtering, grid_gs1, grid_gs2, grid_vs]

    '''plot'''
    ax_ewh_length = 0.5
    ax_ewh_height = 0.4

    ax_cb_length = 0.8
    ax_cb_height = 0.05

    transform = ccrs.PlateCarree()
    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))

    axes = [
        fig.add_axes([i, j, ax_ewh_length, ax_ewh_height], projection=projection)
        for j in [
            (1 - ax_cb_height - ax_ewh_height * 2) * 2 / 3 + ax_ewh_height + ax_cb_height,
            (1 - ax_cb_height - ax_ewh_height * 2) / 3 + ax_cb_height
        ]
        for i in [
            (1 - ax_ewh_length * 2) / 3,
            (1 - ax_ewh_length * 2) * 2 / 3 + ax_ewh_length
        ]
    ]
    axes_title = [
        '(a) no filtering',
        f'(b) Gaussian {gs_r1}km',
        f'(c) Gaussian {gs_r2}km',
        f'(d) Variable-scale {vs_r_min}-{vs_r_max}km, $\Sigma$=diag({vs_sigma[0, 0]}, {vs_sigma[1, 1]})',
    ]

    ax = fig.add_axes([(1 - ax_cb_length) / 2, ax_cb_height / 2,
                       ax_cb_length, ax_cb_height])

    vmin, vmax, vcenter = -40, 40, 0
    norm_ewh = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    lon2d, lat2d = np.meshgrid(lon, lat)
    for i in range(len(axes)):
        p = axes[i].pcolormesh(lon2d, lat2d, grid_list[i], cmap="RdBu_r", transform=transform, norm=norm_ewh)
        axes[i].coastlines()

        axes[i].set_title(axes_title[i])

        if i == 0:
            cb = fig.colorbar(p, cax=ax, orientation='horizontal')
            cb.ax.tick_params(direction='in')

    plt.show()


if __name__ == '__main__':
    demo()
