import pathlib

import cmaps
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import rcParams
import matplotlib

from readfile.readfile import loadCS
from filter.VariableScale import VariableScale
from harmonic.Harmonic import Harmonic
from auxiliary.GeoMathKit import GeoMathKit


def plot(grids, subtitles):
    """
    Plot grids given by this demo code if you want to confirm the results more intuitively.
    Please do not change the content if you are unsure of its specific usage.
    """

    print('plotting...')

    '''Font style setting'''
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)

    '''Define figure and sub-figures'''
    fig = plt.figure(figsize=(10, 6))

    projection = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    ''''''
    ax_sizes = (
        (0.4, 0.4),
        (0.4, 0.4),
        (0.4, 0.4),
        (0.4, 0.4),

        (0.4, 0.033),
        (0.4, 0.033),
        (0.4, 0.033),
        (0.4, 0.033),

        (0.9, 0.1),
    )

    ax_locs = (
        (0.05, 0.6),
        (0.55, 0.6),
        (0.05, 0.15),
        (0.55, 0.15),

        (0.05, 0.55),
        (0.55, 0.55),
        (0.05, 0.1),
        (0.55, 0.1),

        (0., 0.),

    )

    ax_insiders = (
        (0.95, 0.95),
        (0.95, 0.95),
        (0.95, 0.95),
        (0.95, 0.95),

        (0.95, 0.95),
        (0.95, 0.95),
        (0.95, 0.95),
        (0.95, 0.95),

        (0.8, 0.8),
    )

    ax_list = []
    for i in range(len(ax_locs)):
        this_projection = projection if i in (0, 1, 2, 3, 4, 5) else None

        ax_list.append(
            fig.add_axes(
                [ax_locs[i][0] + ax_sizes[i][0] * (1 - ax_insiders[i][0]) / 2,
                 ax_locs[i][1] + ax_sizes[i][1] * (1 - ax_insiders[i][1]) / 2,
                 ax_sizes[i][0] * ax_insiders[i][0], ax_sizes[i][1] * ax_insiders[i][1]],
                projection=this_projection)
        )

    '''Necessary prepare for plotting later'''
    grid_space = 180 / np.shape(grids[0])[0]
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
    lon2d, lat2d = np.meshgrid(lon, lat)

    '''Define numerical range in plotting'''
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-40, vmax=40, vcenter=0)

    '''Plot'''
    p = []
    for i in range(len(grids)):
        ax = ax_list[i]

        p.append(ax.pcolormesh(lon2d, lat2d,
                               grids[i],
                               cmap='RdBu_r',
                               transform=transform,
                               norm=norm))

        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)

    axes_subtitle = ax_list[4:8]

    for i in range(len(axes_subtitle)):
        axes_subtitle[i].axis('off')
        axes_subtitle[i].set_xlim(-1, 1)
        axes_subtitle[i].set_ylim(-1, 1)

        axes_subtitle[i].text(
            0, 0, f'{subtitles[i]}',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=20
        )

    ax_cb = ax_list[8]
    ax_cb.axis('off')
    ax_cb.set_xlim(-1, 1)
    ax_cb.set_ylim(-1, 1)
    ax_cb.text(
        1.2, -0.2, f'(cm)',
        verticalalignment='top',
        horizontalalignment='center',
        fontsize=20
    )

    cb = fig.colorbar(p[0]
                      , orientation='horizontal'
                      , fraction=1, ax=ax_cb
                      , ticks=np.arange(-40, 41, 20)
                      , aspect=30
                      , extend='both'
                      )

    cb.ax.tick_params(direction='in')

    plt.show()


def r_lat_cons500(lat):
    return 500 * 1000  # unit [m]


def r_lat_cons75(lat):
    return 75 * 1000  # unit [m]


def r_lat_sin(lat):
    return ((500 - 200) * np.cos(lat) + 200) * 1000  # unit [m]


def r_lat_substellar(lat):
    r_lat = (-0.05718 * np.degrees(np.abs(lat)) ** 2 + 1.03394 * np.degrees(np.abs(lat)) + 500) * 1000  # unit [m]
    return r_lat


def demo_global_distribution():
    """An example using Variable-scale Gaussian-variant Convolution (VGC) filter"""

    '''Load spherical harmonic coefficients (SHCs) of time-variable gravity field.'''
    filepath = pathlib.Path('../data/EWH_anomaly_201211.gfc')
    lmax = 60
    clm, slm = loadCS(filepath, key='gfc', lmax=60)

    '''Define spatial resolution (degree) of a grid'''
    grid_space = 1

    '''Define a harmonic tool which will be used to transform SHCs to grid format.'''
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
    pilm = GeoMathKit.getLegendre(lat, lmax, option=1)
    har = Harmonic(lat, lon, pilm, lmax, option=1)

    '''Define VGC filters with different parameters'''
    '''
    Attribute r_function gives the way how the half-wavelength in E-W direction changes with latitude.
    Attribute r_function should be function that returns the filtering radius [m] by inputting longitude [degree].
    Attribute sigma is used to tune the anisotropy of the convolution kernel.
    As VGC filter works in spatial domain, an attribute harmonic need giving to transform SHCs into grid format,
    which is given by Harmonic().
    The VGC filter would be degenerated into traditional Gaussian filter if
    sigma is set 1 and r_function is set as a constant.
    '''
    gs200 = VariableScale(r_function=r_lat_cons75, sigma=1, harmonic=har)
    gs500 = VariableScale(r_function=r_lat_cons500, sigma=1, harmonic=har)
    vs2 = VariableScale(r_function=r_lat_substellar, sigma=0.7, harmonic=har)

    '''Filter the SHCs with different filters defined before.'''
    cqlm_gs200, sqlm_gs200 = gs200.apply_to(np.array([clm]), np.array([slm]))
    cqlm_gs500, sqlm_gs500 = gs500.apply_to(np.array([clm]), np.array([slm]))
    cqlm_vs, sqlm_vs = vs2.apply_to(np.array([clm]), np.array([slm]))

    '''Transform the filtered SHCs into grids, if needed.'''
    grids = har.synthesis(np.array([clm, cqlm_gs200[0], cqlm_gs500[0], cqlm_vs[0]]),
                          np.array([slm, sqlm_gs200[0], sqlm_gs500[0], sqlm_vs[0]]))

    '''Plot the filtered results in spatial domain, if needed.'''
    plot(grids * 100, subtitles=('(a) Unfiltered', '(b) Gaussian 75km', '(c) Gaussian 500km', '(d) VGC'))


if __name__ == '__main__':
    demo_global_distribution()
