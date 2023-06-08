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

config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)


def r_lat_cons500(lat):
    return 500 * 1000  # unit [m]


def r_lat_cons75(lat):
    return 75 * 1000  # unit [m]


def r_lat_sin(lat):
    return ((500 - 200) * np.cos(lat) + 200) * 1000  # unit [m]


def r_lat_substellar(lat):
    r_lat = (-0.05718 * np.degrees(np.abs(lat)) ** 2 + 1.03394 * np.degrees(np.abs(lat)) + 500) * 1000  # unit [m]
    return r_lat


def global_distribution():
    filepath = pathlib.Path('../data/EWH_anomaly_201211.gfc')
    lmax = 60
    clm, slm = loadCS(filepath, key='gfc', lmax=60)

    grid_space = 1
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)

    pilm = GeoMathKit.getLegendre(lat, lmax, option=1)
    har = Harmonic(lat, lon, pilm, lmax, option=1)

    gs200 = VariableScale(r_function=r_lat_cons75, sigma=1, harmonic=har)
    gs500 = VariableScale(r_function=r_lat_cons500, sigma=1, harmonic=har)
    vs2 = VariableScale(r_function=r_lat_substellar, sigma=0.7, harmonic=har)

    cqlm_gs200, sqlm_gs200 = gs200.apply_to(np.array([clm]), np.array([slm]))
    cqlm_gs500, sqlm_gs500 = gs500.apply_to(np.array([clm]), np.array([slm]))
    cqlm_vs, sqlm_vs = vs2.apply_to(np.array([clm]), np.array([slm]))

    grids = har.synthesis(np.array([clm, cqlm_gs200[0], cqlm_gs500[0], cqlm_vs[0]]),
                          np.array([slm, sqlm_gs200[0], sqlm_gs500[0], sqlm_vs[0]]))

    grids *= 100  # unit [cm]

    """plot"""

    fig = plt.figure(figsize=(10, 5))

    projection = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    ax_sizes = (
        (0.35, 0.4),
        (0.35, 0.4),
        (0.35, 0.4),
        (0.35, 0.4),

        (0.35, 0.033),
        (0.35, 0.033),
        (0.35, 0.033),
        (0.35, 0.033),
        (0.3, 0.033),

        (0.7, 0.1),

        (0.22, 0.9)
    )

    ax_locs = (
        (0, 0.6),
        (0.35, 0.6),
        (0, 0.15),
        (0.35, 0.15),

        (0, 0.55),
        (0.35, 0.55),
        (0, 0.1),
        (0.35, 0.1),
        (0.71, 0.05),

        (0., 0.),

        (0.76, 0.15)
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
        (0.95, 0.95),

        (0.8, 0.8),

        (0.9, 0.7)
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

    lon2d, lat2d = np.meshgrid(lon, lat)

    norm = matplotlib.colors.TwoSlopeNorm(vmin=-40, vmax=40, vcenter=0)
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

    axes_subtitle = ax_list[4:9]
    subtitles = ['(a) Unfiltered', '(b) Gaussian 75km', '(c) Gaussian 500km', '(d) VGC', '(e)']
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

    ax_cb = ax_list[-2]
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

    label_list = ['(a)', '(b)', '(c)', '(d)']
    ax_xy = ax_list[-1]
    ax_xy.set_xlim(0, 10)
    ax_xy.set_xticks((0, 5, 10))
    ax_xy.set_xlabel('RMS of $\Delta$EWH (cm)', fontsize=16)

    ax_xy.set_ylim(-91, 91)
    ax_xy.set_yticks(np.arange(-90, 91, 45))
    ax_xy.set_ylabel('latitude ($^\circ$)',
                     fontsize=18,
                     labelpad=-15
                     )

    for i in range(len(grids)):
        this_grid = grids[i]
        rms_by_lat = np.sqrt((np.sum(this_grid, axis=1) ** 2)) / np.shape(this_grid)[1]

        ax_xy.plot(rms_by_lat, np.arange(len(rms_by_lat)) - 90, label=label_list[i])

    ax_xy.legend(fontsize=14, loc='center right', bbox_to_anchor=(1.025, 0.31))

    plt.show()


if __name__ == '__main__':
    global_distribution()
