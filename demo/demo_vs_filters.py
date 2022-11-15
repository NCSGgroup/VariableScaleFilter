import numpy as np

from pysrc.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.GeoMathKit import GeoMathKit

# from pysrc.filter.VariableScale import VariableScale, VaryRadiusWay

from pysrc.filter.VariableScaleGPU import VariableScale, VaryRadiusWay

"""
To speed up the calculation when the data is too big,
we have provided the GPU version of variable-scale (VS) filter (i.e., VariableScaleGPU.py).
The implementation of VariableScaleGPU.py depends on CUDA.

These two programs has the same usage, and a simple change of import would work.

Due to the limitation of calculation accuracy, 
the results of these two programs (CPU and GPU ones) may have a RMSE of ~1e-11 on the final EWH anomaly
"""


def demo():
    """an example for using VS filter"""

    """
    Usually, one just need make changes here to adjust the filter parameters, which includes
    
    vs_r_min: the minimum smoothing radius in polar regions;
    
    vs_r_max: the minimum smoothing radius in equatorial regions;
    
    vs_sigma: the covariance matrix to adjust the anisotropy of VS filter, 
              usually supposed to be a 2*2 diagonal matrix, 
              in which the 1st diagonal element is set to be 1, 
              and the 2nd one is set to no greater than 1;
              
    vary_radius_mode: to decide how the smoothing radius varies from the pole to equatorial.
                      At present, sin(theta) and sin(theta)^2 are available;
              
    grid_space: to represent the spatial resolution (in unit [degree]).
    """

    vs_r_min = 100
    vs_r_max = 500
    vs_sigma = np.array([[1, 0], [0, 1]])
    vary_radius_mode = VaryRadiusWay.sin

    grid_space = 1

    """start program"""

    """load EWHA file"""
    """
    file '../data/GRACE_GSM/ewh_201501_p3m10' is calculated using GRACE Level-2 GSM (CSR Release 06), 
    with degree1/c20/c30 replaced by TN-13 and TN-14 files,
    to which adding back GAD model,
    from which removing GIA model,
    deducting a long-term average field over the period from 2005 to 2015,
    then converted the physical quantity geoid anomaly into equivalent water height anomaly (EWHA).
    """
    shc_ewha = np.load('../data/GRACE_GSM/ewh_201501_p3m10.npy')
    Clm, Slm = shc_ewha[0], shc_ewha[1]

    """define harmonic tool"""
    nmax = 60
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
    Pilm = GeoMathKit.getLegendre(lat, nmax, option=1)
    har = Harmonic(lat, lon, Pilm, nmax, option=1)

    '''define filterer'''
    vs_filter = VariableScale(vs_r_min, vs_r_max, vs_sigma, harmonic=har, vary_radius_mode=vary_radius_mode)

    '''apply filtering to Clm, Clm'''
    CS_vs = vs_filter.apply_to(np.array([Clm]), np.array([Slm]))

    '''harmonic synthesis'''
    grid_vs = har.synthesis(*CS_vs)[0]

    '''make a validation'''

    validation = np.load('../results/validation/ewh_grid_201501_p3m10_vs75to500.npy')

    flat = ((grid_vs - validation) / validation).flatten()
    rmse = np.sqrt(np.sum(flat ** 2) / len(flat))

    if rmse < 1e-11:
        print('The code is correctly configured!')
    else:
        print(rmse)


if __name__ == '__main__':
    demo()
