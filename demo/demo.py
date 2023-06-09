import pathlib
import numpy as np

from preference.Enumclasses import FieldType
from readfile.readfile import loadCS
from Love_number.LoveNumber import LoveNumber, LoveNumberType
from convert_field_type.ConvertSHC import ConvertSHC
from filter.VariableScale import VariableScale
from harmonic.Harmonic import Harmonic
from auxiliary.GeoMathKit import GeoMathKit


def demo():
    """This demo gives an example using Variable-scale Gaussian-variant Convolution (VGC) filter"""

    '''
    Load spherical harmonic coefficients (SHCs) of GRACE GSM product and GIF48 model as background field by 
        function loadCS.
    Then subtract the background field from GRACE GSM product.
    Function loadCS requires three parameters filepath, key and lmax.
    Attribute filepath is the path of SHCs file, which describe the SHCs of the Earth's gravity field in the gfc format.
    Attribute key is the identification of valid data in the SHCs file.
    Attribute lmax is the max degree/order you want to read, not greater than that given in the file itself.
    Returned result is a tuple containing two elements, that is, clm and slm, and each of them is two-dimension array.
    '''
    lmax = 60

    filepath_grace = pathlib.Path('../data/GSM/GSM-2_2005001-2005031_GRAC_UTCSR_BA01_0600')
    clm, slm = loadCS(filepath=filepath_grace, key='GRCOF2', lmax=lmax)

    filepath_grace = pathlib.Path('../data/Auxiliary/GIF48.gfc')
    clm_bg, slm_bg = loadCS(filepath=filepath_grace, key='gfc', lmax=lmax)

    clm -= clm_bg
    slm -= slm_bg

    '''
    Convert SHCs from Dimensionless to EWH with class ConvertSHC.
    Class ConvertSHC requires four initialization parameters, c, s, field_type and ln.
    Attribute c and s are SHCs.
    Attribute field_type requires an enumeration type FieldType, and describe the field type input c and s.
        Generally speaking, the GRACE GSM file provides a dimensionless gravity field model,
        here field_type=FieldType.Dimensionless.
    Attribute ln is Love number, given by class LoveNumber as follow. Please see the source code for more details.
    '''

    ln = LoveNumber('../data/Auxiliary/').getNumber(lmax, LoveNumberType.Wang)
    convert = ConvertSHC(clm, slm, field_type=FieldType.Dimensionless, ln=ln)
    convert.convert_to(FieldType.EWH)
    clm, slm = convert.getCS()

    '''
    Define a harmonic tool which will be used to transform SHCs to grid (spatial distribution) format by class Harmonic.
    Harmonic requires three initialization parameters, that is lat, lon, pilm, lmax, and option.
    Attributes grid_space as follow defines the spatial resolution [degree] of a grid.
    Attributes lat and lon are the Geographic longitudes and latitudes used in following programs.
    Attributes pilm is the legendre the associative Legendre polynomials polynomials given by tool function 
        GeoMathKit.getLegendre. Please see the source code for more details.
    Attributes option prescribed the form of input lat and lon,
        which denote co-latitude and longitude in unit[rad] if option=0;
        or latitude and longitude in unit[degree] elsewhere.
    '''
    grid_space = 1
    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
    pilm = GeoMathKit.getLegendre(lat, lmax, option=1)
    har = Harmonic(lat, lon, pilm, lmax, option=1)

    '''
    Define VGC filters with different parameters.
    Attribute r_function gives the way how the half-wavelength in E-W direction changes with latitude.
    Attribute r_function should be function that returns the filtering radius [m] by inputting longitude [degree].
    Attribute sigma is used to tune the anisotropy of the convolution kernel.
    As VGC filter works in spatial domain, an additional attribute harmonic need giving to transform SHCs into grid format if it 
        will be applied to SHCs, which is given by Harmonic().
    The VGC filter would be degenerated into traditional Gaussian filter if
        sigma is set 1 and r_function is set as a constant.
    '''
    a, b, c = -0.05718, 1.03394, 500
    vs = VariableScale(
        r_function=lambda x: (a * np.degrees(np.abs(x)) ** 2 + b * np.degrees(np.abs(x)) + c) * 1000,
        sigma=0.7,
        harmonic=har
    )

    '''
    Convert SHCs to grids by harmonic synthesis.
    Function .systhesis(c, s) requires SHC C and S.
    Returned result is grid.
    '''
    grid = har.synthesis(clm, slm)

    '''
    Filter the SHCs with different filters defined before.
    The input and output of .apply_to(*params, option) requires different format with different setting of option.
    If option=0, *params need two parameters, that is, SHC C and S.
    Otherwise, *params need one parameter, that is, grid.
    Returned result is consistent with the form of input data.
    '''
    grid_filtered = vs.apply_to(grid, option=1)


if __name__ == '__main__':
    demo()
