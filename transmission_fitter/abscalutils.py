import numpy as np
import math

def get_zenith_from_airmass(airmass):

    return math.degrees(math.acos(1./airmass))

def get_airmass_from_zenith(z_):
        ## z angle in degrees
    
        z_ = np.radians(z_)
    
        return 1./math.cos(z_)
def make_wvl_array(min_int=300.,max_int=1100.,num=401):
    ## 401 to mimic Gaia sampling (dLambda = 2nm)
    ## 81 to do dLambda = 10nm
    wvl_arr = np.linspace(min_int,max_int, num, endpoint=True)

    return wvl_arr

def make_wvl_array_Gaia():
    """
    Generate a wavelength array specific to Gaia observations.

    Returns:
        wvl_arr_gaia (numpy.ndarray): Wavelength array for Gaia observations.
        mask_gaia_ir (numpy.ndarray): Boolean mask for wavelengths greater than or equal to max_int_gaia.
        mask_gaia_uv (numpy.ndarray): Boolean mask for wavelengths less than or equal to min_int_gaia.
    """
    wvl_arr = make_wvl_array()

    min_int_gaia = 336
    max_int_gaia = 1020
    mask_gaia = (wvl_arr >= min_int_gaia) & (wvl_arr <= max_int_gaia)
    mask_gaia_uv = (wvl_arr <= min_int_gaia)
    mask_gaia_ir = (wvl_arr >= max_int_gaia)

    wvl_arr_gaia = wvl_arr[mask_gaia] 

    return wvl_arr_gaia,mask_gaia, mask_gaia_ir, mask_gaia_uv


