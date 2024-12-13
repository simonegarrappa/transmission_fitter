
import astropy.units as u
import astropy.io.fits as pyfit
from astropy.coordinates import SkyCoord,EarthLocation,AltAz
from astropy.time import Time
import pandas as pd
import numpy as np


class LastCatUtils(object):
    def __init__(self):
        neot_semadar = EarthLocation(lat=30.0529838 * u.deg, lon=35.0407331 * u.deg, height=415.4 * u.m)
        self.neot_semadar = neot_semadar
        pass

    def tables_from_lastcat(self, catfile):
        """
        Creates a pandas DataFrame from a LAST catalog.
        
        Parameters:
            catfile (str): The path to the LAST catalog.
            
        Returns:
            last_cat (numpy.ndarray): The LAST catalog as a numpy array.
            info_cat (astropy.io.fits.Header): The header information of the LAST catalog.
        """
        
        hdu = pyfit.open(catfile)
        last_cat = hdu[1].data
        info_cat = hdu[2]
        
        return last_cat, info_cat
    
    def get_airmass_from_cat(self, info_cat):
        """
        Calculate the airmass from the given catalog information.

        Parameters:
        - info_cat: The catalog information containing the header information.

        Returns:
        - airmass_last: The calculated airmass value.

        """
        ra_ = info_cat.header['RA']
        dec_ = info_cat.header['DEC']
        time_ = info_cat.header['DATE-OBS']
        img_coord = SkyCoord(ra=ra_ * u.degree, dec=dec_ * u.degree, frame='icrs')

        img_altaz = img_coord.transform_to(AltAz(obstime=time_, location=self.neot_semadar))

        airmass_last = img_altaz.secz.value

        return airmass_last
    
    def get_zenith_from_cat(self,info_cat):
        """
        Get the zenith angle from the given info_cat.

        Parameters:
        info_cat (object): The info_cat object containing the header information.

        Returns:
        float: The zenith angle calculated from the airmass value.
        """
        airmass_last = self.get_airmass_from_cat(info_cat)
        zenith_last = abscalutils.get_zenith_from_airmass(airmass_last)
        return zenith_last
    
    def get_exptime_from_cat(self,info_cat):
        """
        Get the exposure time from the given info_cat.

        Parameters:
        info_cat (object): The info_cat object containing the header information.

        Returns:
        float: The exposure time extracted from the header.
        """
        exptime_last = info_cat.header['EXPTIME']
        return exptime_last
    
    def get_temperature_from_cat(self,info_cat):
        """
        Get the temperature from the given info_cat.

        Parameters:
        info_cat (object): The info_cat object containing the header information.

        Returns:
        float: The temperature extracted from the header.
        """
        temperature_last = info_cat.header['MNTTEMP']
        return temperature_last
    
    def get_jd_from_cat(self,info_cat):
        """
        Get the Julian Date from the given info_cat.

        Parameters:
        info_cat (object): The info_cat object containing the header information.

        Returns:
        float: The Julian Date extracted from the header.
        """
        jd_last = info_cat.header['JD']
        return jd_last
    
    def binbits(self,x, n):
        """Return binary representation of x with at least n bits"""
        bits = bin(x).split('b')[1]

        if len(bits) < n:
            return '0b' + '0' * (n - len(bits)) + bits
    
    def get_flags_keyword(self,decflag):
        """
        Get the flags corresponding to the decimal flag value.

        Parameters:
        decflag (int): The decimal flag value.

        Returns:
        list: The list of flags corresponding to the decimal flag value.
        """
        flags_dict = {
        '0': 'Saturated', 
        '1': 'LowRN',
        '2': 'HighRN',
        '3': 'DarkHighVal',
        '4': 'DarkLowVal',
        '5': 'BiasFlaring',
        '6': 'NaN',
        '7':'FlatHighStd',
        '8':'FlatLowVal',
        '9':'LowQE',
        '10':'Negative',
        '11':'Interpolated',
        '12':'Hole',
        '13':'Spike',
        '14':'CR_DeltaHT',
        '15':'CR_Laplacian',
        '16':'CR_Streak',
        '17':'Ghost',
        '18':'Persistent',
        '19':'Xtalk',
        '20':'Streak',
        '21':'ColumnLow',
        '22':'ColumnHigh',
        '23':'NearEdge',
        '24':'NonLinear',
        '25':'Bleeding',
        '26':'Overlap',
        '27':'SrcNoiseDominated',
        '28':'GainHigh',
        '29':'CoaddLessImages',
        '30':'SrcDetected'}
        binary_number = self.binbits(int(decflag),32)[2:] #bin(int(decflag))[2:]
        indices_flags = np.abs(np.where(np.array(list(binary_number)) == '1')[0] -31)

        flags = []
        for item in indices_flags:
            flags.append(flags_dict[str(item)])
        return flags
    
    def find_source_in_cat(self, coor_target, catfile):
        """
        Find the source in the LAST catalog that is closest to the given coordinates.

        Parameters:
        ra (float): The right ascension of the source.
        dec (float): The declination of the source.
        catfile (str): The filepath of the LAST catalog.

        Returns:
        int: The index of the source in the LAST catalog that is closest to the given coordinates.
        """
        last_cat, info_cat = self.tables_from_lastcat(catfile)
        last_coords = SkyCoord(ra=last_cat['RA']*u.deg, dec=last_cat['DEC']*u.deg)
        source_coords = SkyCoord(ra=coor_target.ra.deg*u.deg, dec=coor_target.dec.deg*u.deg)
        idx, d2d, d3d = source_coords.match_to_catalog_sky(last_coords)

        if d2d.arcsec < 2:
            print('Found source in LAST subframe {} at distance of {} arcsec'.format(int(info_cat.header['CROPID']),d2d.arcsec))
            return idx
        else:
            print('No source found in LAST subframe {} at distance of {} arcsec'.format(int(info_cat.header['CROPID']),d2d.arcsec))
            return None
    

        
    

