

import os
import math
from .abscalutils import make_wvl_array
import numpy as np
from scipy.interpolate import interp1d

"""
The current module contains classes that calculate the transmission of the atmosphere 
for different atmospheric components.

It is based on the SMARTS 2.9.5 model and the Gueymard (2019) paper.
The python implementation of the code is based on the existing code in the repository https://github.com/jararias/dast.git

"""


NLOSCHMIDT = 2.6867811e19  # cm-3, number of particles in a volume of ideal gas

class Atmospheric_Component(object):
    def __init__(self,z_):
        """
        Initialize the Atmospheric_Component class.

        Parameters:
        - z_ (float): The zenith angle in degrees.

        """
        self.wvl_arr = make_wvl_array()
        self.z_ = z_
        pass

    
    def Airmass_from_SMARTS(self,z_, constituent='rayleigh'):
        """
        Calculate the airmass using SMARTS2.9.5 tabulated values.

        Parameters:
        - z_ (float): The zenith angle in degrees.
        - constituent (str): The atmospheric constituent. Default is 'rayleigh'.

        Returns:
        - airmass (float): The calculated airmass.

        Raises:
        - ValueError: If the constituent is not valid.

        """
        coefs = {
            'rayleigh': [0.48353, 0.095846,  96.741, -1.754 ],   
            'aerosol' : [0.16851, 0.18198 ,  95.318, -1.9542],   
            'o3':       [1.0651 , 0.6379  , 101.8  , -2.2694],   
            'h2o':      [0.10648, 0.11423 ,  93.781, -1.9203],   
            'o2':       [0.65779, 0.064713,  96.974, -1.8084],   
            'ch4':      [0.49381, 0.35569 ,  98.23 , -2.1616],   
            'co':       [0.505  , 0.063191,  95.899, -1.917 ],   
            'n2o':      [0.61696, 0.060787,  96.632, -1.8279],   
            'co2':      [0.65786, 0.064688,  96.974, -1.8083],   
            'n2':       [0.38155, 8.871e-05, 95.195, -1.8053],   
            'hno3':     [1.044  , 0.78456 , 103.15 , -2.4794],   
            'no2':      [1.1212 , 1.6132  , 111.55 , -3.2629],   
            'no':       [0.77738, 0.11075 , 100.34 , -1.5794],   
            'so2':      [0.63454, 0.00992 ,  95.804, -2.0573],   
            'nh3':      [0.32101, 0.010793,  94.337, -2.0548]    
        }

        coefs['no3'] = coefs['no2'].copy()
        coefs['bro'] = coefs['o3'].copy()
        coefs['ch2o'] = coefs['n2o'].copy()
        coefs['hno2'] = coefs['hno3'].copy()
        coefs['clno'] = coefs['no2'].copy()
        coefs['ozone'] = coefs['o3'].copy()
        coefs['water'] = coefs['h2o'].copy()

        p = coefs.get(constituent.lower(), None)
        if p is None:
            raise ValueError(f'{constituent} is not a valid constituent.')

        cosz = np.cos(np.radians(z_))
        return 1. / (cosz + p[0]*(z_**p[1])*(p[2]-z_)**p[3]) 
    
class Rayleigh_Transmission(Atmospheric_Component):
    def __init__(self,z_,p_):
        """
        Initialize the Rayleigh_Transmission class.

        Parameters:
        - z_ (float): The zenith angle in degrees.
        - p_ (float): The pressure in mbar.

        """
        super().__init__(z_)
        self.p_ = p_
        pass

    def make_transmission(self):
        '''
        Calculate the Rayleigh transmission using Eq (3) in Gueymard (2019).

        Returns:
        - transmission (numpy.ndarray): The calculated transmission values.

        '''
        z_ = self.z_
        wvl_arr = self.wvl_arr/1e3
        p_ = self.p_
        
        airmass_ = self.Airmass_from_SMARTS(z_,constituent='rayleigh')
        
        pp0 = p_/1013.25
        
        tau_r_l = pp0 / (117.3405 * wvl_arr**4 - 1.5107 * wvl_arr**2 +
                            0.017535 - 8.7743E-4 / wvl_arr**2)
        
        return np.clip(np.exp(-airmass_ * tau_r_l), 0., 1.)
    
class Aerosol_Transmission(Atmospheric_Component):
    def __init__(self,z_,aod_in,alpha_in):
        """
        Initialize the Aerosol_Transmission class.

        Parameters:
        - z_ (float): The zenith angle in degrees.
        - aod_in (float): The aerosol optical depth.
        - alpha_in (float): The alpha parameter.

        """
        super().__init__(z_)
        self.z_ = z_
        self.aod_in = aod_in
        self.alpha_in = alpha_in
        pass
    
    def make_transmission(self):
        '''
        Calculate the aerosol transmission.

        Returns:
        - transmission (numpy.ndarray): The calculated transmission values.

        '''
        z_ = self.z_
        alpha_in = self.alpha_in
        aod_in = self.aod_in
        wvl_arr = self.wvl_arr/1e3

        am_ = self.Airmass_from_SMARTS(z_,constituent='aerosol')
        
        coefs = {
            'moderate': [0.084,0.6],
            'sand storm': [0.54,0.18]
        }
        
        aod_ = aod_in
        alpha_ = alpha_in 
        
        tau_a_l = aod_/((2*wvl_arr)**alpha_)
        
        return np.clip(np.exp(-am_ * tau_a_l), 0., 1.)
    
class Ozone_Transmission(Atmospheric_Component):
    def __init__(self,z_,uo_):
        """
        Initialize the Ozone_Transmission class.

        Parameters:
        - z_ (float): The zenith angle in degrees.
        - uo_ (float): The ozone Dobson units.

        """
        super().__init__(z_)
        self.z_ = z_
        self.uo_ = uo_
        pass
    
    def make_transmission(self):
        '''
        Calculate the ozone transmission.

        Returns:
        - transmission (numpy.ndarray): The calculated transmission values.

        '''
        z_ = self.z_
        uo_ = self.uo_
        wvl_arr = self.wvl_arr

        
        
        uo_ = uo_*0.001 #Convert Dobson units atm-cm 
        
        ## get relative path to the data file
        current_dir = os.path.dirname(__file__)
        
        filename_o3uv = os.path.join(current_dir, 'data' ,'Templates/Abs_O3UV.dat')

        abs_wvl_uv, o3xs_uv = np.loadtxt(
                    filename_o3uv,
                    usecols=(0, 1), skiprows=1, unpack=True)
        
        o3xs_uv = np.interp(wvl_arr, abs_wvl_uv, o3xs_uv, left=0., right=0.)
        
        abo3 = NLOSCHMIDT*o3xs_uv
        
        am_ = self.Airmass_from_SMARTS(z_, 'o3')

        tau_o_l = abo3 * uo_
        
        return np.clip(np.exp(-am_ * tau_o_l), 0., 1.)
    



class WaterTransmittance(Atmospheric_Component):
    """
    Represents a component for calculating the water transmittance of the atmosphere.

    Args:
        z_ (float): The zenith angle in degrees.
        pw_ (float): The precipitable water in cm.
        p_ (float): The pressure in hPa.

    Attributes:
        z_ (float): The zenith angle in degrees.
        pw_ (float): The precipitable water in cm.
        p_ (float): The pressure in hPa.
        h2o (dict): A dictionary containing the water absorption data.

    Methods:
        _calculate_Bw(pw): Calculates the water transmittance factor Bw.
        _calculate_Bm(am): Calculates the molecular transmittance factor Bm.
        _calculate_Bmw(pw, am): Calculates the combined water and molecular transmittance factor Bmw.
        _calculate_Bp(pw, pr, am): Calculates the pressure transmittance factor Bp.
        make_transmission(): Calculates the transmission of the atmosphere.

    """
    def __init__(self,z_,pw_,p_):

        super().__init__(z_)
        self.z_ = z_
        self.pw_ = pw_
        self.p_ = p_

        column_names = (
            'wvl', 'abs',
            'iband',
            'ifitw', 'bwa0', 'bwa1', 'bwa2',
            'ifitm', 'bma0', 'bma1', 'bma2',
            'ifitmw', 'bmwa0', 'bmwa1', 'bmwa2',
            'bpa1', 'bpa2'
        )
        
        current_dir = os.path.dirname(__file__)
        filename_h2o = os.path.join(current_dir, 'data' ,'Templates/Abs_H2O.dat')
        data = np.loadtxt(filename_h2o, skiprows=1)
        self.h2o = dict(zip(column_names, data.T))
        
    
    def _calculate_Bw(self, pw):
        iband = self.h2o['iband']
        pw0 = np.full(iband.shape, 4.11467)
        pw0[iband == 2] = 2.92232
        pw0[iband == 3] = 1.41642
        pw0[iband == 4] = 0.41612
        pw0[iband == 5] = 0.05663
        pww0 = pw - pw0

        bwa0 = self.h2o['bwa0']
        bwa1 = self.h2o['bwa1']
        bwa2 = self.h2o['bwa2']
        Bw = (1. + bwa0 * pww0 + bwa1 * pww0**2)

        ifitw = self.h2o['ifitw']
        ifitw = ifitw * np.ones_like(Bw, dtype=ifitw.dtype)
        Bw[ifitw == 1] = (Bw / (1. + bwa2 * pww0))[ifitw == 1]
        Bw[ifitw == 2] = (Bw / (1. + bwa2 * (pww0**2)))[ifitw == 2]
        Bw[ifitw == 6] = (bwa0 + bwa1 * pww0)[ifitw == 6]

        h2oabs = self.h2o['abs']
        h2oabs = h2oabs * np.ones_like(Bw, dtype=h2oabs.dtype)
        Bw[h2oabs <= 0.] = 1.
        return np.clip(Bw, 0.05, 7.0)

    
    def _calculate_Bm(self, am):
        am1 = am - 1.
        am12 = am1**2

        bma0 = self.h2o['bma0']
        bma1 = self.h2o['bma1']
        bma2 = self.h2o['bma2']
        # Bm = np.ones((bma1.shape[0], am.shape[1], 1))
        Bm = np.ones(tuple([1]*bma1.ndim))

        ifitm = self.h2o['ifitm']
        ifitm = ifitm * np.ones_like(Bm, dtype=ifitm.dtype)
        Bm = np.where(ifitm == 0, bma1*(am**bma2), Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am1)
        Bm = np.where(ifitm == 1, Bmx, Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am12)
        Bm = np.where(ifitm == 2, Bmx, Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am1**.5)
        Bm = np.where(ifitm == 3, Bmx, Bm)
        Bmx = (1. + bma0*am1**.25) / (1. + bma2*am1**.1)
        Bm = np.where(ifitm == 5, Bmx, Bm)

        h2oabs = self.h2o['abs']
        h2oabs = h2oabs * np.ones_like(Bm, dtype=h2oabs.dtype)
        Bm[h2oabs <= 0.] = 1.
        return np.clip(Bm, 0.05, 7.0)

    
    def _calculate_Bmw(self, pw, am):
        Bw = self._calculate_Bw(pw)
        Bm = self._calculate_Bm(am)
        Bmw = Bm * Bw

        ifitm = self.h2o['ifitm']
        ifitm = ifitm * np.ones_like(Bmw, dtype=ifitm.dtype)
        ifitmw = self.h2o['ifitmw']
        ifitmw = ifitmw * np.ones_like(Bmw, dtype=ifitmw.dtype)
        absh2o = self.h2o['abs']
        absh2o = absh2o * np.ones_like(Bmw, dtype=absh2o.dtype)
        Bw = Bw * np.ones(Bmw.shape)
        Bm = Bm * np.ones(Bmw.shape)

        cond1 = np.abs(Bw-1) < 1e-6
        cond2 = ((ifitm != 0) | (absh2o <= 0.)) & (np.abs(Bm - 1.) < 1e-6)
        cond3 = ((ifitm == 0) | (absh2o <= 0.)) & (Bm > 0.968) & (Bm < 1.0441)
        cond4 = (ifitmw == -1) | (absh2o <= 0.)
        cond = cond1 | cond2 | cond3 | cond4

        iband = self.h2o['iband']
        iband = iband * np.ones_like(Bmw, dtype=iband.dtype)
        w0 = 4.11467 * np.ones_like(Bmw, dtype='float')
        w0[iband == 2] = 2.92232
        w0[iband == 3] = 1.41642
        w0[iband == 4] = 0.41612
        w0[iband == 5] = 0.05663

        amw = am*(pw/w0)
        amw1 = amw - 1.
        amw12 = amw1**2
        bmwa0 = self.h2o['bmwa0']
        bmwa1 = self.h2o['bmwa1']
        bmwa2 = self.h2o['bmwa2']

        Bmwx = np.ones(Bmw.shape)
        universe = (ifitmw == 0) & (absh2o > 0.)
        dummy = bmwa1*(amw**bmwa2)
        Bmwx[universe] = dummy[universe]
        universe = (ifitmw == 1) & (absh2o > 0.)
        dummy = (1. + bmwa0*amw1 + bmwa1*amw12) / (1. + bmwa2*amw1)
        Bmwx[universe] = dummy[universe]
        universe = (ifitmw == 2) & (absh2o > 0.)
        dummy = (1. + bmwa0*amw1 + bmwa1*amw12) / (1. + bmwa2*amw12)
        Bmwx[universe] = dummy[universe]

        Bmw = np.where(cond, Bmw, Bmwx)
        return np.clip(Bmw, 0.05, 7)

    def _calculate_Bp(self, pw, pr, am):
        bpa1 = self.h2o['bpa1']
        bpa2 = self.h2o['bpa2']

        pwm = pw*am
        pp0 = pr / 1013.25
        pp01 = np.maximum(0.65, pp0)
        pp02 = pp01**2
        qp = 1. - pp0
        qp1 = np.minimum(0.35, qp)
        qp2 = qp1**2

        absh2o = self.h2o['abs']
        absh2o = absh2o * np.ones(am.shape) * np.ones(pw.shape)
        iband = self.h2o['iband']
        iband = iband * np.ones(absh2o.shape)

        Bp = (1. + 0.1623*qp) * np.ones(iband.shape)
        universe = (iband == 2) & (absh2o > 0.)
        Bpx = (1. + 0.08721*qp1) * np.ones(iband.shape)
        Bp[universe] = Bpx[universe]
        universe = (iband == 3) & (absh2o > 0.)
        A = (1. - bpa1*qp1 - bpa2*qp2) * np.ones(iband.shape)
        Bp[universe] = A[universe]
        universe = (iband == 4) & (absh2o > 0.)
        B = 1. - pwm*np.exp(-0.63486 + 6.9149*pp01 - 13.853*pp02)
        Bp[universe] = (A*B)[universe]
        universe = (iband == 5) & (absh2o > 0.)
        B = 1. - pwm*np.exp(8.9243 - 18.197*pp01 + 2.4141*pp02)
        Bp[universe] = (A*B)[universe]

        Bp[(np.abs(qp*np.ones(Bp.shape)) < 1e-5) | (absh2o <= 0)] = 1
        return np.clip(Bp, 0.3, 1.7)

    def make_transmission(self):
        """
        Calculates the transmission of the atmosphere.

        Returns:
            numpy.ndarray: The calculated transmission values.
        """
        z_ = self.z_
        pw = self.pw_
        pressure = self.p_
        wvl_arr = self.wvl_arr

        am_ = np.array(self.Airmass_from_SMARTS(z_, 'h2o'), ndmin=1)
        pw_ = np.array(pw, ndmin=1)
        pr = np.array(pressure, ndmin=1)

        Bmw = self._calculate_Bmw(pw_, am_)
        Bp = self._calculate_Bp(pw_, pr, am_)
        pwm = (pw_*am_)**0.9426
        tauw_l = Bmw*Bp * self.h2o['abs']*pwm
        tauw_l = interp1d(
            self.h2o['wvl'], tauw_l, kind='linear',
            bounds_error=False, fill_value=0., axis=-1)(wvl_arr)

        return np.clip(np.exp(-tauw_l), 0., 1.)






class UMGTransmittance(Atmospheric_Component):

    """
    UMGTransmittance class represents the atmospheric transmittance model based on the USSA atmosphere.

    Args:
        z_ (float): The altitude in meters.
        tair (float): The air temperature in degrees Celsius.
        p_ (float): The atmospheric pressure in hPa.
        co2_ppm (float, optional): The concentration of CO2 in parts per million (ppm). Defaults to 395.
        with_trace_gases (bool, optional): Flag indicating whether to include trace gases in the model. Defaults to True.
    """

    def __init__(self,z_, tair,p_, co2_ppm=395., with_trace_gases=True):
        assert np.isscalar(tair)
        assert np.isscalar(co2_ppm)

        super().__init__(z_)

        self.z_ = z_
        self.co2_ppm = co2_ppm
        self.tair = tair
        self.p_ = p_
        self.with_trace_gases = with_trace_gases
        




    def read_gas(self,gas_name):

        current_dir = os.path.dirname(__file__)
        filepath_gas = os.path.join(current_dir, 'data' ,f'Templates/Abs_{gas_name}.dat')
        data = np.loadtxt(filepath_gas, skiprows=1)
        gas_wvl = data[:, 0]
        if data.shape[1] == 2:
            return np.interp(self.wvl_arr, gas_wvl, data[:, 1],left=0., right=0.)
        else:
            return [
                np.interp(self.wvl_arr, gas_wvl, data[:, k],left=0., right=0.)
                for k in range(1, data.shape[1])
            ]

    def make_transmission(self):

        # absorption coeffs | cross sections
        self.o2abs = self.read_gas('O2')
        self.n2abs = self.read_gas('N2')
        self.coabs = self.read_gas('CO')
        self.co2abs = self.read_gas('CO2')
        self.ch4abs = self.read_gas('CH4')
        self.o4abs = 1e-46 * self.read_gas('O4')
        self.n2oabs = self.read_gas('N2O')

        if self.with_trace_gases is True:
            self.nh3abs = self.read_gas('NH3')
            self.noabs = self.read_gas('NO')
            sigma, b0 = self.read_gas('NO2')
            self.no2abs = NLOSCHMIDT*(sigma + b0*(228.7-220.))
            sigma, b0 = self.read_gas('SO2U')
            self.so2abs = NLOSCHMIDT*(sigma + b0*(247.-213))
            self.so2abs += self.read_gas('SO2I')
            xs, b0 = self.read_gas('HNO3')
            self.hno3abs = 1e-20*NLOSCHMIDT*xs*np.exp(1e-3*b0*(234.2-298.))
            xs, b0 = self.read_gas('NO3')
            self.no3abs = NLOSCHMIDT*(xs+b0*(225.3-230.))
            self.hno2abs = NLOSCHMIDT*self.read_gas('HNO2')
            xs, b0 = self.read_gas('CH2O')
            self.ch2oabs = NLOSCHMIDT*(xs+b0*(264.-293.))
            self.broabs = NLOSCHMIDT*self.read_gas('BrO')
            xs, b0, b1 = self.read_gas('ClNO')
            TCl = 230.  # K
            self.clnoabs = xs*NLOSCHMIDT*(1.+b0*(TCl-296)+b1*(TCl-296)**2)
        
        pressure = self.p_
        z_ = self.z_

        pp0 = np.array(pressure, ndmin=1)[:, None] / 1013.25
        z_arr = np.array(z_, ndmin=1)[:, None]
       
        taug_l = self._optical_depth(pp0, z_arr)
        return np.clip(np.exp(-taug_l), 0., 1.)[0]

    def _optical_depth(self, pp0, sza):

        tt0 = np.zeros_like(pp0) + (self.tair + 273.15) / 273.15  # noqa

        
        taug_l = np.zeros((len(sza), len(self.wvl_arr)))

        def getam(constituent):
            return self.Airmass_from_SMARTS(sza, constituent)

        def getabs(constituent):
            
            return getattr(self, f'{constituent}abs')[None, :]

        # Uniformly Mixed Gases

        # 1. Oxygen, O2
        abundance = 1.67766e5 * pp0
        taug_l += getabs('o2') * abundance * getam('o2')
        # 2. Methane, CH4
        abundance = 1.3255 * (pp0 ** 1.0574)
        taug_l += getabs('ch4') * abundance * getam('ch4')
        # 3. Carbon Monoxide, CO
        abundance = .29625 * (pp0**2.4480) * \
            np.exp(.54669 - 2.4114 * pp0 + .65756 * (pp0**2))
        taug_l += getabs('co') * abundance * getam('co')
        # 4. Nitrous Oxide, N2O
        abundance = .24730 * (pp0**1.0791)
        taug_l += getabs('n2o') * abundance * getam('n2o')
        # 5. Carbon Dioxide, CO2
        abundance = 0.802685 * self.co2_ppm * pp0
        taug_l += getabs('co2') * abundance * getam('co2')
        # 6. Nitrogen, N2
        abundance = 3.8269 * (pp0**1.8374)
        taug_l += getabs('n2') * abundance * getam('n2')
        # 7. Oxygen-Oxygen, O4
        abundance = 1.8171e4 * (NLOSCHMIDT**2) * (pp0**1.7984) / (tt0**.344)
        taug_l += getabs('o4') * abundance * getam('o2')

        # Misc. Trace Gases

        if self.with_trace_gases is True:
            # 1. Nitric Acid, HNO3
            abundance = 1e-4*3.637*(pp0**0.12319)
            taug_l += getabs('hno3') * abundance * getam('hno3')
            # 2. Nitrogen Dioxide, NO2
            abundance = 1e-4*np.minimum(1.8599+0.18453*pp0, 41.771*pp0)
            taug_l += getabs('no2') * abundance * getam('no2')
            # 3. Nitrogen Trioxide, NO3
            abundance = 5e-5
            taug_l += getabs('no3') * abundance * getam('no3')
            # 4. Nitric Oxide, NO
            abundance = 1e-4*np.minimum(0.74307+2.4015*pp0, 57.079*pp0)
            taug_l += getabs('no') * abundance * getam('no')
            # 5. Sulfur Dioxide, SO2
            abundance = 1e-4*0.11133*(pp0**.812) * np.exp(
                .81319+3.0557*(pp0**2)-1.578*(pp0**3))
            taug_l += getabs('so2') * abundance * getam('so2')
            # 6. Ozone, O3
            # ...implemented as independent transmittance
            # 7. Ammonia, NH3
            lpp0 = np.log(pp0)
            abundance = np.exp(
                - 8.6499 + 2.1947*lpp0 - 2.5936*(lpp0**2)
                - 1.819*(lpp0**3) - 0.65854*(lpp0**4))
            taug_l += getabs('nh3') * abundance * getam('nh3')
            # 8. Bromine Monoxide, BrO
            abundance = 2.5e-6
            taug_l += getabs('bro') * abundance * getam('bro')
            # 9. Formaldehyde, CH2O
            abundance = 3e-4
            taug_l += getabs('ch2o') * abundance * getam('ch2o')
            # 10. Nitrous Acid, HNO2
            abundance = 1e-4
            taug_l += getabs('hno2') * abundance * getam('hno2')
            # 11. Chlorine Nitrate, ClNO3
            abundance = 1.2e-4
            taug_l += getabs('clno') * abundance * getam('clno')

        return taug_l   

    