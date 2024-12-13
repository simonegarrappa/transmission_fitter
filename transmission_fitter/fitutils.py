"""
This module contains the `AbsoluteCalibration` class, which is used for absolute calibration of astronomical data.
"""
import scipy
from astropy.constants import c,h
from astropy.stats import sigma_clip
import math
import numpy as np
import lmfit
from lmfit import Parameters,Model
from lmfit import Minimizer, fit_report
from .gaiaquery import GaiaQuery
from .lastcatutils import LastCatUtils
from .atmospheric_models import Rayleigh_Transmission,Aerosol_Transmission,Ozone_Transmission,WaterTransmittance,UMGTransmittance
from .abscalutils import make_wvl_array,make_wvl_array_Gaia
import pandas as pd
from scipy.interpolate import CubicSpline,interp1d
from scipy.special import legendre
from numpy.polynomial.chebyshev import Chebyshev
import warnings
import os
np.random.seed(6)


class AbsoluteCalibration(object):
    def __init__(self, catfile, useHTM=False, use_atm=True, band='LAST'):
        """
        Initialize the FitUtils class.

        Parameters:
        - catfile (str): The path to the catalog file.
        - useHTM (bool): Flag indicating whether to use HTM (Hierarchical Triangular Mesh) indexing. Default is False.
        - use_atm (bool): Flag indicating whether to use atmospheric correction. Default is True.
        - band (str): The band to use. Default is 'LAST'.

        Returns:
        None
        """
        ## Init general info
        wvl_arr = make_wvl_array()
        self.wvl_arr = wvl_arr
        self.catfile = catfile
        last_cat, info_cat = LastCatUtils().tables_from_lastcat(self.catfile)
        mnttemp = LastCatUtils().get_temperature_from_cat(info_cat)
        z_ = LastCatUtils().get_zenith_from_cat(info_cat)
        jd_ = LastCatUtils().get_jd_from_cat(info_cat)
        self.exptime = info_cat.header['EXPTIME']
        self.z_ = z_
        self.mnttemp = mnttemp
        self.jd_ = jd_
        self.use_atm = use_atm
        self.use_orig_xlt = True
        self.get_residuals = False
        self.band = band
        self.Ageom = math.pi*(0.1397**2)
        self.ErrorEstimation = 'ErrProp'
        self.current_dir = os.path.dirname(__file__)

        print('Using band: ' + self.band)
        #Mirror Reflectivity
        if self.use_orig_xlt:
            filename_xlt_mirror = os.path.join(self.current_dir,'data','Templates/StarBrightXLT_Mirror_Reflectivity.csv')
            df_xlt_mirror = pd.read_csv(filename_xlt_mirror,names = ['Wavelength','Reflectivity'])
        else:
            filename_xlt_mirror = os.path.join(self.current_dir,'data','Templates/StarBrightXLT_Mirror_Reflectivity_Composite.csv')
            df_xlt_mirror = pd.read_csv(filename_xlt_mirror)
        df_xlt_mirror['Reflectivity'] = df_xlt_mirror['Reflectivity']/100.

        #Corrector transmission
        filename_xlt_corrector = os.path.join(self.current_dir,'data','Templates/StarBrightXLT_Corrector_Trasmission.csv')
        df_xlt_corrector = pd.read_csv(filename_xlt_corrector,names = ['Wavelength','Transmission'])
        df_xlt_corrector['Transmission'] = df_xlt_corrector['Transmission']/100.

        if self.use_orig_xlt:
            Ref_mirror = np.polyfit(df_xlt_mirror['Wavelength'],df_xlt_mirror['Reflectivity'],2) 
        else:
            Ref_mirror = np.interp(self.wvl_arr, df_xlt_mirror['Wavelength'].values, df_xlt_mirror['Reflectivity'].values)
        Trasm_corrector = np.polyfit(df_xlt_corrector['Wavelength'],df_xlt_corrector['Transmission'],2)

        self.Ref_mirror = Ref_mirror
        self.Trasm_corrector = Trasm_corrector

        ## Precomputed transmission for error estimation
        filename_transm = os.path.join(self.current_dir,'data','Templates/Transmission_Template_v0_3_3.csv')

        df_transm = pd.read_csv(filename_transm)
        transmission_jolly_tmpl = df_transm['Transmission'].values
        transmission_jolly_wvl = df_transm['Wavelength'].values
        CS_transmission_jolly = CubicSpline(transmission_jolly_wvl,transmission_jolly_tmpl)
        transmission_jolly = CS_transmission_jolly(wvl_arr)

        self.transmission_jolly = transmission_jolly

        self.source_ids = None
        self.tables = None
        self.df_match = None

        self.calibrated_spectra = None
        self.sampling = None

        ## utils
        self.min_int_gaia = 336
        self.max_int_gaia = 1020

        self.useHTM = useHTM

        pass
    
    def Initialize_Params(self):
        """
        Initialize the parameters for the fit.

        Returns:
        - params (lmfit.Parameters): The initialized parameters.

        """
        params = Parameters()
        ### QHY QE ########
        params.add('norm',value = 0.7,min = 0.,max = 1.,vary = False)
        
        params.add('kx0',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('ky0',value = 0.,min = -10.,max = 10.,vary = False)

        params.add('kx',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('ky',value = 0.,min = -10.,max = 10.,vary = False)
        
        params.add('kx2',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('kx3',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('ky2',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('ky3',value = 0.,min = -10.,max = 10.,vary = False)

        params.add('kx4',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('ky4',value = 0.,min = -10.,max = 10.,vary = False)

        params.add('kxy',value = 0.,min = -10.,max = 10.,vary = False)



        ## Best-fit of DESY data
        #####
        #params.add('amplitude',value=293.195050,min = 10.,max = 1000.,vary=False) ## new v0.3.6
        #params.add('center', value = 390.116266,min = 300.,max = 1000.,vary=False)
        #params.add('sigma',value=242.294579,min=1e-2,max = 500.,vary=False)
        #params.add('gamma',value=6.80264968,min =1e-4,max =10.,vary=False) ## new v0.3.6
        #####

        ## Best-fit of Ofek+23 data
        #####
        params.add('amplitude',value=328.1936,min = 10.,max = 1000.,vary=False) 
        params.add('center', value = 570.973,min = 300.,max = 1000.,vary=False)
        params.add('sigma',value=139.77,min=1e-2,max = 500.,vary=False)
        params.add('gamma',value=-0.1517,min =-1,max =10.,vary=False) 

        params.add('l0',value=-0.30,vary=False)
        params.add('l1',value=0.34,vary=False)
        params.add('l2',value=-1.89,vary=False)
        params.add('l3',value=-0.82,vary=False)
        params.add('l4',value=-3.73,vary=False)
        params.add('l5',value=-0.669,vary=False)
        params.add('l6',value=-2.06,vary=False)
        params.add('l7',value=-0.24,vary=False)
        params.add('l8',value=-0.60,vary=False)
        #####



        ### ATM Model
        params.add('pressure',value = 965.,min=960.,max=970.,vary=False)
        params.add('AOD',value=0.084,min=0.01,max=1.0,vary=False)
        params.add('alpha',value=0.6,min=1e-5,max=5.0,vary=False)
        params.add('ozone_col',value=300.,min=10.,max=1000.,vary=False)
        params.add('PW',value=1.4,min=0.1,max=10.,vary=False)
        params.add('temperature',value=self.mnttemp,min=self.mnttemp-5.,max=self.mnttemp+5.,vary=False)

        ### Legendre Polynomials coefficients
        params.add('r0',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('r1',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('r2',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('r3',value = 0.,min = -10.,max = 10.,vary = False)
        params.add('r4',value = 0.,min = -10.,max = 10.,vary = False)

        

        return params
    
    def LegendreModel(self,x,l0,l1,l2,l3,l4,l5,l6,l7,l8):

        newlambda = self.Get_newLambda(x)

        leg_0 = legendre(0)
        leg_1 = legendre(1)
        leg_2 = legendre(2)
        leg_3 = legendre(3)
        leg_4 = legendre(4)
        leg_5 = legendre(5)
        leg_6 = legendre(6)
        leg_7 = legendre(7)
        leg_8 = legendre(8)

        leg_model = np.exp(l0*leg_0(newlambda) + l1*leg_1(newlambda) + l2*leg_2(newlambda) + l3*leg_3(newlambda) + l4*leg_4(newlambda)
        + l5*leg_5(newlambda) + l6*leg_6(newlambda) + l7*leg_7(newlambda) + l8*leg_8(newlambda))

        return leg_model
    

    def Get_newLambda(self, x, min_1=-1., max_1=+1.):
        """
        Calculates the new lambda value based on the input x value and the minimum and maximum values of the wvl_arr.

        Parameters:
        x (float): The input value.
        min_1 (float, optional): The minimum value for the new lambda. Default is -1.
        max_1 (float, optional): The maximum value for the new lambda. Default is +1.

        Returns:
        float: The calculated new lambda value.

        """
        min_0 = np.min(self.wvl_arr)
        max_0 = np.max(self.wvl_arr)

        newlambda = (max_1 - min_1) / (max_0 - min_0) * (x - max_0) + max_1

        return newlambda
   


    def Calculate_OTA_Transmission_from_Model(self,params):
        """
        Calculate the OTA transmission from the model parameters.

        Parameters:
        - params (lmfit.Parameters): The model parameters.

        Returns:
        - OTA_Transmission (numpy.ndarray): The OTA transmission.

        """
        model_sg = lmfit.models.SkewedGaussianModel(independent_vars=['x'])
        #model_pol = lmfit.models.PolynomialModel(degree=3)
        legModel = Model(self.LegendreModel,independent_vars=['x'])

        Model_QE = model_sg*legModel #*model_pol
        
        QE_qhy_Model = Model_QE.eval(params,x = self.wvl_arr)

        Ref_mirror = self.Ref_mirror
        Trasm_corrector = self.Trasm_corrector

        if self.use_orig_xlt:
            OTA_Transmission = QE_qhy_Model*np.polyval(Ref_mirror,self.wvl_arr)*np.polyval(Trasm_corrector,self.wvl_arr)  
        else:
            OTA_Transmission = QE_qhy_Model*Ref_mirror*np.polyval(Trasm_corrector,self.wvl_arr)
    

        return OTA_Transmission
    
    def Calculate_Full_Transmission_from_params(self,params):
        """
        Calculate the full transmission from the model parameters.

        Parameters:
        - params (lmfit.Parameters): The model parameters.

        Returns:
        - transm_full (numpy.ndarray): The full transmission.  
        """

        parvals = params.valuesdict()

        
        params_ota = Parameters()
        params_ota.add('amplitude',value=parvals['amplitude'])
        params_ota.add('center', value = parvals['center'])
        params_ota.add('sigma',value= parvals['sigma'])
        params_ota.add('gamma',value= parvals['gamma'])

        params_ota.add('l0',value = parvals['l0'])
        params_ota.add('l1',value = parvals['l1'])
        params_ota.add('l2',value = parvals['l2'])
        params_ota.add('l3',value = parvals['l3'])
        params_ota.add('l4',value = parvals['l4'])
        params_ota.add('l5',value = parvals['l5'])
        params_ota.add('l6',value = parvals['l6'])
        params_ota.add('l7',value = parvals['l7'])
        params_ota.add('l8',value = parvals['l8'])

        if self.band == 'LAST':
            
            OTA_transmission = self.Calculate_OTA_Transmission_from_Model(params_ota)
        elif self.band == 'SDSS_u':
            filename_transm = os.path.join(self.current_dir,'data','Templates/sdss_u.csv')
            template_transm = pd.read_csv(filename_transm)
        elif self.band == 'SDSS_g':
            filename_transm = os.path.join(self.current_dir,'data','Templates/sdss_g.csv')
            template_transm = pd.read_csv(filename_transm)
            
        elif self.band == 'SDSS_r':
            filename_transm = os.path.join(self.current_dir,'data','Templates/sdss_r.csv')
            template_transm = pd.read_csv(filename_transm)
            
        elif self.band == 'SDSS_i':
            filename_transm = os.path.join(self.current_dir,'data','Templates/sdss_i.csv')
            template_transm = pd.read_csv(filename_transm)
        elif self.band == 'SDSS_z':
            filename_transm = os.path.join(self.current_dir,'data','Templates/sdss_z.csv')
            template_transm = pd.read_csv(filename_transm)
        if self.band != 'LAST':    
            
            interp_transm = interp1d(template_transm['Wavelength'],template_transm['Throughput'],bounds_error=False,fill_value=0.)
            OTA_transmission = interp_transm(self.wvl_arr)

        

        ##Atmospheric components
        rayleigh_transm_Fit = Rayleigh_Transmission(self.z_,parvals['pressure']).make_transmission()
        
        aerosol_transm_Fit = Aerosol_Transmission(self.z_,aod_in=parvals['AOD'],alpha_in=parvals['alpha']).make_transmission()
        
        ozone_transm_Fit = Ozone_Transmission(self.z_,uo_=parvals['ozone_col']).make_transmission()
        
        h2o_transm_Fit = WaterTransmittance(self.z_,pw_=parvals['PW'],p_=parvals['pressure']).make_transmission()
        
        GM_transm_Fit = UMGTransmittance(self.z_,p_=parvals['pressure'],tair=parvals['temperature']).make_transmission()

        ## Chebyshev polynomials term (transmission)
        

        newlambda = self.Get_newLambda(self.wvl_arr)

        cheb_transm = Chebyshev([parvals['r0'],parvals['r1'],parvals['r2'],parvals['r3'],parvals['r4']])
        pol_term = np.exp(cheb_transm(newlambda))
        if self.use_atm:
            transm_full = OTA_transmission*rayleigh_transm_Fit*aerosol_transm_Fit*ozone_transm_Fit*h2o_transm_Fit*GM_transm_Fit*pol_term
        else:
            transm_full = OTA_transmission*pol_term

        return transm_full

    
    def ResidFunc(self,params, x_in, data = None, dataerr = None,magres = False,calc_zp = False,field_corr_=False):
        """
        Calculate the residuals of the fit.

        Parameters:
        - params (lmfit.Parameters): The model parameters.
        - x_in (numpy.ndarray): The input data.
        - data (numpy.ndarray, optional): The observed data. Default is None.
        - dataerr (numpy.ndarray, optional): The error in the observed data. Default is None.
        - magres (bool, optional): Whether to return the residuals in magnitude units. Default is False.
        - calc_zp (bool, optional): Whether to calculate the zero point. Default is False.

        Returns:
        - residuals (numpy.ndarray): The residuals of the fit.

        """

        transm_full = self.Calculate_Full_Transmission_from_params(params)
        parvals = params.valuesdict()
        # Calculate flux (model)
        min_coor = 0.
        max_coor = 1726.

        min_coortr = -1.
        max_coortr = +1.

        if calc_zp:
            Fnu = 3631.e-26 ## zero-flux for AB system
            a = scipy.integrate.trapz(Fnu*transm_full/self.wvl_arr,x=self.wvl_arr)
            b = h.value
            xcoor_ = (max_coortr - min_coortr)/(max_coor-min_coor)*(x_in[0]-max_coor)+max_coortr #(x_in[0] - 863.)/1726. 
            ycoor_ = (max_coortr - min_coortr)/(max_coor-min_coor)*(x_in[1]-max_coor)+max_coortr #(x_in[1] - 863.)/1726. 
        else:    
            a = scipy.integrate.trapz(transm_full*x_in[:,0:-2]*self.wvl_arr,x=self.wvl_arr)
            b = h.value*c.value*1e9
            xcoor_ = (max_coortr - min_coortr)/(max_coor-min_coor)*(x_in[:,-2]-max_coor)+max_coortr #(x_in[:,-2] - 863.)/1726. 
            ycoor_ = (max_coortr - min_coortr)/(max_coor-min_coor)*(x_in[:,-1]-max_coor)+max_coortr #(x_in[:,-1] - 863.)/1726. 
        
        dt = 1. #20.
        Ageom = self.Ageom

        Cheb_x = Chebyshev([0.,parvals['kx'],parvals['kx2'],parvals['kx3'],parvals['kx4']])
        Cheb_y = Chebyshev([0.,parvals['ky'],parvals['ky2'],parvals['ky3'],parvals['ky4']])

        Cheb_xy_x = Chebyshev([0.,parvals['kxy']])
        Cheb_xy_y = Chebyshev([0.,parvals['kxy']])

        model = 2.5*np.log10(parvals['norm']*dt*Ageom*a/b) + Cheb_x(xcoor_) + Cheb_y(ycoor_) + parvals['kx0'] + Cheb_xy_x(xcoor_)*Cheb_xy_y(ycoor_)
        if calc_zp and not field_corr_:
            return model
        if field_corr_ and calc_zp:
            fc_ = Cheb_x(xcoor_) + Cheb_y(ycoor_) + parvals['kx0'] + Cheb_xy_x(xcoor_)*Cheb_xy_y(ycoor_)
            return fc_
        if magres:
            return np.abs(data - model)
        if data is None:
            return model
        if dataerr is None:
            return model - data
        return (model - data)/dataerr
    
    
    
    def Prepare_Spectra_for_Fit_ErrProp(self,source_ids, calibrated_spectra, sampling, df_match):
        """
        Prepare the spectra for fitting.

        Parameters:
        - source_ids (list): The list of source IDs.
        - tables (list): The list of tables.
        - df_match (pandas.DataFrame): The matching dataframe.

        Returns: 
        - Mspectra (numpy.ndarray): The prepared spectra.
        - empirical_flux_error (list): The empirical flux errors.

        """
        print('ATM flag is: ' + str(self.use_atm))
        wvl_arr = self.wvl_arr

        wvl_arr_gaia, mask_gaia, mask_gaia_ir, mask_gaia_uv = make_wvl_array_Gaia()

        #source_ids, calibrated_spectra, sampling, df_match

        
        Mspectra = np.zeros((len(source_ids),len(wvl_arr)))
        Mspectra_sigma = np.zeros((len(source_ids),len(wvl_arr)))
        #Mspectra_pert = np.zeros((len(source_ids),len(wvl_arr)))
        n_sigma = 3.

        tot_residuals = []
        i = 0
        empirical_flux_error = []

        source_index_in_df = np.array(df_match.index)
        #print(source_index_in_df)

        const_fact = 0.5*self.Ageom/(h.value*c.value*1e9)
        
        for ci in source_index_in_df:
            #print(ci)
                
            #df_i = tables[ci]
                
                
            #mask_wvl = (np.asarray(sampling) >= self.min_int_gaia -10.) & (np.asarray(sampling) <= self.max_int_gaia +10.)
            #calibrated_spectra_ci = calibrated_spectra.loc[ci]
            if self.useHTM:
                cols_flux = calibrated_spectra.columns[6:349]
                cols_sigma = calibrated_spectra.columns[349:]
                mu_array = np.asarray(calibrated_spectra[cols_flux].iloc[ci].values)
                sigma_array = np.asarray(calibrated_spectra[cols_sigma].iloc[ci].values)
            else:
            
                mu_array = np.asarray(calibrated_spectra['flux'][ci])
                    
                sigma_array = np.asarray(calibrated_spectra['flux_error'][ci])

            try:
                CS_spectrum = interp1d(sampling,mu_array,bounds_error=True)
                CS_sigma = interp1d(sampling,sigma_array,bounds_error=True)
            except ValueError:
                print('ValueError in CubicSpline')
                print(type(sampling))
                print(type(mu_array))
                continue
            
            
            #new_wvl_arr = np.linspace(280.,max_int, num=500, endpoint=True)

            Mspectra[i,mask_gaia] = CS_spectrum(wvl_arr_gaia)
            Mspectra[i,mask_gaia_uv] = CS_spectrum(wvl_arr_gaia[0])
            Mspectra[i,mask_gaia_ir] = CS_spectrum(wvl_arr_gaia[-1])
            
            Mspectra_sigma[i,mask_gaia] = CS_sigma(wvl_arr_gaia)
            Mspectra_sigma[i,mask_gaia_uv] = CS_sigma(wvl_arr_gaia[0])
            Mspectra_sigma[i,mask_gaia_ir] = CS_sigma(wvl_arr_gaia[-1])

            empirical_sigma_i = const_fact*np.sqrt(np.sum((n_sigma*Mspectra_sigma[i]*self.transmission_jolly*wvl_arr*(wvl_arr[1]-wvl_arr[0]))**2))

            empirical_flux_error.append(empirical_sigma_i)
            
            i += 1


        ##Adding sensor coordinate
        last_x_arr = df_match['LAST_X'].values[np.newaxis]
        last_y_arr = df_match['LAST_Y'].values[np.newaxis]

        Mspectra = np.hstack((Mspectra,last_x_arr.T,last_y_arr.T))

        return Mspectra, empirical_flux_error
    
    
    
    
    def Prepare_Spectra_for_Fit_MC(self,source_ids, calibrated_spectra, sampling, df_match):
        """
        Prepare the spectra for fitting.

        Parameters:
        - source_ids (list): The list of source IDs.
        - tables (list): The list of tables.
        - df_match (pandas.DataFrame): The matching dataframe.

        Returns: 
        - Mspectra (numpy.ndarray): The prepared spectra.
        - empirical_flux_error (list): The empirical flux errors.

        """
        print('ATM flag is: ' + str(self.use_atm))
        wvl_arr = self.wvl_arr

        wvl_arr_gaia, mask_gaia, mask_gaia_ir, mask_gaia_uv = make_wvl_array_Gaia()

        #source_ids, calibrated_spectra, sampling, df_match

        
        Mspectra = np.zeros((len(source_ids),len(wvl_arr)))
        #Mspectra_pert = np.zeros((len(source_ids),len(wvl_arr)))
        n_sigma = 3.5

        tot_residuals = []
        i = 0
        empirical_flux_error = []

        source_index_in_df = np.array(df_match.index)
        #print(source_index_in_df)

        
        for ci in source_index_in_df:
            #print(ci)
                
            #df_i = tables[ci]
                
                
            #mask_wvl = (np.asarray(sampling) >= self.min_int_gaia -10.) & (np.asarray(sampling) <= self.max_int_gaia +10.)
            #calibrated_spectra_ci = calibrated_spectra.loc[ci]
            if self.useHTM:
                cols_flux = calibrated_spectra.columns[6:349]
                cols_sigma = calibrated_spectra.columns[349:]
                mu_array = np.asarray(calibrated_spectra[cols_flux].iloc[ci].values)
                sigma_array = np.asarray(calibrated_spectra[cols_sigma].iloc[ci].values)
            else:
            
                mu_array = np.asarray(calibrated_spectra['flux'][ci])
                    
                sigma_array = np.asarray(calibrated_spectra['flux_error'][ci])

            try:
                CS_spectrum = interp1d(sampling,mu_array,bounds_error=True)
            except ValueError:
                print('ValueError in CubicSpline')
                print(type(sampling))
                print(type(mu_array))
                continue
            
            
            #new_wvl_arr = np.linspace(280.,max_int, num=500, endpoint=True)

            Mspectra[i,mask_gaia] = CS_spectrum(wvl_arr_gaia)
            Mspectra[i,mask_gaia_uv] = CS_spectrum(wvl_arr_gaia[0])
            Mspectra[i,mask_gaia_ir] = CS_spectrum(wvl_arr_gaia[-1])
            
            ## ESTIMATE FLUX ERROR FROM SPECTRUM PERTURBATIONS
            n_samp = 10
            pert_spectra_matrix = np.zeros((n_samp,len(wvl_arr)))
            for bb in range(n_samp):
                pert_spectrum =[]
                for zj in range(len(mu_array)):
                    pert_point = np.random.normal(mu_array[zj],n_sigma*sigma_array[zj],1)
                    pert_spectrum.append(pert_point[0])
                CS_spectrum_pert = CubicSpline(np.asarray(sampling),np.asarray(pert_spectrum),extrapolate=False)
                pert_spectra_matrix[bb,mask_gaia] = CS_spectrum_pert(wvl_arr_gaia)
                pert_spectra_matrix[bb,mask_gaia_uv] = np.random.normal(CS_spectrum_pert(wvl_arr_gaia[0]),n_sigma*sigma_array[0],np.sum(mask_gaia_uv))
                pert_spectra_matrix[bb,mask_gaia_ir] = np.random.normal(CS_spectrum_pert(wvl_arr_gaia[-1]),n_sigma*sigma_array[-1],np.sum(mask_gaia_ir))
                
            
            
            
            estimated_fluxes = self.EstimatePerturbedFluxes(pert_spectra_matrix,wvl_arr_=wvl_arr,transmission=self.transmission_jolly)
            empirical_flux_error.append(np.std(estimated_fluxes))
            ##################################################

            #tot_residuals.append(np.sum(CS_spectrum(df_i['wavelength'][mask_wvl]) - np.asarray(df_i['flux'][mask_wvl])) )
            i += 1


        ##Adding sensor coordinate
        last_x_arr = df_match['LAST_X'].values[np.newaxis]
        last_y_arr = df_match['LAST_Y'].values[np.newaxis]

        Mspectra = np.hstack((Mspectra,last_x_arr.T,last_y_arr.T))

        return Mspectra, empirical_flux_error
    

    def EstimatePerturbedFluxes(self, x_in,wvl_arr_,transmission = None,ret_trans = False):
        """
        Estimate the perturbed fluxes.

        Parameters:
        - x_in (numpy.ndarray): The input data.
        - wvl_arr_ (numpy.ndarray): The wavelength spectrum.
        - transmission (numpy.ndarray, optional): The transmission spectrum. Default is None.
        - ret_trans (bool, optional): Whether to return the transmission spectrum. Default is False.

        Returns:
        - fluxes (numpy.ndarray): The estimated fluxes.

        """
        
        
    
        transm_full = transmission
        
        # Calculate flux (model)
        a = scipy.integrate.trapz(transm_full*x_in*wvl_arr_,x=wvl_arr_)
        
        b = h.value*c.value*1e9
        dt = 1.#20.
        Ageom = self.Ageom
       

        model = 0.5*dt*Ageom*a/b 
        
        if ret_trans:
            return transm_full
        return model
    
    def match_Gaia(self):
        """
        Match with GAIA.

        Returns:
        - source_ids (list): The list of source IDs.
        - tables (list): The list of tables.
        - df_match (pandas.DataFrame): The matching dataframe.

        """
        source_ids, calibrated_spectra, sampling, df_match = GaiaQuery(self.catfile).retrieve_gaia_spectra(useHTM=self.useHTM)
        self.source_ids = source_ids
        self.calibrated_spectra = calibrated_spectra
        self.sampling = sampling
        self.df_match = df_match
        print('Spectra retrieved for catalog file: ' + self.catfile)
        return source_ids, calibrated_spectra, sampling, df_match
    
    def match_Gaia_OLD(self):
        """
        Match with GAIA.

        Returns:
        - source_ids (list): The list of source IDs.
        - tables (list): The list of tables.
        - df_match (pandas.DataFrame): The matching dataframe.

        """
        source_ids, tables, df_match = GaiaQuery(self.catfile).retrieve_gaia_spectra()
        self.source_ids = source_ids
        self.tables = tables
        self.df_match = df_match
        print('Spectra retrieved for catalog file: ' + self.catfile)
        return source_ids, tables, df_match
    
    def fit_transmission(self):
        """
        Fit the transmission.

        Returns:
        - result (lmfit.MinimizerResult): The result of the fit.

        """
        
        params = self.Initialize_Params()

        ## Match with GAIA
        #self.source_ids,self.tables,self.df_match
        #source_ids, tables, df_match = GaiaQuery(self.catfile).retrieve_gaia_spectra()
        #self.source_ids = source_ids
        #self.tables = tables
        #self.df_match = df_match
        #print('Spectra retrieved for catalog file: ' + self.catfile)
        
        df_match = self.df_match
        if self.ErrorEstimation == 'ErrProp':

            Mspectra, empirical_flux_error = self.Prepare_Spectra_for_Fit_ErrProp(self.source_ids, self.calibrated_spectra, self.sampling, self.df_match)
        elif self.ErrorEstimation == 'MC':
            Mspectra, empirical_flux_error = self.Prepare_Spectra_for_Fit_MC(self.source_ids, self.calibrated_spectra, self.sampling, self.df_match)
        
        


        
    
        x = Mspectra
        y_c = np.zeros(len(df_match))
        for i in range(len(df_match)):
            if df_match['G_mag'].values[i] < 16.:
                y_c[i] = df_match['LAST_FLUX_APER_3'].values[i]
            else:
                y_c[i] = df_match['LAST_FLUX_PSF'].values[i]
        
        yerr_emp = np.array(empirical_flux_error)
        yerr = 2.5*np.log10(1 + yerr_emp/y_c)
        y = 2.5*np.log10(y_c)
        
        if np.any(np.isnan(x)):
            raise ValueError('NaN values found in x!')
        if np.any(np.isnan(y)):
            raise ValueError('NaN values found in y!')
        if np.any(np.isnan(yerr)):
            raise ValueError('NaN values found in yerr!')

        print('Fit starts...')
        #nc_orig = len(x)
        #print('Number of calibrators: ' + str(nc_orig))

        flag_0 = False
        flag_1 = False
        flag_2 = False
        flag_3 = False
        flag_4 = False
        flag_5 = False
        
        

        #First fit, only normalisation
        params['norm'].set(vary = True)
        fitter = Minimizer(self.ResidFunc, params, fcn_args=(x, y,yerr))
        out0 = fitter.minimize(method='leastsq')

        for ij in range(3):
            residuals_ = self.ResidFunc(out0.params,x_in=x,data=y,magres=True)
            sigma_clip_out = sigma_clip(residuals_,masked=True,sigma = 3)
            mask_ = sigma_clip_out.mask
            
            x = x[~mask_]
            y = y[~mask_]
            yerr = yerr[~mask_]
            df_match = df_match[~mask_]

            fitter = Minimizer(self.ResidFunc, out0.params, fcn_args=(x, y,yerr))
            out0 = fitter.minimize(method='leastsq')
        
        
        flag_0 = True

        # Second fit, fit only center of QE model and normalisation

        params_fit = out0.params
        
        params_fit['amplitude'].set(vary = False) 
        params_fit['center'].set(vary = True)
        params_fit['sigma'].set(vary = False)
        params_fit['gamma'].set(vary = False) 

        fitter = Minimizer(self.ResidFunc, params_fit, fcn_args=(x, y,yerr))
        out1 = fitter.minimize(method='leastsq')

        flag_1 = True


        params_fit = out1.params
        
        params_fit['norm'].set(vary = False)
        params_fit['center'].set(vary = False)
            


        residuals_ = self.ResidFunc(params_fit,x_in=x,data=y,magres=True)
        sigma_clip_out = sigma_clip(residuals_,masked=True,sigma = 3)
        mask_ = sigma_clip_out.mask
            
        x = x[~mask_]
        y = y[~mask_]
        yerr = yerr[~mask_]
        df_match = df_match[~mask_]

        ## Legendre polynomials term (still tentative)
        leg_flag = False
        
        if leg_flag:
            params_fit['r0'].set(vary = True)
            params_fit['r1'].set(vary = True)
            params_fit['r2'].set(vary = True)
            params_fit['r3'].set(vary = True)
            params_fit['r4'].set(vary = True)

            params_fit['norm'].set(vary = False)
            params_fit['center'].set(vary = False)

        fitter4 = Minimizer(self.ResidFunc, params_fit, fcn_args=(x, y,yerr))
        out4 = fitter4.minimize(method='leastsq')
        flag_4 = True
        
        params_fit = out4.params

        if leg_flag:
            params_fit['r0'].set(vary = False)
            params_fit['r1'].set(vary = False)
            params_fit['r2'].set(vary = False)
            params_fit['r3'].set(vary = False)
            params_fit['r4'].set(vary = False)



        if len(df_match) > 8:
            params_fit['kx0'].set(value = np.random.normal(0., 0.3, 1)[0],vary=True)
            #params_fit['ky0'].set(value =np.random.normal(0., 0.3, 1)[0],vary=False)
            params_fit['kx'].set(value = np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['ky'].set(value =np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['kx2'].set(value = np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['ky2'].set(value =np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['kx3'].set(value = np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['ky3'].set(value =np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['kx4'].set(value = np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['ky4'].set(value =np.random.normal(0., 0.3, 1)[0],vary=True)
            params_fit['kxy'].set(value =np.random.normal(0., 0.3, 1)[0],vary=True)
            

        

        fitter4_bis = Minimizer(self.ResidFunc, params_fit, fcn_args=(x, y,yerr))
        out4_bis = fitter4_bis.minimize(method='leastsq')
        if len(x) > 30:
            for ij in range(3):
                residuals_ = self.ResidFunc(out4_bis.params,x_in=x,data=y,magres=True)
                sigma_clip_out = sigma_clip(residuals_,masked=True,sigma = 2)
                mask_ = sigma_clip_out.mask
                if sum(~mask_) < 30:
                    break
                
                x = x[~mask_]
                y = y[~mask_]
                yerr = yerr[~mask_]
                df_match = df_match[~mask_]

                fitter = Minimizer(self.ResidFunc, out4_bis.params, fcn_args=(x, y,yerr))
                out4_bis = fitter.minimize(method='leastsq')



        params_fit = out4_bis.params

        params_fit['kx0'].set(vary=False)
        params_fit['ky0'].set(vary=False)
        params_fit['kx'].set(vary=False)
        params_fit['ky'].set(vary=False)
        params_fit['kx2'].set(vary=False)
        params_fit['ky2'].set(vary=False)
        params_fit['kx3'].set(vary=False)
        params_fit['ky3'].set(vary=False)
        params_fit['kx4'].set(vary=False)
        params_fit['ky4'].set(vary=False)
        params_fit['kxy'].set(vary=False)

        params_fit['norm'].set(vary = True)
        params_fit['center'].set(vary = False) 

        fitter5 = Minimizer(self.ResidFunc, params_fit, fcn_args=(x, y,yerr))
        out5 = fitter5.minimize(method='leastsq')
        #print(fit_report(out5))
        flag_5 = True


        params_fit = out5.params
        params_fit['norm'].set(vary = False)
        params_fit['ozone_col'].set(vary = False)
        params_fit['PW'].set(vary = True)
        params_fit['AOD'].set(vary = True)

        fitter6 = Minimizer(self.ResidFunc, params_fit, fcn_args=(x, y,yerr))  
        out6 = fitter6.minimize(method='leastsq')
        print(fit_report(out6))

        if flag_0 and flag_1 and flag_2 and flag_3 and flag_4 and flag_5:
            df_match['FIT_STATUS'] = np.ones(len(df_match))
        else:
            df_match['FIT_STATUS'] = np.zeros(len(df_match))

        df_match['MAG_PREDICTED'] = np.array(self.ResidFunc(out6.params,x))

        #df_match.to_pickle(RES_DYR+'/PostChecks_DB_subframe'+ str('{0:03}'.format(sf_num)) +'.pkl')

        params_to_save = out6.params
        #print('Remaining calibrators: ' + str(len(x)))
        #print('We have lost' + str(1-len(x)/nc_orig) + ' percent calibrators')


        df_match_fit = df_match

       
        return params_to_save,df_match_fit