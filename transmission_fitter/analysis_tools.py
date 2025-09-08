
from .fitutils import AbsoluteCalibration
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .lastcatutils import LastCatUtils
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import mad_std
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import scipy
#from tqdm import trange, tqdm
import itertools
from .blazarutils import BlazarQuery
from astropy.constants import h,c
from scipy.interpolate import interp1d
from .abscalutils import make_wvl_array
import os
import glob

class LAST_ABSCAL_Analysis(object):
    def __init__(self,useHTM = True,use_atm = True):
        self.params_cal = None
        self.df_match_cal = None
        self.catfile = None
        self.catlist = None
        self.df_lc = None
        self.match_radius = 2. #in arcsec
        self.df_matchedsources = None
        self.dict_lastframe = {'1': (5,0),'2':(4,0),'3':(3,0),'4':(2,0),'5':(1,0),'6':(0,0),
                  '7': (5,1),'8':(4,1),'9':(3,1),'10':(2,1),'11':(1,1),'12':(0,1),
                  '13': (5,2),'14':(4,2),'15':(3,2),'16':(2,2),'17':(1,2),'18':(0,2),
                  '19': (5,3),'20':(4,3),'21':(3,3),'22':(2,3),'23':(1,3),'24':(0,3)}
        self.useHTM = useHTM
        self.use_atm = use_atm
        wvl_arr = make_wvl_array()
        self.wvl_arr = wvl_arr
        self.current_dir = os.path.dirname(__file__)
        self.cal_results_dir = None
        self.single_output = True
        

        pass

    def calibrate_single_catalog(self, catfile):
            """
            Calibrates a single catalog.

            Args:
                catfile (str): The path to the catalog file.

            Returns:
                tuple: A tuple containing the parameters to save and the DataFrame of matched fits.
            """
            print('Calibrating catalog: {}'.format(catfile))
            abscal_obj = AbsoluteCalibration(catfile=catfile,useHTM=self.useHTM,use_atm=self.use_atm)
            abscal_obj.match_Gaia()

            if abscal_obj.df_match is None:
                print('No matches found in Gaia catalog. Skipping this catalog.')
                return None, None

            params_to_save, df_match_fit = abscal_obj.fit_transmission()

            self.params_cal = params_to_save
            self.df_match_cal = df_match_fit
            self.catfile = catfile

            #self.write_products(resfilename=resfilename)

            print('Calibration complete!')

            return params_to_save, df_match_fit
    
    
    def calibrate_single_catalog_Bootstrap(self,catfile,n_boot=100,resfilename='DefaultResultsBS'):
            """
            Calibrates a single catalog using bootstrap resampling.

            Parameters:
            - catfile (str): The path to the catalog file.
            - n_boot (int): The number of bootstrap iterations. Default is 100.
            - resfilename (str): The name of the file to save the results. Default is 'DefaultResultsBS'.

            Returns:
            None
            """
            ## Calibrate original catalog
            print('Calibrating catalog: {}'.format(catfile))
            abscal_obj = AbsoluteCalibration(catfile=catfile,useHTM=self.useHTM,use_atm=self.use_atm)
            abscal_obj.match_Gaia()

            params_to_save,df_match_fit = abscal_obj.fit_transmission()

            self.params_cal = params_to_save
            self.df_match_cal = df_match_fit
            self.catfile = catfile

            source_ids_0 = abscal_obj.source_ids
            df_match_0 = abscal_obj.df_match
            

            list_index_match = np.array(df_match_0.index)
            
            

            params_to_save_list_bz = []
            df_match_fit_list_bz = []

            for bz in range(n_boot):
                print('Bootstrap iteration: {}'.format(bz))
                ## Bootstrap
                choice_index = np.sort(np.random.choice(list_index_match,int(len(source_ids_0)),replace = True))

                
                
                df_match_bz = df_match_0.loc[choice_index]
                source_ids_bz = df_match_bz['GaiaDR3_ID'].values #np.array(source_ids_0)[choice_index] #backup copy and mask

                abscal_obj.source_ids = source_ids_bz
                abscal_obj.df_match = df_match_bz

                params_to_save_bz,df_match_fit_bz = abscal_obj.fit_transmission()

                params_to_save_list_bz.append(params_to_save_bz)
                df_match_fit_list_bz.append(df_match_fit_bz)

            self.params_cal = params_to_save_list_bz
            self.df_match_cal = df_match_fit_list_bz

            self.write_products_Bootstrap(resfilename=resfilename)
            


    def calibrate_list_of_catalogs(self, catfile_list_txt, resfilename='DefaultResults',single_output = True):
        
        """
        Calibrates a list of catalogs.
        Parameters:
        -----------
        catfile_list_txt : str or list
            A string representing the path to a .txt file containing the list of catalog files, 
            or a list of catalog file paths.
        resfilename : str, optional
            The base name for the results file. Default is 'DefaultResults'.
        single_output : bool, optional
            If True, all results are saved in a single output file. If False, each catalog is 
            processed and saved separately. Default is True.
        Raises:
        -------
        ValueError
            If `catfile_list_txt` is not a .txt file when it is a string.
            If `single_output` is False and `cal_results_dir` is not set.
        TypeError
            If `catfile_list_txt` is neither a string nor a list.
        Returns:
        --------
        None
            Prints 'Calibration complete!' upon successful completion.
        """
        
        self.single_output = single_output
        if not single_output and self.cal_results_dir is None:
            raise ValueError("Please set the calibration results directory using LAST_ABSCAL_Analysis.cal_results_dir().")

        if isinstance(catfile_list_txt, str):
            if not catfile_list_txt.endswith('.txt'):
                raise ValueError("catfile_list_txt must be a .txt file.")
            with open(catfile_list_txt, 'r') as f:
                catfile_list = f.readlines()
        elif isinstance(catfile_list_txt, list):
            catfile_list = catfile_list_txt
        else:
            raise TypeError("catfile_list_txt must be a .txt file or a Python list.")

        catfile_list.sort()
        

        params_to_save_list = []
        df_match_fit_list = []
        catfile_list_processed = []

        for j,catfile in enumerate(catfile_list):
            print('Calibrating catalog number: {} of {}'.format(j+1,len(catfile_list)))
            try:
                params_to_save, df_match_fit = self.calibrate_single_catalog(catfile.strip())
            except:
                print('Error in catalog: {}'.format(catfile))
                continue
            if params_to_save is None:
                #skipping catalog
                continue
            if single_output:
                params_to_save_list.append(params_to_save)
                df_match_fit_list.append(df_match_fit)
                catfile_list_processed.append(catfile)
            else:
                self.params_cal = params_to_save
                self.df_match_cal = df_match_fit
                self.catfile = catfile
                catname = os.path.basename(catfile)
                self.write_products(resfilename=self.cal_results_dir + '/Calibrated_' + catname.replace('.fits',''))

        if single_output:
            self.params_cal = params_to_save_list
            self.df_match_cal = df_match_fit_list
            self.catlist = catfile_list_processed
            
            self.write_products(resfilename=self.cal_results_dir + '/Calibrated_' + resfilename)
        else:
            
            list_calibrated_files = glob.glob(self.cal_results_dir + '/Calibrated_*')
            list_calibrated_files.sort()
            self.get_params_from_calibrated_results(resfile = list_calibrated_files)


        return print('Calibration complete!')
    
    def get_params_from_calibrated_results(self, resfile):
        """
        Retrieves the transmission curve from the calibrated results.

        Parameters:
        - resfile (str): The path to the file containing the calibrated results.

        Returns:
        - params_list (list): A list of parameter dictionaries, each containing the values from the results file.

        """
        if isinstance(resfile,list):
            print('Multiple results files detected. Concatenating results.')
            df_results = pd.read_csv(resfile[0])
            if len(resfile)>1:
                for i in range(1,len(resfile)):
                    df_results = pd.concat([df_results,pd.read_csv(resfile[i])],ignore_index=True)
                catalogs_list = df_results['FILENAME'].values
                
            else:
                catalogs_list = df_results['FILENAME']
        elif isinstance(resfile,str):
            print('Single results file detected.')
            df_results = pd.read_csv(resfile)
        
            catalogs_list = df_results['FILENAME']  

        res_columnames = df_results.columns

        params_list = []

        ## Create a default params object
        for i, catalog_i in enumerate(catalogs_list):
            abscal_obj = AbsoluteCalibration(catfile=catalog_i,useHTM=self.useHTM,use_atm=self.use_atm)
            params_i = abscal_obj.Initialize_Params()
            params_names = params_i.keys()
            ## Update the default params object with the values from the results file
            for colname in res_columnames:
                if colname in params_names:
                    params_i[colname].set(value=df_results[colname][i])

            params_list.append(params_i)

        self.catlist = catalogs_list
        self.params_cal = params_list

        print('Calibration parameters retrieved from stored results.')

        return params_list
    
    def get_params_from_calibrated_results_OLD(self, resfile):
        """
        Retrieves the transmission curve from the calibrated results.

        Parameters:
        - resfile (str): The path to the file containing the calibrated results.

        Returns:
        - params_list (list): A list of parameter dictionaries, each containing the values from the results file.

        """
        df_results = pd.read_csv(resfile)
        catalogs_list = df_results['FILENAME']
        res_columnames = df_results.columns

        params_list = []

        ## Create a default params object
        for i, catalog_i in enumerate(catalogs_list):
            abscal_obj = AbsoluteCalibration(catfile=catalog_i,useHTM=self.useHTM,use_atm=self.use_atm)
            params_i = abscal_obj.Initialize_Params()
            params_names = params_i.keys()
            ## Update the default params object with the values from the results file
            for colname in res_columnames:
                if colname in params_names:
                    params_i[colname].set(value=df_results[colname][i])

            params_list.append(params_i)

        self.catlist = catalogs_list
        self.params_cal = params_list

        print('Calibration parameters retrieved from results file: {}'.format(resfile))

        return params_list
            

    def create_image_photometry_output(self, reference_cat = None,resfile = None,output_folder = None):
        """
        Generate and save a photometry output table for sources in a reference catalog.
        This method processes a reference catalog of astronomical sources, applies photometric calibration,
        computes various photometric and image parameters, and outputs the results as a CSV file. The output
        includes calibrated magnitudes, errors, fluxes, and additional source and image properties.
        Parameters
        ----------
        reference_cat : str, optional
            Path to the reference catalog file (e.g., a LAST catalog). Must be provided.
        resfile : str, optional
            Path to a file containing calibration results. If not provided, uses calibration parameters
            from the current object (requires calibration to have been run previously).
        output_folder : str, optional
            Directory where the output CSV file will be saved. If not provided, saves in the same directory
            as the reference catalog.
        Returns
        -------
        matched_sources_df : pandas.DataFrame
            DataFrame containing the photometry output for all sources in the reference catalog. Columns include:
            - SOURCE_ID, RA, Dec, JD, MAG_PSF_AB, MAG_PSF_AB_ERR, MAG_APER_AB, MAG_APER_AB_ERR, AB_ZP, SN,
                MAG_PSF_LAST, MAG_PSF_LAST_ERR, MAG_APER_LAST, MAG_APER_LAST_ERR, LAST_FLAGS, LAST_X, LAST_Y,
                FWHM, ELLIPTICITY, FLUX_APER_3, FLUX_PSF, FIELD_CORR, LAST_X2, LAST_Y2, LAST_XY, LAST_BACK_IM,
                LAST_BACK_ANNULUS
        Raises
        ------
        ValueError
            If `reference_cat` is not provided, or if calibration parameters are unavailable.
        Notes
        -----
        - The method expects the reference catalog to be in a format compatible with `LastCatUtils`.
        - Calibration is performed using the `AbsoluteCalibration` class.
        - The output CSV file is named as `<reference_cat>_PhotometryOutput.csv`.
        """
        
       
        if reference_cat is None:
            raise ValueError('Please provide a catalog file.')
        if resfile is None:
            try:
                params_list = self.params_cal
            except:
                raise ValueError('Please provide a resfile or run the calibration first.')
        else:
            params_list = self.get_params_from_calibrated_results(resfile)

        last_cat_ref, info_cat_ref = LastCatUtils().tables_from_lastcat(reference_cat.strip());
        last_cat_ref_apy = SkyCoord(ra=last_cat_ref['RA'], dec=last_cat_ref['Dec'], unit='deg', frame='icrs')
        colnames_df = ['SOURCE_ID', 'RA', 'Dec', 'JD', 'MAG_PSF_AB', 'MAG_PSF_AB_ERR','MAG_APER_AB','MAG_APER_AB_ERR', 
                       'AB_ZP', 'SN', 'MAG_PSF_LAST', 'MAG_PSF_LAST_ERR','MAG_APER_LAST', 'MAG_APER_LAST_ERR','LAST_FLAGS','LAST_X','LAST_Y','FWHM',
                       'ELLIPTICITY','FLUX_APER_3','FLUX_PSF','FIELD_CORR','LAST_X2','LAST_Y2','LAST_XY','LAST_BACK_IM','LAST_BACK_ANNULUS']
        matched_sources_df = pd.DataFrame(columns=colnames_df)

        matched_sources_df = matched_sources_df.astype({'SOURCE_ID': 'int', 'RA': 'float', 'Dec': 'float', 'JD': 'float',
                                                        'MAG_PSF_AB': 'float', 'MAG_PSF_AB_ERR': 'float',
                                                        'MAG_APER_AB': 'float', 'MAG_APER_AB_ERR': 'float', 'AB_ZP': 'float',
                                                        'SN': 'float', 'MAG_PSF_LAST': 'float', 'MAG_PSF_LAST_ERR': 'float',
                                                        'MAG_APER_LAST': 'float', 'MAG_APER_LAST_ERR': 'float','LAST_FLAGS':'float','LAST_X':'float','LAST_Y':'float','FWHM':'float','ELLIPTICITY':'float',
                                                        'FLUX_APER_3':'float','FLUX_PSF':'float','FIELD_CORR':'float','LAST_X2':'float','LAST_Y2':'float','LAST_XY':'float',
                                                        'LAST_BACK_IM':'float','LAST_BACK_ANNULUS':'float'})
        if 'coadd' in reference_cat:
            n_coadd  = info_cat_ref.header['NCOADD']
        else:
            n_coadd = 1 
        
        sn_ = last_cat_ref['SN']
        flux_psf_ = n_coadd*last_cat_ref['FLUX_PSF'] / info_cat_ref.header['EXPTIME']
        flux_aper_ = n_coadd*last_cat_ref['FLUX_APER_3'] / info_cat_ref.header['EXPTIME']
        last_x = last_cat_ref['X']
        last_y = last_cat_ref['Y']
        last_x2 = last_cat_ref['X2']
        last_y2 = last_cat_ref['Y2']
        last_xy = last_cat_ref['XY']
        last_back_im = last_cat_ref['BACK_IM']
        last_back_annulus = last_cat_ref['BACK_ANNULUS']
        last_flags = last_cat_ref['FLAGS']
        x_c = np.array([last_x, last_y])

        abscal_obj = AbsoluteCalibration(catfile=reference_cat.strip(),useHTM=self.useHTM,use_atm=self.use_atm)

        abzp_ = abscal_obj.ResidFunc(params_list[j], x_c, calc_zp=True)
        fc_ = abscal_obj.ResidFunc(params_list[j],x_c,calc_zp=True,field_corr_ = True)

        abmag_psf_ = abzp_ - 2.5 * np.log10(flux_psf_)
        abmag_psf_err = 1.086 / sn_

        abmag_aper_ = abzp_ - 2.5 * np.log10(flux_aper_)
        abmag_aper_err = 1.086 / sn_

        ellepticity_ = 1 - (info_cat_ref.header['MED_B'] / info_cat_ref.header['MED_A'])
        try:
            fwhm_ = info_cat_ref.header['FWHM']
        except:
            fwhm_ = 999.0
        
        idx_match_ref = range(len(last_cat_ref))

        df_i_dict = {'SOURCE_ID': idx_match_ref, 'RA': last_cat_ref['RA'],
                        'Dec': last_cat_ref['Dec'], 'JD': info_cat_ref.header['JD'] * np.ones(len(idx_match_ref)),
                        'MAG_PSF_AB': abmag_psf_, 'MAG_PSF_AB_ERR': abmag_psf_err,
                        'MAG_APER_AB': abmag_aper_, 'MAG_APER_AB_ERR': abmag_aper_err, 'AB_ZP': abzp_, 'SN': sn_,
                        'MAG_PSF_LAST': last_cat_ref['MAG_PSF'], 'MAG_PSF_LAST_ERR': abmag_psf_err,
                        'MAG_APER_LAST': last_cat_ref['MAG_APER_3'],
                        'MAG_APER_LAST_ERR': last_cat_ref['MAGERR_APER_3'],'LAST_FLAGS':last_flags,'LAST_X':last_x,'LAST_Y':last_y,
                        'FWHM':fwhm_,'ELLIPTICITY':ellepticity_,'FLUX_APER_3':flux_aper_,'FLUX_PSF':flux_psf_,'FIELD_CORR':fc_,
                        'LAST_X2':last_x2,'LAST_Y2':last_y2,'LAST_XY':last_xy,'LAST_BACK_IM':last_back_im,'LAST_BACK_ANNULUS':last_back_annulus}

        matched_sources_df = pd.concat([matched_sources_df, pd.DataFrame(df_i_dict)], ignore_index=True)

        reference_cat_noext = os.path.splitext(reference_cat)[0]
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            reference_cat_noext = os.path.join(output_folder,os.path.basename(reference_cat_noext))
        
        pd.to_csv(matched_sources_df, reference_cat_noext + '_PhotometryOutput.csv', index=False)

        return matched_sources_df




    def create_matchedsource_df(self, resfile=None,min_jd = None,max_jd = None,coor_target = None):
        """
        Create a DataFrame of matched sources from calibrated results.

        Parameters:
        - resfile (str): Path to the calibrated results file.

        Returns:
        - matched_sources_df (pd.DataFrame): DataFrame containing the matched sources.

        """
        if resfile is None:
            try:
                params_list = self.params_cal
            except:
                raise ValueError('Please provide a resfile or run the calibration first.')
        else:
            params_list = self.get_params_from_calibrated_results(resfile)
        

        catlist = self.catlist

        n_sources_in_cat = []
        limmag_list = []
        catlist_selection = []
        for cat_i in catlist:
            last_cat, info_cat = LastCatUtils().tables_from_lastcat(cat_i.strip());

            if coor_target is not None:
                last_cat_coor = SkyCoord(ra=info_cat.header['RA'], dec=info_cat.header['Dec'], unit='deg', frame='icrs')
                if coor_target.separation(last_cat_coor).to(u.deg).value > 1.:
                    n_sources_in_cat.append(0)
                    continue

            if min_jd is not None:
                if info_cat.header['JD'] < min_jd:
                    n_sources_in_cat.append(0)
                    continue
            if max_jd is not None:
                if info_cat.header['JD'] > max_jd:
                    n_sources_in_cat.append(0)
                    continue
                
            n_sources_in_cat.append(len(last_cat))
            limmag_list.append(info_cat.header['LIMMAG'])
            catlist_selection.append(cat_i)


        #catlist = catlist_selection
        reference_cat = catlist[np.argmax(n_sources_in_cat)]
        print('Catalog with the most sources:', reference_cat)
        
        last_cat_ref, info_cat_ref = LastCatUtils().tables_from_lastcat(reference_cat.strip());
        last_cat_ref_apy = SkyCoord(ra=last_cat_ref['RA'], dec=last_cat_ref['Dec'], unit='deg', frame='icrs')
        colnames_df = ['SOURCE_ID', 'RA', 'Dec', 'JD', 'MAG_PSF_AB', 'MAG_PSF_AB_ERR','MAG_APER_AB','MAG_APER_AB_ERR', 
                       'AB_ZP', 'SN', 'MAG_PSF_LAST', 'MAG_PSF_LAST_ERR','MAG_APER_LAST', 'MAG_APER_LAST_ERR','LAST_FLAGS','LAST_X','LAST_Y','FWHM',
                       'ELLIPTICITY','FLUX_APER_3','FLUX_PSF','FIELD_CORR','LAST_X2','LAST_Y2','LAST_XY','LAST_BACK_IM','LAST_BACK_ANNULUS']
        matched_sources_df = pd.DataFrame(columns=colnames_df)

        matched_sources_df = matched_sources_df.astype({'SOURCE_ID': 'int', 'RA': 'float', 'Dec': 'float', 'JD': 'float',
                                                        'MAG_PSF_AB': 'float', 'MAG_PSF_AB_ERR': 'float',
                                                        'MAG_APER_AB': 'float', 'MAG_APER_AB_ERR': 'float', 'AB_ZP': 'float',
                                                        'SN': 'float', 'MAG_PSF_LAST': 'float', 'MAG_PSF_LAST_ERR': 'float',
                                                        'MAG_APER_LAST': 'float', 'MAG_APER_LAST_ERR': 'float','LAST_FLAGS':'float','LAST_X':'float','LAST_Y':'float','FWHM':'float','ELLIPTICITY':'float',
                                                        'FLUX_APER_3':'float','FLUX_PSF':'float','FIELD_CORR':'float','LAST_X2':'float','LAST_Y2':'float','LAST_XY':'float',
                                                        'LAST_BACK_IM':'float','LAST_BACK_ANNULUS':'float'})

        match_radius = self.match_radius
        print('Matching sources with a radius of {} arcsec'.format(match_radius))
        
        for j, cat_ in enumerate(list(catlist)):
            
            print('Matching sources from catalog: {}'.format(cat_))
            last_cat, info_cat = LastCatUtils().tables_from_lastcat(cat_.strip());
            if 'coadd' in cat_:
                n_coadd  = info_cat.header['NCOADD']
            else:
                n_coadd = 1 
            last_cat_apy = SkyCoord(ra=last_cat['RA'], dec=last_cat['Dec'], unit='deg', frame='icrs')
            idx, sep2d, d3d = last_cat_apy.match_to_catalog_sky(last_cat_ref_apy)

            mask_match = sep2d.to(u.arcsec) < match_radius * u.arcsec
            idx_match_ref = idx[mask_match]

            # Calculate MAG_AB, MAG_AB_ERR and AB_ZP

            sn_ = last_cat['SN'][mask_match]
            flux_psf_ = n_coadd*last_cat['FLUX_PSF'][mask_match] / info_cat.header['EXPTIME']
            flux_aper_ = n_coadd*last_cat['FLUX_APER_3'][mask_match] / info_cat.header['EXPTIME']
            last_x = last_cat['X'][mask_match]
            last_y = last_cat['Y'][mask_match]
            last_x2 = last_cat['X2'][mask_match]
            last_y2 = last_cat['Y2'][mask_match]
            last_xy = last_cat['XY'][mask_match]
            last_back_im = last_cat['BACK_IM'][mask_match]
            last_back_annulus = last_cat['BACK_ANNULUS'][mask_match]
            last_flags = last_cat['FLAGS'][mask_match]
            x_c = np.array([last_x, last_y])

            abscal_obj = AbsoluteCalibration(catfile=cat_.strip(),useHTM=self.useHTM,use_atm=self.use_atm)

            abzp_ = abscal_obj.ResidFunc(params_list[j], x_c, calc_zp=True)
            fc_ = abscal_obj.ResidFunc(params_list[j],x_c,calc_zp=True,field_corr_ = True)

            abmag_psf_ = abzp_ - 2.5 * np.log10(flux_psf_)
            abmag_psf_err = 1.086 / sn_

            abmag_aper_ = abzp_ - 2.5 * np.log10(flux_aper_)
            abmag_aper_err = 1.086 / sn_

            ellepticity_ = 1 - (info_cat.header['MED_B'] / info_cat.header['MED_A'])
            try:
                fwhm_ = info_cat.header['FWHM']
            except:
                fwhm_ = 999.0
            

            df_i_dict = {'SOURCE_ID': idx_match_ref, 'RA': last_cat['RA'][mask_match],
                         'Dec': last_cat['Dec'][mask_match], 'JD': info_cat.header['JD'] * np.ones(len(idx_match_ref)),
                         'MAG_PSF_AB': abmag_psf_, 'MAG_PSF_AB_ERR': abmag_psf_err,
                         'MAG_APER_AB': abmag_aper_, 'MAG_APER_AB_ERR': abmag_aper_err, 'AB_ZP': abzp_, 'SN': sn_,
                         'MAG_PSF_LAST': last_cat['MAG_PSF'][mask_match], 'MAG_PSF_LAST_ERR': abmag_psf_err,
                         'MAG_APER_LAST': last_cat['MAG_APER_3'][mask_match],
                         'MAG_APER_LAST_ERR': last_cat['MAGERR_APER_3'][mask_match],'LAST_FLAGS':last_flags,'LAST_X':last_x,'LAST_Y':last_y,
                         'FWHM':fwhm_,'ELLIPTICITY':ellepticity_,'FLUX_APER_3':flux_aper_,'FLUX_PSF':flux_psf_,'FIELD_CORR':fc_,
                         'LAST_X2':last_x2,'LAST_Y2':last_y2,'LAST_XY':last_xy,'LAST_BACK_IM':last_back_im,'LAST_BACK_ANNULUS':last_back_annulus}

            matched_sources_df = pd.concat([matched_sources_df, pd.DataFrame(df_i_dict)], ignore_index=True)

        self.df_matchedsources = matched_sources_df
        #self.catlist = catlist

        return matched_sources_df
    

    def make_Synthetic_Photometry_LC(self, wvl_array, spectrum_):
        """
        Calculates the synthetic photometry using the given parameters, wavelength array, and spectrum.

        Parameters:
        - params: A dictionary-like object containing the parameter values.
        - wvl_array: An array of wavelengths.
        - spectrum_: An array of corresponding spectrum values.

        Returns:
        - flux_syn: The calculated synthetic photometry flux.

        """
        params = self.params_cal
        flux_syn_list = []
        jd_syn_list = []
        for j, cat_ in enumerate(list(self.catlist)):
            abscal_obj = AbsoluteCalibration(catfile=cat_.strip(),useHTM=self.useHTM,use_atm=self.use_atm)

            transm_full = abscal_obj.Calculate_Full_Transmission_from_params(params[j])
            parvals = params[j].valuesdict()

            ## Interpolate spectrum and resample with self.wvl_arr
            CS_spectrum = interp1d(wvl_array, spectrum_, bounds_error=False,fill_value="extrapolate")
            spectrum_interp = CS_spectrum(self.wvl_arr)

            a = scipy.integrate.trapz(transm_full * spectrum_interp * abscal_obj.wvl_arr, x=abscal_obj.wvl_arr)
            b = h.value * c.value * 1e9

            Ageom = abscal_obj.Ageom

            flux_syn = parvals['norm'] * Ageom * a / b
            flux_syn_list.append(flux_syn)
            jd_syn_list.append(abscal_obj.jd_)
        
        df_lc_syn = pd.DataFrame({'JD': jd_syn_list, 'FLUX_SYN': flux_syn_list})
        df_lc_syn = df_lc_syn.sort_values(by='JD')

        return df_lc_syn

    def get_lc_from_matched_sources(self, coor_target):
        """
        Retrieves the light curve data for the closest source to the target coordinates.

        Parameters:
        - coor_target (SkyCoord): The target coordinates.

        Returns:
        - df_source_id (DataFrame): The light curve data for the closest source, sorted by JD.

        Raises:
        - None

        """
        df_matchedsources = self.df_matchedsources
        match_radius = self.match_radius

        unique_list = df_matchedsources['SOURCE_ID'].unique()
        df_crossmatch = pd.DataFrame(columns=['SOURCE_ID', 'RA', 'Dec'])
        ra_list = []
        dec_list = []
        for source_id in unique_list:
            ra_list.append(df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]['RA'].values[0])
            dec_list.append(df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]['Dec'].values[0])

        df_crossmatch['SOURCE_ID'] = unique_list
        df_crossmatch['RA'] = ra_list
        df_crossmatch['Dec'] = dec_list

        ## Find the closest source to the target
        coor_crossmatch = SkyCoord(ra=df_crossmatch['RA'], dec=df_crossmatch['Dec'], unit='deg', frame='icrs')
        idx, sep2d, d3d = coor_target.match_to_catalog_sky(coor_crossmatch)
        mask_sep = sep2d < match_radius * u.arcsec

        if not (mask_sep[0]):
            print("No match found in catalog.")
            return None
        else:
            source_id = df_crossmatch['SOURCE_ID'][idx]
            df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]
            df_source_id = df_source_id.sort_values(by='JD')

            return df_source_id
        




    def plot_rms(self, cat_start=None, cat_end=None, outfile=None):
        """
        Plots the robust standard deviation of magnitude measurements against the AB magnitude for aperture and PSF photometry.

        Parameters:
        - cat_start (optional): Start index of the catalog. Default is None.
        - cat_end (optional): End index of the catalog. Default is None.
        - outfile (optional): Filepath to save the plot. Default is None.

        Raises:
        - ValueError: If self.df_matchedsources is None.

        Returns:
        - None
        """
        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None, create df_matchedsources object first!")

        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()

        sns.set_context('talk', font_scale=1.4, rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True

        fig, ax = plt.subplots(1, figsize=(12, 10))

        for i, item in enumerate(['MAG_APER_AB', 'MAG_PSF_AB']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]
                if len(df_source_id) < 8:
                    continue
                df_source_id = df_source_id.sort_values(by='JD')
                mag_sel = df_source_id[item].values

                pc_ = np.nanpercentile(mag_sel, [15.87, 50., 84.13])
                sigma_ = (pc_[2] - pc_[0]) / 2.
                rms_arr.append(mad_std(mag_sel, ignore_nan=True))
                ab_mag_arr.append(pc_[1])

            if item == 'MAG_APER_AB':
                c_ = 'C0'
                l_ = 'Aperture photometry'
            elif item == 'MAG_PSF_AB':
                c_ = 'C1'
                l_ = 'PSF photometry'

            ax.scatter(ab_mag_arr, rms_arr, s=3, label=l_, color=c_)

        ax.set_xlabel('AB MAG [mag]')
        ax.set_xlim(10., 21.)
        ax.semilogy()
        ax.set_ylabel('Robust Standard Deviation [mag]')
        ax.legend()
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        plt.show()

        return None
    
    def plot_rms_bestphotometry(self, cat_start=None, cat_end=None, outfile=None):
        """
        Plots the robust standard deviation of the best photometry (either aperture or PSF) 
        against the AB magnitude for each source.

        Args:
            cat_start (optional): Start index of the catalog. Defaults to None.
            cat_end (optional): End index of the catalog. Defaults to None.
            outfile (optional): File path to save the plot. Defaults to None.

        Returns:
            ab_mag_arr_best (ndarray): Array of AB magnitudes for the best photometry.
            rms_arr_best (ndarray): Array of robust standard deviations for the best photometry.
        """

        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None, create df_matchedsources object first!")

        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()

        sns.set_context('paper', font_scale=1.4, rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True

        fig, ax = plt.subplots(1, figsize=(12, 10))

        for i, item in enumerate(['MAG_APER_AB', 'MAG_PSF_AB']):
            rms_arr = []
            ab_mag_arr = []
            n_timestamps = len(df_matchedsources['JD'].unique())
            for source_id in unique_list:
                df_source_id = df_matchedsources[(df_matchedsources['SOURCE_ID'] == source_id) & (df_matchedsources['FWHM'] < 3.)]
                if len(df_source_id) < 0.5 * n_timestamps:
                    continue
                df_source_id = df_source_id.sort_values(by='JD')
                mag_sel = df_source_id[item].values

                pc_ = np.nanpercentile(mag_sel, [15.87, 50., 84.13])
                sigma_ = (pc_[2] - pc_[0]) / 2.
                rms_arr.append(mad_std(mag_sel, ignore_nan=True))
                ab_mag_arr.append(pc_[1])

            if item == 'MAG_APER_AB':
                c_ = 'C0'
                l_ = 'Aperture photometry'
                ab_mag_arr_aper = ab_mag_arr
                rms_arr_aper = rms_arr
            elif item == 'MAG_PSF_AB':
                c_ = 'C1'
                l_ = 'PSF photometry'
                ab_mag_arr_psf = ab_mag_arr
                rms_arr_psf = rms_arr

        mask_aper = (np.array(rms_arr_aper) < np.array(rms_arr_psf)) & (np.array(ab_mag_arr_aper) < 16.)
        ab_mag_arr_best = np.concatenate((np.array(ab_mag_arr_aper)[mask_aper], np.array(ab_mag_arr_psf)[~mask_aper]))
        rms_arr_best = np.concatenate((np.array(rms_arr_aper)[mask_aper], np.array(rms_arr_psf)[~mask_aper]))

        ax.scatter(ab_mag_arr_best, rms_arr_best, s=5, color='C0')

        ax.set_xlabel('AB MAG [mag]')
        ax.set_xlim(10., np.max(ab_mag_arr_psf) + 0.3)
        ax.semilogy()
        ax.set_ylabel('Robust Standard Deviation [mag]')

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        return ab_mag_arr_best, rms_arr_best

    def plot_rms_landoltcal(self, cat_start=None, cat_end=None, outfile=None):
        """
        Plots the RMS (Robust Standard Deviation) of stars in the image against their AB magnitude.
        Also includes the RMS of LANDOLT-STETSON calibrators in the image.

        Parameters:
            cat_start (int): The starting index of the catalog.
            cat_end (int): The ending index of the catalog.
            outfile (str): The path to save the plot as an image file.

        Raises:
            ValueError: If self.df_matchedsources is None.

        Returns:
            None
        """
        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None, create df_matchedsources object first!")

        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()

        # ALL STARS IN THE IMAGE
        # Get the RMS of all stars in the image
        sns.set_context('talk', font_scale=1.4, rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True

        fig, ax = plt.subplots(1, figsize=(12, 10))

        for i, item in enumerate(['MAG_APER_AB', 'MAG_PSF_AB']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]
                if len(df_source_id) < 10.:
                    continue
                df_source_id = df_source_id.sort_values(by='JD')
                mag_sel = df_source_id[item].values

                pc_ = np.nanpercentile(mag_sel, [15.87, 50., 84.13])
                rms_arr.append(mad_std(mag_sel, ignore_nan=True))
                ab_mag_arr.append(pc_[1])

            if item == 'MAG_APER_AB':
                c_ = 'C0'
            elif item == 'MAG_PSF_AB':
                c_ = 'C1'

            ax.scatter(ab_mag_arr, rms_arr, s=3, alpha=0.5, color=c_)

        # LANDOLT-STETSON CALIBRATORS IN THE IMAGE
        # Get the RMS of all LANDOLT-STETSON calibrators in the image
        sourcequery = BlazarQuery()
        ed3_calibrators_apy = sourcequery.edr3_calibrators_apy
        df_matchedsources_all_apy = SkyCoord(ra=df_matchedsources['RA'].values * u.deg,
                                             dec=df_matchedsources['Dec'].values * u.deg, frame='icrs')
        idx, sep2d, d3d = df_matchedsources_all_apy.match_to_catalog_sky(ed3_calibrators_apy)

        mask_match = sep2d.to(u.arcsec) < 2. * u.arcsec
        arr_idx_all = df_matchedsources.index.to_numpy()
        idx_match = arr_idx_all[mask_match]
        df_matchedsources_cal = df_matchedsources.loc[idx_match]

        unique_list = df_matchedsources_cal['SOURCE_ID'].unique()
        for i, item in enumerate(['MAG_APER_AB', 'MAG_PSF_AB']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources_cal[df_matchedsources_cal['SOURCE_ID'] == source_id]
                if len(df_source_id) < 10.:
                    continue
                df_source_id = df_source_id.sort_values(by='JD')
                mag_sel = df_source_id[item].values

                pc_ = np.nanpercentile(mag_sel, [15.87, 50., 84.13])
                rms_arr.append(mad_std(mag_sel, ignore_nan=True))
                ab_mag_arr.append(pc_[1])

            if item == 'MAG_APER_AB':
                c_ = 'C0'
                l_ = 'Aperture photometry'
            elif item == 'MAG_PSF_AB':
                c_ = 'C1'
                l_ = 'PSF photometry'

            ax.scatter(ab_mag_arr, rms_arr, s=3, label=l_, alpha=1.0, color=c_)

        ax.set_xlabel('AB MAG [mag]')
        ax.set_xlim(10., 21.)
        ax.semilogy()
        ax.set_ylabel('Robust Standard Deviation [mag]')
        ax.legend()
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        plt.show()

        return None
    
    def plot_rms_LAST(self):

        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None,create df_matchedsources object first!")
        
        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()
        

        

        sns.set_context('talk',font_scale=1.4,rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        
        fig, ax = plt.subplots(1,figsize=(12,10))
        
        for i,item in enumerate(['MAG_APER_LAST','MAG_PSF_LAST']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]  
                if len(df_source_id) < 10.: continue 
                df_source_id = df_source_id.sort_values(by='JD')
                pc_ = np.nanpercentile(df_source_id[item],[15.87,50.,84.13])
                sigma_ = (pc_[2]-pc_[0])/2.
                rms_arr.append(mad_std(df_source_id[item].values,ignore_nan=True))
                ab_mag_arr.append(pc_[1])
            
            ax.scatter(ab_mag_arr,rms_arr,s=3,label =item)
            
            
            
        ax.set_xlabel('LAST MAG')
        ax.set_xlim(10.,21.)
        ax.semilogy()
        ax.set_ylabel('Robust Standard Deviation [mag]')
        ax.legend()

        plt.show()


        return None
    
    def plot_rms_AB_vs_LAST(self):
        """
        Plots the Robust Standard Deviation (RMS) of different magnitude measurements (MAG_APER_AB, MAG_PSF_AB, MAG_APER_LAST, MAG_PSF_LAST) 
        against the average magnitude (AB magnitude) for each unique source ID in the dataframe df_matchedsources.

        Raises:
            ValueError: If self.df_matchedsources is None, create df_matchedsources object first.
        """
        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None, create df_matchedsources object first!")
        
        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()

        sns.set_context('talk', font_scale=1.4, rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        
        fig, ax = plt.subplots(1, figsize=(12, 10))
        
        for i, item in enumerate(['MAG_APER_AB', 'MAG_PSF_AB', 'MAG_APER_LAST', 'MAG_PSF_LAST']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]
                if len(df_source_id) < 10.:
                    continue 
                df_source_id = df_source_id.sort_values(by='JD')
                pc_ = np.nanpercentile(df_source_id[item], [15.87, 50., 84.13])
                sigma_ = (pc_[2] - pc_[0]) / 2.
                rms_arr.append(mad_std(df_source_id[item].values, ignore_nan=True))
                ab_mag_arr.append(pc_[1])
            
            ax.scatter(ab_mag_arr, rms_arr, s=3, label=item)
            
        ax.set_xlabel('MAG')
        ax.set_xlim(10., 21.)
        ax.semilogy()
        ax.set_ylabel('Robust Standard Deviation [mag]')
        ax.legend()

        plt.show()

        return None
    
    def plot_sem(self):

        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None,create df_matchedsources object first!")
        
        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()
        

        

        sns.set_context('talk',font_scale=1.4,rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        
        fig, ax = plt.subplots(1,figsize=(12,10))
        
        for i,item in enumerate(['MAG_APER_AB','MAG_PSF_AB']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]   
                if len(df_source_id) < 10.: continue 
                df_source_id = df_source_id.sort_values(by='JD')
                pc_ = np.nanpercentile(df_source_id[item],[15.87,50.,84.13])
                rms_arr.append(scipy.stats.sem(df_source_id[item].values,nan_policy = 'omit'))
                ab_mag_arr.append(pc_[1])
            
            rms_arr = np.array(rms_arr)
            ab_mag_arr = np.array(ab_mag_arr)
            ax.scatter(ab_mag_arr[~np.isnan(rms_arr)],rms_arr[~np.isnan(rms_arr)],s=3,label =item)
            
            
        ax.set_xlabel('AB MAG')
        ax.set_xlim(10.,21.)
        ax.semilogy()
        ax.set_ylabel('Standard Error on the Mean [mag]')
        ax.legend()

        plt.show()


        return None
    
    def plot_sem_LAST(self):

        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None,create df_matchedsources object first!")
        
        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()
        

        

        sns.set_context('talk',font_scale=1.4,rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        
        fig, ax = plt.subplots(1,figsize=(12,10))
        
        for i,item in enumerate(['MAG_APER_LAST','MAG_PSF_LAST']):
            rms_arr = []
            ab_mag_arr = []

            for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]   
                if len(df_source_id) < 10.: continue 
                df_source_id = df_source_id.sort_values(by='JD')
                pc_ = np.nanpercentile(df_source_id[item],[15.87,50.,84.13])
                rms_arr.append(scipy.stats.sem(df_source_id[item].values,nan_policy = 'omit'))
                ab_mag_arr.append(pc_[1])
            
            rms_arr = np.array(rms_arr)
            ab_mag_arr = np.array(ab_mag_arr)
            ax.scatter(ab_mag_arr[~np.isnan(rms_arr)],rms_arr[~np.isnan(rms_arr)],s=3,label =item)
            
            
        ax.set_xlabel('LAST MAG')
        ax.set_xlim(10.,21.)
        ax.semilogy()
        ax.set_ylabel('Standard Error on the Mean [mag]')
        ax.legend()

        plt.show()


        return None
    
    def plot_transmissions(self, resfile=None):
        """
        Plots the transmissions for each item in the catlist.

        Parameters:
        - resfile (str): Path to the results file. If None, the method will read from self.params_cal.

        Returns:
        None
        """

        if self.params_cal is None:
            print("self.params_cal is None, reading from results file.")
            self.get_params_from_calibrated_results(resfile)

        params_list = self.params_cal
        catlist = self.catlist

        sns.set_context('talk', font_scale=1.4, rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True

        fig, ax = plt.subplots(1, figsize=(12, 10))

        for i, item in enumerate(self.catlist):
            abscal_obj = AbsoluteCalibration(catfile=item, useHTM=self.useHTM, use_atm=self.use_atm)
            transm_full = abscal_obj.Calculate_Full_Transmission_from_params(params_list[i])
            parvals = params_list[i].valuesdict()

            ax.plot(abscal_obj.wvl_arr, transm_full * parvals['norm'], label=str(i))

        ax.set_xlabel('Wavelegth [nm]')
        ax.set_ylim(0.,)
        ax.set_xlim(np.min(abscal_obj.wvl_arr), np.max(abscal_obj.wvl_arr))

        ax.set_ylabel('Transmission')
        ax.legend(ncol=4)

        plt.show()


    
 
    def get_lc(self,coor_target):
        """
        Retrieves the light curve at the specified coordinates.

        Parameters:
        - coor_target: SkyCoord object representing the target coordinates.

        Returns:
        - df_lc: Pandas DataFrame containing the light curve data.
        """
        params_cal = self.params_cal
        #df_match_cal = self.df_match_cal


        #if not isinstance(df_match_cal, list):
        #    raise TypeError("df_match_cal must be a list.")
        


        df_lc = pd.DataFrame(columns=['SOURCE_ID', 'RA', 'Dec', 'JD', 'MAG_PSF_AB', 'MAG_PSF_AB_ERR','MAG_APER_AB','MAG_APER_AB_ERR', 
                       'AB_ZP', 'SN', 'MAG_PSF_LAST', 'MAG_PSF_LAST_ERR','MAG_APER_LAST', 'MAG_APER_LAST_ERR','LAST_FLAGS','LAST_X','LAST_Y','FWHM','ELLIPTICITY'])

        df_lc = df_lc.astype({'SOURCE_ID': 'int', 'RA': 'float', 'Dec': 'float', 'JD': 'float',
                                                        'MAG_PSF_AB': 'float', 'MAG_PSF_AB_ERR': 'float',
                                                        'MAG_APER_AB': 'float', 'MAG_APER_AB_ERR': 'float', 'AB_ZP': 'float',
                                                        'SN': 'float', 'MAG_PSF_LAST': 'float', 'MAG_PSF_LAST_ERR': 'float',
                                                        'MAG_APER_LAST': 'float', 'MAG_APER_LAST_ERR': 'float','LAST_FLAGS':'float','LAST_X':'float','LAST_Y':'float','FWHM':'float','ELLIPTICITY':'float'})

        print('Getting light curve at coordinates: {}'.format(coor_target))

        match_radius = self.match_radius

        for i,params_ in enumerate(params_cal):
            last_cat,info_cat = LastCatUtils().tables_from_lastcat(self.catlist[i].strip())
            abscal_obj = AbsoluteCalibration(catfile=self.catlist[i].strip(),useHTM=self.useHTM,use_atm=self.use_atm)
            coord_last_cat = SkyCoord(ra = last_cat['RA'],dec = last_cat['DEC'], unit = 'deg',frame = 'icrs')
            if 'coadd' in self.catlist[i]:
                n_coadd  = info_cat.header['NCOADD']
            else:
                n_coadd = 1 
            sep = coor_target.separation(coord_last_cat)
            
            mask_sep = sep < match_radius*u.arcsec

        
            if sum(mask_sep) == 0: 
                print("No match found in catalog.")
                continue
                
            elif sum(mask_sep) == 1:
                last_idx_ = np.where(mask_sep == True)[0][0]
                jd_ = info_cat.header['JD']
                sn_ = last_cat['SN'][last_idx_]
                flux_aper_ = n_coadd*last_cat['FLUX_APER_3'][last_idx_]/info_cat.header['EXPTIME']
                #flux_aper_err_ = last_cat['FLUXERR_APER_3'][last_idx_]/info_cat.header['EXPTIME']
                flux_psf_ = n_coadd*last_cat['FLUX_PSF'][last_idx_]/info_cat.header['EXPTIME']
                #flux_psf_err_ = last_cat['FLUXERR_APER_3'][last_idx_]/info_cat.header['EXPTIME']
                last_x = last_cat['X'][last_idx_]
                last_y = last_cat['Y'][last_idx_]
                last_flags = last_cat['FLAGS'][last_idx_]
                
                x_c = np.array([last_x,last_y])

                abzp_ = abscal_obj.ResidFunc(params_, x_c, calc_zp=True)

                abmag_psf_ = abzp_ - 2.5 * np.log10(flux_psf_)
                abmag_psf_err = 1.086 / sn_

                abmag_aper_ = abzp_ - 2.5 * np.log10(flux_aper_)
                abmag_aper_err = 1.086 / sn_

                ellepticity_ = 1 - (info_cat.header['MED_B'] / info_cat.header['MED_A'])

                try:
                    fwhm_ = info_cat.header['FWHM']
                except:
                    fwhm_ = 999.0

                df_i_dict = {'SOURCE_ID': last_idx_, 'RA': last_cat['RA'][last_idx_],
                            'Dec': last_cat['Dec'][last_idx_], 'JD': info_cat.header['JD'],
                            'MAG_PSF_AB': abmag_psf_, 'MAG_PSF_AB_ERR': abmag_psf_err,
                            'MAG_APER_AB': abmag_aper_, 'MAG_APER_AB_ERR': abmag_aper_err, 'AB_ZP': abzp_, 'SN': sn_,
                            'MAG_PSF_LAST': last_cat['MAG_PSF'][last_idx_], 'MAG_PSF_LAST_ERR': abmag_psf_err,
                            'MAG_APER_LAST': last_cat['MAG_APER_3'][last_idx_],
                            'MAG_APER_LAST_ERR': last_cat['MAGERR_APER_3'][last_idx_],'LAST_FLAGS':last_flags,'LAST_X':last_x,'LAST_Y':last_y,'FWHM':fwhm_,'ELLIPTICITY':ellepticity_}

                  
                df_lc = pd.concat([df_lc,pd.DataFrame([df_i_dict])],ignore_index=True)  
                
            else:
                print('Too many coincidences! Skipping this catalog.')
                continue
        df_lc.reset_index(drop=True,inplace=True)
        self.df_lc = df_lc

        return df_lc
        
    def plot_lc(self, coor_target):
        """
        Plots the light curve for a given target.

        Parameters:
        coor_target (str): The coordinates of the target.

        Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
        ax (matplotlib.axes._subplots.AxesSubplot): The generated axes object.
        """
        df_lc = self.get_lc(coor_target)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.errorbar(df_lc['JD'] - df_lc['JD'].values[0], df_lc['AB_MAG'], yerr=df_lc['AB_MAG_ERR'], fmt='o')
        ax.set_xlabel('JD')
        ax.set_ylabel('AB_MAG')
        plt.gca().invert_yaxis()
        plt.show()

        return fig, ax



    

    def write_products(self,resfilename='DefaultResults'):
        """
        Write the calibration results to a CSV file.

        Parameters:
        - resfilename (str): The name of the CSV file to save the results. Default is 'DefaultResults'.

        Returns:
        - None
        """
        params_cal = self.params_cal
        df_match_cal = self.df_match_cal
        catfile = self.catfile
        catfile_list = self.catlist

        columns = ['FILENAME','norm', 'kx0','ky0', 'kx', 'ky', 'kx2', 'kx3', 'ky2', 'ky3','kx4','ky4','kxy',
                'amplitude', 'center', 'sigma', 'gamma', 'pressure', 'AOD', 'alpha', 'ozone_col', 'PW', 'temperature','r0','r1','r2','r3','r4']
        
        df_results = pd.DataFrame(columns=columns)
        if catfile_list is not None:
            for i in range(len(catfile_list)):
                row = [os.path.abspath(catfile_list[i].strip()),params_cal[i]['norm'].value,params_cal[i]['kx0'].value,params_cal[i]['ky0'].value,params_cal[i]['kx'].value,params_cal[i]['ky'].value,
                    params_cal[i]['kx2'].value,params_cal[i]['kx3'].value,params_cal[i]['ky2'].value,params_cal[i]['ky3'].value,params_cal[i]['kx4'].value,params_cal[i]['ky4'].value,params_cal[i]['kxy'].value,
                    params_cal[i]['amplitude'].value,params_cal[i]['center'].value,params_cal[i]['sigma'].value,params_cal[i]['gamma'].value,
                    params_cal[i]['pressure'].value,params_cal[i]['AOD'].value,params_cal[i]['alpha'].value,params_cal[i]['ozone_col'].value,
                    params_cal[i]['PW'].value,params_cal[i]['temperature'].value,params_cal[i]['r0'].value,params_cal[i]['r1'].value,params_cal[i]['r2'].value,params_cal[i]['r3'].value,params_cal[i]['r4'].value]
                df_results.loc[i] = row
        else:
            row = [os.path.abspath(catfile.strip()),params_cal['norm'].value,params_cal['kx0'].value,params_cal['ky0'].value,params_cal['kx'].value,params_cal['ky'].value,
                params_cal['kx2'].value,params_cal['kx3'].value,params_cal['ky2'].value,params_cal['ky3'].value,params_cal['kx4'].value,params_cal['ky4'].value,params_cal['kxy'].value,
                params_cal['amplitude'].value,params_cal['center'].value,params_cal['sigma'].value,params_cal['gamma'].value,
                params_cal['pressure'].value,params_cal['AOD'].value,params_cal['alpha'].value,params_cal['ozone_col'].value,
                params_cal['PW'].value,params_cal['temperature'].value,params_cal['r0'].value,params_cal['r1'].value,params_cal['r2'].value,params_cal['r3'].value,params_cal['r4'].value]
            df_results.loc[0] = row

        df_results.to_csv(resfilename+'.csv',index=False)
        return print('Results saved to {}.csv'.format(resfilename))
    
    def write_products_Bootstrap(self, resfilename='DefaultResultsBS'):
        """
        Write the bootstrap analysis results to a CSV file.

        Args:
            resfilename (str): The name of the CSV file to save the results to. Default is 'DefaultResultsBS'.

        Returns:
            None
        """
        params_cal = self.params_cal
        catfile = self.catfile

        columns = ['FILENAME', 'norm', 'kx0','ky0', 'kx', 'ky', 'kx2', 'kx3', 'ky2', 'ky3','kx4','ky4',
                   'amplitude', 'center', 'sigma', 'gamma', 'pressure', 'AOD', 'alpha', 'ozone_col', 'PW', 'temperature','r0','r1','r2','r3','r4']
        df_results = pd.DataFrame(columns=columns)

        for i in range(len(params_cal)):
            row = [catfile.strip(), params_cal[i]['norm'].value, params_cal[i]['kx0'].value,params_cal[i]['ky0'].value, params_cal[i]['kx'].value,
                   params_cal[i]['ky'].value, params_cal[i]['kx2'].value,
                   params_cal[i]['kx3'].value, params_cal[i]['ky2'].value, params_cal[i]['ky3'].value,params_cal[i]['kx4'].value,params_cal[i]['ky4'].value,
                   params_cal[i]['amplitude'].value, params_cal[i]['center'].value, params_cal[i]['sigma'].value,
                   params_cal[i]['gamma'].value, params_cal[i]['pressure'].value, params_cal[i]['AOD'].value,
                   params_cal[i]['alpha'].value, params_cal[i]['ozone_col'].value, params_cal[i]['PW'].value,
                   params_cal[i]['temperature'].value, params_cal[i]['r0'].value, params_cal[i]['r1'].value,params_cal[i]['r2'].value,params_cal[i]['r3'].value,params_cal[i]['r4'].value]
            df_results.loc[i] = row

        df_results.to_csv(resfilename + '.csv', index=False)
        return print('Results saved to {}.csv'.format(resfilename))
    

    def plot_BS_results(self,cropid=99,outfile=None):
        # IN PROGRESS
        if self.df_matchedsources is None:
            raise ValueError("self.df_matchedsources is None,create df_matchedsources object first!")
        
        df_matchedsources = self.df_matchedsources

        unique_list = df_matchedsources['SOURCE_ID'].unique()
        rms_arr = []
        X_ = []
        Y_ = []
        for source_id in unique_list:
                df_source_id = df_matchedsources[df_matchedsources['SOURCE_ID'] == source_id]   
                
                #rms_arr.append(mad_std(df_source_id['AB_ZP'].values))
                rms_arr.append(np.sqrt(np.sum((df_source_id['AB_ZP'].values -np.mean(df_source_id['AB_ZP'].values))**2)/(len(df_source_id['AB_ZP'].values) - 1)))
        
                X_.append(df_source_id['LAST_X'].mean())
                Y_.append(df_source_id['LAST_Y'].mean())
        
        X_ = np.array(X_)
        Y_ = np.array(Y_)
        Pi = np.hstack((X_.reshape(len(X_),1),Y_.reshape(len(Y_),1)))
        N = 1700
        grid_x, grid_y = np.mgrid[0:N:30j, 0:N:30j]
        Z_cubic = scipy.interpolate.griddata(Pi, rms_arr, (grid_x, grid_y), method='linear',fill_value=0.)
        
        sns.set_context('paper',font_scale=1.4,rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        fig, ax = plt.subplots(1,figsize=(8,8))

        im = ax.imshow(Z_cubic.T, extent=(0, N, 0, N), origin='lower',cmap='jet',aspect='auto')

        ax.set_xlim(0,N)
        ax.set_ylim(0,N)
        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')
        ax.set_title('Bootstrap CropID: {}'.format(cropid))
        fig.colorbar(im, ax=ax)

        

        if outfile is not None:
            fig.savefig(outfile+'.pdf',bbox_inches='tight')

        plt.show()
        return rms_arr 
    
    def plot_BS_results_grid(self,cropid=99,outfile=None):
        # IN PROGRESS
        X_g = np.array(range(1727))
        Y_g = np.array(range(1727))

        combinations_ = list(itertools.product(X_g,Y_g))

        X_ = []
        Y_ = []

        for comb in combinations_:
            X_.append(comb[0])
            Y_.append(comb[1])
        
        X_ = np.array(X_)
        Y_ = np.array(Y_)
        x_in_grid = np.array([X_,Y_])

        
        abscalobj = AbsoluteCalibration(self.catfile,useHTM=self.useHTM,use_atm=self.use_atm)
        zp_matrix = np.zeros((len(X_),len(self.params_cal)))
        for i,item in enumerate(self.params_cal):
            zp_matrix[:,i] = abscalobj.ResidFunc(params=item,x_in=x_in_grid,calc_zp=True)
        
        rms_arr = np.std(zp_matrix,axis=1)/1.4826

            
        Pi = np.hstack((X_.reshape(len(X_),1),Y_.reshape(len(Y_),1)))
        N = 1727
        grid_x, grid_y = np.mgrid[0:N:30j, 0:N:30j]
        Z_cubic = scipy.interpolate.griddata(Pi, rms_arr, (grid_x, grid_y), method='linear',fill_value=0.)
        
        sns.set_context('paper',font_scale=1.4,rc={"lines.linewidth": 2.5})
        rcParams['text.usetex'] = True
        fig, ax = plt.subplots(1,figsize=(8,8))

        im = ax.imshow(Z_cubic.T, extent=(0, N, 0, N), origin='lower',cmap='jet',aspect='auto')

        ax.set_xlim(0,1667)
        ax.set_ylim(0,1667)
        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')
        ax.set_title('Bootstrap CropID: {}'.format(cropid))
        fig.colorbar(im, ax=ax)

        

        if outfile is not None:
            fig.savefig(outfile+'.pdf',bbox_inches='tight')

        plt.show()
        return rms_arr 
    
    def plot_residuals_2D_image(self,resfilename):
        sns.set_context('paper',font_scale=1.4,rc={"lines.linewidth": 2.5})
        dict_lastframe = self.dict_lastframe
        fig,axs = plt.subplots(6,4,sharex=True,sharey=True,figsize=(20,20))

        for i,item in enumerate(self.df_match_cal):
            framenum = i+1
            frame_tup = dict_lastframe[str(framenum)]

            df_i = self.df_match_cal[i]
            
            mag_obs = []
            mag_pred = df_i['MAG_PREDICTED'].values

            for j in range(len(df_i)):
                if df_i['G_mag'].values[j] > 16:
                    mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_PSF'].values[j]))
                else:
                    mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_APER_3'].values[j]))

            mag_obs = np.array(mag_obs)

            mag_res = mag_obs - mag_pred

            axs_frame = axs[frame_tup[0],frame_tup[1]]

            ## LAST XY residuals
            #axs_frame.scatter(df_i['LAST_Y'].values,df_i['MAG_RES'].values)
            X_ =  df_i['LAST_X'].values
            Y_ = df_i['LAST_Y'].values
            Pi = np.hstack((X_.reshape(len(X_),1),Y_.reshape(len(Y_),1)))
            N=1700
            grid_x, grid_y = np.mgrid[0:1725:50j, 0:1725:50j]
            
            Z_cubic = scipy.interpolate.griddata(Pi, mag_res,(grid_x,grid_y), method = "cubic",fill_value=0.)
            #if framenum == 15: print(Z_cubic)
            #im = axs_frame.imshow(Z_cubic.T, extent=(0,1725,0,1725), origin='lower',cmap = 'jet',aspect='auto',vmin = -0.05,vmax = 0.05)
            im = axs_frame.imshow(Z_cubic.T, extent=(0,1725,0,1725), origin='lower',cmap = 'jet',aspect='auto',
                                  norm=mcolors.SymLogNorm(linthresh=0.03, linscale=1., vmin=-0.05, vmax=0.05, base=10))
            axs_frame.set_ylim(0.,1725.)
            axs_frame.set_xlim(0.,1725.)


            axs_frame.plot(X_,Y_,color = 'k',marker = '*',ms = 3.,ls = 'None')

        fig.subplots_adjust(right=0.82,wspace=0.05,hspace=0.05)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Residuals [mag]')    
            
        fig.add_subplot(111, frameon=False)

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        
        plt.xlabel('LAST X coord [pixel]')
        plt.ylabel('LAST Y coord [pixel]',labelpad=20)
        plt.savefig(resfilename,bbox_inches='tight')


        #plt.show()
        return None


    def plot_residuals_diagnostic_singleframe(self,cropid,resfilename=None):
        sns.set_context('paper',font_scale=2.)
        fig,axs = plt.subplots(2,2,sharex=False,sharey=False,figsize=(14,14./1.6))

        df_i = self.df_match_cal[cropid-1]

        ## Hist residuals 
        axs_ = axs[0,0]
        mag_obs = []
        mag_pred = df_i['MAG_PREDICTED'].values

        for j in range(len(df_i)):
            if df_i['G_mag'].values[j] > 16:
                mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_PSF'].values[j]))
            else:
                mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_APER_3'].values[j]))

        mag_obs = np.array(mag_obs)

        mag_res = mag_obs - mag_pred

        bin_edges = np.histogram_bin_edges(mag_res, bins='auto')
        axs_.hist(mag_res,bins=bin_edges,density=False,edgecolor='k',histtype='step',facecolor = 'C0',fill=True,linewidth=1.5)
        axs_.set_xlabel('Residuals [mag]')
        axs_.set_ylabel('Counts')
        #axs_.set_xlim(np.min(bin_edges),np.max(bin_edges))
        mu,median,sigma = astropy.stats.sigma_clipped_stats(mag_res,sigma=5)
        axs_.set_xlim(-0.05,0.05)
        textstr = '\n'.join((
            r'$\mu_{1/2}=%.4f$' % (median, ),
            r'$\sigma=%.4f$' % (sigma, )))

    
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    

        axs_.text(0.05, 0.95, textstr, transform=axs_.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


        ## Residuals vs G_color
        axs_ = axs[0,1]
        axs_.scatter(df_i['G_color'],mag_res,s=3,c = 'C1')
        #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
        axs_.set_xlabel('Gaia color')
        axs_.set_ylim(-0.05,0.05)
        axs_.set_ylabel('Residuals [mag]')

        ## Residuals vs Xcoor
        axs_ = axs[1,0]
        axs_.scatter(df_i['LAST_X'],mag_res,s=3, c='C2')
        #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
        axs_.set_xlabel('LAST X coord [pixel]')
        axs_.set_ylim(-0.05,0.05)
        axs_.set_ylabel('Residuals [mag]')

        ## Residuals vs Ycoor
        axs_ = axs[1,1]
        axs_.scatter(df_i['LAST_Y'],mag_res,s=3, c='C3')
        #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
        axs_.set_xlabel('LAST Y coord [pixel]')
        axs_.set_ylim(-0.05,0.05)
        axs_.set_ylabel('Residuals [mag]')

        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        plt.savefig(resfilename)

        plt.show()
    
    


    def plot_residuals_diagnostic_fullimage(self,resfilename=None,figtitle=None):
        sns.set_context('paper',font_scale=2.)
        fig,axs = plt.subplots(3,2,sharex=False,sharey=False,figsize=(14,14./1.6))
        
        mag_res_all = []
        
        if figtitle is not None:
            fig.suptitle(figtitle, y=0.92)
        for cropid in range(1,25):
            df_i = self.df_match_cal[cropid-1]

            ## Hist residuals 
            
            mag_obs = []
            mag_pred = df_i['MAG_PREDICTED'].values

            for j in range(len(df_i)):
                if df_i['G_mag'].values[j] > 16:
                    mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_PSF'].values[j]))
                else:
                    mag_obs.append(2.5*np.log10(df_i['LAST_FLUX_APER_3'].values[j]))

            mag_obs = np.array(mag_obs)

            mag_res = mag_obs - mag_pred

            for item in mag_res:
                mag_res_all.append(item)

            


            ## Residuals vs G_color
            axs_ = axs[2,0]
            axs_.scatter(df_i['G_color'],mag_res,s=3,c = 'C1')
            #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
            axs_.set_xlabel('Gaia color')
            axs_.set_ylim(-0.05,0.05)
            axs_.set_ylabel('Residuals [mag]')

            ## Residuals vs Xcoor
            axs_ = axs[1,0]
            axs_.scatter(df_i['LAST_X'],mag_res,s=3, c='C2')
            #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
            axs_.set_xlabel('LAST X coord [pixel]')
            axs_.set_ylim(-0.05,0.05)
            axs_.set_ylabel('Residuals [mag]')

            ## Residuals vs Ycoor
            axs_ = axs[1,1]
            axs_.scatter(df_i['LAST_Y'],mag_res,s=3, c='C3')
            #axs_.set_ylim(np.min(bin_edges),np.max(bin_edges))
            axs_.set_xlabel('LAST Y coord [pixel]')
            axs_.set_ylim(-0.05,0.05)
            axs_.set_ylabel('Residuals [mag]')

            axs_ = axs[2,1]
            axs_.scatter(df_i['G_mag'],np.abs(mag_res),s=3, c='C4')
            axs_.set_xlabel('Gaia mag')
            axs_.semilogy()
            axs_.set_ylim(None,1e-1)
            axs_.set_ylabel('Residuals (abs.) [mag]')


        axs_ = axs[0,0]
        bin_edges = np.histogram_bin_edges(mag_res_all, bins='auto')
        axs_.hist(mag_res_all,bins=bin_edges,density=False,edgecolor='k',histtype='step',facecolor = 'C0',fill=True,linewidth=1.5)
        axs_.set_xlabel('Residuals [mag]')
        axs_.set_ylabel('Counts')
        #axs_.set_xlim(np.min(bin_edges),np.max(bin_edges))
        mu,median,sigma = astropy.stats.sigma_clipped_stats(mag_res_all,sigma=5)
        axs_.set_xlim(-0.05,0.05)
        #textstr = '\n'.join((
        #    r'$\mu_{1/2}=%.3f$' % (median, ),
        #    r'$\sigma=%.3f$' % (sigma, )))
        textstr = r'$\sigma=%.3f$' % (sigma, )
    
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    

        axs_.text(0.05, 0.95, textstr, transform=axs_.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

        axs_ = axs[0,1]
        params_list = self.params_cal
        for i,item in enumerate(self.catlist):
            abscal_obj = AbsoluteCalibration(catfile=item,useHTM=self.useHTM,use_atm=self.use_atm)
            transm_full = abscal_obj.Calculate_Full_Transmission_from_params(params_list[i])
            parvals = params_list[i].valuesdict()
            axs_.plot(abscal_obj.wvl_arr,transm_full*parvals['norm'],label = str(i))
        axs_.set_xlabel('Wavelegth [nm]')
        axs_.set_ylim(0.,)
        axs_.set_xlim(np.min(abscal_obj.wvl_arr),np.max(abscal_obj.wvl_arr))
        axs_.set_ylabel('Transmission')
        #axs_.legend(ncol=4)

        fig.subplots_adjust(hspace=0.5,wspace=0.3)
        plt.savefig(resfilename)

        plt.show()

    
    def calibrate_example_catalogs(self, cattype = 'single',resfilename = None):

        if cattype == 'single':
            catfiles_jolly = os.path.join(self.current_dir,'data','Image_Test/LAST*_sci_proc_Cat_1.fits') 
            catfiles = glob.glob(catfiles_jolly)
            if len(catfiles) == 0:
                raise ValueError('No catalog files found in {}'.format(catfiles_jolly))
            else:
                print('Found {} catalog files'.format(len(catfiles)))
                print('Calibrating...')
                catfiles.sort()
        
                self.calibrate_list_of_catalogs(catfiles,resfilename=resfilename)
                print('Calibration done.')
        elif cattype == 'coadd':  
            catfiles_jolly = os.path.join(self.current_dir,'data','Image_Test_Coadd/LAST*_sci_coadd_Cat_1.fits') 
            catfiles = glob.glob(catfiles_jolly)
            if len(catfiles) == 0:
                raise ValueError('No catalog files found in {}'.format(catfiles_jolly))
            else:
                print('Found {} catalog files'.format(len(catfiles)))
                print('Calibrating...')
                catfiles.sort()
        
                self.calibrate_list_of_catalogs(catfiles,resfilename=resfilename)
                print('Calibration done.')  

        return None








                

            




    


                
            
            
















