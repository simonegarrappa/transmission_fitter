
import astropy.units as u
import astropy.io.fits as pyfit
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from .lastcatutils import LastCatUtils
from tqdm import tqdm

class BlazarQuery(object):

    def __init__(self):
            """
            Initializes the BlazarUtils class.

            This method prepares the 'romabz5' and 'romabz5_apy' attributes by calling the 'prepare_romabz5' method.
            It also sets the 'match_radius' attribute to 1.0 and initializes the 'columns_bz5' attribute with a list of column names.

            Parameters:
                None

            Returns:
                None
            """
            
            ## ROMABZ5 catalog
            romabz5, romabz5_apy = self.prepare_romabz5()

            self.romabz5 = romabz5
            self.romabz5_apy = romabz5_apy

            ## Landolt-Stetson calibrators from Gaia-EDR3
            df_calibrators, edr3_calibrators_apy = self.prepare_gaiaedr3_calibrators()
            self.edr3_calibrators = df_calibrators
            self.edr3_calibrators_apy = edr3_calibrators_apy
            
            
            self.match_radius = 2.0
            self.columns_bz5 = ['LASTCatalog','JD','RomaBZ5_NAME','RA','DEC',
                                'Redshift','Class','Rmag','LAST_MAG_APER','LAST_MAGERR_APER',
                                'LAST_MAG_PSF','LAST_MAGERR_PSF','LAST_RA','LAST_DEC']
            self.columns_ed3calibrators = ['LASTCatalog','JD','LS_CALIBRATOR_NAME','RA','DEC',
                                           'Class','Vmag','LAST_MAG_APER','LAST_MAGERR_APER',
                                           'LAST_MAG_PSF','LAST_MAGERR_PSF','LAST_RA','LAST_DEC']

            pass
    
    def prepare_gaiaedr3_calibrators(self):
        """
        Prepare Gaia EDR3 calibrators.

        Reads the calibrators data from a CSV file and selects only sources with Vmag < 18.
        Converts the coordinates of the calibrators to SkyCoord object.

        Returns:
            df_calibrators (pandas.DataFrame): DataFrame containing the calibrators data.
            edr3_calibrators_apy (SkyCoord): SkyCoord object containing the coordinates of the calibrators.
        """
        df_calibrators = pd.read_csv('./SourceCatalogs/Calibrators_Landolt_Stetson.csv')
        ## Select only sources with Vmag < 18
        #df_calibrators = df_calibrators[df_calibrators['Vmag'] < 18]

        edr3_calibrators_apy = SkyCoord(ra=df_calibrators['RAJ2000'], dec=df_calibrators['DEJ2000'], unit='deg', frame='fk5')

        return df_calibrators, edr3_calibrators_apy


    def prepare_romabz5(self):
        """
        Prepare the RomaBZ5 catalog by reading the FITS file, extracting the necessary data,
        and converting the coordinates to SkyCoord objects.

        Returns:
            romabz5 (astropy.io.fits.FITS_rec): The RomaBZ5 catalog data.
            romabz5_apy (astropy.coordinates.SkyCoord): The RomaBZ5 catalog coordinates as SkyCoord objects.
        """
        hdu = pyfit.open('./SourceCatalogs/RomaBZ5.fits')
        romabz5 = hdu[1].data

        coor_array = []

        for i in range(len(romabz5)):
            ra_ = str(format(romabz5['Rah'][i],'02'))+ ' ' + str(format(romabz5['Ram'][i],'02')) + ' ' + str(romabz5['Ras'][i])
            dec_ = str(romabz5['DE-'][i]) + str(format(romabz5['DEd'][i],'02'))+ ' ' + str(format(romabz5['DEm'][i],'02')) + ' ' + str(romabz5['DEs'][i])

            coor_array.append(ra_ + ' ' + dec_)

        romabz5_apy = SkyCoord(coor_array, unit=(u.hourangle, u.deg),frame='fk5')

        return romabz5, romabz5_apy
    
    def prepare_4fgl(self):
        """
        Prepare the 4FGL catalog for further analysis.

        Returns:
            fgl (astropy.table.Table): The 4FGL catalog data.
            fgl_apy (astropy.coordinates.SkyCoord): The astropy SkyCoord object containing the coordinates of the 4FGL sources.
        """
        hdu = pyfit.open('./SourceCatalogs/gll_psc_v33.fit')
        fgl = hdu[1].data

        fgl_apy = SkyCoord(ra=fgl['RAJ2000'], dec=fgl['DEJ2000'], unit='deg', frame='fk5')

        return fgl, fgl_apy


    def search_romabz5_singleimage(self, catfile):
        """
        Search for the blazar candidates in the ROMABZ5 catalog.

        Parameters:
        - catfile (str): The path to the catalog file.

        Returns:
        - df_match (pd.DataFrame): A DataFrame containing the matched blazar candidates from the ROMABZ5 catalog.
            The DataFrame includes the following columns:
            - 'LASTCatalog': The name of the LAST catalog file.
            - 'JD': The Julian Date.
            - 'RomaBZ5_NAME': The name of the blazar candidate in the ROMABZ5 catalog.
            - 'RA': The right ascension of the blazar candidate.
            - 'DEC': The declination of the blazar candidate.
            - 'Redshift': The redshift of the blazar candidate.
            - 'Class': The class of the blazar candidate.
            - 'Rmag': The R magnitude of the blazar candidate.
        """
        romabz5, romabz5_apy = self.romabz5, self.romabz5_apy

        last_cat, info_cat = LastCatUtils().tables_from_lastcat(catfile.strip())

        last_cat_apy = SkyCoord(ra=last_cat['RA'], dec=last_cat['Dec'], unit='deg', frame='icrs')
        idx, sep2d, d3d = romabz5_apy.match_to_catalog_sky(last_cat_apy)
        

        mask_match = sep2d.to(u.arcsec) < self.match_radius * u.arcsec
        idx_match_ref = idx[mask_match]

        columns_ = self.columns_bz5
        df_match = pd.DataFrame(columns=columns_)
        if len(idx_match_ref) > 0:
            last_cat_names = [catfile.strip()] * len(idx_match_ref)
            last_cat_jd = [info_cat.header['JD']] * len(idx_match_ref)
            last_cat_rbz5_name = romabz5['Name'][mask_match]
            ra_match = romabz5_apy.ra.deg[mask_match]
            dec_match = romabz5_apy.dec.deg[mask_match]
            redshift_match = romabz5['z'][mask_match]
            class_match = romabz5['Class'][mask_match]
            rmag_match = romabz5['Rmag'][mask_match]

            df_match['LASTCatalog'] = last_cat_names
            df_match['JD'] = last_cat_jd
            df_match['RomaBZ5_NAME'] = last_cat_rbz5_name
            df_match['RA'] = ra_match
            df_match['DEC'] = dec_match
            df_match['Redshift'] = redshift_match
            df_match['Class'] = class_match
            df_match['Rmag'] = rmag_match
            
            last_mag_aper = []
            last_magerr_aper = []
            last_mag_psf = []
            last_magerr_psf = []
            last_ra = []
            last_dec = []
            for i in range(len(df_match)):
                rmbz_src_coord_i = SkyCoord(ra=df_match['RA'][i], dec=df_match['DEC'][i], unit='deg', frame='fk5')
                idx_last, sep2d_last, d3d_last = rmbz_src_coord_i.match_to_catalog_sky(last_cat_apy)
                #mask_match_last = sep2d_last.to(u.arcsec) < self.match_radius * u.arcsec
                idx_match_ref_last = idx_last #[mask_match_last]
                

                
                last_mag_aper.append(last_cat['MAG_APER_3'][idx_match_ref_last])
                last_magerr_aper.append(last_cat['MAGERR_APER_3'][idx_match_ref_last])
                last_mag_psf.append(last_cat['MAG_PSF'][idx_match_ref_last])
                last_magerr_psf.append(last_cat['MAGERR_PSF'][idx_match_ref_last])
                last_ra.append(last_cat['RA'][idx_match_ref_last])
                last_dec.append(last_cat['Dec'][idx_match_ref_last])

            df_match['LAST_MAG_APER'] = last_mag_aper
            df_match['LAST_MAGERR_APER'] = last_magerr_aper
            df_match['LAST_MAG_PSF'] = last_mag_psf
            df_match['LAST_MAGERR_PSF'] = last_magerr_psf
            df_match['LAST_RA'] = last_ra
            df_match['LAST_DEC'] = last_dec
                
            
        return df_match
    
    def search_romabz5_multimage(self, catfile_list_txt):
        """
        Searches for matches with RomaBZ5 catalog in multiple LAST images.

        Args:
            catfile_list_txt (str): Path to the text file containing a list of catalog files.

        Returns:
            pd.DataFrame: A DataFrame containing the matched results.
        """
        if not isinstance(catfile_list_txt, str):
            catfile_list = catfile_list_txt
        else:
            with open(catfile_list_txt, 'r') as f:
                catfile_list = f.readlines()

        columns_ = self.columns_bz5
        df_match = pd.DataFrame(columns=columns_)

        for nn, catfile in enumerate(catfile_list):

            if (nn+1) % 1000 == 0:
                print('Searching in image {} of {}'.format(nn+1, len(catfile_list)))

            df_match_ = self.search_romabz5_singleimage(catfile)
            if len(df_match_) > 0:
                print('Found {} matches in image {}'.format(len(df_match_), nn+1))
                df_match = pd.concat([df_match, df_match_], ignore_index=True)

        return df_match


    def search_gaiaedr3_calibrators_singleimage(self, catfile):
        
        
        df_calibrators, edr3_calibrators_apy = self.edr3_calibrators, self.edr3_calibrators_apy

        last_cat, info_cat = LastCatUtils().tables_from_lastcat(catfile.strip())

        last_cat_apy = SkyCoord(ra=last_cat['RA'], dec=last_cat['Dec'], unit='deg', frame='icrs')
        idx, sep2d, d3d = edr3_calibrators_apy.match_to_catalog_sky(last_cat_apy)
        

        mask_match = sep2d.to(u.arcsec) < self.match_radius * u.arcsec
        idx_match_ref = idx[mask_match]

        columns_ = self.columns_ed3calibrators
        df_match = pd.DataFrame(columns=columns_)
        if len(idx_match_ref) > 0:
            last_cat_names = [catfile.strip()] * len(idx_match_ref)
            last_cat_jd = [info_cat.header['JD']] * len(idx_match_ref)
            last_cat_rbz5_name = df_calibrators['Name'].values[mask_match]
            ra_match = edr3_calibrators_apy.ra.deg[mask_match]
            dec_match = edr3_calibrators_apy.dec.deg[mask_match]
            
            class_match = df_calibrators['StarType'].values[mask_match]
            vmag_match = df_calibrators['Vmag'].values[mask_match]

            df_match['LASTCatalog'] = last_cat_names
            df_match['JD'] = last_cat_jd
            df_match['LS_CALIBRATOR_NAME'] = last_cat_rbz5_name
            df_match['RA'] = ra_match
            df_match['DEC'] = dec_match
            df_match['Class'] = class_match
            df_match['Vmag'] = vmag_match

            last_mag_aper = []
            last_magerr_aper = []
            last_mag_psf = []
            last_magerr_psf = []
            last_ra = []
            last_dec = []
            for i in range(len(df_match)):
                rmbz_src_coord_i = SkyCoord(ra=df_match['RA'][i], dec=df_match['DEC'][i], unit='deg', frame='fk5')
                idx_last, sep2d_last, d3d_last = rmbz_src_coord_i.match_to_catalog_sky(last_cat_apy)
                #mask_match_last = sep2d_last.to(u.arcsec) < self.match_radius * u.arcsec
                idx_match_ref_last = idx_last #[mask_match_last]
                

                
                last_mag_aper.append(last_cat['MAG_APER_3'][idx_match_ref_last])
                last_magerr_aper.append(last_cat['MAGERR_APER_3'][idx_match_ref_last])
                last_mag_psf.append(last_cat['MAG_PSF'][idx_match_ref_last])
                last_magerr_psf.append(last_cat['MAGERR_PSF'][idx_match_ref_last])
                last_ra.append(last_cat['RA'][idx_match_ref_last])
                last_dec.append(last_cat['Dec'][idx_match_ref_last])

            df_match['LAST_MAG_APER'] = last_mag_aper
            df_match['LAST_MAGERR_APER'] = last_magerr_aper
            df_match['LAST_MAG_PSF'] = last_mag_psf
            df_match['LAST_MAGERR_PSF'] = last_magerr_psf
            df_match['LAST_RA'] = last_ra
            df_match['LAST_DEC'] = last_dec
                
            
        return df_match
    
    def search_gaiaedr3_calibrators_multimage(self, catfile_list_txt):
        
        if not isinstance(catfile_list_txt, str):
            catfile_list = catfile_list_txt
        else:
            with open(catfile_list_txt, 'r') as f:
                catfile_list = f.readlines()

        columns_ = self.columns_ed3calibrators
        df_match = pd.DataFrame(columns=columns_)

        for nn, catfile in tqdm(enumerate(catfile_list)):

            if (nn+1) % 1000 == 0:
                print('Searching in image {} of {}'.format(nn+1, len(catfile_list)))

            df_match_ = self.search_gaiaedr3_calibrators_singleimage(catfile)
            if len(df_match_) > 0:
                print('Found {} matches in image {}'.format(len(df_match_), nn+1))
                df_match = pd.concat([df_match, df_match_], ignore_index=True)

        return df_match
    

    def search_Blazars_in_LAST_Visits(self,last_visits_file,blazar_catalog = 'ROMABZ5'):

        if blazar_catalog == 'ROMABZ5':
            romabz5, romabz5_apy = self.romabz5, self.romabz5_apy
        elif blazar_catalog == '4FGL':
            df_match = self.search_4fgl_multimage(last_visits_file)
        else:
            print('Blazar catalog not found!')
            return None

        return df_match