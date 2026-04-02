
from astroquery.gaia import Gaia
import astropy.units as u
import astropy.io.fits as pyfit
from astropy.coordinates import SkyCoord
from astropy.time import Time
import pandas as pd
import numpy as np
from .lastcatutils import LastCatUtils
import gaiaxpy
import catsHTM
import time
import os


class GaiaQuery(object):
    """
    A class for querying Gaia catalog and matching sources with the LAST catalog.

    Attributes:
        catfile (str): The path to the catalog file.

    Methods:
        create_query: Creates a query string for querying Gaia catalog.
        run_query_to_pandas: Runs the Gaia query and returns the results as a pandas DataFrame.
        match_last_and_gaia: Matches sources from the LAST catalog with sources from the Gaia catalog based on their coordinates.
        retrieve_gaia_spectra: Retrieves Gaia spectra for the matched sources.
    """

    def __init__(self, catfile = None,login_gaia=False):
        """
        Initializes a GaiaQuery object.

        Parameters:
            catfile (str): The path to the catalog file.
        """
        if login_gaia:
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            self.logged_in = False
            user = os.environ.get('GAIA_USER') or os.environ.get('ESA_GAIA_USER')
            password = os.environ.get('GAIA_PASSWORD') or os.environ.get('ESA_GAIA_PASSWORD')
            try:
                if user and password:
                    Gaia.login(user=user, password=password, verbose=True)
                    self.logged_in = True
                else:
                    try:
                        Gaia.login(verbose=False)  # will use keyring/.netrc if configured; no prompt in non-interactive
                        self.logged_in = True
                    except Exception:
                        print('Gaia login skipped: no credentials found, continuing anonymously.')
            except Exception as e:
                print(f'Gaia login failed: {e}. Continuing anonymously.')

        self.catfile = catfile

        if catfile is None:
            self.nonvalid_catalog = True
            print('No catalog file provided, I will work only with utilities.')
            return
        
        last_cat, info_cat = LastCatUtils().tables_from_lastcat(catfile)

        self.last_cat = last_cat
        self.info_cat = info_cat

        if sum(np.isnan(last_cat['RA']))>0 or sum(np.isnan(last_cat['Dec']))>0:
            self.nonvalid_catalog = True
        else:
            self.nonvalid_catalog = False

        self.cRa = None #in degrees
        self.cDec = None #in degrees
        self.sep_subframe = None #astropy quantity
        self.path_HTM = '/mnt/euclid/catsHTM'
        self.sampling = np.linspace(336.,1020.,343,endpoint=True)

        if 'coadd' in catfile:
            self.ncoadd = info_cat.header['NCOADD']
        else:
            self.ncoadd = 1.
        
    def create_query(self):
        """
        Creates a query string for querying Gaia catalog based on the catalog information.

        Returns:
            query (str): The query string for querying Gaia catalog.
        """
        info_cat = self.info_cat

        Ra_min = info_cat.header['RA1']
        Ra_max = info_cat.header['RA2']

        Dec_min = info_cat.header['DEC1']
        Dec_max = info_cat.header['DEC4']
        cRa = info_cat.header['RA']
        cDec = info_cat.header['DEC']
        coor_1 = SkyCoord(ra=info_cat.header['RA1'], dec=info_cat.header['DEC1'], unit='deg', frame='icrs')
        
        coor_center = SkyCoord(ra=cRa, dec=cDec, unit='deg', frame='icrs')
        
        sep_subframe = coor_center.separation(coor_1)

        self.cRa = coor_center.ra
        self.cDec = coor_center.dec
        self.sep_subframe = sep_subframe
        print('Create query with proper motions...')
        if Ra_max < 360.:
            query = "SELECT gaia.source_id, gaia.ra AS g_ra, gaia.dec AS g_dec, gaia.pmra AS g_pmra, gaia.pmdec AS g_pmdec, teff_gspphot AS g_teff, phot_g_mean_mag AS g_mag, bp_rp AS g_color \
            FROM gaiadr3.gaia_source AS gaia \
            WHERE DISTANCE(POINT("+str(cRa)+","+str(cDec)+"),POINT(gaia.ra, gaia.dec)) < "+str(sep_subframe.deg)+" AND \
            has_xp_sampled = 'TRUE' AND \
            phot_g_mean_mag > 12 AND \
            classprob_dsc_combmod_star > 0.9 AND \
            phot_g_mean_mag < 16."
        else:
            print('Ra_max > 360 deg!')
            Ra_max = Ra_max - 360.
            query = "SELECT gaia.source_id, gaia.ra AS g_ra, gaia.dec AS g_dec, gaia.pmra AS g_pmra, gaia.pmdec AS g_pmdec, ra_error AS g_ra_err, dec_error AS g_dec_err, teff_gspphot AS g_teff, phot_g_mean_mag AS g_mag, bp_rp AS g_color \
            FROM gaiadr3.gaia_source AS gaia \
            WHERE (gaia.ra BETWEEN "+str(0.)+" AND "+str(Ra_max)+" OR \
            gaia.ra BETWEEN "+str(Ra_min)+" AND "+str(360.)+") AND \
            gaia.dec BETWEEN "+str(Dec_min)+" AND "+str(Dec_max)+" AND \
            has_xp_sampled = 'TRUE' AND \
            phot_g_mean_mag > 12 AND \
            classprob_dsc_combmod_star > 0.9 AND \
            phot_g_mean_mag < 16."

        return query
    
    def create_general_query(self,cRa,cDec,sep_subframe):
        """
        Construct an ADQL query string to retrieve Gaia DR3 sources near a given sky position.
        The query selects the following columns from gaiadr3.gaia_source:
        - source_id
        - ra AS g_ra
        - dec AS g_dec
        - pmra AS g_pmra
        - pmdec AS g_pmdec
        - teff_gspphot AS g_teff
        - phot_g_mean_mag AS g_mag
        - bp_rp AS g_color
        It applies a spatial filter using ADQL's DISTANCE and POINT functions to include only
        sources within the specified angular separation from the provided center.
        Parameters
        ----------
        cRa : float
            Right ascension of the search center in degrees (ICRS).
        cDec : float
            Declination of the search center in degrees (ICRS).
        sep_subframe : object
            Angular radius of the search region; must provide a .deg attribute (degrees).
        Returns
        -------
        str
            An ADQL query string targeting gaiadr3.gaia_source with the selected columns and
            a distance-based spatial constraint.
        Side Effects
        ------------
        Prints a status message to stdout indicating that the query is being created.
        """
           
        
        print('Create query with proper motions...')
        
        query = "SELECT gaia.source_id, gaia.ra AS g_ra, gaia.dec AS g_dec, gaia.pmra AS g_pmra, gaia.pmdec AS g_pmdec, teff_gspphot AS g_teff, phot_g_mean_mag AS g_mag, bp_rp AS g_color \
            FROM gaiadr3.gaia_source AS gaia \
            WHERE DISTANCE(POINT("+str(cRa)+","+str(cDec)+"),POINT(gaia.ra, gaia.dec)) < "+str(sep_subframe.deg)
        

        return query
    
    
        
    def run_query_to_pandas(self, query):
        """
        Runs the Gaia query and returns the results as a pandas DataFrame.

        Parameters:
            query (str): The query string for querying Gaia catalog.

        Returns:
            df_gaia (DataFrame): The Gaia catalog data as a pandas DataFrame.
        """
        job = Gaia.launch_job_async(query)
        results = job.get_results()

        df_gaia = results.to_pandas()
        df_gaia['g_pmra'].fillna(0., inplace=True)
        df_gaia['g_pmdec'].fillna(0.,inplace=True)
        return df_gaia
    
    
        
    def match_last_and_gaia(self):
        """
        Matches sources from the LAST catalog with sources from the Gaia catalog based on their coordinates.

        Returns:
            df_match (DataFrame): DataFrame containing the matched sources from both catalogs.
        """
        start_query = time.time()
        
        df_gaia_raw = self.run_query_to_pandas(self.create_query())
        
        print('Query time: ',time.time()-start_query)

        last_cat = self.last_cat
        info_cat = self.info_cat


        dt = info_cat.header['EXPTIME']

        df_match = pd.DataFrame(columns=['GaiaDR3_ID','LAST_num','JD','LAST_SN','LAST_FLUX_APER_3','LAST_FLUXERR_APER_3','LAST_FLUX_PSF','LAST_FLUXERR_PSF','ang_sep','LAST_X','LAST_Y','G_color','LAST_FLAGS','G_mag','g_ra','g_dec'])

        df_match = df_match.astype({'GaiaDR3_ID':'int','LAST_num':'int','JD':'float','LAST_SN':'float',
                                    'LAST_FLUX_APER_3':'float','LAST_FLUXERR_APER_3':'float','LAST_FLUX_PSF':'float','LAST_FLUXERR_PSF':'float','ang_sep':'float','LAST_X':'float',
                                    'LAST_Y':'float','G_color':'float','LAST_FLAGS':'float','G_mag':'float','g_ra':'float','g_dec':'float'})

        coord_last_cat = SkyCoord(ra=last_cat['RA'], dec=last_cat['DEC'], unit='deg', frame='icrs')

        coor_gaia_raw_2016 = SkyCoord(ra=df_gaia_raw['g_ra'].values*u.deg, dec=df_gaia_raw['g_dec'].values*u.deg,
                                 pm_ra_cosdec=df_gaia_raw['g_pmra'].values * u.mas/u.yr,
                                 pm_dec=df_gaia_raw['g_pmdec'].values * u.mas/u.yr,
                                 frame='icrs', obstime=Time(2016.0, format="jyear"))
       
        coor_gaia_raw = coor_gaia_raw_2016.apply_space_motion(new_obstime=Time(info_cat.header['JD'], format='jd'))
        idx_raw, d2d_raw, d3d_raw = coord_last_cat.match_to_catalog_sky(coor_gaia_raw)
        mask_sep_raw = d2d_raw < 2.*u.arcsec
        
        idx_match_raw = idx_raw[mask_sep_raw]
        df_gaia = df_gaia_raw.iloc[idx_match_raw].reset_index(drop=True)


        gaiaid_list =[]
        lastid_list =[]

        start = time.time()
        for i in range(len(df_gaia)):

            coord_0 = SkyCoord(ra=df_gaia['g_ra'][i], dec=df_gaia['g_dec'][i], unit='deg', frame='icrs', obstime=Time(info_cat.header['JD'], format='jd'))

            sep = coord_0.separation(coord_last_cat)
                
            mask_sep = sep < 2.*u.arcsec

            if sum(mask_sep) == 0: 
                
                continue
            elif sum(mask_sep) == 1:
                
                last_idx_ = np.where(mask_sep == True)[0][0]

                last_flags = last_cat['FLAGS'][last_idx_]

                #binary_number = bin(int(last_flags))[2:]
                #indices_flags = np.where(np.array(list(binary_number)) == '1')[0]

                last_flags_keys = LastCatUtils().get_flags_keyword(last_flags)
                
                #if len(indices_flags) > 0:
                if any(ff in ['Saturated','NaN','Negative','CR_DeltaHT','NearEdge'] for ff in last_flags_keys):    
                    continue

                jd_ = info_cat.header['JD']    
                sn_ = last_cat['SN'][last_idx_]
                    
                flux_3_ = self.ncoadd*last_cat['FLUX_APER_3'][last_idx_]/dt
                flux_3_err_ = self.ncoadd*last_cat['FLUXERR_APER_3'][last_idx_]/dt

                flux_psf_ = self.ncoadd*last_cat['FLUX_PSF'][last_idx_]/dt
                flux_psf_err_ = self.ncoadd*last_cat['FLUXERR_APER_3'][last_idx_]/dt

                last_x = last_cat['X'][last_idx_]
                last_y = last_cat['Y'][last_idx_]
                last_flags = last_cat['FLAGS'][last_idx_]

                gaia_color = df_gaia['g_color'][i]
                gaia_mag = df_gaia['g_mag'][i]
                g_Ra = df_gaia['g_ra'][i]
                g_Dec = df_gaia['g_dec'][i]
                    
                gaiaid_list.append(df_gaia['source_id'][i])
                lastid_list.append(int(last_idx_))
                    
                new_row = {'GaiaDR3_ID':int(df_gaia['source_id'][i]),'ang_sep':sep[last_idx_].to(u.arcsec).value,
                            'LAST_SN':sn_,'LAST_FLUX_APER_3':flux_3_,'LAST_FLUXERR_APER_3':flux_3_err_,'LAST_num':int(last_idx_),'LAST_X':last_x,'LAST_Y':last_y,'G_color':gaia_color,'LAST_FLAGS':last_flags,
                            'G_mag':gaia_mag,'LAST_FLUX_PSF':flux_psf_,'LAST_FLUXERR_PSF':flux_psf_err_,'JD':jd_,'g_ra':g_Ra,'g_dec':g_Dec}
               
                
                #df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)    
                df_match = pd.concat([df_match,pd.DataFrame([new_row])],ignore_index=True)     
                    
            else:
                print('Too many coincidences!! Abort!')
                continue
        df_match = df_match.astype({'GaiaDR3_ID':'int','LAST_num':'int','JD':'float','LAST_SN':'float',
                                    'LAST_FLUX_APER_3':'float','LAST_FLUXERR_APER_3':'float','LAST_FLUX_PSF':'float','LAST_FLUXERR_PSF':'float',
                                    'ang_sep':'float','LAST_X':'float','LAST_Y':'float','G_color':'float','LAST_FLAGS':'float','G_mag':'float','g_ra':'float','g_dec':'float'})    

        df_match['GaiaDR3_ID'] = gaiaid_list
        df_match['LAST_num'] = lastid_list

        df_match = df_match[(df_match['LAST_SN']>5) & (df_match['LAST_SN']<1000) & (df_match['LAST_FLUX_APER_3']>0) & (df_match['LAST_FLUX_PSF']>0)].reset_index()
        print('Matching time: ',time.time()-start)
        return df_match
    

    
        

        

        

    def retrieve_gaia_spectra(self,useHTM=False):
        """
        Retrieves Gaia spectra using the last match and Gaia query.

        Returns:
            source_ids (list): List of Gaia source IDs.
            calibrated_spectra (array): Calibrated spectra.
            sampling (float): Sampling rate.
            df_match (DataFrame): DataFrame containing the last match.
        """
        if self.nonvalid_catalog:
            return None, None, None, None
        df_match = self.match_last_and_gaia()
        source_ids = list(df_match['GaiaDR3_ID'].astype(str))
         
        start = time.time()
        if useHTM:
            cat,colcell, colunits=catsHTM.cone_search('GAIADR3spec',self.cRa.rad,self.cDec.rad,self.sep_subframe.arcsec,catalogs_dir=self.path_HTM)
            df_cat = pd.DataFrame(cat, columns=colcell)
            #df_cat['source_id'] = df_cat['source_id'].astype(int)
            #df_cat['source_id'] = df_cat['source_id'].astype(str)
            #calibrated_spectra = df_cat
            #calibrated_spectra = df_cat[df_cat['source_id'].isin(source_ids)]
            #print(calibrated_spectra)
            #calibrated_spectra.reset_index(drop=True, inplace=True)

            coord_df_match = SkyCoord(ra=df_match['g_ra'], dec=df_match['g_dec'], unit='deg', frame='icrs',equinox='J2016')
            coord_df_cat = SkyCoord(ra=df_cat["['RA']"], dec=df_cat["['Dec']"], unit='rad', frame='icrs',equinox='J2016')
            
            #First match to select elements of df_match with spectra
            idx, d2d, d3d = coord_df_cat.match_to_catalog_sky(coord_df_match)
            mask_sep = d2d < 1.*u.arcsec
            idx_match = idx[mask_sep]
            df_match_spectra = df_match.iloc[idx_match].reset_index(drop=True)
            source_ids = list(df_match_spectra['GaiaDR3_ID'].astype(str))
            #Second match, inverted, to select spectra with counterpart
            coord_df_match_spectra = SkyCoord(ra=df_match_spectra['g_ra'], dec=df_match_spectra['g_dec'], unit='deg', frame='icrs',equinox='J2016')
            idx_s, d2d_s, d3d_s = coord_df_match_spectra.match_to_catalog_sky(coord_df_cat)
            mask_sep_s = d2d_s < 1.*u.arcsec
            idx_match_s = idx_s[mask_sep_s]
            calibrated_spectra = df_cat.iloc[idx_match_s].reset_index(drop=True)




            sampling = self.sampling
            return source_ids, calibrated_spectra, sampling, df_match_spectra

        
        else:

            calibrated_spectra_unsrt, sampling = self.retrieve_gaia_spectra_from_ids(source_ids)
            print('Spectra calibration time: ',time.time()-start)
            print('Number of calibrators (Gmag < 16): ',len(df_match))
            idx_gaiaid = df_match['GaiaDR3_ID'].values
            calibrated_spectra = calibrated_spectra_unsrt.loc[idx_gaiaid].reset_index(drop=True)

            calibrated_spectra['GaiaDR3_ID'] = idx_gaiaid

            calibrated_spectra = calibrated_spectra.sort_values('GaiaDR3_ID').reset_index(drop=True)
            df_match = df_match.sort_values('GaiaDR3_ID').reset_index(drop=True)
            source_ids = list(df_match['GaiaDR3_ID'].astype(str))

            #sort calibrated_spectra by GaiaDR3_ID

            #if len(df_match[df_match['G_mag'] < 15.]) > 29:
            #    df_match_sel = df_match[df_match['G_mag'] < 15.]
            #    source_ids_sel = df_match_sel['GaiaDR3_ID'].values
            #    idx_sel = df_match_sel.index
            #    calibrated_spectra_sel = calibrated_spectra.loc[source_ids_sel]
            #    calibrated_spectra_sel.reset_index(drop=True, inplace=True)
            #    df_match_sel.reset_index(drop=True, inplace=True)

            #    print('Reduced to Gmag < 15, number of calibrators: ',len(df_match_sel))

            #    return source_ids_sel, calibrated_spectra_sel, sampling, df_match_sel



            return source_ids, calibrated_spectra, sampling, df_match

    
    

    def retrieve_gaia_spectra_from_ids(self, source_ids, chunk_size=50):
        """
        Retrieves Gaia XP sampled spectra for the given source IDs using the
        Gaia archive TAP service (astroquery), without relying on gaiaxpy.

        The returned flux arrays are sampled on a fixed wavelength grid of 343
        points from 336 to 1020 nm (2 nm steps), consistent with
        gaiadr3.xp_sampled_mean_spectrum.

        IDs are sent in chunks to avoid HTTP 500 errors from large IN clauses.

        Parameters:
            source_ids (list): List of Gaia DR3 source IDs (int or str).
            chunk_size (int): Number of IDs per TAP query (default 50).

        Returns:
            spectra_df (DataFrame): DataFrame indexed by source_id with columns
                'flux' (np.ndarray, length 343) and 'flux_error' (np.ndarray, length 343).
            sampling (np.ndarray): Wavelength array in nm, shape (343,).
        """
        sampling = np.linspace(336., 1020., 343, endpoint=True)

        ids = [int(sid) for sid in source_ids]
        chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

        rows = []
        sampling = None
        for i, chunk in enumerate(chunks):
            print(f'Querying chunk {i+1}/{len(chunks)} ({len(chunk)} IDs)...')
            datalink = Gaia.load_data(
                ids=chunk,
                data_release='Gaia DR3',
                data_structure='INDIVIDUAL',
                retrieval_type='XP_SAMPLED',
                format='votable',
                overwrite_output_file=True,
            )
            # Keys are like 'XP_SAMPLED-Gaia DR3 <source_id>.xml'
            for key, table_list in datalink.items():
                src_id = int(key.split('DR3 ')[1].replace('.xml', ''))
                for tbl in table_list:
                    t = tbl.to_table()
                    if sampling is None:
                        sampling = np.array(t['wavelength'])
                    rows.append({
                        'source_id': src_id,
                        'flux': np.array(t['flux']),
                        'flux_error': np.array(t['flux_error']),
                    })
            time.sleep(1)

        if sampling is None:
            sampling = np.linspace(336., 1020., 343, endpoint=True)

        spectra_df = pd.DataFrame(rows).set_index('source_id')
        print(f'Retrieved spectra for {len(spectra_df)} sources.')
        return spectra_df, sampling
    

    
    
