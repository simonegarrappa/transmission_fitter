"""Tests for ``LAST_ABSCAL_Analysis``, the high-level driver and product writer.

The calibration itself is covered in ``test_fitutils.py``; here the fitting
engine is stubbed where the point is orchestration (caching, bookkeeping,
product I/O), and exercised for real only where the numbers matter.
"""

import os

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from transmission_fitter import analysis_tools
from transmission_fitter.analysis_tools import LAST_ABSCAL_Analysis
from transmission_fitter.fitutils import AbsoluteCalibration
from transmission_fitter.lastcatutils import LastCatUtils


@pytest.fixture
def analysis():
    """Analysis object configured for offline, atmosphere-free work."""
    return LAST_ABSCAL_Analysis(useHTM=False, use_atm=False)


@pytest.fixture
def params_for(single_catalog):
    """Factory returning fitted-looking parameter sets for a catalog."""

    def _make(norm=0.62, center=575.0, catfile=single_catalog):
        obj = AbsoluteCalibration(catfile=catfile, use_atm=False)
        params = obj.Initialize_Params()
        params["norm"].set(value=norm)
        params["center"].set(value=center)
        return params

    return _make


class TestInitialisation:
    def test_defaults(self, analysis):
        assert analysis.match_radius == 2.0
        assert analysis.params_cal is None
        assert analysis.df_matchedsources is None
        assert analysis.wvl_arr.shape == (401,)
        assert analysis.cal_results_dir is None

    def test_subframe_layout_covers_the_full_24_frame_mosaic(self, analysis):
        layout = analysis.dict_lastframe
        assert sorted(int(k) for k in layout) == list(range(1, 25))
        positions = list(layout.values())
        assert len(set(positions)) == 24  # every subframe has its own cell
        assert {p[0] for p in positions} == set(range(6))
        assert {p[1] for p in positions} == set(range(4))

    def test_observatory_location_matches_neot_semadar(self, analysis):
        assert analysis.obs_location.lat.deg == pytest.approx(30.0529838)
        assert analysis.obs_location.lon.deg == pytest.approx(35.0407331)


class TestCalibrateSingleCatalog:
    """The Gaia/fit machinery is stubbed; only the caching flow is under test."""

    @pytest.fixture
    def stub_abscal(self, monkeypatch, make_spectra, make_df_match):
        spectra, sampling = make_spectra(n_sources=4)
        df_match = make_df_match(n_sources=4)

        class _StubAbsCal:
            instances = []

            def __init__(self, catfile, useHTM=False, use_atm=True):
                self.catfile = catfile
                self.df_gaia_raw = None
                self._gaia_cache_region = None
                self.calibrated_spectra = None
                self.sampling = None
                self.df_match = df_match.copy()
                self.match_calls = []
                self.fit_calls = 0
                _StubAbsCal.instances.append(self)

            def match_Gaia(self, get_spectra=False):
                self.match_calls.append(get_spectra)
                self._gaia_cache_region = (1.0, 2.0, 0.5)
                self.df_gaia_raw = pd.DataFrame({"source_id": [1]})
                return (
                    list(df_match["GaiaDR3_ID"].astype(str)),
                    spectra.copy(),
                    sampling,
                    self.df_match,
                )

            def fit_transmission(self):
                self.fit_calls += 1
                return "params", self.df_match

        _StubAbsCal.instances = []
        monkeypatch.setattr(analysis_tools, "AbsoluteCalibration", _StubAbsCal)
        return _StubAbsCal

    def test_first_call_downloads_spectra_and_caches_them(
        self, analysis, stub_abscal, single_catalog
    ):
        params, df_match_fit = analysis.calibrate_single_catalog(single_catalog)
        assert stub_abscal.instances[0].match_calls == [True]
        assert params == "params"
        assert len(df_match_fit) == 4
        assert analysis.calibrated_spectra is not None
        assert analysis.catfile == single_catalog
        assert analysis._gaia_cache_region == (1.0, 2.0, 0.5)

    def test_subsequent_call_reuses_the_cached_spectra(
        self, analysis, stub_abscal, single_catalogs
    ):
        analysis.calibrate_single_catalog(single_catalogs[0])
        analysis.calibrate_single_catalog(single_catalogs[1])
        assert stub_abscal.instances[0].match_calls == [True]
        assert stub_abscal.instances[1].match_calls == [False]

    def test_htm_mode_always_downloads(self, stub_abscal, single_catalogs):
        analysis = LAST_ABSCAL_Analysis(useHTM=True, use_atm=False)
        analysis.calibrate_single_catalog(single_catalogs[0])
        analysis.calibrate_single_catalog(single_catalogs[1])
        assert [inst.match_calls for inst in stub_abscal.instances] == [[True], [True]]

    def test_catalog_without_matches_is_skipped(
        self, analysis, stub_abscal, single_catalog, monkeypatch
    ):
        class _NoMatch(stub_abscal):
            def match_Gaia(self, get_spectra=False):
                out = super().match_Gaia(get_spectra=get_spectra)
                self.df_match = None
                return out

        monkeypatch.setattr(analysis_tools, "AbsoluteCalibration", _NoMatch)
        params, df_match_fit = analysis.calibrate_single_catalog(single_catalog)
        assert params is None and df_match_fit is None
        assert _NoMatch.instances[-1].fit_calls == 0


class TestCalibrateListOfCatalogsValidation:
    def test_rejects_a_non_txt_path(self, analysis, tmp_path):
        analysis.cal_results_dir = str(tmp_path)
        with pytest.raises(ValueError, match=".txt file"):
            analysis.calibrate_list_of_catalogs("catalogs.csv")

    def test_rejects_unsupported_input_types(self, analysis, tmp_path):
        analysis.cal_results_dir = str(tmp_path)
        with pytest.raises(TypeError, match="must be a .txt file or a Python list"):
            analysis.calibrate_list_of_catalogs(42)

    def test_per_catalog_output_requires_a_results_directory(self, analysis, single_catalogs):
        with pytest.raises(ValueError, match="calibration results directory"):
            analysis.calibrate_list_of_catalogs(single_catalogs, single_output=False)


class TestProductRoundTrip:
    def test_single_catalog_results_round_trip_through_csv(
        self, analysis, params_for, single_catalog, tmp_path
    ):
        analysis.params_cal = params_for(norm=0.61, center=572.5)
        analysis.catfile = single_catalog
        analysis.catlist = None
        resfile = tmp_path / "Calibrated_single"

        analysis.write_products(resfilename=str(resfile))
        df = pd.read_csv(str(resfile) + ".csv")

        assert len(df) == 1
        assert df["FILENAME"].iloc[0] == os.path.abspath(single_catalog)
        assert df["norm"].iloc[0] == pytest.approx(0.61)
        assert df["center"].iloc[0] == pytest.approx(572.5)

    def test_catalog_list_results_round_trip_through_csv(
        self, analysis, params_for, single_catalogs, tmp_path
    ):
        norms = [0.55, 0.60, 0.65]
        analysis.params_cal = [params_for(norm=n, catfile=c) for n, c in zip(norms, single_catalogs)]
        analysis.catlist = single_catalogs
        resfile = tmp_path / "Calibrated_list"

        analysis.write_products(resfilename=str(resfile))
        recovered = LAST_ABSCAL_Analysis(useHTM=False, use_atm=False)
        params_list = recovered.get_params_from_calibrated_results(str(resfile) + ".csv")

        assert len(params_list) == 3
        assert [p["norm"].value for p in params_list] == pytest.approx(norms)
        assert list(recovered.catlist) == [os.path.abspath(c) for c in single_catalogs]
        assert recovered.params_cal is params_list

    def test_results_from_several_files_are_concatenated(
        self, analysis, params_for, single_catalogs, tmp_path
    ):
        written = []
        for i, catfile in enumerate(single_catalogs[:2]):
            analysis.params_cal = params_for(norm=0.5 + 0.1 * i, catfile=catfile)
            analysis.catfile = catfile
            analysis.catlist = None
            name = str(tmp_path / f"Calibrated_{i}")
            analysis.write_products(resfilename=name)
            written.append(name + ".csv")

        params_list = analysis.get_params_from_calibrated_results(written)
        assert len(params_list) == 2
        assert [p["norm"].value for p in params_list] == pytest.approx([0.5, 0.6])

    def test_bootstrap_products_have_one_row_per_realisation(
        self, analysis, params_for, single_catalog, tmp_path
    ):
        analysis.params_cal = [params_for(norm=0.5 + 0.01 * i) for i in range(5)]
        analysis.catfile = single_catalog
        resfile = tmp_path / "Bootstrap"

        analysis.write_products_Bootstrap(resfilename=str(resfile))
        df = pd.read_csv(str(resfile) + ".csv")

        assert len(df) == 5
        assert df["norm"].values == pytest.approx([0.5 + 0.01 * i for i in range(5)])
        assert (df["FILENAME"] == single_catalog).all()

    def test_unfitted_parameters_survive_the_round_trip(
        self, analysis, params_for, single_catalog, tmp_path
    ):
        params = params_for()
        analysis.params_cal = params
        analysis.catfile = single_catalog
        analysis.catlist = None
        resfile = str(tmp_path / "Calibrated_defaults")
        analysis.write_products(resfilename=resfile)

        recovered = analysis.get_params_from_calibrated_results(resfile + ".csv")[0]
        for name in ("amplitude", "sigma", "gamma", "pressure", "AOD", "alpha", "ozone_col", "PW"):
            assert recovered[name].value == pytest.approx(params[name].value)


class TestSyntheticPhotometry:
    def test_light_curve_has_one_point_per_catalog_sorted_in_time(
        self, analysis, params_for, single_catalogs, catalog_tables
    ):
        analysis.catlist = single_catalogs
        analysis.params_cal = [params_for(catfile=c) for c in single_catalogs]
        wvl = np.linspace(300.0, 1100.0, 200)
        spectrum = 1e-16 * np.ones_like(wvl)

        df_lc = analysis.make_Synthetic_Photometry_LC(wvl, spectrum)

        assert list(df_lc.columns) == ["JD", "FLUX_SYN"]
        assert len(df_lc) == len(single_catalogs)
        assert df_lc["JD"].is_monotonic_increasing
        assert (df_lc["FLUX_SYN"] > 0).all()

        _, info_cat = catalog_tables
        assert df_lc["JD"].iloc[0] == pytest.approx(info_cat.header["JD"])

    def test_synthetic_flux_is_linear_in_the_input_spectrum(
        self, analysis, params_for, single_catalog
    ):
        analysis.catlist = [single_catalog]
        analysis.params_cal = [params_for()]
        wvl = np.linspace(300.0, 1100.0, 200)

        faint = analysis.make_Synthetic_Photometry_LC(wvl, 1e-16 * np.ones_like(wvl))
        bright = analysis.make_Synthetic_Photometry_LC(wvl, 3e-16 * np.ones_like(wvl))
        assert bright["FLUX_SYN"].iloc[0] / faint["FLUX_SYN"].iloc[0] == pytest.approx(3.0)


class TestAirmassCorrection:
    @pytest.fixture
    def airmass_dataset(self):
        """Sources whose magnitudes follow an exactly colour-dependent extinction."""
        slope_color, intercept_color = 0.1, 0.05
        airmasses = np.linspace(1.0, 1.8, 10)
        colors = {1: 0.5, 2: 1.0, 3: 1.5}
        base_mag = {1: 14.0, 2: 14.5, 3: 15.0}

        rows = []
        for source_id, color in colors.items():
            k = slope_color * color + intercept_color
            for j, airmass in enumerate(airmasses):
                mag = base_mag[source_id] + k * (airmass - 1.0)
                rows.append(
                    {
                        "SOURCE_ID": source_id,
                        "GaiaDR3_ID": 10**18 + source_id,
                        "JD": 2460381.0 + j * 0.01,
                        "AIRMASS": airmass,
                        "BP_RP": color,
                        "MAG_APER_AB": mag,
                        "MAG_PSF_AB": mag + 0.01,
                        "MAG_APER_AB_CORR": np.nan,
                        "MAG_PSF_AB_CORR": np.nan,
                    }
                )
        return pd.DataFrame(rows), base_mag, slope_color, intercept_color

    def test_returns_none_without_input(self, analysis):
        assert analysis.make_airmass_corr_magnitudes(None) is None

    def test_corrected_magnitudes_are_flat_in_airmass(self, analysis, airmass_dataset):
        df, base_mag, _, _ = airmass_dataset
        out = analysis.make_airmass_corr_magnitudes(df)

        assert out["MAG_APER_AB_CORR"].notna().all()
        for source_id, expected in base_mag.items():
            corrected = out.loc[out["SOURCE_ID"] == source_id, "MAG_APER_AB_CORR"]
            np.testing.assert_allclose(corrected.values, expected, atol=1e-8)

    def test_psf_magnitudes_get_the_same_correction(self, analysis, airmass_dataset):
        df, _, _, _ = airmass_dataset
        out = analysis.make_airmass_corr_magnitudes(df)
        delta_aper = out["MAG_APER_AB"] - out["MAG_APER_AB_CORR"]
        delta_psf = out["MAG_PSF_AB"] - out["MAG_PSF_AB_CORR"]
        np.testing.assert_allclose(delta_aper.values, delta_psf.values, atol=1e-10)

    def test_correction_scatter_is_reduced(self, analysis, airmass_dataset):
        df, _, _, _ = airmass_dataset
        out = analysis.make_airmass_corr_magnitudes(df)
        for source_id in out["SOURCE_ID"].unique():
            sel = out[out["SOURCE_ID"] == source_id]
            assert sel["MAG_APER_AB_CORR"].std() < sel["MAG_APER_AB"].std()

    def test_diagnostic_plot_is_written_when_requested(
        self, analysis, airmass_dataset, tmp_path, no_usetex
    ):
        df, _, _, _ = airmass_dataset
        outfile = tmp_path / "airmass_slopes.png"
        analysis.make_airmass_corr_magnitudes(df, outfile=str(outfile))
        assert outfile.exists() and outfile.stat().st_size > 0


class TestGetLightCurve:
    def test_extracts_the_photometry_of_a_known_source(
        self, analysis, params_for, single_catalog, catalog_tables, clean_sources
    ):
        last_cat, info_cat = catalog_tables
        row = int(clean_sources[0])
        target = SkyCoord(ra=last_cat["RA"][row] * u.deg, dec=last_cat["Dec"][row] * u.deg)

        analysis.catlist = [single_catalog]
        params = params_for()
        analysis.params_cal = [params]

        df_lc = analysis.get_lc(target)

        assert len(df_lc) == 1
        entry = df_lc.iloc[0]
        assert entry["SOURCE_ID"] == row
        assert entry["JD"] == pytest.approx(info_cat.header["JD"])
        assert entry["MAG_APER_LAST"] == pytest.approx(last_cat["MAG_APER_3"][row])

        # AB magnitude is the fitted zero point applied to the count rate.
        flux_rate = last_cat["FLUX_APER_3"][row] / info_cat.header["EXPTIME"]
        assert entry["MAG_APER_AB"] == pytest.approx(
            entry["AB_ZP"] - 2.5 * np.log10(flux_rate)
        )
        assert entry["MAG_PSF_AB_ERR"] == pytest.approx(1.086 / last_cat["SN"][row])
        assert entry["ELLIPTICITY"] == pytest.approx(
            1 - info_cat.header["MED_B"] / info_cat.header["MED_A"]
        )
        assert analysis.df_lc is df_lc

    def test_no_source_at_the_position_gives_an_empty_frame(
        self, analysis, params_for, single_catalog
    ):
        analysis.catlist = [single_catalog]
        analysis.params_cal = [params_for()]
        df_lc = analysis.get_lc(SkyCoord(ra=300.0 * u.deg, dec=-45.0 * u.deg))
        assert len(df_lc) == 0


class TestPlots:
    @pytest.fixture
    def matched_sources(self):
        rng = np.random.default_rng(3)
        rows = []
        for source_id in range(6):
            mag = 13.0 + source_id * 0.7
            for epoch in range(12):
                rows.append(
                    {
                        "SOURCE_ID": source_id,
                        "JD": 2460381.0 + epoch * 0.01,
                        "MAG_APER_AB": mag + rng.normal(0.0, 0.01),
                        "MAG_PSF_AB": mag + rng.normal(0.0, 0.012),
                        "LAST_X": rng.uniform(0, 1726),
                        "LAST_Y": rng.uniform(0, 1726),
                        "AB_ZP": 22.5 + rng.normal(0.0, 0.01),
                    }
                )
        return pd.DataFrame(rows)

    def test_plot_rms_requires_matched_sources(self, analysis):
        with pytest.raises(ValueError, match="df_matchedsources is None"):
            analysis.plot_rms()

    def test_plot_rms_writes_a_figure(self, analysis, matched_sources, tmp_path, no_usetex):
        analysis.df_matchedsources = matched_sources
        outfile = tmp_path / "rms.png"
        assert analysis.plot_rms(outfile=str(outfile)) is None
        assert outfile.exists() and outfile.stat().st_size > 0

    def test_plot_transmissions_runs_for_a_catalog_list(
        self, analysis, params_for, single_catalogs, no_usetex
    ):
        analysis.catlist = single_catalogs
        analysis.params_cal = [params_for(catfile=c) for c in single_catalogs]
        assert analysis.plot_transmissions() is None
