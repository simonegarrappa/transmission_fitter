"""Tests for ``AbsoluteCalibration``, the transmission-fitting engine.

The unit tests below drive the model pieces (parameter setup, transmission
model, residual function, spectra preparation) directly. The final class runs a
closed-loop fit on synthetic photometry generated from the model itself, which
is the only way to check that ``fit_transmission`` recovers what it should.
"""

import math

import numpy as np
import pandas as pd
import pytest
from astropy.constants import c, h

from transmission_fitter import fitutils
from transmission_fitter.abscalutils import make_wvl_array
from transmission_fitter.fitutils import AbsoluteCalibration

WVL = make_wvl_array()


@pytest.fixture(scope="module")
def abscal(single_catalog):
    """An ``AbsoluteCalibration`` built on the example single-exposure catalog."""
    return AbsoluteCalibration(catfile=single_catalog)


@pytest.fixture
def abscal_noatm(single_catalog):
    """Same, with the atmosphere switched off (fast, deterministic model)."""
    return AbsoluteCalibration(catfile=single_catalog, use_atm=False)


def at(array, wavelength_nm):
    return float(array[np.argmin(np.abs(WVL - wavelength_nm))])


class TestInitialisation:
    def test_pulls_observing_conditions_from_the_catalog_header(self, abscal, catalog_tables):
        _, info_cat = catalog_tables
        assert abscal.exptime == info_cat.header["EXPTIME"]
        assert abscal.mnttemp == info_cat.header["MNTTEMP"]
        assert abscal.jd_ == info_cat.header["JD"]
        assert 0.0 < abscal.z_ < 90.0

    def test_default_flags_and_geometry(self, abscal):
        assert abscal.band == "LAST"
        assert abscal.use_atm is True
        assert abscal.useHTM is False
        assert abscal.ErrorEstimation == "ErrProp"
        # Collecting area of a 27.94 cm aperture, in m^2.
        assert abscal.Ageom == pytest.approx(math.pi * 0.1397**2)

    def test_templates_are_loaded_onto_the_model_grid(self, abscal):
        assert abscal.wvl_arr.shape == (401,)
        assert abscal.transmission_jolly.shape == (401,)
        assert np.all(abscal.transmission_jolly >= 0.0)
        assert len(abscal.Ref_mirror) == 3  # quadratic polyfit coefficients
        assert len(abscal.Trasm_corrector) == 3


class TestInitializeParams:
    def test_all_parameters_start_frozen(self, abscal):
        params = abscal.Initialize_Params()
        assert all(not p.vary for p in params.values())

    def test_expected_parameter_families_are_present(self, abscal):
        params = abscal.Initialize_Params()
        for name in ("norm", "amplitude", "center", "sigma", "gamma"):
            assert name in params
        for name in ("pressure", "AOD", "alpha", "ozone_col", "PW", "temperature"):
            assert name in params
        for i in range(9):
            assert f"l{i}" in params
        for i in range(5):
            assert f"r{i}" in params
        for name in ("kx0", "ky0", "kx", "ky", "kx2", "ky2", "kx3", "ky3", "kx4", "ky4", "kxy"):
            assert name in params

    def test_temperature_is_bracketed_around_the_mount_temperature(self, abscal):
        params = abscal.Initialize_Params()
        assert params["temperature"].value == pytest.approx(abscal.mnttemp)
        assert params["temperature"].min == pytest.approx(abscal.mnttemp - 5.0)
        assert params["temperature"].max == pytest.approx(abscal.mnttemp + 5.0)

    def test_normalisation_is_bounded_to_a_physical_range(self, abscal):
        params = abscal.Initialize_Params()
        assert 0.0 <= params["norm"].value <= 1.0
        assert (params["norm"].min, params["norm"].max) == (0.0, 1.0)

    def test_returns_an_independent_object_each_call(self, abscal):
        first = abscal.Initialize_Params()
        first["norm"].set(value=0.123)
        assert abscal.Initialize_Params()["norm"].value != pytest.approx(0.123)


class TestGetNewLambda:
    def test_maps_the_model_grid_onto_minus_one_to_one(self, abscal):
        mapped = abscal.Get_newLambda(np.array([300.0, 700.0, 1100.0]))
        np.testing.assert_allclose(mapped, [-1.0, 0.0, 1.0], atol=1e-12)

    def test_is_affine_and_increasing(self, abscal):
        mapped = abscal.Get_newLambda(WVL)
        assert np.all(np.diff(mapped) > 0)
        assert np.allclose(np.diff(mapped), mapped[1] - mapped[0])

    def test_custom_output_range(self, abscal):
        mapped = abscal.Get_newLambda(np.array([300.0, 1100.0]), min_1=0.0, max_1=10.0)
        np.testing.assert_allclose(mapped, [0.0, 10.0], atol=1e-12)


class TestOpticalModel:
    def test_legendre_model_is_positive_definite(self, abscal):
        params = abscal.Initialize_Params().valuesdict()
        model = abscal.LegendreModel(WVL, *[params[f"l{i}"] for i in range(9)])
        assert model.shape == WVL.shape
        assert np.all(model > 0.0)  # it is an exponential of a polynomial

    def test_ota_transmission_peaks_in_the_visible(self, abscal):
        params = abscal.Initialize_Params()
        ota = abscal.Calculate_OTA_Transmission_from_Model(params)
        assert ota.shape == WVL.shape
        assert np.all(ota >= 0.0)
        assert 400.0 < WVL[np.argmax(ota)] < 700.0
        assert ota.max() < 1.0

    def test_full_transmission_is_a_valid_throughput(self, abscal):
        transm = abscal.Calculate_Full_Transmission_from_params(abscal.Initialize_Params())
        assert transm.shape == WVL.shape
        assert np.all((transm >= 0.0) & (transm <= 1.0))
        assert 400.0 < WVL[np.argmax(transm)] < 700.0

    def test_atmosphere_only_removes_light(self, abscal, abscal_noatm):
        params = abscal.Initialize_Params()
        with_atm = abscal.Calculate_Full_Transmission_from_params(params)
        without_atm = abscal_noatm.Calculate_Full_Transmission_from_params(params)
        assert np.all(with_atm <= without_atm + 1e-12)
        assert with_atm.max() < without_atm.max()

    @pytest.mark.parametrize(
        "band, peak_nm",
        [
            ("SDSS_u", 366.0),
            ("SDSS_g", 480.0),
            ("SDSS_r", 625.0),
            ("SDSS_i", 765.0),
            ("SDSS_z", 900.0),
        ],
    )
    def test_sdss_bands_load_their_own_throughput_curves(self, single_catalog, band, peak_nm):
        obj = AbsoluteCalibration(catfile=single_catalog, band=band, use_atm=False)
        transm = obj.Calculate_Full_Transmission_from_params(obj.Initialize_Params())
        assert np.all((transm >= 0.0) & (transm <= 1.0))
        assert WVL[np.argmax(transm)] == pytest.approx(peak_nm, abs=60.0)

    def test_sdss_bands_are_ordered_in_wavelength(self, single_catalog):
        peaks = []
        for band in ("SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"):
            obj = AbsoluteCalibration(catfile=single_catalog, band=band, use_atm=False)
            transm = obj.Calculate_Full_Transmission_from_params(obj.Initialize_Params())
            peaks.append(WVL[np.argmax(transm)])
        assert peaks == sorted(peaks)

    def test_chebyshev_term_scales_the_throughput(self, abscal_noatm):
        params = abscal_noatm.Initialize_Params()
        baseline = abscal_noatm.Calculate_Full_Transmission_from_params(params)
        params["r0"].set(value=-1.0)
        damped = abscal_noatm.Calculate_Full_Transmission_from_params(params)
        np.testing.assert_allclose(damped, baseline * math.exp(-1.0), rtol=1e-10)


class TestResidFunc:
    @staticmethod
    def _design_matrix(fluxes, x_coords, y_coords):
        """Stack per-source spectra with their sensor coordinates."""
        return np.hstack(
            [
                np.asarray(fluxes),
                np.asarray(x_coords)[:, None],
                np.asarray(y_coords)[:, None],
            ]
        )

    def test_zero_point_is_a_plausible_ab_magnitude(self, abscal):
        params = abscal.Initialize_Params()
        zp = abscal.ResidFunc(params, np.array([863.0, 863.0]), calc_zp=True)
        assert 18.0 < float(zp) < 26.0

    def test_zero_point_scales_with_the_normalisation(self, abscal):
        params = abscal.Initialize_Params()
        zp1 = float(abscal.ResidFunc(params, np.array([863.0, 863.0]), calc_zp=True))
        params["norm"].set(value=params["norm"].value / 2.0)
        zp2 = float(abscal.ResidFunc(params, np.array([863.0, 863.0]), calc_zp=True))
        assert zp1 - zp2 == pytest.approx(2.5 * math.log10(2.0))

    def test_field_correction_is_flat_until_spatial_terms_are_set(self, abscal):
        params = abscal.Initialize_Params()
        x_in = np.array([100.0, 1600.0])
        assert float(
            abscal.ResidFunc(params, x_in, calc_zp=True, field_corr_=True)
        ) == pytest.approx(0.0)

        params["kx0"].set(value=0.2)
        params["kx"].set(value=0.1)
        fc = float(abscal.ResidFunc(params, x_in, calc_zp=True, field_corr_=True))
        assert fc != pytest.approx(0.0)

    def test_field_correction_is_the_zero_point_offset(self, abscal):
        params = abscal.Initialize_Params()
        params["kx0"].set(value=0.15)
        params["ky"].set(value=-0.05)
        x_in = np.array([300.0, 1200.0])
        zp_flat_params = abscal.Initialize_Params()
        zp_flat = float(abscal.ResidFunc(zp_flat_params, x_in, calc_zp=True))
        zp = float(abscal.ResidFunc(params, x_in, calc_zp=True))
        fc = float(abscal.ResidFunc(params, x_in, calc_zp=True, field_corr_=True))
        assert zp - zp_flat == pytest.approx(fc)

    def test_model_matches_the_hand_computed_synthetic_magnitude(self, abscal_noatm):
        params = abscal_noatm.Initialize_Params()
        spectrum = np.full((1, WVL.size), 1e-16)
        x_in = self._design_matrix(spectrum, [863.0], [863.0])

        model = abscal_noatm.ResidFunc(params, x_in)

        transm = abscal_noatm.Calculate_Full_Transmission_from_params(params)
        integral = np.trapezoid(transm * spectrum[0] * WVL, x=WVL)
        expected = 2.5 * np.log10(
            params["norm"].value * abscal_noatm.Ageom * integral / (h.value * c.value * 1e9)
        )
        assert float(model[0]) == pytest.approx(expected)

    def test_brighter_sources_give_brighter_model_values(self, abscal_noatm):
        params = abscal_noatm.Initialize_Params()
        spectra = np.vstack([np.full(WVL.size, 1e-16), np.full(WVL.size, 2e-16)])
        x_in = self._design_matrix(spectra, [863.0, 863.0], [863.0, 863.0])
        model = abscal_noatm.ResidFunc(params, x_in)
        assert model[1] - model[0] == pytest.approx(2.5 * math.log10(2.0))

    def test_residual_variants(self, abscal_noatm):
        params = abscal_noatm.Initialize_Params()
        spectra = np.vstack([np.full(WVL.size, 1e-16), np.full(WVL.size, 2e-16)])
        x_in = self._design_matrix(spectra, [400.0, 1200.0], [400.0, 1200.0])

        model = abscal_noatm.ResidFunc(params, x_in)
        data = model + np.array([0.1, -0.2])
        dataerr = np.array([0.05, 0.10])

        np.testing.assert_allclose(abscal_noatm.ResidFunc(params, x_in, data=data), model - data)
        np.testing.assert_allclose(
            abscal_noatm.ResidFunc(params, x_in, data=data, dataerr=dataerr),
            (model - data) / dataerr,
        )
        np.testing.assert_allclose(
            abscal_noatm.ResidFunc(params, x_in, data=data, magres=True), np.abs(data - model)
        )


class TestEstimatePerturbedFluxes:
    def test_flux_is_linear_in_the_input_spectrum(self, abscal):
        transmission = abscal.transmission_jolly
        spectra = np.vstack([np.full(WVL.size, 1e-16), np.full(WVL.size, 3e-16)])
        fluxes = abscal.EstimatePerturbedFluxes(spectra, WVL, transmission=transmission)
        assert fluxes.shape == (2,)
        assert fluxes[1] / fluxes[0] == pytest.approx(3.0)

    def test_matches_the_analytic_integral(self, abscal):
        transmission = abscal.transmission_jolly
        spectrum = np.full((1, WVL.size), 1e-16)
        flux = abscal.EstimatePerturbedFluxes(spectrum, WVL, transmission=transmission)
        expected = (
            0.5
            * abscal.Ageom
            * np.trapezoid(transmission * spectrum[0] * WVL, x=WVL)
            / (h.value * c.value * 1e9)
        )
        assert float(flux[0]) == pytest.approx(expected)

    def test_ret_trans_returns_the_transmission_unchanged(self, abscal):
        transmission = abscal.transmission_jolly
        out = abscal.EstimatePerturbedFluxes(
            np.full((1, WVL.size), 1e-16), WVL, transmission=transmission, ret_trans=True
        )
        np.testing.assert_array_equal(out, transmission)


class TestPrepareSpectraForFit:
    @pytest.fixture
    def prepared(self, abscal_noatm, make_spectra, make_df_match):
        spectra, sampling = make_spectra(n_sources=5)
        df_match = make_df_match(n_sources=5)
        source_ids = list(spectra["GaiaDR3_ID"].astype(str))
        matrix, errors = abscal_noatm.Prepare_Spectra_for_Fit_ErrProp(
            source_ids, spectra, sampling, df_match
        )
        return matrix, errors, spectra, sampling, df_match

    def test_matrix_shape_carries_spectra_plus_sensor_coordinates(self, prepared):
        matrix, _, _, _, df_match = prepared
        assert matrix.shape == (5, WVL.size + 2)
        np.testing.assert_allclose(matrix[:, -2], df_match["LAST_X"].values)
        np.testing.assert_allclose(matrix[:, -1], df_match["LAST_Y"].values)

    def test_spectra_are_interpolated_onto_the_model_grid(self, prepared):
        matrix, _, spectra, sampling, _ = prepared
        expected = np.interp(WVL[(WVL >= 336.0) & (WVL <= 1020.0)], sampling, spectra["flux"][0])
        got = matrix[0, : WVL.size][(WVL >= 336.0) & (WVL <= 1020.0)]
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_outside_gaia_coverage_the_edge_value_is_held(self, prepared):
        matrix, _, spectra, _, _ = prepared
        row = matrix[0, : WVL.size]
        uv = WVL < 336.0
        ir = WVL > 1020.0
        assert np.allclose(row[uv], row[np.argmin(np.abs(WVL - 336.0))])
        assert np.allclose(row[ir], row[np.argmin(np.abs(WVL - 1020.0))])

    def test_flux_errors_are_positive_and_scale_with_the_spectral_errors(
        self, abscal_noatm, make_spectra, make_df_match
    ):
        spectra, sampling = make_spectra(n_sources=4, rel_error=0.01)
        noisier, _ = make_spectra(n_sources=4, rel_error=0.05)
        df_match = make_df_match(n_sources=4)
        ids = list(spectra["GaiaDR3_ID"].astype(str))

        _, errors = abscal_noatm.Prepare_Spectra_for_Fit_ErrProp(ids, spectra, sampling, df_match)
        _, errors_noisy = abscal_noatm.Prepare_Spectra_for_Fit_ErrProp(
            ids, noisier, sampling, df_match
        )

        errors = np.asarray(errors)
        assert errors.shape == (4,)
        assert np.all(errors > 0)
        np.testing.assert_allclose(np.asarray(errors_noisy) / errors, 5.0, rtol=1e-8)

    def test_monte_carlo_variant_produces_comparable_errors(
        self, abscal_noatm, make_spectra, make_df_match
    ):
        spectra, sampling = make_spectra(n_sources=3)
        df_match = make_df_match(n_sources=3)
        ids = list(spectra["GaiaDR3_ID"].astype(str))

        matrix_mc, errors_mc = abscal_noatm.Prepare_Spectra_for_Fit_MC(
            ids, spectra, sampling, df_match
        )
        matrix_ep, errors_ep = abscal_noatm.Prepare_Spectra_for_Fit_ErrProp(
            ids, spectra, sampling, df_match
        )

        assert matrix_mc.shape == matrix_ep.shape
        np.testing.assert_allclose(matrix_mc, matrix_ep)
        assert np.all(np.asarray(errors_mc) > 0)
        # Both estimators work off the same 1% spectral errors, so they should
        # agree to within an order of magnitude.
        ratio = np.asarray(errors_mc) / np.asarray(errors_ep)
        assert np.all((ratio > 0.05) & (ratio < 20.0))


class TestMatchGaia:
    """``match_Gaia`` only orchestrates ``GaiaQuery``; the query is stubbed."""

    @pytest.fixture
    def stub_gaia_query(self, monkeypatch, make_spectra, make_df_match):
        spectra, sampling = make_spectra(n_sources=4)
        df_match = make_df_match(n_sources=4)

        class _StubGaiaQuery:
            instances = []

            def __init__(self, catfile):
                self.catfile = catfile
                self.sampling = sampling
                self.df_gaia_raw = None
                self._gaia_cache_region = None
                self.calls = []
                _StubGaiaQuery.instances.append(self)

            def retrieve_gaia_spectra(self, useHTM=False):
                self.calls.append(("retrieve", useHTM))
                self.df_gaia_raw = pd.DataFrame({"source_id": [1]})
                self._gaia_cache_region = (1.0, 2.0, 0.5)
                return (
                    list(df_match["GaiaDR3_ID"].astype(str)),
                    spectra.copy(),
                    sampling,
                    df_match.copy(),
                )

            def match_last_and_gaia(self):
                self.calls.append(("match",))
                self.df_gaia_raw = pd.DataFrame({"source_id": [1]})
                self._gaia_cache_region = (1.0, 2.0, 0.5)
                return df_match.copy()

        _StubGaiaQuery.instances = []
        monkeypatch.setattr(fitutils, "GaiaQuery", _StubGaiaQuery)
        return _StubGaiaQuery, spectra, sampling, df_match

    def test_downloading_path_stores_results_and_cache(self, abscal_noatm, stub_gaia_query):
        stub_cls, spectra, sampling, df_match = stub_gaia_query
        source_ids, calibrated, returned_sampling, returned_match = abscal_noatm.match_Gaia(
            get_spectra=True
        )

        assert stub_cls.instances[0].calls == [("retrieve", False)]
        assert source_ids == list(df_match["GaiaDR3_ID"].astype(str))
        assert len(calibrated) == len(df_match)
        np.testing.assert_allclose(returned_sampling, sampling)
        assert abscal_noatm.df_match is returned_match
        assert abscal_noatm._gaia_cache_region == (1.0, 2.0, 0.5)
        assert abscal_noatm.df_gaia_raw is not None

    def test_htm_mode_forces_spectra_retrieval(self, single_catalog, stub_gaia_query):
        stub_cls, _, _, _ = stub_gaia_query
        obj = AbsoluteCalibration(catfile=single_catalog, useHTM=True, use_atm=False)
        obj.match_Gaia(get_spectra=False)
        assert stub_cls.instances[0].calls == [("retrieve", True)]

    def test_reuse_path_matches_against_already_downloaded_spectra(
        self, abscal_noatm, stub_gaia_query
    ):
        stub_cls, spectra, _, df_match = stub_gaia_query
        abscal_noatm.calibrated_spectra = spectra.copy()

        source_ids, calibrated, _, returned_match = abscal_noatm.match_Gaia(get_spectra=False)

        assert stub_cls.instances[0].calls == [("match",)]
        assert set(source_ids) == set(df_match["GaiaDR3_ID"].astype(str))
        assert len(calibrated) == len(returned_match)
        np.testing.assert_array_equal(
            calibrated["GaiaDR3_ID"].values, returned_match["GaiaDR3_ID"].values
        )


@pytest.mark.slow
class TestFitTransmissionClosedLoop:
    """End-to-end fit on photometry generated from the model itself."""

    @pytest.fixture(scope="class")
    def fit_result(self, single_catalog):
        n_sources = 40
        true_norm = 0.55

        obj = AbsoluteCalibration(catfile=single_catalog, use_atm=False)
        sampling = np.linspace(336.0, 1020.0, 343, endpoint=True)
        base = 1e-16 * (sampling / 500.0) ** -1.5
        spectra = pd.DataFrame(
            {
                "GaiaDR3_ID": np.arange(n_sources, dtype=np.int64) + 10**18,
                "flux": [base * (1.0 + 0.1 * i / n_sources) for i in range(n_sources)],
                "flux_error": [0.01 * base * (1.0 + 0.1 * i / n_sources) for i in range(n_sources)],
            }
        )
        rng = np.random.default_rng(7)
        df_match = pd.DataFrame(
            {
                "GaiaDR3_ID": spectra["GaiaDR3_ID"].values,
                "G_mag": np.full(n_sources, 14.0),
                "LAST_X": rng.uniform(10.0, 1700.0, n_sources),
                "LAST_Y": rng.uniform(10.0, 1700.0, n_sources),
            }
        )

        obj.source_ids = list(spectra["GaiaDR3_ID"].astype(str))
        obj.calibrated_spectra = spectra
        obj.sampling = sampling

        matrix, _ = obj.Prepare_Spectra_for_Fit_ErrProp(
            obj.source_ids, spectra, sampling, df_match
        )
        truth = obj.Initialize_Params()
        truth["norm"].set(value=true_norm)
        synthetic_mag = obj.ResidFunc(truth, matrix)
        df_match["LAST_FLUX_APER_3"] = 10 ** (synthetic_mag / 2.5)
        df_match["LAST_FLUX_PSF"] = 10 ** (synthetic_mag / 2.5)

        obj.df_match = df_match
        params_out, df_match_fit = obj.fit_transmission()
        return obj, params_out, df_match_fit, true_norm, n_sources

    def test_recovers_the_injected_normalisation(self, fit_result):
        _, params_out, _, true_norm, _ = fit_result
        assert params_out["norm"].value == pytest.approx(true_norm, rel=1e-3)

    def test_spatial_terms_stay_near_zero_for_a_flat_field(self, fit_result):
        _, params_out, _, _, _ = fit_result
        for name in ("kx0", "kx", "ky", "kx2", "ky2", "kx3", "ky3", "kx4", "ky4", "kxy"):
            assert abs(params_out[name].value) < 1e-2, name

    def test_output_frame_is_annotated_and_sigma_clipped(self, fit_result):
        _, _, df_match_fit, _, n_sources = fit_result
        for col in ("FIT_STATUS", "MAG_PREDICTED", "SYNTHETIC_ERR"):
            assert col in df_match_fit.columns
        assert 0 < len(df_match_fit) <= n_sources
        assert np.all(np.isfinite(df_match_fit["MAG_PREDICTED"].values))

    def test_predicted_magnitudes_reproduce_the_synthetic_photometry(self, fit_result):
        _, _, df_match_fit, _, _ = fit_result
        observed = 2.5 * np.log10(df_match_fit["LAST_FLUX_APER_3"].values)
        np.testing.assert_allclose(
            df_match_fit["MAG_PREDICTED"].values, observed, atol=1e-3
        )

    def test_nan_photometry_is_rejected_loudly(self, single_catalog, make_spectra, make_df_match):
        obj = AbsoluteCalibration(catfile=single_catalog, use_atm=False)
        spectra, sampling = make_spectra(n_sources=5)
        df_match = make_df_match(n_sources=5)
        df_match["LAST_FLUX_APER_3"] = np.nan
        df_match["LAST_FLUX_PSF"] = np.nan

        obj.source_ids = list(spectra["GaiaDR3_ID"].astype(str))
        obj.calibrated_spectra = spectra
        obj.sampling = sampling
        obj.df_match = df_match

        with pytest.raises(ValueError, match="NaN values found"):
            obj.fit_transmission()
