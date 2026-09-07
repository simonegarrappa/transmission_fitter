"""Offline tests for ``GaiaQuery``.

Every TAP/archive call is stubbed out, so these tests exercise the query
building, caching, matching and chunking logic without touching the network.
Live-service checks live in ``test_gaiaquery_connectivity.py`` (marked
``network``).
"""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table

from transmission_fitter import gaiaquery
from transmission_fitter.gaiaquery import ARI_TAP_URL, GaiaQuery
from transmission_fitter.lastcatutils import LastCatUtils

N_SAMPLES = 343


# --------------------------------------------------------------------------
# TAP / archive stubs
# --------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, table):
        self._table = table

    def to_table(self):
        return self._table


class _FakeTAPService:
    """Stand-in for ``pyvo.dal.TAPService`` that records the queries it gets."""

    def __init__(self, url, table=None, error=None, calls=None):
        self.url = url
        self._table = table
        self._error = error
        self.calls = calls if calls is not None else []

    def search(self, query):
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return _FakeResult(self._table)


@pytest.fixture
def patch_tap(monkeypatch):
    """Install a fake TAPService; returns the list of queries it received."""

    def _install(table=None, error=None):
        calls = []
        monkeypatch.setattr(
            gaiaquery.pyvo.dal,
            "TAPService",
            lambda url: _FakeTAPService(url, table=table, error=error, calls=calls),
        )
        return calls

    return _install


def gaia_source_table(n=3, with_nan_pm=True):
    """A ``gaiadr3.gaia_source`` result table with the aliased column names."""
    pmra = np.linspace(-5.0, 5.0, n)
    pmdec = np.linspace(5.0, -5.0, n)
    if with_nan_pm:
        pmra[0] = np.nan
        pmdec[-1] = np.nan
    return Table(
        {
            "source_id": np.arange(n, dtype=np.int64) + 10**18,
            "g_ra": np.linspace(122.7, 123.1, n),
            "g_dec": np.linspace(2.7, 3.1, n),
            "g_pmra": pmra,
            "g_pmdec": pmdec,
            "g_teff": np.full(n, 5500.0),
            "g_mag": np.linspace(13.0, 15.5, n),
            "g_color": np.linspace(0.5, 1.5, n),
        }
    )


def xp_spectra_table(source_ids):
    """An ``xp_sampled_mean_spectrum`` result table for the given ids."""
    n = len(source_ids)
    flux = np.tile(np.linspace(1e-16, 2e-16, N_SAMPLES), (n, 1))
    return Table(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "flux": flux,
            "flux_error": 0.01 * flux,
        }
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
class TestInitialisation:
    def test_utilities_only_mode_without_a_catalog(self):
        gq = GaiaQuery()
        assert gq.catfile is None
        assert gq.nonvalid_catalog is True

    def test_single_exposure_catalog_is_valid_and_not_coadded(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        assert gq.nonvalid_catalog is False
        assert gq.ncoadd == 1.0
        assert gq.sampling.shape == (N_SAMPLES,)
        assert gq.sampling[0] == pytest.approx(336.0)
        assert gq.sampling[-1] == pytest.approx(1020.0)
        assert gq.df_gaia_raw is None and gq._gaia_cache_region is None

    def test_coadd_catalog_picks_up_ncoadd(self, coadd_catalog):
        gq = GaiaQuery(coadd_catalog)
        assert gq.ncoadd == pytest.approx(20.0)


# --------------------------------------------------------------------------
# ADQL construction
# --------------------------------------------------------------------------
class TestQueryConstruction:
    def test_create_query_uses_the_field_centre_and_buffer(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        query = gq.create_query()

        header = gq.info_cat.header
        assert "gaiadr3.gaia_source" in query
        assert "CONTAINS(POINT('ICRS'" in query and "CIRCLE('ICRS'" in query
        assert str(header["RA"]) in query and str(header["DEC"]) in query

        # The search circle is the subframe half-diagonal plus the drift buffer.
        centre = SkyCoord(ra=header["RA"], dec=header["DEC"], unit="deg", frame="icrs")
        corner = SkyCoord(ra=header["RA1"], dec=header["DEC1"], unit="deg", frame="icrs")
        expected_radius = centre.separation(corner).deg + gq.gaia_query_buffer
        assert str(expected_radius) in query

    def test_create_query_applies_the_calibrator_selection(self, single_catalog):
        query = GaiaQuery(single_catalog).create_query()
        assert "has_xp_sampled = 'TRUE'" in query
        assert "phot_g_mean_mag > 12" in query
        assert "phot_g_mean_mag < 16." in query
        assert "classprob_dsc_combmod_star > 0.9" in query

    def test_create_query_caches_the_field_geometry_on_the_instance(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        assert gq.cRa is None and gq.cDec is None and gq.sep_subframe is None
        gq.create_query()
        assert gq.cRa.deg == pytest.approx(gq.info_cat.header["RA"])
        assert gq.cDec.deg == pytest.approx(gq.info_cat.header["DEC"])
        assert gq.sep_subframe.deg > 0

    def test_create_general_query_embeds_the_requested_cone(self):
        gq = GaiaQuery()
        query = gq.create_general_query(163.0, 23.0, Angle(0.3 * u.deg))
        assert "CIRCLE('ICRS',163.0,23.0,0.3" in query.replace(" ", "")
        assert "gaiadr3.gaia_source" in query


# --------------------------------------------------------------------------
# Catalog query execution
# --------------------------------------------------------------------------
class TestRunQueryToPandas:
    def test_uses_the_ari_mirror_first(self, patch_tap):
        calls = patch_tap(table=gaia_source_table())
        df = GaiaQuery().run_query_to_pandas("SELECT 1")
        assert calls == ["SELECT 1"]
        assert isinstance(df, pd.DataFrame) and len(df) == 3

    def test_missing_proper_motions_are_zero_filled(self, patch_tap):
        patch_tap(table=gaia_source_table(with_nan_pm=True))
        df = GaiaQuery().run_query_to_pandas("SELECT 1")
        assert df["g_pmra"].notna().all()
        assert df["g_pmdec"].notna().all()
        assert df["g_pmra"].iloc[0] == 0.0
        assert df["g_pmdec"].iloc[-1] == 0.0

    def test_falls_back_to_the_esa_archive_when_ari_fails(self, patch_tap, monkeypatch):
        patch_tap(error=RuntimeError("ARI down"))
        launched = []

        class _FakeJob:
            def get_results(self):
                return gaia_source_table()

        def fake_launch(query, *args, **kwargs):
            launched.append(query)
            return _FakeJob()

        monkeypatch.setattr(gaiaquery.Gaia, "launch_job_async", fake_launch)

        df = GaiaQuery().run_query_to_pandas("SELECT 2")
        assert launched == ["SELECT 2"]
        assert len(df) == 3

    def test_retries_then_raises_when_both_services_fail(self, patch_tap, monkeypatch):
        patch_tap(error=RuntimeError("ARI down"))
        attempts = []

        def fake_launch(query, *args, **kwargs):
            attempts.append(query)
            raise ConnectionResetError("Connection reset by peer")

        monkeypatch.setattr(gaiaquery.Gaia, "launch_job_async", fake_launch)
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)

        with pytest.raises(RuntimeError, match="Gaia query failed after 3 attempts"):
            GaiaQuery().run_query_to_pandas("SELECT 3", max_retries=3)
        assert len(attempts) == 3


# --------------------------------------------------------------------------
# Query caching
# --------------------------------------------------------------------------
class TestGaiaCache:
    def test_cache_is_invalid_before_any_query(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        assert not gq._is_cache_valid()

    def test_cache_is_valid_for_an_enclosing_region(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        gq.create_query()
        gq.df_gaia_raw = pd.DataFrame({"source_id": [1]})
        gq._gaia_cache_region = (gq.cRa.deg, gq.cDec.deg, gq.sep_subframe.deg + 0.5)
        assert gq._is_cache_valid()

    def test_cache_is_invalid_for_a_smaller_or_offset_region(self, single_catalog):
        gq = GaiaQuery(single_catalog)
        gq.create_query()
        gq.df_gaia_raw = pd.DataFrame({"source_id": [1]})

        gq._gaia_cache_region = (gq.cRa.deg, gq.cDec.deg, gq.sep_subframe.deg * 0.5)
        assert not gq._is_cache_valid()

        gq._gaia_cache_region = (gq.cRa.deg + 5.0, gq.cDec.deg, gq.sep_subframe.deg + 0.5)
        assert not gq._is_cache_valid()


# --------------------------------------------------------------------------
# LAST <-> Gaia matching
# --------------------------------------------------------------------------
@pytest.fixture
def fake_gaia_for_catalog(single_catalog, catalog_tables, clean_sources):
    """A synthetic Gaia catalog placed on real LAST sources.

    Contains six well-behaved sources (expected to match) plus, when the
    example catalog provides them, one source carrying a rejected FLAGS bit and
    one with SN outside the accepted range (both expected to be dropped).
    """
    last_cat, _ = catalog_tables
    lcu = LastCatUtils()

    good_rows = list(clean_sources[:6])
    rejected = {"Saturated", "NaN", "Negative", "CR_DeltaHT", "NearEdge"}
    bad_rows = [
        i
        for i in range(len(last_cat))
        if any(f in rejected for f in lcu.get_flags_keyword(last_cat["FLAGS"][i]))
    ][:1]
    low_sn_rows = [i for i in range(len(last_cat)) if 0 < last_cat["SN"][i] < 5.0][:1]

    rows = good_rows + bad_rows + low_sn_rows
    n = len(rows)
    df = pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64) + 10**18,
            "g_ra": np.asarray([last_cat["RA"][i] for i in rows], dtype=float),
            "g_dec": np.asarray([last_cat["Dec"][i] for i in rows], dtype=float),
            "g_pmra": np.zeros(n),
            "g_pmdec": np.zeros(n),
            "g_teff": np.full(n, 5500.0),
            "g_mag": np.full(n, 14.0),
            "g_color": np.linspace(0.4, 1.4, n),
        }
    )
    expected_ids = df["source_id"].values[: len(good_rows)]
    dropped_ids = df["source_id"].values[len(good_rows) :]
    return df, expected_ids, dropped_ids, good_rows


class TestMatchLastAndGaia:
    @pytest.fixture
    def matched(self, single_catalog, fake_gaia_for_catalog, monkeypatch):
        df_gaia, expected_ids, dropped_ids, good_rows = fake_gaia_for_catalog
        gq = GaiaQuery(single_catalog)
        calls = []

        def fake_query(query, **kwargs):
            calls.append(query)
            return df_gaia.copy()

        monkeypatch.setattr(gq, "run_query_to_pandas", fake_query)
        return gq, gq.match_last_and_gaia(), expected_ids, dropped_ids, good_rows, calls

    def test_recovers_the_injected_calibrators(self, matched):
        _, df_match, expected_ids, _, _, _ = matched
        assert set(df_match["GaiaDR3_ID"]) == set(expected_ids)

    def test_rejects_flagged_and_low_signal_sources(self, matched):
        _, df_match, _, dropped_ids, _, _ = matched
        assert not set(df_match["GaiaDR3_ID"]) & set(dropped_ids)

    def test_separations_are_within_the_match_radius(self, matched):
        _, df_match, _, _, _, _ = matched
        assert (df_match["ang_sep"] < 2.0).all()

    def test_fluxes_are_normalised_by_exposure_time(self, matched, catalog_tables):
        gq, df_match, _, _, _, _ = matched
        last_cat, info_cat = catalog_tables
        exptime = info_cat.header["EXPTIME"]
        row = df_match.iloc[0]
        last_idx = int(row["LAST_num"])
        assert row["LAST_FLUX_APER_3"] == pytest.approx(
            gq.ncoadd * last_cat["FLUX_APER_3"][last_idx] / exptime
        )
        assert row["LAST_FLUX_PSF"] == pytest.approx(
            gq.ncoadd * last_cat["FLUX_PSF"][last_idx] / exptime
        )

    def test_carries_the_epoch_and_sensor_coordinates(self, matched, catalog_tables):
        _, df_match, _, _, _, _ = matched
        last_cat, info_cat = catalog_tables
        assert (df_match["JD"] == info_cat.header["JD"]).all()
        row = df_match.iloc[0]
        last_idx = int(row["LAST_num"])
        assert row["LAST_X"] == pytest.approx(last_cat["X"][last_idx])
        assert row["LAST_Y"] == pytest.approx(last_cat["Y"][last_idx])

    def test_second_call_reuses_the_cached_catalog(self, matched):
        gq, _, _, _, _, calls = matched
        assert len(calls) == 1
        gq.match_last_and_gaia()
        assert len(calls) == 1, "cached Gaia catalog should not be re-queried"

    def test_a_different_sky_region_invalidates_the_cache(self, matched, single_catalogs):
        gq, _, _, _, _, calls = matched
        gq._gaia_cache_region = (0.0, 0.0, 0.01)  # pretend the cache is elsewhere
        gq.match_last_and_gaia()
        assert len(calls) == 2


# --------------------------------------------------------------------------
# XP spectra retrieval
# --------------------------------------------------------------------------
class TestRetrieveGaiaSpectraFromIds:
    def test_returns_the_fixed_sampling_grid(self, patch_tap, monkeypatch):
        patch_tap(table=xp_spectra_table([10**18]))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)
        _, sampling = GaiaQuery().retrieve_gaia_spectra_from_ids([10**18])
        assert sampling.shape == (N_SAMPLES,)
        np.testing.assert_allclose(sampling[[0, -1]], [336.0, 1020.0])

    def test_frame_is_indexed_by_source_id_with_array_columns(self, patch_tap, monkeypatch):
        ids = [10**18, 10**18 + 1]
        patch_tap(table=xp_spectra_table(ids))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)
        spectra, _ = GaiaQuery().retrieve_gaia_spectra_from_ids(ids)
        assert list(spectra.index) == ids
        assert set(spectra.columns) == {"flux", "flux_error"}
        assert np.asarray(spectra.loc[ids[0], "flux"]).shape == (N_SAMPLES,)

    def test_ids_are_split_into_chunks(self, patch_tap, monkeypatch):
        ids = [10**18 + i for i in range(5)]
        calls = patch_tap(table=xp_spectra_table(ids[:2]))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)
        GaiaQuery().retrieve_gaia_spectra_from_ids(ids, chunk_size=2)
        assert len(calls) == 3  # 2 + 2 + 1
        assert all("xp_sampled_mean_spectrum" in q for q in calls)
        assert str(ids[0]) in calls[0] and str(ids[-1]) in calls[-1]

    def test_accepts_string_ids(self, patch_tap, monkeypatch):
        ids = [10**18, 10**18 + 1]
        calls = patch_tap(table=xp_spectra_table(ids))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)
        spectra, _ = GaiaQuery().retrieve_gaia_spectra_from_ids([str(i) for i in ids])
        assert list(spectra.index) == ids
        assert str(ids[0]) in calls[0]

    def test_falls_back_to_the_esa_datalink(self, patch_tap, monkeypatch):
        patch_tap(error=RuntimeError("ARI down"))
        source_id = 10**18

        class _FakeDatalinkTable:
            def to_table(self):
                # ESA datalink returns one row per wavelength sample.
                flux = np.linspace(1e-16, 2e-16, N_SAMPLES)
                return Table(
                    {
                        "wavelength": np.linspace(336.0, 1020.0, N_SAMPLES),
                        "flux": flux,
                        "flux_error": 0.01 * flux,
                    }
                )

        def fake_load_data(**kwargs):
            assert kwargs["retrieval_type"] == "XP_SAMPLED"
            return {f"XP_SAMPLED_COMBINED-Gaia DR3 {source_id}.xml": [_FakeDatalinkTable()]}

        monkeypatch.setattr(gaiaquery.Gaia, "load_data", fake_load_data)
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)

        spectra, sampling = GaiaQuery().retrieve_gaia_spectra_from_ids([source_id])
        assert list(spectra.index) == [source_id]
        assert np.asarray(spectra.loc[source_id, "flux"]).shape == (N_SAMPLES,)


class TestRetrieveGaiaSpectra:
    def test_invalid_catalog_short_circuits(self):
        gq = GaiaQuery()  # no catfile -> nonvalid_catalog
        assert gq.retrieve_gaia_spectra() == (None, None, None, None)

    def test_end_to_end_without_htm(
        self, single_catalog, fake_gaia_for_catalog, patch_tap, monkeypatch
    ):
        df_gaia, expected_ids, _, _ = fake_gaia_for_catalog
        gq = GaiaQuery(single_catalog)
        monkeypatch.setattr(gq, "run_query_to_pandas", lambda query, **kw: df_gaia.copy())
        patch_tap(table=xp_spectra_table(expected_ids))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)

        source_ids, spectra, sampling, df_match = gq.retrieve_gaia_spectra(useHTM=False)

        assert set(source_ids) == {str(i) for i in expected_ids}
        assert len(spectra) == len(df_match) == len(expected_ids)
        # Both tables are sorted on GaiaDR3_ID so rows line up positionally.
        np.testing.assert_array_equal(
            spectra["GaiaDR3_ID"].values, df_match["GaiaDR3_ID"].values
        )
        assert sampling.shape == (N_SAMPLES,)

    def test_sources_without_spectra_are_dropped(
        self, single_catalog, fake_gaia_for_catalog, patch_tap, monkeypatch
    ):
        df_gaia, expected_ids, _, _ = fake_gaia_for_catalog
        gq = GaiaQuery(single_catalog)
        monkeypatch.setattr(gq, "run_query_to_pandas", lambda query, **kw: df_gaia.copy())
        # The archive only knows about half of the matched calibrators.
        available = expected_ids[: len(expected_ids) // 2]
        patch_tap(table=xp_spectra_table(available))
        monkeypatch.setattr(gaiaquery.time, "sleep", lambda s: None)

        source_ids, spectra, _, df_match = gq.retrieve_gaia_spectra(useHTM=False)
        assert set(source_ids) == {str(i) for i in available}
        assert len(df_match) == len(available) == len(spectra)
