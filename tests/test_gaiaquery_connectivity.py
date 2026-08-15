"""
Connectivity tests for the TAP server used by gaiaquery.

These tests hit the live ARI-Gaia Heidelberg TAP mirror
(``gaiaquery.ARI_TAP_URL``), which hosts ``gaiadr3.xp_sampled_mean_spectrum``
and is the primary source for Gaia XP sampled spectra (the ESA archive is the
fallback, see ``GaiaQuery.retrieve_gaia_spectra_from_ids``).

They require network access and are marked with the ``network`` marker so they
can be skipped offline::

    pytest tests/test_gaiaquery_connectivity.py
    pytest -m "not network"          # skip them
"""

import numpy as np
import pyvo
import pytest

from transmission_fitter.gaiaquery import ARI_TAP_URL, GaiaQuery

# Fixed XP sampling grid: 336-1020 nm, 343 points, 2 nm step.
N_SAMPLES = 343

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tap_service():
    """A TAPService pointed at the ARI-Gaia mirror, or skip if unreachable."""
    try:
        return pyvo.dal.TAPService(ARI_TAP_URL)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"Could not reach ARI-Gaia TAP service {ARI_TAP_URL}: {exc}")


def test_tap_service_reachable(tap_service):
    """The TAP service responds and advertises the spectra table."""
    try:
        tables = tap_service.tables
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"TAP /tables endpoint unavailable: {exc}")

    assert "gaiadr3.xp_sampled_mean_spectrum" in tables.keys(), (
        "Expected table gaiadr3.xp_sampled_mean_spectrum not found on "
        f"{ARI_TAP_URL}; available tables changed?"
    )


def test_tap_query_returns_spectrum(tap_service):
    """A minimal query returns a source with the expected flux columns/shape."""
    try:
        result = tap_service.search(
            "SELECT TOP 1 source_id, flux, flux_error "
            "FROM gaiadr3.xp_sampled_mean_spectrum"
        )
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"ARI-Gaia query failed (server-side issue?): {exc}")

    tbl = result.to_table()
    assert len(tbl) == 1, "Expected exactly one row from SELECT TOP 1"

    row = tbl[0]
    assert int(row["source_id"]) > 0

    flux = np.asarray(row["flux"])
    flux_error = np.asarray(row["flux_error"])
    assert flux.shape == (N_SAMPLES,)
    assert flux_error.shape == (N_SAMPLES,)
    assert np.isfinite(flux).any(), "Flux array is entirely non-finite"


def test_source_catalog_query_roundtrip(tap_service):
    """The gaiadr3.gaia_source cone query (used for catalog matching) works.

    This exercises the same ADQL geometry dialect that create_query builds and
    that run_query_to_pandas sends to ARI-Gaia. It is a distinct service path
    from the XP-spectra queries above (the catalog query used to go to the ESA
    archive, which fails with 'Connection reset by peer').
    """
    import astropy.units as u
    from astropy.coordinates import Angle

    gq = GaiaQuery()  # utilities-only mode
    query = gq.create_general_query(163.0, 23.0, Angle(0.3 * u.deg))

    df = gq.run_query_to_pandas(query)
    assert len(df) > 0, "Expected sources in a 0.3 deg cone around (163, 23)"
    for col in ("source_id", "g_ra", "g_dec", "g_mag", "g_pmra", "g_pmdec"):
        assert col in df.columns
    assert df["g_pmra"].notna().all(), "g_pmra should be NaN-filled to 0"


def test_retrieve_gaia_spectra_from_ids_roundtrip(tap_service):
    """End-to-end: GaiaQuery retrieves a spectrum for a real source_id."""
    # Grab a real id that is known to have an XP sampled spectrum.
    seed = tap_service.search(
        "SELECT TOP 1 source_id FROM gaiadr3.xp_sampled_mean_spectrum"
    ).to_table()
    source_id = int(seed[0]["source_id"])

    gq = GaiaQuery()  # utilities-only mode, no catalog file needed
    spectra_df, sampling = gq.retrieve_gaia_spectra_from_ids([source_id])

    assert sampling.shape == (N_SAMPLES,)
    np.testing.assert_allclose(sampling[[0, -1]], [336.0, 1020.0])

    assert source_id in spectra_df.index, (
        "Spectrum for the requested source_id was not retrieved"
    )
    flux = np.asarray(spectra_df.loc[source_id, "flux"])
    assert flux.shape == (N_SAMPLES,)
