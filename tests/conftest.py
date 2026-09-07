"""Shared fixtures for the transmission_fitter test-suite.

All fixtures here are offline: they build on the example LAST catalogs and
templates shipped in ``transmission_fitter/data``. Tests that need a live
service are marked ``network`` (see ``pytest.ini``).
"""

import os

import matplotlib

matplotlib.use("Agg")  # no interactive backend during tests

import astropy.io.fits as pyfit
import numpy as np
import pandas as pd
import pytest

import transmission_fitter
from transmission_fitter.lastcatutils import LastCatUtils

PACKAGE_DIR = os.path.dirname(transmission_fitter.__file__)
DATA_DIR = os.path.join(PACKAGE_DIR, "data")

# Gaia XP sampled spectra grid: 336-1020 nm, 343 points, 2 nm step.
GAIA_SAMPLING = np.linspace(336.0, 1020.0, 343, endpoint=True)


@pytest.fixture(scope="session")
def data_dir():
    """Absolute path to the packaged example-data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def single_catalog():
    """A single-exposure LAST catalog (subframe 1) from the example data."""
    return os.path.join(
        DATA_DIR,
        "Image_Test",
        "LAST.01.10.04_20240311.194154.510_clear_923_010_001_001_sci_proc_Cat_1.fits",
    )


@pytest.fixture(scope="session")
def single_catalogs():
    """The first three single-exposure catalogs, sorted by subframe."""
    base = os.path.join(DATA_DIR, "Image_Test")
    names = [
        "LAST.01.10.04_20240311.194154.510_clear_923_010_001_{0:03d}_sci_proc_Cat_1.fits".format(i)
        for i in (1, 2, 3)
    ]
    return [os.path.join(base, n) for n in names]


@pytest.fixture(scope="session")
def coadd_catalog():
    """A coadded LAST catalog (carries the NCOADD header keyword)."""
    return os.path.join(
        DATA_DIR,
        "Image_Test_Coadd",
        "LAST.01.10.04_20240311.193844.471_clear_923_000_001_001_sci_coadd_Cat_1.fits",
    )


@pytest.fixture(scope="session")
def catalog_tables(single_catalog):
    """``(last_cat, info_cat)`` for the single-exposure example catalog."""
    return LastCatUtils().tables_from_lastcat(single_catalog)


@pytest.fixture(scope="session")
def clean_sources(catalog_tables):
    """Row indices of unflagged, well-measured sources in the example catalog.

    These are the rows ``GaiaQuery.match_last_and_gaia`` is expected to keep:
    no rejected FLAGS bits, 5 < SN < 1000 and positive aperture/PSF fluxes.
    """
    last_cat, _ = catalog_tables
    mask = (
        (last_cat["FLAGS"] == 0)
        & (last_cat["SN"] > 5.0)
        & (last_cat["SN"] < 1000.0)
        & (last_cat["FLUX_APER_3"] > 0)
        & (last_cat["FLUX_PSF"] > 0)
    )
    return np.where(mask)[0]


@pytest.fixture(scope="session")
def gaia_sampling():
    """The fixed Gaia XP wavelength grid used by the retrieval code."""
    return GAIA_SAMPLING.copy()


@pytest.fixture
def make_spectra():
    """Factory for a synthetic ``(calibrated_spectra, sampling)`` pair.

    The spectra are smooth power laws on the Gaia XP grid, so interpolation in
    ``Prepare_Spectra_for_Fit_*`` is exact and the resulting fluxes are
    predictable.
    """

    def _make(n_sources=10, amplitude=1e-16, slope=-1.5, rel_error=0.01):
        sampling = GAIA_SAMPLING.copy()
        base = amplitude * (sampling / 500.0) ** slope
        fluxes = [base * (1.0 + 0.1 * i / max(n_sources, 1)) for i in range(n_sources)]
        spectra = pd.DataFrame(
            {
                "GaiaDR3_ID": np.arange(n_sources, dtype=np.int64) + 10**18,
                "flux": fluxes,
                "flux_error": [rel_error * f for f in fluxes],
            }
        )
        return spectra, sampling

    return _make


@pytest.fixture
def make_df_match():
    """Factory for a minimal ``df_match`` as consumed by the fitting code."""

    def _make(n_sources=10, seed=42, g_mag=14.0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "GaiaDR3_ID": np.arange(n_sources, dtype=np.int64) + 10**18,
                "G_mag": np.full(n_sources, g_mag),
                "LAST_X": rng.uniform(10.0, 1700.0, n_sources),
                "LAST_Y": rng.uniform(10.0, 1700.0, n_sources),
            }
        )

    return _make


@pytest.fixture
def no_usetex(monkeypatch):
    """Neutralise the ``rcParams['text.usetex'] = True`` set by plot helpers.

    The plotting methods in ``analysis_tools`` unconditionally switch on LaTeX
    rendering, which needs a TeX installation. This fixture swallows that one
    setting while leaving every other rcParam assignment intact.
    """
    from transmission_fitter import analysis_tools

    class _RcShim:
        def __setitem__(self, key, value):
            if key == "text.usetex":
                return
            matplotlib.rcParams[key] = value

        def __getitem__(self, key):
            return matplotlib.rcParams[key]

    monkeypatch.setattr(analysis_tools, "rcParams", _RcShim())
    monkeypatch.setattr(analysis_tools.plt, "show", lambda *a, **k: None)


def inject_source(catfile, ra, dec, out_path, row=0):
    """Copy ``catfile`` to ``out_path`` with one source moved to (ra, dec).

    Used to exercise the "a match exists" branch of the catalog-matching code,
    which the example fields do not otherwise trigger.
    """
    with pyfit.open(catfile) as hdul:
        new = pyfit.HDUList([hdu.copy() for hdu in hdul])
    new[1].data["RA"][row] = ra
    new[1].data["Dec"][row] = dec
    new.writeto(out_path, overwrite=True)
    new.close()
    return str(out_path)
