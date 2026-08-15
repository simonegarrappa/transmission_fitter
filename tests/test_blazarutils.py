"""Tests for ``BlazarQuery``, the catalog cross-matching helper.

``BlazarQuery`` loads its reference catalogs through paths relative to the
*current working directory* (``./SourceCatalogs/...``), so every test here runs
with the packaged data directory as cwd; ``test_requires_the_data_directory_as_cwd``
pins that requirement down.
"""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from transmission_fitter.blazarutils import BlazarQuery

from conftest import inject_source


@pytest.fixture(scope="module")
def blazar_query():
    """A ``BlazarQuery`` built with the packaged catalogs (loads ~240k rows)."""
    from conftest import DATA_DIR

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(DATA_DIR)
        yield BlazarQuery()


class TestCatalogLoading:
    def test_requires_the_data_directory_as_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            BlazarQuery()

    def test_romabz5_is_loaded_with_matching_coordinates(self, blazar_query):
        assert len(blazar_query.romabz5) > 3000
        assert len(blazar_query.romabz5_apy) == len(blazar_query.romabz5)
        for col in ("Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "z", "Class", "Rmag"):
            assert col in blazar_query.romabz5.columns.names

    def test_sexagesimal_coordinates_are_parsed_correctly(self, blazar_query):
        cat = blazar_query.romabz5
        coords = blazar_query.romabz5_apy
        for i in (0, 100, 1000):
            expected_ra = (cat["RAh"][i] + cat["RAm"][i] / 60.0 + cat["RAs"][i] / 3600.0) * 15.0
            expected_dec = cat["DEd"][i] + cat["DEm"][i] / 60.0 + cat["DEs"][i] / 3600.0
            if str(cat["DE-"][i]).strip() == "-":
                expected_dec = -expected_dec
            assert coords.ra.deg[i] == pytest.approx(expected_ra, abs=1e-6)
            assert coords.dec.deg[i] == pytest.approx(expected_dec, abs=1e-6)

    def test_landolt_stetson_calibrators_are_loaded(self, blazar_query):
        assert len(blazar_query.edr3_calibrators) > 1000
        assert len(blazar_query.edr3_calibrators_apy) == len(blazar_query.edr3_calibrators)
        for col in ("Name", "RAJ2000", "DEJ2000", "Vmag", "StarType"):
            assert col in blazar_query.edr3_calibrators.columns

    def test_4fgl_catalog_is_loaded_on_demand(self, blazar_query, data_dir, monkeypatch):
        monkeypatch.chdir(data_dir)
        fgl, fgl_apy = blazar_query.prepare_4fgl()
        assert len(fgl) > 5000
        assert len(fgl_apy) == len(fgl)
        assert np.all(np.abs(fgl_apy.dec.deg) <= 90.0)

    def test_default_matching_configuration(self, blazar_query):
        assert blazar_query.match_radius == 2.0
        assert blazar_query.columns_bz5[:5] == [
            "LASTCatalog",
            "JD",
            "RomaBZ5_NAME",
            "RA",
            "DEC",
        ]
        assert "LS_CALIBRATOR_NAME" in blazar_query.columns_ed3calibrators


class TestSearchRomaBZ5:
    def test_field_without_blazars_returns_an_empty_typed_frame(
        self, blazar_query, single_catalog
    ):
        df = blazar_query.search_romabz5_singleimage(single_catalog)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == blazar_query.columns_bz5

    @pytest.fixture
    def catalog_with_blazar(self, blazar_query, single_catalog, tmp_path):
        """A copy of the example catalog with one source moved onto a blazar."""
        blazar_idx = 500
        coord = blazar_query.romabz5_apy[blazar_idx]
        path = inject_source(
            single_catalog, coord.ra.deg, coord.dec.deg, tmp_path / "with_blazar.fits"
        )
        return path, blazar_idx, coord

    def test_finds_an_injected_blazar(self, blazar_query, catalog_with_blazar, catalog_tables):
        path, blazar_idx, coord = catalog_with_blazar
        _, info_cat = catalog_tables

        df = blazar_query.search_romabz5_singleimage(path)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["RomaBZ5_NAME"] == blazar_query.romabz5["Name"][blazar_idx]
        assert row["RA"] == pytest.approx(coord.ra.deg)
        assert row["DEC"] == pytest.approx(coord.dec.deg)
        assert row["JD"] == pytest.approx(info_cat.header["JD"])
        assert row["LASTCatalog"] == path
        assert row["Redshift"] == pytest.approx(blazar_query.romabz5["z"][blazar_idx])

    def test_reports_the_last_photometry_of_the_match(
        self, blazar_query, catalog_with_blazar, catalog_tables
    ):
        path, _, _ = catalog_with_blazar
        last_cat, _ = catalog_tables  # row 0 is the one that was moved
        row = blazar_query.search_romabz5_singleimage(path).iloc[0]
        assert float(np.atleast_1d(row["LAST_MAG_APER"])[0]) == pytest.approx(
            last_cat["MAG_APER_3"][0]
        )
        assert float(np.atleast_1d(row["LAST_MAG_PSF"])[0]) == pytest.approx(
            last_cat["MAG_PSF"][0]
        )

    def test_multi_image_search_concatenates_per_image_results(
        self, blazar_query, catalog_with_blazar, single_catalog
    ):
        path, _, _ = catalog_with_blazar
        df = blazar_query.search_romabz5_multimage([single_catalog, path])
        assert len(df) == 1
        assert list(df.columns) == blazar_query.columns_bz5

    def test_multi_image_search_reads_a_list_file(
        self, blazar_query, catalog_with_blazar, single_catalog, tmp_path
    ):
        path, _, _ = catalog_with_blazar
        listfile = tmp_path / "catalogs.txt"
        listfile.write_text(f"{single_catalog}\n{path}\n")
        df = blazar_query.search_romabz5_multimage(str(listfile))
        assert len(df) == 1


class TestSearchLandoltStetsonCalibrators:
    def test_field_without_calibrators_returns_an_empty_typed_frame(
        self, blazar_query, single_catalog
    ):
        df = blazar_query.search_gaiaedr3_calibrators_singleimage(single_catalog)
        assert len(df) == 0
        assert list(df.columns) == blazar_query.columns_ed3calibrators

    def test_finds_an_injected_calibrator(self, blazar_query, single_catalog, tmp_path):
        cal_idx = 1000
        coord = blazar_query.edr3_calibrators_apy[cal_idx]
        path = inject_source(
            single_catalog, coord.ra.deg, coord.dec.deg, tmp_path / "with_calibrator.fits"
        )

        df = blazar_query.search_gaiaedr3_calibrators_singleimage(path)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["LS_CALIBRATOR_NAME"] == blazar_query.edr3_calibrators["Name"].values[cal_idx]
        assert row["Vmag"] == pytest.approx(
            blazar_query.edr3_calibrators["Vmag"].values[cal_idx], nan_ok=True
        )
        assert row["RA"] == pytest.approx(coord.ra.deg)

    def test_multi_image_search_over_a_list(self, blazar_query, single_catalogs):
        df = blazar_query.search_gaiaedr3_calibrators_multimage(single_catalogs)
        assert list(df.columns) == blazar_query.columns_ed3calibrators
        assert len(df) == 0


class TestSearchBlazarsInVisits:
    def test_unknown_catalog_name_returns_none(self, blazar_query, single_catalogs):
        assert blazar_query.search_Blazars_in_LAST_Visits(single_catalogs, "UNKNOWN") is None
