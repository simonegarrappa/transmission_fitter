"""Tests for ``LastCatUtils``, the reader for LAST catalog products."""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from transmission_fitter.abscalutils import get_airmass_from_zenith
from transmission_fitter.lastcatutils import LastCatUtils


@pytest.fixture(scope="module")
def lcu():
    return LastCatUtils()


class TestTablesFromLastCat:
    def test_returns_source_table_and_header_hdu(self, lcu, single_catalog):
        last_cat, info_cat = lcu.tables_from_lastcat(single_catalog)
        assert len(last_cat) > 0
        for col in ("RA", "Dec", "FLUX_APER_3", "FLUX_PSF", "SN", "FLAGS", "X", "Y"):
            assert col in last_cat.columns.names
        for key in ("RA", "DEC", "JD", "EXPTIME", "MNTTEMP", "DATE-OBS"):
            assert key in info_cat.header

    def test_column_access_is_case_insensitive(self, lcu, single_catalog):
        """Callers use both ``Dec`` and ``DEC``; FITS_rec must accept either."""
        last_cat, _ = lcu.tables_from_lastcat(single_catalog)
        np.testing.assert_array_equal(last_cat["Dec"], last_cat["DEC"])


class TestHeaderAccessors:
    def test_exptime_temperature_and_jd_come_from_the_header(self, lcu, catalog_tables):
        _, info_cat = catalog_tables
        assert lcu.get_exptime_from_cat(info_cat) == info_cat.header["EXPTIME"]
        assert lcu.get_temperature_from_cat(info_cat) == info_cat.header["MNTTEMP"]
        assert lcu.get_jd_from_cat(info_cat) == info_cat.header["JD"]

    def test_airmass_matches_an_independent_altaz_computation(self, lcu, catalog_tables):
        _, info_cat = catalog_tables
        coord = SkyCoord(
            ra=info_cat.header["RA"] * u.deg, dec=info_cat.header["DEC"] * u.deg, frame="icrs"
        )
        altaz = coord.transform_to(
            AltAz(obstime=info_cat.header["DATE-OBS"], location=lcu.neot_semadar)
        )
        expected = lcu.get_hardie_airmass(altitude=altaz.alt.degree)
        assert float(lcu.get_airmass_from_cat(info_cat)) == pytest.approx(float(expected))

    def test_airmass_is_physical_for_the_example_field(self, lcu, catalog_tables):
        _, info_cat = catalog_tables
        airmass = float(lcu.get_airmass_from_cat(info_cat))
        assert 1.0 < airmass < 3.0

    def test_zenith_angle_is_consistent_with_the_airmass(self, lcu, catalog_tables):
        _, info_cat = catalog_tables
        zenith = lcu.get_zenith_from_cat(info_cat)
        airmass = float(lcu.get_airmass_from_cat(info_cat))
        assert 0.0 < zenith < 90.0
        assert get_airmass_from_zenith(zenith) == pytest.approx(airmass)

    def test_observatory_location_is_neot_semadar(self, lcu):
        assert lcu.neot_semadar.lat.deg == pytest.approx(30.053072)
        assert lcu.neot_semadar.lon.deg == pytest.approx(35.040858)
        assert lcu.neot_semadar.height.to_value(u.m) == pytest.approx(415.4)


class TestHardieAirmass:
    def test_unity_at_the_zenith(self, lcu):
        assert float(lcu.get_hardie_airmass(zenith_angle=0.0)) == pytest.approx(1.0)
        assert float(lcu.get_hardie_airmass(altitude=90.0)) == pytest.approx(1.0)

    def test_zenith_angle_and_altitude_are_equivalent_inputs(self, lcu):
        from_zenith = float(lcu.get_hardie_airmass(zenith_angle=35.0))
        from_altitude = float(lcu.get_hardie_airmass(altitude=55.0))
        assert from_zenith == pytest.approx(from_altitude)

    def test_correction_reduces_the_plane_parallel_secant(self, lcu):
        for z in (30.0, 45.0, 60.0, 70.0):
            hardie = float(lcu.get_hardie_airmass(zenith_angle=z))
            secz = 1.0 / math.cos(math.radians(z))
            assert hardie < secz
            assert hardie == pytest.approx(secz, rel=2e-2)

    def test_accepts_astropy_quantities(self, lcu):
        assert float(lcu.get_hardie_airmass(zenith_angle=45 * u.deg)) == pytest.approx(
            float(lcu.get_hardie_airmass(zenith_angle=45.0))
        )

    def test_result_is_dimensionless(self, lcu):
        assert lcu.get_hardie_airmass(zenith_angle=30.0).unit == u.dimensionless_unscaled

    @pytest.mark.parametrize("kwargs", [{}, {"zenith_angle": 30.0, "altitude": 60.0}])
    def test_exactly_one_angle_must_be_given(self, lcu, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            lcu.get_hardie_airmass(**kwargs)


class TestFlagDecoding:
    def test_binbits_pads_to_the_requested_width(self, lcu):
        bits = lcu.binbits(5, 32)
        assert bits.startswith("0b")
        assert len(bits) == 34
        assert bits[2:].endswith("101")
        assert set(bits[2:-3]) == {"0"}

    @pytest.mark.parametrize(
        "decflag, expected",
        [
            (0, []),
            (1, ["Saturated"]),
            (2, ["LowRN"]),
            (3, ["Saturated", "LowRN"]),
            (2**6, ["NaN"]),
            (2**14, ["CR_DeltaHT"]),
            (2**23, ["NearEdge"]),
            (2**30, ["SrcDetected"]),
        ],
    )
    def test_known_flag_values_decode_to_the_documented_names(self, lcu, decflag, expected):
        assert sorted(lcu.get_flags_keyword(decflag)) == sorted(expected)

    def test_every_flag_value_in_the_example_catalog_decodes(self, lcu, catalog_tables):
        last_cat, _ = catalog_tables
        for value in np.unique(last_cat["FLAGS"]):
            flags = lcu.get_flags_keyword(value)
            assert isinstance(flags, list)
            assert all(isinstance(f, str) for f in flags)


class TestFindSourceInCat:
    def test_finds_a_source_at_its_own_coordinates(self, lcu, single_catalog, catalog_tables):
        last_cat, _ = catalog_tables
        row = 10
        target = SkyCoord(ra=last_cat["RA"][row] * u.deg, dec=last_cat["Dec"][row] * u.deg)
        idx = lcu.find_source_in_cat(target, single_catalog)
        assert idx is not None
        assert int(np.atleast_1d(idx)[0]) == row

    def test_returns_none_when_nothing_is_within_two_arcsec(self, lcu, single_catalog):
        # A position on the opposite side of the sky from the example field.
        target = SkyCoord(ra=300.0 * u.deg, dec=-45.0 * u.deg)
        assert lcu.find_source_in_cat(target, single_catalog) is None
