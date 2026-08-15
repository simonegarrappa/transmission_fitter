"""Tests for the wavelength-grid and airmass helpers in ``abscalutils``."""

import math

import numpy as np
import pytest

from transmission_fitter.abscalutils import (
    get_airmass_from_zenith,
    get_zenith_from_airmass,
    make_wvl_array,
    make_wvl_array_Gaia,
)


class TestAirmassZenithConversions:
    def test_airmass_at_zenith_is_one(self):
        assert get_airmass_from_zenith(0.0) == pytest.approx(1.0)

    def test_airmass_is_secant_of_zenith_angle(self):
        assert get_airmass_from_zenith(60.0) == pytest.approx(2.0)
        assert get_airmass_from_zenith(45.0) == pytest.approx(math.sqrt(2.0))

    def test_zenith_from_airmass_inverts_airmass_from_zenith(self):
        for z in (0.0, 15.0, 31.6, 60.0, 75.0):
            assert get_zenith_from_airmass(get_airmass_from_zenith(z)) == pytest.approx(z)

    def test_airmass_increases_with_zenith_angle(self):
        zeniths = np.arange(0.0, 80.0, 5.0)
        airmasses = np.array([get_airmass_from_zenith(z) for z in zeniths])
        assert np.all(np.diff(airmasses) > 0)


class TestMakeWvlArray:
    def test_default_grid_matches_gaia_sampling(self):
        wvl = make_wvl_array()
        assert wvl.shape == (401,)
        assert wvl[0] == pytest.approx(300.0)
        assert wvl[-1] == pytest.approx(1100.0)
        # 2 nm steps, as documented in the module.
        assert np.allclose(np.diff(wvl), 2.0)

    def test_custom_range_and_sampling(self):
        wvl = make_wvl_array(min_int=400.0, max_int=800.0, num=81)
        assert wvl.shape == (81,)
        assert wvl[0] == pytest.approx(400.0)
        assert wvl[-1] == pytest.approx(800.0)
        assert np.allclose(np.diff(wvl), 5.0)


class TestMakeWvlArrayGaia:
    def test_gaia_subgrid_is_within_gaia_coverage(self):
        wvl_gaia, mask_gaia, mask_ir, mask_uv = make_wvl_array_Gaia()
        assert wvl_gaia.min() == pytest.approx(336.0)
        assert wvl_gaia.max() == pytest.approx(1020.0)
        assert wvl_gaia.shape == (343,)
        assert mask_gaia.sum() == wvl_gaia.size

    def test_masks_cover_the_full_grid(self):
        wvl = make_wvl_array()
        _, mask_gaia, mask_ir, mask_uv = make_wvl_array_Gaia()
        assert np.all(mask_gaia | mask_ir | mask_uv)
        assert mask_gaia.shape == wvl.shape

        # The UV/IR masks are inclusive of the Gaia edges, so they overlap the
        # Gaia mask at exactly those two wavelengths and nowhere else.
        assert wvl[mask_gaia & mask_uv].tolist() == [336.0]
        assert wvl[mask_gaia & mask_ir].tolist() == [1020.0]
        assert not np.any(mask_uv & mask_ir)

    def test_uv_and_ir_masks_select_the_expected_side(self):
        wvl = make_wvl_array()
        _, _, mask_ir, mask_uv = make_wvl_array_Gaia()
        assert wvl[mask_uv].max() == pytest.approx(336.0)
        assert wvl[mask_ir].min() == pytest.approx(1020.0)
