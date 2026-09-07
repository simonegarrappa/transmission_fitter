"""Tests for the SMARTS-based atmospheric transmission components.

The assertions are physical rather than numerical wherever possible: every
component must return a transmission in [0, 1] on the 401-point grid, must
attenuate more at larger airmass / larger column, and must put its absorption
features at the right wavelengths (ozone in the UV, the H2O band near 940 nm,
the O2 A-band at 760 nm).
"""

import numpy as np
import pytest

from transmission_fitter.abscalutils import make_wvl_array
from transmission_fitter.atmospheric_models import (
    NLOSCHMIDT,
    Aerosol_Transmission,
    Atmospheric_Component,
    Ozone_Transmission,
    Rayleigh_Transmission,
    UMGTransmittance,
    WaterTransmittance,
)

WVL = make_wvl_array()
N_WVL = WVL.size

ALL_CONSTITUENTS = [
    "rayleigh", "aerosol", "o3", "h2o", "o2", "ch4", "co", "n2o", "co2", "n2",
    "hno3", "no2", "no", "so2", "nh3", "no3", "bro", "ch2o", "hno2", "clno",
    "ozone", "water",
]


def at(transmission, wavelength_nm):
    """Transmission value at a wavelength that lies on the model grid."""
    return float(transmission[np.argmin(np.abs(WVL - wavelength_nm))])


class TestAirmassFromSMARTS:
    @pytest.mark.parametrize("constituent", ALL_CONSTITUENTS)
    def test_airmass_is_unity_at_zenith(self, constituent):
        comp = Atmospheric_Component(0.0)
        assert comp.Airmass_from_SMARTS(0.0, constituent) == pytest.approx(1.0, abs=1e-6)

    def test_airmass_approximates_secant_z_at_moderate_angles(self):
        comp = Atmospheric_Component(60.0)
        # SMARTS airmass stays slightly below sec(z) because of refraction.
        am = comp.Airmass_from_SMARTS(60.0, "rayleigh")
        assert am == pytest.approx(2.0, rel=5e-3)
        assert am < 1.0 / np.cos(np.radians(60.0)) + 1e-9

    def test_airmass_increases_monotonically_with_zenith_angle(self):
        comp = Atmospheric_Component(0.0)
        airmasses = [comp.Airmass_from_SMARTS(z, "rayleigh") for z in range(0, 85, 5)]
        assert np.all(np.diff(airmasses) > 0)

    def test_aliased_constituents_share_coefficients(self):
        comp = Atmospheric_Component(45.0)
        assert comp.Airmass_from_SMARTS(45.0, "ozone") == comp.Airmass_from_SMARTS(45.0, "o3")
        assert comp.Airmass_from_SMARTS(45.0, "water") == comp.Airmass_from_SMARTS(45.0, "h2o")
        assert comp.Airmass_from_SMARTS(45.0, "no3") == comp.Airmass_from_SMARTS(45.0, "no2")

    def test_constituent_name_is_case_insensitive(self):
        comp = Atmospheric_Component(45.0)
        assert comp.Airmass_from_SMARTS(45.0, "O3") == comp.Airmass_from_SMARTS(45.0, "o3")

    def test_unknown_constituent_raises(self):
        comp = Atmospheric_Component(0.0)
        with pytest.raises(ValueError, match="not a valid constituent"):
            comp.Airmass_from_SMARTS(0.0, "unobtanium")


class TestRayleighTransmission:
    def test_shape_and_bounds(self):
        transm = Rayleigh_Transmission(30.0, 965.0).make_transmission()
        assert transm.shape == (N_WVL,)
        assert np.all((transm >= 0.0) & (transm <= 1.0))

    def test_scattering_is_much_stronger_in_the_blue(self):
        transm = Rayleigh_Transmission(30.0, 965.0).make_transmission()
        assert at(transm, 400.0) < 0.75
        assert at(transm, 800.0) > 0.95
        # Monotonic increase with wavelength across the whole grid.
        assert np.all(np.diff(transm) > 0)

    def test_higher_pressure_attenuates_more(self):
        low = Rayleigh_Transmission(30.0, 800.0).make_transmission()
        high = Rayleigh_Transmission(30.0, 1013.25).make_transmission()
        assert np.all(high <= low + 1e-12)
        assert at(high, 400.0) < at(low, 400.0)

    def test_larger_zenith_angle_attenuates_more(self):
        near_zenith = Rayleigh_Transmission(0.0, 965.0).make_transmission()
        low_altitude = Rayleigh_Transmission(70.0, 965.0).make_transmission()
        assert np.all(low_altitude <= near_zenith + 1e-12)


class TestAerosolTransmission:
    def test_shape_and_bounds(self):
        transm = Aerosol_Transmission(30.0, aod_in=0.084, alpha_in=0.6).make_transmission()
        assert transm.shape == (N_WVL,)
        assert np.all((transm >= 0.0) & (transm <= 1.0))

    def test_zero_optical_depth_is_transparent(self):
        transm = Aerosol_Transmission(30.0, aod_in=0.0, alpha_in=0.6).make_transmission()
        assert np.allclose(transm, 1.0)

    def test_higher_aod_attenuates_more(self):
        thin = Aerosol_Transmission(30.0, aod_in=0.084, alpha_in=0.6).make_transmission()
        thick = Aerosol_Transmission(30.0, aod_in=0.54, alpha_in=0.6).make_transmission()
        assert np.all(thick < thin)

    def test_larger_angstrom_exponent_steepens_the_wavelength_dependence(self):
        flat = Aerosol_Transmission(30.0, aod_in=0.084, alpha_in=0.2).make_transmission()
        steep = Aerosol_Transmission(30.0, aod_in=0.084, alpha_in=1.5).make_transmission()
        blue_to_red_flat = at(flat, 400.0) / at(flat, 900.0)
        blue_to_red_steep = at(steep, 400.0) / at(steep, 900.0)
        assert blue_to_red_steep < blue_to_red_flat < 1.0


class TestOzoneTransmission:
    def test_shape_and_bounds(self):
        transm = Ozone_Transmission(30.0, uo_=300.0).make_transmission()
        assert transm.shape == (N_WVL,)
        assert np.all((transm >= 0.0) & (transm <= 1.0))

    def test_zero_column_is_transparent(self):
        transm = Ozone_Transmission(30.0, uo_=0.0).make_transmission()
        assert np.allclose(transm, 1.0)

    def test_absorption_is_confined_to_the_uv_huggins_band(self):
        transm = Ozone_Transmission(30.0, uo_=300.0).make_transmission()
        assert at(transm, 310.0) < 0.5  # strong Huggins-band absorption
        assert at(transm, 700.0) > 0.98  # essentially transparent in the red

    def test_larger_column_attenuates_more(self):
        thin = Ozone_Transmission(30.0, uo_=200.0).make_transmission()
        thick = Ozone_Transmission(30.0, uo_=400.0).make_transmission()
        assert at(thick, 310.0) < at(thin, 310.0)
        assert np.all(thick <= thin + 1e-12)

    @pytest.mark.parametrize(
        "wavelength_nm, expected",
        [(300.0, 0.0577013), (310.0, 0.5022164), (320.0, 0.8114372), (600.0, 0.9587160)],
    )
    def test_reference_values_for_300_dobson_at_zenith(self, wavelength_nm, expected):
        """Absolute anchor: pins the Dobson-to-atm-cm conversion and the

        SMARTS cross-section interpolation, which relative comparisons cannot
        detect on their own.
        """
        transm = Ozone_Transmission(0.0, uo_=300.0).make_transmission()
        assert at(transm, wavelength_nm) == pytest.approx(expected, rel=1e-5)


class TestWaterTransmittance:
    def test_shape_and_bounds(self):
        transm = WaterTransmittance(30.0, pw_=1.4, p_=965.0).make_transmission()
        assert transm.shape == (N_WVL,)
        assert np.all((transm >= 0.0) & (transm <= 1.0))

    def test_strongest_band_is_the_940nm_water_band(self):
        transm = WaterTransmittance(30.0, pw_=1.4, p_=965.0).make_transmission()
        assert 900.0 <= WVL[np.argmin(transm)] <= 970.0
        # The blue continuum is untouched by water vapour.
        assert at(transm, 500.0) > 0.999

    def test_more_precipitable_water_attenuates_more(self):
        dry = WaterTransmittance(30.0, pw_=0.5, p_=965.0).make_transmission()
        humid = WaterTransmittance(30.0, pw_=5.0, p_=965.0).make_transmission()
        assert humid.min() < dry.min()
        assert np.all(humid <= dry + 1e-9)

    def test_larger_zenith_angle_attenuates_more(self):
        overhead = WaterTransmittance(0.0, pw_=1.4, p_=965.0).make_transmission()
        low = WaterTransmittance(70.0, pw_=1.4, p_=965.0).make_transmission()
        assert low.min() < overhead.min()

    def test_reference_value_in_the_940nm_band(self):
        """Absolute anchor for the Bw/Bm/Bmw/Bp correction chain."""
        transm = WaterTransmittance(0.0, pw_=1.4, p_=1013.25).make_transmission()
        assert at(transm, 940.0) == pytest.approx(0.6565820, rel=1e-5)


class TestUMGTransmittance:
    def test_shape_and_bounds(self):
        transm = UMGTransmittance(30.0, tair=15.0, p_=965.0).make_transmission()
        assert transm.shape == (N_WVL,)
        assert np.all((transm >= 0.0) & (transm <= 1.0))

    def test_strongest_feature_is_the_oxygen_a_band(self):
        transm = UMGTransmittance(30.0, tair=15.0, p_=965.0).make_transmission()
        assert WVL[np.argmin(transm)] == pytest.approx(760.0, abs=4.0)

    def test_trace_gases_only_add_absorption(self):
        without = UMGTransmittance(
            30.0, tair=15.0, p_=965.0, with_trace_gases=False
        ).make_transmission()
        with_trace = UMGTransmittance(
            30.0, tair=15.0, p_=965.0, with_trace_gases=True
        ).make_transmission()
        assert np.all(with_trace <= without + 1e-12)
        assert np.any(with_trace < without)

    def test_higher_pressure_attenuates_more(self):
        low = UMGTransmittance(30.0, tair=15.0, p_=700.0).make_transmission()
        high = UMGTransmittance(30.0, tair=15.0, p_=1013.25).make_transmission()
        assert high.min() < low.min()

    def test_reference_value_in_the_oxygen_a_band(self):
        """Absolute anchor for the gas abundance formulae at 1 atm, zenith."""
        transm = UMGTransmittance(0.0, tair=15.0, p_=1013.25).make_transmission()
        assert at(transm, 760.0) == pytest.approx(0.3648276, rel=1e-5)

    def test_more_co2_attenuates_more_in_the_co2_bands(self):
        low = UMGTransmittance(30.0, tair=15.0, p_=965.0, co2_ppm=280.0).make_transmission()
        high = UMGTransmittance(30.0, tair=15.0, p_=965.0, co2_ppm=800.0).make_transmission()
        assert np.all(high <= low + 1e-12)
        assert np.any(high < low)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tair": np.array([10.0, 15.0])},
            {"tair": 15.0, "co2_ppm": np.array([395.0, 400.0])},
        ],
    )
    def test_non_scalar_temperature_or_co2_is_rejected(self, kwargs):
        kwargs.setdefault("co2_ppm", 395.0)
        with pytest.raises(AssertionError):
            UMGTransmittance(30.0, p_=965.0, **kwargs)

    def test_read_gas_interpolates_onto_the_model_grid(self):
        umg = UMGTransmittance(30.0, tair=15.0, p_=965.0)
        o2 = umg.read_gas("O2")
        assert np.asarray(o2).shape == (N_WVL,)
        assert np.all(np.asarray(o2) >= 0.0)

        # Files with several columns come back as a list of per-column arrays.
        no2 = umg.read_gas("NO2")
        assert isinstance(no2, list) and len(no2) == 2
        assert all(np.asarray(col).shape == (N_WVL,) for col in no2)


def test_loschmidt_constant_value():
    """The number density used to convert cross-sections to optical depths."""
    assert NLOSCHMIDT == pytest.approx(2.6867811e19)
