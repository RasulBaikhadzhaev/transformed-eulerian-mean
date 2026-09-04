import numpy as np
import pytest
import pandas as pd
import xarray as xr
from metpy.units import units

from pathlib import Path
from transformed_eulerian_mean.utils import (
    nanGradient,
    is_equal_or_shorter_than_month, is_equal_or_shorter_than_day,
    format_seconds, binData, addRatioUnits,
)
from transformed_eulerian_mean.interpolation import alt2press, press2alt
from transformed_eulerian_mean.file_io import extractTimeFromFileNames
from transformed_eulerian_mean.constants import P0, H

addRatioUnits()


# ── nanGradient ───────────────────────────────────────────────────────────────

def test_nangradient_clean_matches_numpy():
    """No NaN: result must equal np.gradient."""
    y = np.array([1.0, 4.0, 9.0, 16.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = nanGradient(y, x, axis=0)
    np.testing.assert_allclose(result, np.gradient(y, x), rtol=1e-12)


def test_nangradient_interior_nan_finite_elsewhere():
    """Interior NaN: gradient at surrounding valid points should be finite."""
    y = np.array([1.0, np.nan, 9.0, 16.0])
    x = np.array([1.0, 2.0,   3.0,  4.0])
    result = nanGradient(y, x, axis=0)
    assert np.isfinite(result[0])
    assert np.isfinite(result[2])
    assert np.isfinite(result[3])


def test_nangradient_all_nan_returns_nan():
    y = np.full(5, np.nan)
    x = np.arange(5, dtype=float)
    result = nanGradient(y, x, axis=0)
    assert np.all(np.isnan(result))


def test_nangradient_single_valid_returns_nan():
    """Only one valid point → gradient undefined everywhere."""
    y = np.array([np.nan, 5.0, np.nan, np.nan])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    result = nanGradient(y, x, axis=0)
    assert np.all(np.isnan(result))


def test_nangradient_units_y_and_x():
    """y in K, x in m → output has units K/m."""
    y = np.array([300.0, 302.0, 304.0]) * units('K')
    x = np.array([0.0, 1000.0, 2000.0]) * units('m')
    result = nanGradient(y, x, axis=0)
    assert hasattr(result, 'units')
    np.testing.assert_allclose(result.to('K/m').magnitude, 0.002, rtol=1e-6)


def test_nangradient_units_y_only():
    """y has units, x bare → output has same units as y."""
    y = np.array([1.0, 2.0, 3.0]) * units('m/s')
    x = np.array([0.0, 1.0, 2.0])
    result = nanGradient(y, x, axis=0)
    assert hasattr(result, 'units')
    assert str(result.units) == str(y.units)


def test_nangradient_units_x_only():
    """x has units, y bare → output has units 1/x_units."""
    y = np.array([0.0, 1.0, 4.0])
    x = np.array([0.0, 1.0, 2.0]) * units('s')
    result = nanGradient(y, x, axis=0)
    assert hasattr(result, 'units')


def test_nangradient_no_units():
    """No units on either y or x → plain numpy array."""
    y = np.array([0.0, 1.0, 4.0, 9.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    result = nanGradient(y, x, axis=0)
    assert not hasattr(result, 'units')
    assert isinstance(result, np.ndarray)


def test_nangradient_axis1():
    """Gradient along axis=1 for a 2D array."""
    y = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]])
    x = np.array([0.0, 1.0, 2.0])
    result = nanGradient(y, x, axis=1)
    np.testing.assert_allclose(result, np.gradient(y, x, axis=1), rtol=1e-12)


# ── alt2press / press2alt ─────────────────────────────────────────────────────

def test_alt2press_at_zero_altitude():
    """At z=0 km, pressure must equal P0."""
    p = alt2press(0.0 * units.km)
    np.testing.assert_allclose(p.to('hPa').magnitude, P0.to('hPa').magnitude, rtol=1e-10)


def test_press2alt_at_p0():
    """At p=P0, log-pressure altitude must be 0 km."""
    z = press2alt(P0)
    np.testing.assert_allclose(z.to('km').magnitude, 0.0, atol=1e-10)


def test_alt2press_press2alt_round_trip():
    """press2alt(alt2press(z)) must recover the original altitude."""
    z = np.array([5.0, 15.0, 25.0, 35.0]) * units.km
    p = alt2press(z)
    z_recovered = press2alt(p)
    np.testing.assert_allclose(z_recovered.to('km').magnitude, z.magnitude, rtol=1e-10)


def test_alt2press_monotonically_decreasing():
    """Pressure must decrease monotonically with altitude."""
    z = np.linspace(0, 50, 20) * units.km
    p = alt2press(z)
    assert np.all(np.diff(p.magnitude) < 0)


# ── is_equal_or_shorter_than_month ───────────────────────────────────────────

def test_month_check_6h_is_true():
    assert is_equal_or_shorter_than_month('6H') is True


def test_month_check_daily_is_true():
    assert is_equal_or_shorter_than_month('D') is True


def test_month_check_weekly_is_true():
    assert is_equal_or_shorter_than_month('W') is True


def test_month_check_monthly_code_is_true():
    assert is_equal_or_shorter_than_month('MS') is True


def test_month_check_quarterly_is_false():
    assert is_equal_or_shorter_than_month('Q') is False


def test_month_check_dateoffset_monthly_is_true():
    assert is_equal_or_shorter_than_month(pd.DateOffset(months=1)) is True


def test_month_check_timedelta_15days_is_true():
    assert is_equal_or_shorter_than_month(pd.Timedelta('15 days')) is True


# ── is_equal_or_shorter_than_day ─────────────────────────────────────────────

def test_day_check_6h_is_true():
    assert is_equal_or_shorter_than_day('6H') is True


def test_day_check_daily_is_true():
    assert is_equal_or_shorter_than_day('D') is True


def test_day_check_weekly_is_false():
    assert is_equal_or_shorter_than_day('W') is False


def test_day_check_monthly_code_is_false():
    assert is_equal_or_shorter_than_day('MS') is False


def test_day_check_quarterly_is_false():
    assert is_equal_or_shorter_than_day('Q') is False


def test_day_check_timedelta_12h_is_true():
    assert is_equal_or_shorter_than_day(pd.Timedelta('12 hours')) is True


def test_day_check_timedelta_2days_is_false():
    assert is_equal_or_shorter_than_day(pd.Timedelta('2 days')) is False


# ── format_seconds ────────────────────────────────────────────────────────────

def test_format_seconds_full():
    assert format_seconds(3661) == "1h 1m 1s"


def test_format_seconds_under_minute():
    assert format_seconds(59) == "59s"


def test_format_seconds_exact_hour():
    assert format_seconds(3600) == "1h 0m 0s"


def test_format_seconds_zero():
    assert format_seconds(0) == "0s"


def test_format_seconds_negative():
    assert format_seconds(-1) == "calculating..."


# ── binData ───────────────────────────────────────────────────────────────────

def _make_ds(nlat, nlon, nalt, fill=1.0):
    data = np.full((nalt, nlat, nlon), fill)
    return xr.Dataset(
        {'X': (['alt', 'lat', 'lon'], data, {'units': 'K'})},
        coords={
            'alt': np.linspace(5, 50, nalt),
            'lat': np.linspace(-80, 80, nlat),
            'lon': np.linspace(0, 350, nlon),
        },
    )


def test_bindata_halves_resolution():
    """Binning by 2 should halve lat and lon dimension sizes."""
    ds = _make_ds(nlat=8, nlon=16, nalt=5)
    result = binData(ds, 2, 2)
    assert result.sizes['lat'] == 4
    assert result.sizes['lon'] == 8


def test_bindata_uniform_field_unchanged():
    """Averaging a constant field must leave its value unchanged."""
    ds = _make_ds(nlat=8, nlon=16, nalt=5, fill=42.0)
    result = binData(ds, 2, 2)
    np.testing.assert_allclose(result['X'].values, 42.0)


def test_bindata_identity_with_factor_one():
    """binData with factor=1 is a no-op."""
    rng = np.random.default_rng(7)
    data = rng.random((3, 4, 8))
    ds = xr.Dataset(
        {'X': (['alt', 'lat', 'lon'], data, {'units': 'K'})},
        coords={
            'alt': np.linspace(5, 50, 3),
            'lat': np.linspace(-80, 80, 4),
            'lon': np.linspace(0, 350, 8),
        },
    )
    result = binData(ds, 1, 1)
    np.testing.assert_array_equal(result['X'].values, data)


# ── nanGradient extra edge cases ──────────────────────────────────────────────

def test_nangradient_leading_nan_finite_interior():
    """NaN at the start: interior gradient at valid points should be finite."""
    y = np.array([np.nan, 4.0, 9.0, 16.0])
    x = np.array([1.0,    2.0, 3.0,  4.0])
    result = nanGradient(y, x, axis=0)
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])
    assert np.isfinite(result[3])


def test_nangradient_trailing_nan_finite_interior():
    """NaN at the end: interior gradient at valid points should be finite."""
    y = np.array([1.0, 4.0, 9.0, np.nan])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = nanGradient(y, x, axis=0)
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])


def test_nangradient_two_valid_points_both_finite():
    """Two valid points (minimum to compute gradient): both should be finite."""
    y = np.array([np.nan, 3.0, 7.0, np.nan])
    x = np.array([0.0,    1.0, 2.0, 3.0])
    result = nanGradient(y, x, axis=0)
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])
    assert np.isnan(result[0])
    assert np.isnan(result[3])


def test_nangradient_2d_scattered_nan():
    """2D array with NaN scattered in one row: other rows unchanged."""
    y = np.array([
        [1.0, 4.0, 9.0],
        [np.nan, np.nan, np.nan],
        [2.0, 6.0, 12.0],
    ])
    x = np.array([1.0, 2.0, 3.0])
    result = nanGradient(y, x, axis=1)
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isnan(result[1]))
    assert np.all(np.isfinite(result[2]))


# ── addRatioUnits ─────────────────────────────────────────────────────────────

def test_addRatioUnits_ppmv_is_accessible():
    """After addRatioUnits(), 'ppmv' must be a valid unit."""
    addRatioUnits()
    q = 1.0 * units('ppmv')
    assert str(q.units) == 'ppmv'


def test_addRatioUnits_ppbv_is_1000th_of_ppmv():
    """1000 ppbv == 1 ppmv."""
    addRatioUnits()
    q_ppbv = 1000.0 * units('ppbv')
    q_ppmv = q_ppbv.to('ppmv')
    np.testing.assert_allclose(q_ppmv.magnitude, 1.0, rtol=1e-10)


def test_addRatioUnits_ppmm_is_accessible():
    """After addRatioUnits(), 'ppmm' must be a valid unit."""
    addRatioUnits()
    q = 1.0 * units('ppmm')
    assert str(q.units) == 'ppmm'


def test_addRatioUnits_idempotent():
    """Calling addRatioUnits() twice should not raise."""
    addRatioUnits()
    addRatioUnits()  # must not crash


# ── extractTimeFromFileNames ──────────────────────────────────────────────────

def _fake_paths(names):
    """Create Path objects from plain name strings (no real files needed)."""
    return [Path(n) for n in names]


def test_extracttime_yyyy_mm_dd_suffix_wildcard():
    """'*YYYYMMDD???' extracts year, month, day from the filename tail."""
    paths = _fake_paths(['model_19990315.nc', 'model_20031201.nc'])
    result = extractTimeFromFileNames(paths, '*YYYYMMDD???')
    assert result[0].year == 1999
    assert result[0].month == 3
    assert result[0].day == 15
    assert result[1].year == 2003
    assert result[1].month == 12
    assert result[1].day == 1


def test_extracttime_yy_century_inference_pre50():
    """YY < 50 should be treated as 20YY."""
    paths = _fake_paths(['data_230115.nc'])   # YYMMDD -> 2023-01-15
    result = extractTimeFromFileNames(paths, '*YYMMDD???')
    assert result[0].year == 2023


def test_extracttime_yy_century_inference_post50():
    """YY >= 50 should be treated as 19YY."""
    paths = _fake_paths(['data_780115.nc'])   # YYMMDD -> 1978-01-15
    result = extractTimeFromFileNames(paths, '*YYMMDD???')
    assert result[0].year == 1978


def test_extracttime_prefix_position():
    """Pattern without leading '*': year read from fixed offset from filename start."""
    paths = _fake_paths(['19990315.nc'])
    result = extractTimeFromFileNames(paths, 'YYYYMMDD')
    assert result[0].year == 1999
    assert result[0].month == 3
    assert result[0].day == 15
