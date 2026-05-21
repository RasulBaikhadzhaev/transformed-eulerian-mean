"""
Unit tests for tem_pkg.interpolation.

Covers all previously uncovered branches:
  - alt2press / press2alt (round-trip)
  - interpolateToLogPressure: 'pressure', 'log-pressure', 'skip', error,
                              saveInterpolatedZonalMeanVars, saveZonalMeanVars
  - interpolateToTheta: 'other', 'theta', error
  - interpolateToThetaAndCombineData: tracer 'other'/'theta', met 'other'/'theta', errors
  - interpolateToPressureAndCombineData: tracer 'other'/'log-pressure', met 'pressure'/'log-pressure', errors
"""

import numpy as np
import pytest
import xarray as xr
from metpy.units import units

from tem_pkg.utils import addRatioUnits
from tem_pkg.interpolation import (
    alt2press,
    press2alt,
    interpolateToLogPressure,
    interpolateToTheta,
    interpolateToThetaAndCombineData,
    interpolateToPressureAndCombineData,
)

addRatioUnits()

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

_NLAT, _NLON = 4, 6
_LATS = np.linspace(-60.0, 60.0, _NLAT)
_LONS = np.linspace(0.0, 300.0, _NLON)

# Pressure levels (hPa) and their log-pressure altitude equivalents (~0, 16, 32 km)
_PRESS_HPA = np.array([1000.0, 100.0, 10.0])

# Log-pressure altitude coordinate (km)
_ALT_KM = np.array([0.0, 16.1, 32.2])

# Target interpolation levels (km) — passed as Python list to avoid
# numpy array != 'skip' ambiguity in the production code (line 67/85).
_TARGET_LP = [5.0, 15.0, 25.0]

# Theta levels (K) and interpolation targets
_THETA_SRC = np.array([300.0, 400.0, 500.0, 600.0])
_TARGET_THETA = np.array([350.0, 450.0, 550.0])

_NPRESS = len(_PRESS_HPA)
_NALT   = len(_ALT_KM)
_NTHETA = len(_THETA_SRC)

_RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _press_ds(extra_vars=None):
    """Dataset with 'press' vertical dim (numeric hPa), lat, lon."""
    shape = (_NPRESS, _NLAT, _NLON)
    ds = xr.Dataset(
        {'U': (['press', 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'}),
         'V': (['press', 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'})},
        coords={'press': _PRESS_HPA, 'lat': _LATS, 'lon': _LONS},
    )
    if extra_vars:
        ds = ds.assign(extra_vars)
    return ds


def _alt_ds(vert_name='myalt', vars_=('U', 'V')):
    """Dataset already in log-pressure alt coordinates."""
    shape = (_NALT, _NLAT, _NLON)
    return xr.Dataset(
        {v: ([vert_name, 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'})
         for v in vars_},
        coords={vert_name: _ALT_KM, 'lat': _LATS, 'lon': _LONS},
    )


def _other_logpress_ds():
    """Dataset with alt coord + 3-D pressure variable (vertDimType='other' source)."""
    nalt = 5
    alt = np.linspace(5.0, 45.0, nalt)
    press_3d = (1000.0 * np.exp(-alt[:, None, None] / 7.0) *
                np.ones((nalt, _NLAT, _NLON))).astype('f4')
    shape = (nalt, _NLAT, _NLON)
    return xr.Dataset(
        {'U':     (['alt', 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'}),
         'PRESS': (['alt', 'lat', 'lon'], press_3d, {'units': 'hPa'})},
        coords={'alt': alt, 'lat': _LATS, 'lon': _LONS},
    )


def _theta_coord_ds(vert_name='theta', tracer_var='BA', include_theta_dot=True):
    """Dataset already on isentropic (theta) coordinate."""
    shape = (_NTHETA, _NLAT, _NLON)
    data = {tracer_var: ([vert_name, 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'})}
    if include_theta_dot:
        data['THETA_DOT'] = ([vert_name, 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'K/s'})
    return xr.Dataset(data, coords={vert_name: _THETA_SRC, 'lat': _LATS, 'lon': _LONS})


def _other_theta_ds(tracer_var='BA'):
    """Dataset with a 3-D THETA field for interpolation to theta levels."""
    nvert = 6
    vert = np.linspace(5.0, 50.0, nvert)
    theta_3d = (300.0 + vert[:, None, None] * 8.0 *
                np.ones((nvert, _NLAT, _NLON))).astype('f4')
    shape = (nvert, _NLAT, _NLON)
    return xr.Dataset(
        {tracer_var: (['vert', 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'}),
         'THETA':    (['vert', 'lat', 'lon'], theta_3d, {'units': 'K'})},
        coords={'vert': vert, 'lat': _LATS, 'lon': _LONS},
    )


def _other_press_tracer_ds():
    """Tracer dataset with a 3-D pressure variable (for 'other' press interp)."""
    nvert = 6
    vert = np.linspace(5.0, 50.0, nvert)
    press_3d = (1000.0 * np.exp(-vert[:, None, None] / 7.0) *
                np.ones((nvert, _NLAT, _NLON))).astype('f4')
    shape = (nvert, _NLAT, _NLON)
    return xr.Dataset(
        {'BA':    (['vert', 'lat', 'lon'], _RNG.standard_normal(shape).astype('f4'), {'units': 'm/s'}),
         'PRESS': (['vert', 'lat', 'lon'], press_3d, {'units': 'hPa'})},
        coords={'vert': vert, 'lat': _LATS, 'lon': _LONS},
    )


# ---------------------------------------------------------------------------
# alt2press / press2alt
# ---------------------------------------------------------------------------

def test_alt2press_press2alt_roundtrip():
    alts = np.array([0.0, 7.0, 14.0, 21.0]) * units.km
    np.testing.assert_allclose(
        press2alt(alt2press(alts)).magnitude,
        alts.magnitude,
        rtol=1e-5,
    )


def test_alt2press_known_value():
    """At alt=0 km pressure should equal P0 (1000 hPa)."""
    result = alt2press(0 * units.km).to('hPa')
    np.testing.assert_allclose(result.magnitude, 1000.0, rtol=1e-6)


def test_press2alt_known_value():
    """At P0 (1000 hPa) altitude should be 0 km."""
    result = press2alt(1000 * units.hPa)
    np.testing.assert_allclose(result.magnitude, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# interpolateToLogPressure — 'pressure' branch (lines 62–68)
# ---------------------------------------------------------------------------

def test_interp_to_logpress_pressure_branch_coords():
    ds = _press_ds()
    result = interpolateToLogPressure(
        ds, ['U', 'V'], 'pressure', _TARGET_LP, 'press', 'lat', 'lon')
    assert 'alt' in result.dims
    np.testing.assert_allclose(result['alt'].values, _TARGET_LP, rtol=1e-5)


def test_interp_to_logpress_pressure_branch_skip():
    """targetLevels='skip' keeps original converted alt coordinate without re-gridding."""
    ds = _press_ds()
    result = interpolateToLogPressure(
        ds, ['U', 'V'], 'pressure', 'skip', 'press', 'lat', 'lon')
    assert 'alt' in result.dims
    assert len(result['alt']) == _NPRESS


# ---------------------------------------------------------------------------
# interpolateToLogPressure — 'log-pressure' branch (lines 83–86)
# ---------------------------------------------------------------------------

def test_interp_to_logpress_logpress_branch_with_target():
    ds = _alt_ds(vert_name='myalt')
    result = interpolateToLogPressure(
        ds, ['U', 'V'], 'log-pressure', _TARGET_LP, 'myalt', 'lat', 'lon')
    assert 'alt' in result.dims
    np.testing.assert_allclose(result['alt'].values, _TARGET_LP, rtol=1e-5)


def test_interp_to_logpress_logpress_branch_skip():
    ds = _alt_ds(vert_name='myalt')
    result = interpolateToLogPressure(
        ds, ['U', 'V'], 'log-pressure', 'skip', 'myalt', 'lat', 'lon')
    assert 'alt' in result.dims
    assert len(result['alt']) == _NALT


# ---------------------------------------------------------------------------
# interpolateToLogPressure — invalid vertDimType (line 89–91)
# ---------------------------------------------------------------------------

def test_interp_to_logpress_invalid_type_exits():
    ds = _alt_ds()
    with pytest.raises(SystemExit):
        interpolateToLogPressure(
            ds, ['U'], 'bad_type', _TARGET_LP, 'myalt', 'lat', 'lon')


# ---------------------------------------------------------------------------
# interpolateToLogPressure — saveInterpolatedZonalMeanVars / saveZonalMeanVars (94, 97)
# ---------------------------------------------------------------------------

def test_interp_to_logpress_save_interpolated_zonal_mean_var():
    """Variable listed in saveInterpolatedZonalMeanVars should have no 'lon' dim."""
    ds = _other_logpress_ds()
    result = interpolateToLogPressure(
        ds, ['U'], 'other', _TARGET_LP, 'alt', 'lat', 'lon',
        pressureVarName='PRESS',
        saveInterpolatedZonalMeanVars=['U'],
    )
    assert 'U' in result
    assert 'lon' not in result['U'].dims


def test_interp_to_logpress_save_zonal_mean_var():
    """Variable listed in saveZonalMeanVars should have no 'lon' dim in result."""
    ds = _other_logpress_ds()
    result = interpolateToLogPressure(
        ds, ['U'], 'other', _TARGET_LP, 'alt', 'lat', 'lon',
        pressureVarName='PRESS',
        saveZonalMeanVars=['U'],
    )
    assert 'U' in result
    assert 'lon' not in result['U'].dims


# ---------------------------------------------------------------------------
# interpolateToTheta — 'other' branch (lines 106–116)
# ---------------------------------------------------------------------------

def test_interp_to_theta_other_branch():
    ds = _other_theta_ds(tracer_var='BA')
    cfg = {
        'verticalDimensionType': 'other',
        'thetaName': 'THETA',
        'targetLevels': _TARGET_THETA,
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToTheta(ds, ['BA'], cfg)
    assert 'theta' in result.dims
    assert len(result['theta']) == len(_TARGET_THETA)


# ---------------------------------------------------------------------------
# interpolateToTheta — 'theta' branch (line 119)
# ---------------------------------------------------------------------------

def test_interp_to_theta_theta_branch():
    ds = _theta_coord_ds(vert_name='myth', tracer_var='BA')
    cfg = {
        'verticalDimensionType': 'theta',
        'vertDim': 'myth',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToTheta(ds, ['BA'], cfg)
    assert 'theta' in result.dims
    assert 'lat' in result.dims


# ---------------------------------------------------------------------------
# interpolateToTheta — error (lines 121–124)
# ---------------------------------------------------------------------------

def test_interp_to_theta_invalid_type_exits():
    ds = _theta_coord_ds(tracer_var='BA')
    with pytest.raises(SystemExit):
        interpolateToTheta(ds, ['BA'], {'verticalDimensionType': 'bad'})


# ---------------------------------------------------------------------------
# interpolateToThetaAndCombineData — tracer 'other' + met 'other' (135–139)
# ---------------------------------------------------------------------------

def test_combine_theta_both_other():
    tracer = _other_theta_ds(tracer_var='BA')
    met    = _other_theta_ds(tracer_var='V')
    cfg = {
        'tracerVerticalDimensionType': 'other',
        'tracerNames': ['BA'],
        'tracerThetaName': 'THETA',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'targetLevels': _TARGET_THETA,
        'verticalDimensionType': 'other',
        'thetaName': 'THETA',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToThetaAndCombineData(tracer, met, ['V'], cfg)
    assert 'theta' in result.dims
    assert 'BA' in result


# ---------------------------------------------------------------------------
# interpolateToThetaAndCombineData — tracer 'theta' + met 'theta' (166–167)
# ---------------------------------------------------------------------------

def test_combine_theta_both_theta():
    tracer = _theta_coord_ds(vert_name='myth', tracer_var='BA')
    met    = _theta_coord_ds(vert_name='myth', tracer_var='V', include_theta_dot=False)
    cfg = {
        'tracerVerticalDimensionType': 'theta',
        'tracerVertDim': 'myth',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'theta',
        'vertDim': 'myth',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToThetaAndCombineData(tracer, met, ['V'], cfg)
    assert 'theta' in result.dims


# ---------------------------------------------------------------------------
# interpolateToThetaAndCombineData — error branches (150–152, 169–172)
# ---------------------------------------------------------------------------

def test_combine_theta_invalid_tracer_type_exits():
    tracer = _theta_coord_ds(tracer_var='BA')
    met    = _theta_coord_ds(tracer_var='V')
    cfg = {'tracerVerticalDimensionType': 'bad'}
    with pytest.raises(SystemExit):
        interpolateToThetaAndCombineData(tracer, met, ['V'], cfg)


def test_combine_theta_invalid_met_type_exits():
    tracer = _theta_coord_ds(vert_name='myth', tracer_var='BA')
    met    = _theta_coord_ds(tracer_var='V')
    cfg = {
        'tracerVerticalDimensionType': 'theta',
        'tracerVertDim': 'myth',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'bad',
    }
    with pytest.raises(SystemExit):
        interpolateToThetaAndCombineData(tracer, met, ['V'], cfg)


# ---------------------------------------------------------------------------
# interpolateToPressureAndCombineData — tracer 'other' (192–204)
# ---------------------------------------------------------------------------

def test_combine_press_tracer_other():
    tracer = _other_press_tracer_ds()
    met    = _other_logpress_ds()
    cfg = {
        'tracerVerticalDimensionType': 'other',
        'tracerNames': ['BA'],
        'pressureName': 'PRESS',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'targetLevels': _TARGET_LP,
        'verticalDimensionType': 'other',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToPressureAndCombineData(tracer, met, ['U'], cfg)
    assert 'alt' in result.dims
    assert 'BA' in result


# ---------------------------------------------------------------------------
# interpolateToPressureAndCombineData — tracer 'log-pressure' (206–208)
# ---------------------------------------------------------------------------

def test_combine_press_tracer_logpress():
    tracer = _alt_ds(vert_name='myalt', vars_=('BA',))
    met    = _other_logpress_ds()
    cfg = {
        'tracerVerticalDimensionType': 'log-pressure',
        'tracerVertDim': 'myalt',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'other',
        'targetLevels': _TARGET_LP,
        'pressureName': 'PRESS',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToPressureAndCombineData(tracer, met, ['U'], cfg)
    assert 'alt' in result.dims


# ---------------------------------------------------------------------------
# interpolateToPressureAndCombineData — met 'pressure' (215–219)
# ---------------------------------------------------------------------------

def test_combine_press_met_pressure():
    """Met dataset has a pressure coordinate dim (hPa) → converted to log-pressure alt."""
    tracer = _alt_ds(vert_name='myalt', vars_=('BA',))
    met    = _press_ds()
    cfg = {
        'tracerVerticalDimensionType': 'log-pressure',
        'tracerVertDim': 'myalt',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'pressure',
        'vertDim': 'press',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToPressureAndCombineData(tracer, met, ['U', 'V'], cfg)
    assert 'alt' in result.dims


# ---------------------------------------------------------------------------
# interpolateToPressureAndCombineData — met 'log-pressure' (235–241)
# ---------------------------------------------------------------------------

def test_combine_press_met_logpress():
    tracer = _alt_ds(vert_name='myalt', vars_=('BA',))
    met    = _alt_ds(vert_name='metalt', vars_=('U', 'V'))
    cfg = {
        'tracerVerticalDimensionType': 'log-pressure',
        'tracerVertDim': 'myalt',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'log-pressure',
        'vertDim': 'metalt',
        'latDim': 'lat',
        'lonDim': 'lon',
    }
    result = interpolateToPressureAndCombineData(tracer, met, ['U', 'V'], cfg)
    assert 'alt' in result.dims


# ---------------------------------------------------------------------------
# interpolateToPressureAndCombineData — error branches (210–212, 238–241)
# ---------------------------------------------------------------------------

def test_combine_press_invalid_tracer_type_exits():
    tracer = _alt_ds(vars_=('BA',))
    met    = _alt_ds(vars_=('U',))
    cfg = {'tracerVerticalDimensionType': 'bad'}
    with pytest.raises(SystemExit):
        interpolateToPressureAndCombineData(tracer, met, ['U'], cfg)


def test_combine_press_invalid_met_type_exits():
    tracer = _alt_ds(vert_name='myalt', vars_=('BA',))
    met    = _alt_ds(vars_=('U',))
    cfg = {
        'tracerVerticalDimensionType': 'log-pressure',
        'tracerVertDim': 'myalt',
        'tracerLatDim': 'lat',
        'tracerLonDim': 'lon',
        'verticalDimensionType': 'bad',
    }
    with pytest.raises(SystemExit):
        interpolateToPressureAndCombineData(tracer, met, ['U'], cfg)
