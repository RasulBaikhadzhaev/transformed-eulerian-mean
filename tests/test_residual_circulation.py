import multiprocessing

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from metpy.units import units
from transformed_eulerian_mean.residual_circulation import TEMCalcs, mainCalcs, init_worker
from transformed_eulerian_mean.utils import addRatioUnits

addRatioUnits()

EXPECTED_VARS = [
    'THETA', 'U', 'V', 'V_RES_STD', 'W_RES_STD',
    'EPF_vert', 'EPF_lat', 'div_EPF_vert', 'div_EPF_lat',
    'div_EPF', 'MASS_SF_RES_STD',
]
TIMESTAMP = np.datetime64('2000-01-01T00:00:00')


def test_output_variables_present(synthetic_logpress_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    for var in EXPECTED_VARS:
        assert var in dsOut, f"Missing output variable: {var}"


def test_output_values_finite(synthetic_logpress_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    for var in EXPECTED_VARS:
        interior = dsOut[var].values[1:-1, 1:-1]  # skip boundary rows
        assert np.all(np.isfinite(interior)), f"{var} has non-finite interior values"


def test_div_epf_equals_sum_of_components(synthetic_logpress_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    diff = dsOut['div_EPF'].values - (dsOut['div_EPF_lat'].values + dsOut['div_EPF_vert'].values)
    # Results are stored as float32, so tolerance reflects single-precision rounding
    np.testing.assert_allclose(diff, 0.0, atol=1e-6)


def test_zonally_uniform_flow_zero_eddy_fluxes(zonally_uniform_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, zonally_uniform_dataset, TIMESTAMP)
    np.testing.assert_allclose(dsOut['EPF_lat'].values,  0.0, atol=1e-10)
    np.testing.assert_allclose(dsOut['EPF_vert'].values, 0.0, atol=1e-10)
    np.testing.assert_allclose(dsOut['div_EPF'].values,  0.0, atol=1e-10)


def test_zonally_uniform_flow_residual_equals_eulerian(zonally_uniform_dataset, minimal_tem_config):
    # When eddies vanish, vBarStar == vBar
    dsOut = TEMCalcs(minimal_tem_config, zonally_uniform_dataset, TIMESTAMP)
    np.testing.assert_allclose(
        dsOut['V_RES_STD'].values,
        dsOut['V'].values,
        atol=1e-10,
    )


def test_mass_stream_function_top_boundary_zero(synthetic_logpress_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    # Integration starts from the top (array is flipped before cumulative_trapezoid,
    # initial=0 placed at highest altitude), so the top boundary should be zero
    np.testing.assert_allclose(dsOut['MASS_SF_RES_STD'].values[-1, :], 0.0, atol=1e-4)


def test_missing_vertical_wind_no_crash(synthetic_logpress_dataset, minimal_tem_config):
    config = dict(minimal_tem_config)
    config['verticalWindType'] = 'missing'
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'W_RES_STD' in dsOut
    assert np.any(np.isfinite(dsOut['W_RES_STD'].values))


def test_missing_vertical_wind_w_not_saved(synthetic_logpress_dataset, minimal_tem_config):
    config = dict(minimal_tem_config)
    config['verticalWindType'] = 'missing'
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'W' not in dsOut


def test_omega_to_w_conversion_hydrostatic(synthetic_logpress_dataset, minimal_tem_config):
    # With omega=0 everywhere, W should also be zero
    import copy
    import xarray as xr
    ds = synthetic_logpress_dataset.copy(deep=True)
    ds['OMEGA'].values[:] = 0.0
    dsOut = TEMCalcs(minimal_tem_config, ds, TIMESTAMP)
    np.testing.assert_allclose(dsOut['W'].values, 0.0, atol=1e-15)


def test_fourier_wave_range_inclusive(synthetic_logpress_dataset, minimal_tem_config):
    """Sum of individual waves 1,2,3 must equal the range '1-3'."""
    config_range = dict(minimal_tem_config)
    config_range['FourierTransform'] = True
    config_range['verticalWindType'] = 'omega'
    config_range['Waves'] = ['1-3']

    config_individual = dict(config_range)
    config_individual['Waves'] = ['1', '2', '3']

    dsOut_range = TEMCalcs(config_range, synthetic_logpress_dataset, TIMESTAMP)
    dsOut_ind   = TEMCalcs(config_individual, synthetic_logpress_dataset, TIMESTAMP)

    range_vals = dsOut_range['EPFLat_WaveN'].values[:, :, 0]
    ind_sum    = (dsOut_ind['EPFLat_WaveN'].values[:, :, 0] +
                  dsOut_ind['EPFLat_WaveN'].values[:, :, 1] +
                  dsOut_ind['EPFLat_WaveN'].values[:, :, 2])

    np.testing.assert_allclose(range_vals, ind_sum, rtol=1e-5)


def test_zonal_mean_of_primes_is_zero(synthetic_logpress_dataset, minimal_tem_config):
    """vPrime = v - vBar, so mean(vPrime, lon) must be identically zero."""
    config = dict(minimal_tem_config)
    config['saveEddyTerms'] = True
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    # Saved as float32, so tolerance reflects single-precision rounding (~1e-7 relative)
    for var in ['uPrime', 'vPrime', 'thetaPrime']:
        zonal_mean = dsOut[var].values.mean(axis=2)
        np.testing.assert_allclose(zonal_mean, 0.0, atol=1e-6,
                                   err_msg=f"Zonal mean of {var} should be zero")


def test_save_eddy_terms_variables_present(synthetic_logpress_dataset, minimal_tem_config):
    config = dict(minimal_tem_config)
    config['saveEddyTerms'] = True
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)

    zonal_mean_eddy = ['vPrimeUPrimeBar', 'vPrimeThetaPrimeBar', 'wPrimeUPrimeBar']
    lon_eddy        = ['uPrime', 'vPrime', 'thetaPrime', 'vPrimeThetaPrime', 'vPrimeUPrime',
                       'wPrime', 'wPrimeUPrime']
    for var in zonal_mean_eddy + lon_eddy:
        assert var in dsOut, f"Missing eddy term: {var}"


def test_save_eddy_terms_missing_wind_no_w_fields(synthetic_logpress_dataset, minimal_tem_config):
    config = dict(minimal_tem_config)
    config['saveEddyTerms'] = True
    config['verticalWindType'] = 'missing'
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    for var in ['wPrimeUPrimeBar', 'wPrime', 'wPrimeUPrime']:
        assert var not in dsOut, f"{var} should be absent when vertical wind is missing"


def test_residual_differs_from_eulerian_with_eddies(synthetic_logpress_dataset, minimal_tem_config):
    """With zonal variation, V_RES_STD must differ from V."""
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    diff = dsOut['V_RES_STD'].values - dsOut['V'].values
    assert np.any(np.abs(diff) > 1e-12), "V_RES_STD should differ from V when eddies are present"


def test_output_coordinates_match_input(synthetic_logpress_dataset, minimal_tem_config):
    dsOut = TEMCalcs(minimal_tem_config, synthetic_logpress_dataset, TIMESTAMP)
    np.testing.assert_array_equal(dsOut['lat'].values, synthetic_logpress_dataset['lat'].values)
    np.testing.assert_array_equal(dsOut['alt'].values, synthetic_logpress_dataset['alt'].values)
    assert dsOut['time'].values[0] == TIMESTAMP


def test_temperature_type_temperature_no_crash(synthetic_logpress_dataset, minimal_tem_config):
    """When temperatureType='temperature', theta is computed internally from T and p."""
    import xarray as xr
    ds = synthetic_logpress_dataset.copy(deep=True)
    # Rename THETA to TEMP and set realistic temperature values (~240 K at stratosphere)
    ds['TEMP'] = ds['THETA'].copy(deep=True)
    ds['TEMP'].values[:] = 240.0
    ds['TEMP'].attrs['units'] = 'K'

    config = dict(minimal_tem_config)
    config['temperatureName'] = 'TEMP'
    config['temperatureType'] = 'temperature'
    dsOut = TEMCalcs(config, ds, TIMESTAMP)
    assert 'THETA' in dsOut
    assert np.all(np.isfinite(dsOut['THETA'].values[1:-1, 1:-1]))


def test_fourier_all_waves_output_shape(synthetic_logpress_dataset, minimal_tem_config):
    """Waves=['all'] should produce nlon//2 - 1 wavenumbers (wave 0 excluded)."""
    nlon = synthetic_logpress_dataset.sizes['lon']
    config = dict(minimal_tem_config)
    config['FourierTransform'] = True
    config['verticalWindType'] = 'omega'
    config['Waves'] = ['all']
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'EPFLat_WaveN' in dsOut
    # rfft of nlon points gives nlon//2 + 1 bins; wave 0 (DC) excluded → nlon//2 wavenumbers
    assert dsOut['EPFLat_WaveN'].shape[2] == nlon // 2


def test_fourier_open_end_range_no_crash(synthetic_logpress_dataset, minimal_tem_config):
    """Waves=['18-end'] should not crash and should produce a single 2D wavenumber field."""
    config = dict(minimal_tem_config)
    config['FourierTransform'] = True
    config['verticalWindType'] = 'omega'
    config['Waves'] = ['18-end']
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'EPFLat_WaveN' in dsOut
    assert dsOut['EPFLat_WaveN'].shape[2] == 1


def test_vertical_wind_type_w_no_crash(synthetic_logpress_dataset, minimal_tem_config):
    """verticalWindType='W' reads vertical wind directly in m/s without omega conversion."""
    import xarray as xr
    ds = synthetic_logpress_dataset.copy(deep=True)
    # Rename OMEGA to W and mark it as m/s so the 'W' branch is exercised
    ds['W'] = ds['OMEGA'].copy(deep=True)
    ds['W'].attrs['units'] = 'm/s'

    config = dict(minimal_tem_config)
    config['verticalWindName'] = 'W'
    config['verticalWindType'] = 'W'
    dsOut = TEMCalcs(config, ds, TIMESTAMP)
    for var in ['V_RES_STD', 'W_RES_STD', 'EPF_lat', 'EPF_vert', 'div_EPF']:
        assert var in dsOut, f"Missing variable: {var}"
    interior = dsOut['W_RES_STD'].values[1:-1, 1:-1]
    assert np.any(np.isfinite(interior))


def test_vertical_wind_type_w_omega_zero_equal(synthetic_logpress_dataset, minimal_tem_config):
    """With OMEGA=0, 'omega' branch gives w=0; 'W' branch with w=0 must give same wBarStar."""
    import xarray as xr
    ds = synthetic_logpress_dataset.copy(deep=True)
    ds['OMEGA'].values[:] = 0.0

    ds['W'] = ds['OMEGA'].copy(deep=True)
    ds['W'].attrs['units'] = 'm/s'

    config_omega = dict(minimal_tem_config)
    config_omega['verticalWindType'] = 'omega'

    config_w = dict(minimal_tem_config)
    config_w['verticalWindName'] = 'W'
    config_w['verticalWindType'] = 'W'

    dsOut_omega = TEMCalcs(config_omega, ds, TIMESTAMP)
    dsOut_w     = TEMCalcs(config_w, ds, TIMESTAMP)

    np.testing.assert_allclose(
        dsOut_omega['W_RES_STD'].values,
        dsOut_w    ['W_RES_STD'].values,
        atol=1e-12,
    )


def test_mass_stream_function_antisymmetry(minimal_tem_config):
    """For v = A*sin(lat) with no zonal variation, massSF should be antisymmetric about the equator.

    vPrime = 0 so the TEM correction vanishes and vBarStar = vBar = A*sin(lat).
    MSF = -cos(phi)*sin(phi)*f(z) which satisfies MSF(-phi) = -MSF(phi).
    """
    nlat, nlon, nalt = 9, 36, 10
    # Symmetric latitude grid around equator
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)

    # v = A * sin(lat): antisymmetric about equator, no zonal variation → vPrime = 0
    V = 2.0 * np.sin(np.pi * lats[None, :, None] / 180) * np.ones((nalt, nlat, nlon))
    U = np.zeros((nalt, nlat, nlon))
    # THETA depends only on altitude → DThetaBarDZ ≠ 0, thetaPrime = 0
    THETA = (300 + alts[:, None, None] * 3.0) * np.ones((nalt, nlat, nlon))
    OMEGA = np.zeros((nalt, nlat, nlon))

    import xarray as xr
    ds = xr.Dataset(
        {
            'U':     (['alt', 'lat', 'lon'], U,     {'units': 'm/s'}),
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    dsOut = TEMCalcs(minimal_tem_config, ds, TIMESTAMP)
    sf = dsOut['MASS_SF_RES_STD'].values  # shape (nalt, nlat)

    # Northern hemisphere (lat > 0) and southern hemisphere (lat < 0) indices
    # lats is symmetric: lats[i] == -lats[-(i+1)]
    n = nlat // 2
    sf_north = sf[:, n + 1:]   # positive lats
    sf_south = sf[:, :n][:, ::-1]  # negative lats, flipped to match north

    # massSF should be antisymmetric: sf(lat) ≈ -sf(-lat)
    np.testing.assert_allclose(sf_north, -sf_south, atol=1e-5)


def test_flat_theta_profile_no_crash(minimal_tem_config):
    """Flat theta (DThetaBarDZ ≈ 0) should not produce inf — NaN is acceptable but not inf."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)

    rng = np.random.default_rng(1)
    V     = 0.5 * rng.standard_normal((nalt, nlat, nlon))
    U     = np.zeros((nalt, nlat, nlon))
    # Completely flat potential temperature — DThetaBarDZ = 0 everywhere
    THETA = np.full((nalt, nlat, nlon), 300.0)
    OMEGA = 0.01 * rng.standard_normal((nalt, nlat, nlon))

    import xarray as xr
    ds = xr.Dataset(
        {
            'U':     (['alt', 'lat', 'lon'], U,     {'units': 'm/s'}),
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    TIMESTAMP = np.datetime64('2000-01-01T00:00:00')
    dsOut = TEMCalcs(minimal_tem_config, ds, TIMESTAMP)

    # No variable should contain inf (NaN is tolerable for degenerate input)
    for var in ['V_RES_STD', 'W_RES_STD', 'EPF_lat', 'EPF_vert', 'div_EPF']:
        vals = dsOut[var].values
        assert not np.any(np.isinf(vals)), f"{var} contains inf for flat theta profile"


# ---------------------------------------------------------------------------
# mainCalcs — time-dimension routing tests
# ---------------------------------------------------------------------------

def _make_single_timestamp_file(tmp_path, timestamp, filename):
    """Write a minimal ERA5-like NetCDF file with a time dimension containing one timestamp."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    # Realistic hybrid pressure levels (hPa), decreasing with index
    press_1d = np.linspace(900.0, 10.0, nalt)
    rng = np.random.default_rng(seed=int(pd.Timestamp(timestamp).timestamp()) % (2**31))

    shape = (1, nalt, nlat, nlon)  # time=1
    press = press_1d[:, None, None] * np.ones(shape)
    theta = (300.0 + press_1d[:, None, None] * 0.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape)
    u     = (20.0 + 2.0 * rng.standard_normal(shape)).astype('float32')
    v     = (1.0  + 0.5 * rng.standard_normal(shape)).astype('float32')
    omega = (0.01 * rng.standard_normal(shape)).astype('float32')

    ds = xr.Dataset(
        {
            'PRESS': (['time', 'hybrid', 'lat', 'lon'], press, {'units': 'hPa'}),
            'THETA': (['time', 'hybrid', 'lat', 'lon'], theta, {'units': 'K'}),
            'U':     (['time', 'hybrid', 'lat', 'lon'], u,     {'units': 'm s**-1'}),
            'V':     (['time', 'hybrid', 'lat', 'lon'], v,     {'units': 'm s**-1'}),
            'OMEGA': (['time', 'hybrid', 'lat', 'lon'], omega, {'units': 'Pa s**-1'}),
        },
        coords={
            'time':   (['time'],   [np.datetime64(timestamp)]),
            'hybrid': (['hybrid'], np.arange(nalt, dtype=float)),
            'lat':    (['lat'],    lats, {'units': 'degrees_N'}),
            'lon':    (['lon'],    lons, {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def _mainCalcs_config(output_dir):
    return {
        'outputDirectory': str(output_dir),
        'outPrefix': 'Res_circ',
        'outputTemporalMean': False,
        'verticalDimensionType': 'other',
        'verticalWindType': 'omega',
        'temperatureType': 'theta',
        'pressureName': 'PRESS',
        'temperatureName': 'THETA',
        'zonalWindName': 'U',
        'meridionalWindName': 'V',
        'verticalWindName': 'OMEGA',
        'vertDim': 'hybrid',
        'latDim': 'lat',
        'lonDim': 'lon',
        'timeDim': 'time',
        'targetLevels': [5.0, 10.0, 17.35, 25.0, 35.0],
        'FourierTransform': False,
        'saveEddyTerms': False,
        'inputDataDescription': 'unit test',
        'Waves': ['1'],
    }


def _run_mainCalcs(chunk_df, config):
    """Run mainCalcs in-process by initialising the shared counter directly."""
    counter_val = multiprocessing.Value('i', 0)
    init_worker(counter_val)
    mainCalcs(chunk_df, ['PRESS', 'THETA', 'U', 'V', 'OMEGA'], config)


def test_mainCalcs_time_dim_no_averaging_produces_one_file_per_timestamp(tmp_path):
    """A file with 3 timestamps and outputTemporalMean=False must produce 3 output files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    timestamps = ['2000-01-01T00:00:00', '2000-01-01T06:00:00', '2000-01-01T12:00:00']
    path = _make_single_timestamp_file(input_dir, timestamps[0], 'era5_multi.nc')

    # Overwrite with a file that has 3 timestamps in one file
    ds_list = []
    for ts in timestamps:
        nlat, nlon, nalt = 9, 36, 10
        lats = np.linspace(-80, 80, nlat)
        lons = np.linspace(0, 350, nlon)
        press_1d = np.linspace(900.0, 10.0, nalt)
        rng = np.random.default_rng(42)
        shape = (nalt, nlat, nlon)
        press = press_1d[:, None, None] * np.ones(shape)
        theta = (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape)
        u = (20.0 + 2.0 * rng.standard_normal(shape)).astype('float32')
        v = (1.0 + 0.5 * rng.standard_normal(shape)).astype('float32')
        omega = (0.01 * rng.standard_normal(shape)).astype('float32')
        ds_list.append(xr.Dataset(
            {
                'PRESS': (['hybrid', 'lat', 'lon'], press, {'units': 'hPa'}),
                'THETA': (['hybrid', 'lat', 'lon'], theta, {'units': 'K'}),
                'U':     (['hybrid', 'lat', 'lon'], u,     {'units': 'm s**-1'}),
                'V':     (['hybrid', 'lat', 'lon'], v,     {'units': 'm s**-1'}),
                'OMEGA': (['hybrid', 'lat', 'lon'], omega, {'units': 'Pa s**-1'}),
            },
            coords={
                'hybrid': (['hybrid'], np.arange(nalt, dtype=float)),
                'lat':    (['lat'],    lats, {'units': 'degrees_N'}),
                'lon':    (['lon'],    lons, {'units': 'degree'}),
            },
        ))

    combined = xr.concat(ds_list, dim=pd.DatetimeIndex(timestamps, name='time'))
    nc_path = input_dir / 'era5_multi.nc'
    combined.to_netcdf(nc_path)

    chunk_df = pd.DataFrame({'Path': [str(nc_path)]},
                            index=pd.DatetimeIndex(['2000-01-01T00:00:00']))
    config = _mainCalcs_config(output_dir)

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 3, (
        f"Expected 3 output files (one per timestamp), got {len(produced)}: "
        f"{[p.name for p in produced]}"
    )


def test_mainCalcs_empty_timedim_string_no_time_dim_file(tmp_path):
    """timeDim='' with a file that has no time dimension must produce exactly one output file.

    This is the typical use-case: ERA5 files where each file contains one
    timestep with no time dimension in the NetCDF, and the config records
    timeDim='' to reflect that."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    press_1d = np.linspace(900.0, 10.0, nalt)
    rng = np.random.default_rng(0)
    shape = (nalt, nlat, nlon)
    press = press_1d[:, None, None] * np.ones(shape)
    theta = (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape)
    u = (20.0 + 2.0 * rng.standard_normal(shape)).astype('float32')
    v = (1.0 + 0.5 * rng.standard_normal(shape)).astype('float32')
    omega = (0.01 * rng.standard_normal(shape)).astype('float32')

    ds = xr.Dataset(
        {
            'PRESS': (['hybrid', 'lat', 'lon'], press, {'units': 'hPa'}),
            'THETA': (['hybrid', 'lat', 'lon'], theta, {'units': 'K'}),
            'U':     (['hybrid', 'lat', 'lon'], u,     {'units': 'm s**-1'}),
            'V':     (['hybrid', 'lat', 'lon'], v,     {'units': 'm s**-1'}),
            'OMEGA': (['hybrid', 'lat', 'lon'], omega, {'units': 'Pa s**-1'}),
        },
        coords={
            'hybrid': (['hybrid'], np.arange(nalt, dtype=float)),
            'lat':    (['lat'],    lats, {'units': 'degrees_N'}),
            'lon':    (['lon'],    lons, {'units': 'degree'}),
        },
    )
    nc_path = input_dir / 'era5_notime.nc'
    ds.to_netcdf(nc_path)

    chunk_df = pd.DataFrame({'Path': [str(nc_path)]},
                            index=pd.DatetimeIndex(['2000-01-01T00:00:00']))
    config = _mainCalcs_config(output_dir)
    config['timeDim'] = ''  # files have no time dimension

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 output file, got {len(produced)}: {[p.name for p in produced]}"
    )


# ---------------------------------------------------------------------------
# TEMCalcs — additional branch coverage
# ---------------------------------------------------------------------------

def test_tem_calcs_deg_north_lat_units(synthetic_logpress_dataset, minimal_tem_config):
    """'deg N' in lat.units triggers the attribute normalization branch (line 36)."""
    ds = synthetic_logpress_dataset.copy(deep=True)
    ds.lat.attrs['units'] = 'deg N'
    dsOut = TEMCalcs(minimal_tem_config, ds, TIMESTAMP)
    assert 'THETA' in dsOut


def test_fourier_save_eddy_terms_specific_waves(synthetic_logpress_dataset, minimal_tem_config):
    """FourierTransform + saveEddyTerms + specific Waves covers lines 227-234, 288-305."""
    config = dict(minimal_tem_config)
    config['FourierTransform'] = True
    config['verticalWindType'] = 'omega'
    config['saveEddyTerms'] = True
    config['Waves'] = ['1', '2']
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'EPFLat_WaveN' in dsOut
    assert 'vPrimeThetaPrimeWaveN' in dsOut
    assert dsOut['EPFLat_WaveN'].shape[2] == 2


def test_fourier_save_eddy_terms_wave_range(synthetic_logpress_dataset, minimal_tem_config):
    """saveEddyTerms + Waves=['1-3'] exercises the range sub-branch (lines 296-299)."""
    config = dict(minimal_tem_config)
    config['FourierTransform'] = True
    config['verticalWindType'] = 'omega'
    config['saveEddyTerms'] = True
    config['Waves'] = ['1-3']
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'vPrimeThetaPrimeWaveN' in dsOut
    assert dsOut['EPFLat_WaveN'].shape[2] == 1


def test_fourier_save_eddy_terms_open_end_range(synthetic_logpress_dataset, minimal_tem_config):
    """saveEddyTerms + Waves=['18-end'] exercises the open-end range sub-branch (lines 300-301)."""
    config = dict(minimal_tem_config)
    config['FourierTransform'] = True
    config['verticalWindType'] = 'omega'
    config['saveEddyTerms'] = True
    config['Waves'] = ['18-end']
    dsOut = TEMCalcs(config, synthetic_logpress_dataset, TIMESTAMP)
    assert 'vPrimeThetaPrimeWaveN' in dsOut
    assert dsOut['EPFLat_WaveN'].shape[2] == 1


# ---------------------------------------------------------------------------
# _run_tem_and_attach_vars — non-empty attach lists
# ---------------------------------------------------------------------------

def test_run_tem_attach_vars_non_empty(synthetic_logpress_dataset, minimal_tem_config):
    """Non-empty saveInterpolatedZonalMeanVars / saveZonalMeanVars cover lines 323, 325.

    Production code indexes with a single string key, so each list must contain exactly
    one variable name passed as a string (not a list).
    """
    from transformed_eulerian_mean.residual_circulation import _run_tem_and_attach_vars

    ds = synthetic_logpress_dataset.copy(deep=True)
    # Add 2-D (alt, lat) variables that mimic already-zonally-averaged outputs
    ds_zm = synthetic_logpress_dataset.mean(dim='lon')
    ds['BA_zm'] = ds_zm['BA']
    ds['O3_zm'] = ds_zm['O3']

    dsOut = _run_tem_and_attach_vars(
        minimal_tem_config, ds, TIMESTAMP,
        saveInterpolatedZonalMeanVars='BA_zm',
        saveZonalMeanVars='O3_zm',
    )
    assert 'BA_zm' in dsOut
    assert 'O3_zm' in dsOut


# ---------------------------------------------------------------------------
# Helpers for mainCalcs temporal-mean / resample tests
# ---------------------------------------------------------------------------

def _make_no_timedim_file(tmp_path, filename, seed=42):
    """NetCDF with 3-D fields only (no time dimension)."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    press_1d = np.linspace(900.0, 10.0, nalt)
    rng = np.random.default_rng(seed)
    shape = (nalt, nlat, nlon)
    ds = xr.Dataset(
        {
            'PRESS': (['hybrid', 'lat', 'lon'],
                      press_1d[:, None, None] * np.ones(shape), {'units': 'hPa'}),
            'THETA': (['hybrid', 'lat', 'lon'],
                      (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape), {'units': 'K'}),
            'U':     (['hybrid', 'lat', 'lon'],
                      (20.0 + 2.0 * rng.standard_normal(shape)).astype('float32'), {'units': 'm s**-1'}),
            'V':     (['hybrid', 'lat', 'lon'],
                      (1.0 + 0.5 * rng.standard_normal(shape)).astype('float32'), {'units': 'm s**-1'}),
            'OMEGA': (['hybrid', 'lat', 'lon'],
                      (0.01 * rng.standard_normal(shape)).astype('float32'), {'units': 'Pa s**-1'}),
        },
        coords={
            'hybrid': (['hybrid'], np.arange(nalt, dtype=float)),
            'lat':    (['lat'],    lats, {'units': 'degrees_N'}),
            'lon':    (['lon'],    lons, {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def _make_multi_timestamp_file(tmp_path, timestamps, filename):
    """NetCDF with multiple timestamps in the 'time' dimension."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    press_1d = np.linspace(900.0, 10.0, nalt)
    ds_list = []
    for i, _ts in enumerate(timestamps):
        rng = np.random.default_rng(i)
        shape = (nalt, nlat, nlon)
        ds_list.append(xr.Dataset(
            {
                'PRESS': (['hybrid', 'lat', 'lon'],
                          press_1d[:, None, None] * np.ones(shape), {'units': 'hPa'}),
                'THETA': (['hybrid', 'lat', 'lon'],
                          (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape), {'units': 'K'}),
                'U':     (['hybrid', 'lat', 'lon'],
                          (20.0 + 2.0 * rng.standard_normal(shape)).astype('float32'), {'units': 'm s**-1'}),
                'V':     (['hybrid', 'lat', 'lon'],
                          (1.0 + 0.5 * rng.standard_normal(shape)).astype('float32'), {'units': 'm s**-1'}),
                'OMEGA': (['hybrid', 'lat', 'lon'],
                          (0.01 * rng.standard_normal(shape)).astype('float32'), {'units': 'Pa s**-1'}),
            },
            coords={
                'hybrid': (['hybrid'], np.arange(nalt, dtype=float)),
                'lat':    (['lat'],    lats, {'units': 'degrees_N'}),
                'lon':    (['lon'],    lons, {'units': 'degree'}),
            },
        ))
    combined = xr.concat(ds_list, dim=pd.DatetimeIndex(timestamps, name='time'))
    path = tmp_path / filename
    combined.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# mainCalcs — monthly resample (lines 330-339, 346, 355-357, 381-400, 411-413)
# ---------------------------------------------------------------------------

def test_mainCalcs_monthly_resample(tmp_path):
    """Monthly resample: Jan group (2 ts) and Feb group (2 ts) — both take the multi-ts path.

    Covers: _finalize_mean (330-339), monthly filename (346), _accumulate (355-357),
            _process_group multi-ts (387-400), monthly resample loop (411-413).

    Note: the single-ts path in _process_group (lines 381-385) is not exercised here
    because it passes an xarray DataArray scalar as timestamp to TEMCalcs which fails
    at line 177 (ds.coords['time'] = [xr.DataArray]) — a known production edge case
    that only occurs with the monthly/daily resample routes.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Jan 1, Jan 15, Feb 1, Feb 15 → 2 groups × 2 timestamps each → both multi-ts path
    timestamps = [
        '2000-01-01T00:00:00', '2000-01-15T00:00:00',
        '2000-02-01T00:00:00', '2000-02-15T00:00:00',
    ]
    nc_path = _make_multi_timestamp_file(input_dir, timestamps, 'multi.nc')

    chunk_df = pd.DataFrame({'Path': [str(nc_path)]},
                            index=pd.DatetimeIndex([timestamps[0]]))
    config = _mainCalcs_config(output_dir)
    config['outputTemporalMean'] = 'monthly'

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 2, (
        f"Expected 2 monthly output files, got {len(produced)}: "
        f"{[p.name for p in produced]}"
    )
    assert any('monthlyMean_2000_01' in p.name for p in produced)
    assert any('monthlyMean_2000_02' in p.name for p in produced)
    for p in produced:
        ds = xr.open_dataset(p)
        assert 'dU_dt' in ds, f"dU_dt missing from {p.name}"
        assert not np.all(np.isnan(ds['dU_dt'].values)), f"dU_dt is all-NaN in {p.name}"
        ds.close()


# ---------------------------------------------------------------------------
# mainCalcs — daily resample (lines 348, 415-417)
# ---------------------------------------------------------------------------

def test_mainCalcs_daily_resample(tmp_path):
    """Daily resample: 2 timestamps per day on 2 consecutive days — both take multi-ts path.

    Covers: daily filename (348), daily resample loop (415-417), _accumulate (355-357),
            _finalize_mean (330-339).
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Jan 1 00:00 + Jan 1 12:00 + Jan 2 00:00 + Jan 2 12:00 → 2 groups × 2 ts
    timestamps = [
        '2000-01-01T00:00:00', '2000-01-01T12:00:00',
        '2000-01-02T00:00:00', '2000-01-02T12:00:00',
    ]
    nc_path = _make_multi_timestamp_file(input_dir, timestamps, 'daily.nc')

    chunk_df = pd.DataFrame({'Path': [str(nc_path)]},
                            index=pd.DatetimeIndex([timestamps[0]]))
    config = _mainCalcs_config(output_dir)
    config['outputTemporalMean'] = 'daily'

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 2, (
        f"Expected 2 daily output files, got {len(produced)}: "
        f"{[p.name for p in produced]}"
    )
    assert any('dailyMean_2000_01_01' in p.name for p in produced)
    assert any('dailyMean_2000_01_02' in p.name for p in produced)


# ---------------------------------------------------------------------------
# mainCalcs — multiple files → temporal mean (lines 440-478)
# ---------------------------------------------------------------------------

def test_mainCalcs_multiple_files_temporal_mean(tmp_path):
    """chunk_df with 2 no-time-dim files exercises the multi-file averaging path.

    Covers: 355-357 (_accumulate), 330-339 (_finalize_mean), 440-478 (no-timeDim branch).
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    path1 = _make_no_timedim_file(input_dir, 'f1.nc', seed=10)
    path2 = _make_no_timedim_file(input_dir, 'f2.nc', seed=20)

    chunk_df = pd.DataFrame(
        {'Path': [str(path1), str(path2)]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00', '2000-01-01T06:00:00']),
    )
    config = _mainCalcs_config(output_dir)
    config['timeDim'] = ''  # no time dimension in files

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 averaged output file, got {len(produced)}: "
        f"{[p.name for p in produced]}"
    )
    ds = xr.open_dataset(produced[0])
    assert 'dU_dt' in ds, "dU_dt missing from multi-file temporal mean output"
    assert not np.all(np.isnan(ds['dU_dt'].values)), "dU_dt is all-NaN in multi-file temporal mean output"
    ds.close()


def test_mainCalcs_multiple_files_with_timedim(tmp_path):
    """chunk_df with 2 files each having 2 timestamps in the time dim.

    readAndTransposeData squeezes size-1 dims, so each file needs ≥2 timestamps
    to keep the time dimension alive after squeeze.
    Covers: 447-456 (multi-file loop with timeDim in dataset).
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Each file has 2 timestamps so time dim survives .squeeze()
    ts1 = ['2000-01-01T00:00:00', '2000-01-01T06:00:00']
    ts2 = ['2000-01-02T00:00:00', '2000-01-02T06:00:00']
    path1 = _make_multi_timestamp_file(input_dir, ts1, 'f1.nc')
    path2 = _make_multi_timestamp_file(input_dir, ts2, 'f2.nc')

    chunk_df = pd.DataFrame(
        {'Path': [str(path1), str(path2)]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00', '2000-01-02T00:00:00']),
    )
    config = _mainCalcs_config(output_dir)
    # timeDim='time' is already in the default config

    _run_mainCalcs(chunk_df, config)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 averaged output file, got {len(produced)}: "
        f"{[p.name for p in produced]}"
    )
