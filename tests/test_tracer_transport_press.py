import multiprocessing

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from metpy.units import units
from tem_pkg.tracer_transport_press import tracerTransport, mainCalcs, init_worker
from tem_pkg.utils import addRatioUnits

addRatioUnits()

TIMESTAMP = np.datetime64('2000-01-01T00:00:00')

EXPECTED_VARS_PER_TRACER = [
    '{t}_chi_bar', '{t}_dt_sum', '{t}_divm_z', '{t}_divm_lat',
    '{t}_m_z', '{t}_m_lat', '{t}_adv_z', '{t}_adv_lat',
]


def _tracer_vars(tracer):
    return [v.format(t=tracer) for v in EXPECTED_VARS_PER_TRACER]


def test_output_variables_present(synthetic_logpress_dataset, minimal_tracer_press_config):
    result, _, _ = tracerTransport(synthetic_logpress_dataset, minimal_tracer_press_config)
    for var in _tracer_vars('BA') + ['massSF']:
        assert var in result, f"Missing output variable: {var}"


def test_output_values_finite(synthetic_logpress_dataset, minimal_tracer_press_config):
    result, _, _ = tracerTransport(synthetic_logpress_dataset, minimal_tracer_press_config)
    for var in _tracer_vars('BA'):
        data = result[var][0]
        interior = data[1:-1, 1:-1]
        assert np.all(np.isfinite(interior.magnitude if hasattr(interior, 'magnitude') else interior)), \
            f"{var} has non-finite interior values"


def test_dt_sum_equals_component_sum(synthetic_logpress_dataset, minimal_tracer_press_config):
    result, _, _ = tracerTransport(synthetic_logpress_dataset, minimal_tracer_press_config)
    divm_z   = result['BA_divm_z'][0]
    divm_lat = result['BA_divm_lat'][0]
    adv_z    = result['BA_adv_z'][0]
    adv_lat  = result['BA_adv_lat'][0]
    dt_sum   = result['BA_dt_sum'][0]

    expected = divm_z + divm_lat + adv_z + adv_lat  # sinkSource=0
    np.testing.assert_allclose(
        dt_sum.magnitude if hasattr(dt_sum, 'magnitude') else dt_sum,
        expected.magnitude if hasattr(expected, 'magnitude') else expected,
        rtol=1e-6,
    )


def test_zonally_uniform_tracer_zero_eddy_fluxes(zonally_uniform_dataset, minimal_tracer_press_config):
    result, _, _ = tracerTransport(zonally_uniform_dataset, minimal_tracer_press_config)
    for var in ['BA_m_z', 'BA_m_lat', 'BA_divm_z', 'BA_divm_lat']:
        data = result[var][0]
        vals = data.magnitude if hasattr(data, 'magnitude') else np.array(data)
        np.testing.assert_allclose(vals, 0.0, atol=1e-10, err_msg=f"{var} should be zero for uniform tracer")


def test_integer_sink_source_zero_no_contribution(synthetic_logpress_dataset, minimal_tracer_press_config):
    config = dict(minimal_tracer_press_config)
    config['sinksSources'] = ['0']
    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)

    divm_z   = result['BA_divm_z'][0]
    divm_lat = result['BA_divm_lat'][0]
    adv_z    = result['BA_adv_z'][0]
    adv_lat  = result['BA_adv_lat'][0]
    dt_sum   = result['BA_dt_sum'][0]

    expected = divm_z + divm_lat + adv_z + adv_lat
    np.testing.assert_allclose(
        dt_sum.magnitude if hasattr(dt_sum, 'magnitude') else dt_sum,
        expected.magnitude if hasattr(expected, 'magnitude') else expected,
        rtol=1e-6,
    )


def test_half_life_decay_is_negative_for_positive_tracer(synthetic_logpress_dataset, minimal_tracer_press_config):
    config = dict(minimal_tracer_press_config)
    config['sinksSources'] = ['half life, 90, days']

    # Compute dt_sum with decay and without, isolate sinkSource contribution
    config_no_sink = dict(config)
    config_no_sink['sinksSources'] = ['0']

    result_decay,    _, _ = tracerTransport(synthetic_logpress_dataset, config)
    result_no_decay, _, _ = tracerTransport(synthetic_logpress_dataset, config_no_sink)

    dt_decay    = result_decay   ['BA_dt_sum'][0]
    dt_no_decay = result_no_decay['BA_dt_sum'][0]

    sink = dt_decay - dt_no_decay
    sink_vals = sink.magnitude if hasattr(sink, 'magnitude') else np.array(sink)

    # BA is positive everywhere so decay must be negative
    assert np.all(sink_vals < 0), "Half-life decay should produce negative sink for positive tracer"


def test_half_life_decay_magnitude(synthetic_logpress_dataset, minimal_tracer_press_config):
    """sinkSource = -chiBar * ln(2) / halflife_in_seconds"""
    config = dict(minimal_tracer_press_config)
    config['sinksSources'] = ['half life, 90, days']

    config_no_sink = dict(config)
    config_no_sink['sinksSources'] = ['0']

    result_decay,    _, _ = tracerTransport(synthetic_logpress_dataset, config)
    result_no_decay, _, _ = tracerTransport(synthetic_logpress_dataset, config_no_sink)

    chi_bar = result_decay['BA_chi_bar'][0]
    chi_vals = chi_bar.magnitude if hasattr(chi_bar, 'magnitude') else np.array(chi_bar)

    dt_decay    = result_decay   ['BA_dt_sum'][0]
    dt_no_decay = result_no_decay['BA_dt_sum'][0]
    sink_vals   = (dt_decay - dt_no_decay)
    sink_vals   = sink_vals.magnitude if hasattr(sink_vals, 'magnitude') else np.array(sink_vals)

    halflife_s = 90 * 24 * 3600
    expected_sink = -chi_vals * np.log(2) / halflife_s

    np.testing.assert_allclose(sink_vals, expected_sink, rtol=1e-5)


def test_two_tracers_independent_half_lives(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Regression test: each tracer must use its own half-life, not index 2.

    Run twice with BA getting 90-day vs 365-day half-life while O3 gets 0.
    The ratio of BA sink magnitudes must equal 365/90.
    """
    config_90 = dict(minimal_tracer_press_config)
    config_90['tracerNames'] = ['BA', 'O3']
    config_90['sinksSources'] = ['half life, 90, days', '0']

    config_365 = dict(minimal_tracer_press_config)
    config_365['tracerNames'] = ['BA', 'O3']
    config_365['sinksSources'] = ['half life, 365, days', '0']

    config_no_sink = dict(minimal_tracer_press_config)
    config_no_sink['tracerNames'] = ['BA', 'O3']
    config_no_sink['sinksSources'] = ['0', '0']

    result_90,      _, _ = tracerTransport(synthetic_logpress_dataset, config_90)
    result_365,     _, _ = tracerTransport(synthetic_logpress_dataset, config_365)
    result_no_decay,_, _ = tracerTransport(synthetic_logpress_dataset, config_no_sink)

    sink_90  = (result_90 ['BA_dt_sum'][0] - result_no_decay['BA_dt_sum'][0])
    sink_365 = (result_365['BA_dt_sum'][0] - result_no_decay['BA_dt_sum'][0])

    sink_90_vals  = sink_90 .magnitude if hasattr(sink_90,  'magnitude') else np.array(sink_90)
    sink_365_vals = sink_365.magnitude if hasattr(sink_365, 'magnitude') else np.array(sink_365)

    # 90-day half-life decays 365/90 times faster than 365-day
    ratio = np.abs(sink_90_vals).mean() / np.abs(sink_365_vals).mean()
    np.testing.assert_allclose(ratio, 365 / 90, rtol=0.01)


def test_fourier_wave_range_inclusive(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Sum of individual waves 1,2,3 must equal the range '1-3'."""
    config_range = dict(minimal_tracer_press_config)
    config_range['FourierTransform'] = True
    config_range['Waves'] = ['1-3']

    config_individual = dict(config_range)
    config_individual['Waves'] = ['1', '2', '3']

    result_range, _, _ = tracerTransport(synthetic_logpress_dataset, config_range)
    result_ind,   _, _ = tracerTransport(synthetic_logpress_dataset, config_individual)

    fourier_range = result_range['Fourier']
    fourier_ind   = result_ind  ['Fourier']

    key = 'BA_m_lat_WN'
    range_vals = fourier_range[key][0][:, :, 0]
    ind_sum    = (fourier_ind[key][0][:, :, 0] +
                  fourier_ind[key][0][:, :, 1] +
                  fourier_ind[key][0][:, :, 2])

    range_vals = range_vals.magnitude if hasattr(range_vals, 'magnitude') else range_vals
    ind_sum    = ind_sum.magnitude    if hasattr(ind_sum,    'magnitude') else ind_sum

    np.testing.assert_allclose(range_vals, ind_sum, rtol=1e-5)


def test_two_tracers_output_both_present(synthetic_logpress_dataset, minimal_tracer_press_config):
    config = dict(minimal_tracer_press_config)
    config['tracerNames'] = ['BA', 'O3']
    config['sinksSources'] = ['0', '0']

    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)
    for tracer in ['BA', 'O3']:
        for var in _tracer_vars(tracer):
            assert var in result, f"Missing output variable: {var}"


def test_massstream_function_top_boundary_zero(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Integration starts at the top (flipped cumulative_trapezoid, initial=0 at top)."""
    result, lats, alts = tracerTransport(synthetic_logpress_dataset, minimal_tracer_press_config)
    massSF = result['massSF'][0]
    top = massSF[-1, :]
    top_vals = top.magnitude if hasattr(top, 'magnitude') else np.array(top)
    np.testing.assert_allclose(top_vals, 0.0, atol=1e-4)


def test_fourier_divm_wn_equals_sum_of_components(synthetic_logpress_dataset, minimal_tracer_press_config):
    """divm_WN must equal divm_lat_WN + divm_z_WN for every wavenumber."""
    config = dict(minimal_tracer_press_config)
    config['FourierTransform'] = True
    config['Waves'] = ['1', '2', '3']
    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)
    fourier = result['Fourier']

    divm_lat = fourier['BA_divm_lat_WN'][0]
    divm_z   = fourier['BA_divm_z_WN'][0]
    divm_tot = fourier['BA_divm_WN'][0]

    divm_lat = divm_lat.magnitude if hasattr(divm_lat, 'magnitude') else divm_lat
    divm_z   = divm_z  .magnitude if hasattr(divm_z,   'magnitude') else divm_z
    divm_tot = divm_tot.magnitude if hasattr(divm_tot,  'magnitude') else divm_tot

    np.testing.assert_allclose(divm_tot, divm_lat + divm_z, rtol=1e-5)


def test_fourier_open_end_range_no_crash(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Waves=['18-end'] should sum all waves from 18 onwards without crashing."""
    config = dict(minimal_tracer_press_config)
    config['FourierTransform'] = True
    config['Waves'] = ['18-end']
    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)
    fourier = result['Fourier']
    assert 'BA_m_lat_WN' in fourier
    assert fourier['BA_m_lat_WN'][0].shape[2] == 1


def test_fourier_all_waves_output_shape(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Waves=['all'] should produce nlon//2 - 1 wavenumbers."""
    nlon = synthetic_logpress_dataset.sizes['lon']
    config = dict(minimal_tracer_press_config)
    config['FourierTransform'] = True
    config['Waves'] = ['all']
    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)
    fourier = result['Fourier']
    assert 'BA_m_lat_WN' in fourier
    assert fourier['BA_m_lat_WN'][0].shape[2] == nlon // 2


def test_fourier_keys_contain_tracer_name(synthetic_logpress_dataset, minimal_tracer_press_config):
    config = dict(minimal_tracer_press_config)
    config['FourierTransform'] = True
    config['Waves'] = ['1']
    result, _, _ = tracerTransport(synthetic_logpress_dataset, config)
    fourier = result['Fourier']
    expected_keys = ['BA_m_lat_WN', 'BA_m_z_WN', 'BA_divm_lat_WN', 'BA_divm_z_WN', 'BA_divm_WN']
    for key in expected_keys:
        assert key in fourier, f"Missing Fourier key: {key}"


def test_uniform_tracer_gradient_zero_advection(minimal_tracer_press_config):
    """Constant chiBar in both lat and alt → adv_z and adv_lat are zero."""
    import xarray as xr
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)

    rng = np.random.default_rng(0)
    # Wind has zonal variation to generate non-trivial vBarStar/wBarStar
    V     = 2.0 * np.cos(np.pi * lats[None, :, None] / 180) * np.ones((nalt, nlat, nlon))
    V    += 0.5 * rng.standard_normal((nalt, nlat, nlon))
    THETA = (300 + alts[:, None, None] * 3.0 +
             2.0 * np.sin(2 * np.pi * lons[None, None, :] / 360)) * np.ones((nalt, nlat, nlon))
    OMEGA = 0.01 * rng.standard_normal((nalt, nlat, nlon))
    # Tracer: exactly uniform — no lat or alt gradient
    BA    = np.ones((nalt, nlat, nlon)) * 1e-6

    ds = xr.Dataset(
        {
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
            'BA':    (['alt', 'lat', 'lon'], BA,    {'units': 'ppmv'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    result, _, _ = tracerTransport(ds, minimal_tracer_press_config)
    for var in ['BA_adv_z', 'BA_adv_lat']:
        data = result[var][0]
        vals = data.magnitude if hasattr(data, 'magnitude') else np.array(data)
        np.testing.assert_allclose(vals[1:-1, 1:-1], 0.0, atol=1e-20,
                                   err_msg=f"{var} should be zero for spatially uniform tracer")


def test_integer_sink_source_nonzero_shifts_dt_sum(synthetic_logpress_dataset, minimal_tracer_press_config):
    """sinksSources='3' adds 3 ppmv/s to dt_sum relative to sinksSources='0'."""
    config_three = dict(minimal_tracer_press_config)
    config_three['sinksSources'] = ['3']
    config_zero = dict(minimal_tracer_press_config)
    config_zero['sinksSources'] = ['0']

    result_three, _, _ = tracerTransport(synthetic_logpress_dataset, config_three)
    result_zero,  _, _ = tracerTransport(synthetic_logpress_dataset, config_zero)

    dt_three = result_three['BA_dt_sum'][0]
    dt_zero  = result_zero ['BA_dt_sum'][0]

    diff = dt_three - dt_zero
    diff_vals = diff.magnitude if hasattr(diff, 'magnitude') else np.array(diff)
    np.testing.assert_allclose(diff_vals, 3.0, rtol=1e-6)


def test_age_of_air_tracer_time_units_no_crash(synthetic_logpress_dataset, minimal_tracer_press_config):
    """Tracer with time units (Age of Air) is converted to seconds internally."""
    ds = synthetic_logpress_dataset.copy(deep=True)
    ds['BA'].attrs['units'] = 'years'

    result, _, _ = tracerTransport(ds, minimal_tracer_press_config)
    chi_bar = result['BA_chi_bar'][0]
    # After .to_base_units(), years → seconds; values should be large positive numbers
    chi_vals = chi_bar.magnitude if hasattr(chi_bar, 'magnitude') else np.array(chi_bar)
    assert np.all(chi_vals > 0), "Age-of-air tracer should be positive after unit conversion"
    # 1e-6 years in seconds ≈ 31.5 s; values should be much larger than raw 1e-6
    assert np.all(chi_vals > 1.0), "Age-of-air values should be in seconds (>> 1)"


def test_vertical_wind_type_w_no_crash(synthetic_logpress_dataset, minimal_tracer_press_config):
    """verticalWindType='W' reads vertical wind directly as m/s without omega conversion."""
    import xarray as xr
    ds = synthetic_logpress_dataset.copy(deep=True)
    ds['W'] = ds['OMEGA'].copy(deep=True)
    ds['W'].attrs['units'] = 'm/s'

    config = dict(minimal_tracer_press_config)
    config['verticalWindName'] = 'W'
    config['verticalWindType'] = 'W'
    result, _, _ = tracerTransport(ds, config)
    for var in _tracer_vars('BA') + ['massSF']:
        assert var in result, f"Missing output variable: {var}"
    interior = result['BA_dt_sum'][0][1:-1, 1:-1]
    vals = interior.magnitude if hasattr(interior, 'magnitude') else np.array(interior)
    assert np.all(np.isfinite(vals)), "BA_dt_sum has non-finite interior values for verticalWindType='W'"


def test_massstream_function_antisymmetry(minimal_tracer_press_config):
    """For v = A*sin(lat) with no zonal variation, massSF should be antisymmetric about the equator.

    vPrime = 0, so vBarStar = vBar = A*sin(lat). MSF(-lat) = -MSF(lat).
    """
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)

    V = 2.0 * np.sin(np.pi * lats[None, :, None] / 180) * np.ones((nalt, nlat, nlon))
    THETA = (300 + alts[:, None, None] * 3.0) * np.ones((nalt, nlat, nlon))
    OMEGA = np.zeros((nalt, nlat, nlon))
    BA    = np.ones((nalt, nlat, nlon)) * 1e-6

    import xarray as xr
    ds = xr.Dataset(
        {
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
            'BA':    (['alt', 'lat', 'lon'], BA,    {'units': 'ppmv'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    result, _, _ = tracerTransport(ds, minimal_tracer_press_config)
    sf = result['massSF'][0]
    sf_vals = sf.magnitude if hasattr(sf, 'magnitude') else np.array(sf)  # shape (nalt, nlat)

    n = nlat // 2
    sf_north = sf_vals[:, n + 1:]
    sf_south = sf_vals[:, :n][:, ::-1]

    np.testing.assert_allclose(sf_north, -sf_south, atol=1e-5)


# ---------------------------------------------------------------------------
# mainCalcs helpers
# ---------------------------------------------------------------------------

def _make_press_nc_file(tmp_path, filename, seed=42):
    """NetCDF with all pressure-transport fields (tracer + met in same file, no time dim)."""
    nlat, nlon, nalt = 9, 36, 10
    lats     = np.linspace(-80, 80, nlat)
    lons     = np.linspace(0, 350, nlon)
    press_1d = np.linspace(900.0, 10.0, nalt)
    rng = np.random.default_rng(seed)
    shape = (nalt, nlat, nlon)
    press = press_1d[:, None, None] * np.ones(shape)
    theta = (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape)
    ds = xr.Dataset(
        {
            'PRESS': (['hybrid', 'lat', 'lon'], press.astype('float32'), {'units': 'hPa'}),
            'THETA': (['hybrid', 'lat', 'lon'], theta.astype('float32'), {'units': 'K'}),
            'V':     (['hybrid', 'lat', 'lon'],
                      (2.0 * np.cos(np.pi * lats[None, :, None] / 180) * np.ones(shape)
                       + 0.5 * rng.standard_normal(shape)).astype('float32'), {'units': 'm/s'}),
            'OMEGA': (['hybrid', 'lat', 'lon'],
                      (0.01 * rng.standard_normal(shape)).astype('float32'), {'units': 'Pa/s'}),
            'BA':    (['hybrid', 'lat', 'lon'],
                      (1e-6 + 1e-8 * np.sin(2 * np.pi * lons[None, None, :] / 360)
                       * np.ones(shape)).astype('float32'), {'units': 'ppmv'}),
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


def _press_maincalcs_config(output_dir):
    return {
        'outputDirectory': str(output_dir),
        'outPrefix': 'press_transport_',
        'tracerDataInMetFiles': True,
        'verticalDimensionType': 'other',
        'pressureName': 'PRESS',
        'temperatureName': 'THETA',
        'temperatureType': 'theta',
        'vertDim': 'hybrid',
        'latDim': 'lat',
        'lonDim': 'lon',
        'meridionalWindName': 'V',
        'verticalWindName': 'OMEGA',
        'verticalWindType': 'omega',
        'tracerNames': ['BA'],
        'sinksSources': ['0'],
        'massSF': True,
        'FourierTransform': False,
        'Waves': ['1'],
        'binningLat': 1,
        'binningLon': 1,
        'targetLevels': [5.0, 10.0, 17.35, 25.0, 35.0],
    }


def _run_press_maincalcs(pathsAndTime, config, req_vars_with_tracers):
    counter_val = multiprocessing.Value('i', 0)
    init_worker(counter_val)
    mainCalcs(config, 0, pathsAndTime=pathsAndTime, reqVarsWithTracers=req_vars_with_tracers)


# ---------------------------------------------------------------------------
# mainCalcs — tracerDataInMetFiles=True (lines 212-219)
# ---------------------------------------------------------------------------

def test_press_maincalcs_tracer_in_met_produces_output(tmp_path):
    """mainCalcs with tracerDataInMetFiles=True reads one file and writes one output."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    nc_path = _make_press_nc_file(input_dir, 'press_data.nc')
    paths_df = pd.DataFrame(
        {'Path': [str(nc_path)]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00']),
    )
    config   = _press_maincalcs_config(output_dir)
    req_vars = ['PRESS', 'THETA', 'V', 'OMEGA', 'BA']

    _run_press_maincalcs(paths_df, config, req_vars)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 output file, got {len(produced)}: {[p.name for p in produced]}"
    )


# ---------------------------------------------------------------------------
# mainCalcs — tracerDataInMetFiles=False (separate tracer/met files, lines 222-234)
# ---------------------------------------------------------------------------

def _make_press_tracer_nc(tmp_path, filename):
    """Tracer-only NetCDF already in log-pressure altitude coordinates."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)   # km log-pressure alt
    shape = (nalt, nlat, nlon)
    rng = np.random.default_rng(11)
    ds = xr.Dataset(
        {'BA': (['alt', 'lat', 'lon'],
                (1e-6 + 1e-8 * rng.standard_normal(shape)).astype('float32'),
                {'units': 'ppmv'})},
        coords={
            'alt': (['alt'], alts, {'units': 'km'}),
            'lat': (['lat'], lats, {'units': 'degrees_N'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def _make_press_met_nc(tmp_path, filename):
    """Met-only NetCDF on hybrid vertical coordinate."""
    nlat, nlon, nalt = 9, 36, 10
    lats     = np.linspace(-80, 80, nlat)
    lons     = np.linspace(0, 350, nlon)
    press_1d = np.linspace(900.0, 10.0, nalt)
    shape    = (nalt, nlat, nlon)
    press    = press_1d[:, None, None] * np.ones(shape)
    theta    = (300.0 + np.arange(nalt)[:, None, None] * 5.0) * np.ones(shape)
    rng = np.random.default_rng(12)
    ds = xr.Dataset(
        {
            'PRESS': (['hybrid', 'lat', 'lon'], press.astype('float32'), {'units': 'hPa'}),
            'THETA': (['hybrid', 'lat', 'lon'], theta.astype('float32'), {'units': 'K'}),
            'V':     (['hybrid', 'lat', 'lon'],
                      (2.0 * np.cos(np.pi * lats[None, :, None] / 180) * np.ones(shape)
                       + 0.5 * rng.standard_normal(shape)).astype('float32'), {'units': 'm/s'}),
            'OMEGA': (['hybrid', 'lat', 'lon'],
                      (0.01 * rng.standard_normal(shape)).astype('float32'), {'units': 'Pa/s'}),
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


def test_press_maincalcs_separate_tracer_met_produces_output(tmp_path):
    """mainCalcs with tracerDataInMetFiles=False uses pathDictionary (lines 222-234)."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tracer_path = _make_press_tracer_nc(input_dir, 'tracer.nc')
    met_path    = _make_press_met_nc(input_dir, 'met.nc')

    ts = pd.Timestamp('2000-01-01T00:00:00')
    path_dict = {ts: (str(tracer_path), [str(met_path)], [1.0])}

    config = _press_maincalcs_config(output_dir)
    config['tracerDataInMetFiles'] = False
    config['tracerVertDim'] = 'alt'
    config['tracerLatDim']  = 'lat'
    config['tracerLonDim']  = 'lon'
    config['tracerVerticalDimensionType'] = 'log-pressure'

    counter_val = multiprocessing.Value('i', 0)
    init_worker(counter_val)
    req_vars = ['PRESS', 'THETA', 'V', 'OMEGA']
    mainCalcs(config, 0, pathDictionary=path_dict, reqVars=req_vars)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 output file, got {len(produced)}: {[p.name for p in produced]}"
    )
