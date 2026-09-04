import multiprocessing

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from metpy.units import units
from transformed_eulerian_mean.tracer_transport_theta import tracerTransport, mainCalcs, init_worker
from transformed_eulerian_mean.utils import addRatioUnits

addRatioUnits()

EXPECTED_VARS_PER_TRACER = [
    '{t}_chi_bar', '{t}_dt_sum', '{t}_divm_theta', '{t}_divm_lat',
    '{t}_m_theta', '{t}_m_lat', '{t}_adv_theta', '{t}_adv_lat',
]


def _tracer_vars(tracer):
    return [v.format(t=tracer) for v in EXPECTED_VARS_PER_TRACER]


def test_output_variables_present(synthetic_theta_dataset, minimal_tracer_theta_config):
    result, _, _ = tracerTransport(synthetic_theta_dataset, minimal_tracer_theta_config)
    for var in _tracer_vars('BA') + ['massSF', 'PRESS']:
        assert var in result, f"Missing output variable: {var}"


def test_output_values_finite(synthetic_theta_dataset, minimal_tracer_theta_config):
    result, _, _ = tracerTransport(synthetic_theta_dataset, minimal_tracer_theta_config)
    for var in _tracer_vars('BA'):
        data = result[var][0]
        interior = data[1:-1, 1:-1]
        vals = interior.magnitude if hasattr(interior, 'magnitude') else np.array(interior)
        assert np.all(np.isfinite(vals)), f"{var} has non-finite interior values"


def test_dt_sum_equals_component_sum(synthetic_theta_dataset, minimal_tracer_theta_config):
    result, _, _ = tracerTransport(synthetic_theta_dataset, minimal_tracer_theta_config)
    divm_theta = result['BA_divm_theta'][0]
    divm_lat   = result['BA_divm_lat'][0]
    adv_theta  = result['BA_adv_theta'][0]
    adv_lat    = result['BA_adv_lat'][0]
    dt_sum     = result['BA_dt_sum'][0]

    expected = divm_theta + divm_lat + adv_theta + adv_lat  # sinkSource=0
    np.testing.assert_allclose(
        dt_sum.magnitude   if hasattr(dt_sum,   'magnitude') else dt_sum,
        expected.magnitude if hasattr(expected, 'magnitude') else expected,
        rtol=1e-6,
    )


def test_zonally_uniform_tracer_zero_eddy_fluxes(zonally_uniform_theta_dataset, minimal_tracer_theta_config):
    result, _, _ = tracerTransport(zonally_uniform_theta_dataset, minimal_tracer_theta_config)
    for var in ['BA_m_theta', 'BA_m_lat', 'BA_divm_theta', 'BA_divm_lat']:
        data = result[var][0]
        vals = data.magnitude if hasattr(data, 'magnitude') else np.array(data)
        np.testing.assert_allclose(vals, 0.0, atol=1e-10,
                                   err_msg=f"{var} should be zero for uniform tracer")


def test_integer_sink_source_zero_no_contribution(synthetic_theta_dataset, minimal_tracer_theta_config):
    config = dict(minimal_tracer_theta_config)
    config['sinksSources'] = ['0']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)

    divm_theta = result['BA_divm_theta'][0]
    divm_lat   = result['BA_divm_lat'][0]
    adv_theta  = result['BA_adv_theta'][0]
    adv_lat    = result['BA_adv_lat'][0]
    dt_sum     = result['BA_dt_sum'][0]

    expected = divm_theta + divm_lat + adv_theta + adv_lat
    np.testing.assert_allclose(
        dt_sum.magnitude   if hasattr(dt_sum,   'magnitude') else dt_sum,
        expected.magnitude if hasattr(expected, 'magnitude') else expected,
        rtol=1e-6,
    )


def test_half_life_decay_is_negative_for_positive_tracer(synthetic_theta_dataset, minimal_tracer_theta_config):
    config_decay    = dict(minimal_tracer_theta_config)
    config_no_decay = dict(minimal_tracer_theta_config)
    config_decay   ['sinksSources'] = ['half life, 90, days']
    config_no_decay['sinksSources'] = ['0']

    result_decay,    _, _ = tracerTransport(synthetic_theta_dataset, config_decay)
    result_no_decay, _, _ = tracerTransport(synthetic_theta_dataset, config_no_decay)

    dt_decay    = result_decay   ['BA_dt_sum'][0]
    dt_no_decay = result_no_decay['BA_dt_sum'][0]
    sink = dt_decay - dt_no_decay
    sink_vals = sink.magnitude if hasattr(sink, 'magnitude') else np.array(sink)

    assert np.all(sink_vals < 0), "Half-life decay should produce negative sink for positive tracer"


def test_half_life_decay_magnitude(synthetic_theta_dataset, minimal_tracer_theta_config):
    """sinkSource = -chiBar * ln(2) / halflife (theta version keeps original units — ppmv/day)."""
    config_decay    = dict(minimal_tracer_theta_config)
    config_no_decay = dict(minimal_tracer_theta_config)
    config_decay   ['sinksSources'] = ['half life, 90, days']
    config_no_decay['sinksSources'] = ['0']

    result_decay,    _, _ = tracerTransport(synthetic_theta_dataset, config_decay)
    result_no_decay, _, _ = tracerTransport(synthetic_theta_dataset, config_no_decay)

    chi_bar   = result_decay['BA_chi_bar'][0]
    chi_vals  = chi_bar.magnitude if hasattr(chi_bar, 'magnitude') else np.array(chi_bar)

    sink = result_decay['BA_dt_sum'][0] - result_no_decay['BA_dt_sum'][0]
    sink_vals = sink.magnitude if hasattr(sink, 'magnitude') else np.array(sink)

    # halfLifeUnits.to_base_units() converts days → seconds, so result is in ppmv/s;
    # expected = -chi_vals * ln(2) / (90 days in seconds)
    expected_sink = -chi_vals * np.log(2) / (90.0 * 86400.0)
    np.testing.assert_allclose(sink_vals, expected_sink, rtol=1e-5)


def test_two_tracers_output_both_present(synthetic_theta_dataset, minimal_tracer_theta_config):
    config = dict(minimal_tracer_theta_config)
    config['tracerNames']  = ['BA', 'O3']
    config['sinksSources'] = ['0', '0']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)
    for tracer in ['BA', 'O3']:
        for var in _tracer_vars(tracer):
            assert var in result, f"Missing output variable: {var}"


def test_massstream_function_top_boundary_zero(synthetic_theta_dataset, minimal_tracer_theta_config):
    """Integration starts at the top (flipped cumulative_trapezoid, initial=0 at top)."""
    result, _, _ = tracerTransport(synthetic_theta_dataset, minimal_tracer_theta_config)
    massSF = result['massSF'][0]
    top = massSF[-1, :]
    top_vals = top.magnitude if hasattr(top, 'magnitude') else np.array(top)
    np.testing.assert_allclose(top_vals, 0.0, atol=1e-4)


def test_integer_sink_source_nonzero_shifts_dt_sum(synthetic_theta_dataset, minimal_tracer_theta_config):
    """sinksSources='3' adds 3 ppmv/s to dt_sum relative to sinksSources='0'."""
    config_three = dict(minimal_tracer_theta_config)
    config_zero  = dict(minimal_tracer_theta_config)
    config_three['sinksSources'] = ['3']
    config_zero ['sinksSources'] = ['0']

    result_three, _, _ = tracerTransport(synthetic_theta_dataset, config_three)
    result_zero,  _, _ = tracerTransport(synthetic_theta_dataset, config_zero)

    dt_three = result_three['BA_dt_sum'][0]
    dt_zero  = result_zero ['BA_dt_sum'][0]
    diff = dt_three - dt_zero
    diff_vals = diff.magnitude if hasattr(diff, 'magnitude') else np.array(diff)
    np.testing.assert_allclose(diff_vals, 3.0, rtol=1e-6)


def test_age_of_air_tracer_time_units_no_crash(synthetic_theta_dataset, minimal_tracer_theta_config):
    """Tracer with time units is converted to seconds internally."""
    ds = synthetic_theta_dataset.copy(deep=True)
    ds['BA'].attrs['units'] = 'years'
    result, _, _ = tracerTransport(ds, minimal_tracer_theta_config)
    chi_bar  = result['BA_chi_bar'][0]
    chi_vals = chi_bar.magnitude if hasattr(chi_bar, 'magnitude') else np.array(chi_bar)
    assert np.all(chi_vals > 0), "Age-of-air tracer should be positive after unit conversion"
    assert np.all(chi_vals > 1.0), "Age-of-air values should be in seconds (>> 1)"


def test_fourier_keys_contain_tracer_name(synthetic_theta_dataset, minimal_tracer_theta_config):
    config = dict(minimal_tracer_theta_config)
    config['FourierTransform'] = True
    config['Waves'] = ['1']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)
    fourier = result['Fourier']
    expected_keys = ['BA_m_lat_WN', 'BA_m_theta_WN', 'BA_divm_lat_WN', 'BA_divm_theta_WN', 'BA_divm_WN']
    for key in expected_keys:
        assert key in fourier, f"Missing Fourier key: {key}"


def test_fourier_divm_wn_equals_sum_of_components(synthetic_theta_dataset, minimal_tracer_theta_config):
    """divm_WN must equal divm_lat_WN + divm_theta_WN for every wavenumber."""
    config = dict(minimal_tracer_theta_config)
    config['FourierTransform'] = True
    config['Waves'] = ['1', '2', '3']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)
    fourier = result['Fourier']

    divm_lat   = fourier['BA_divm_lat_WN'][0]
    divm_theta = fourier['BA_divm_theta_WN'][0]
    divm_tot   = fourier['BA_divm_WN'][0]

    divm_lat   = divm_lat  .magnitude if hasattr(divm_lat,   'magnitude') else divm_lat
    divm_theta = divm_theta.magnitude if hasattr(divm_theta, 'magnitude') else divm_theta
    divm_tot   = divm_tot  .magnitude if hasattr(divm_tot,   'magnitude') else divm_tot

    np.testing.assert_allclose(divm_tot, divm_lat + divm_theta, rtol=1e-5)


def test_fourier_all_waves_output_shape(synthetic_theta_dataset, minimal_tracer_theta_config):
    """Waves=['all'] should produce nlon//2 wavenumbers."""
    nlon = synthetic_theta_dataset.sizes['lon']
    config = dict(minimal_tracer_theta_config)
    config['FourierTransform'] = True
    config['Waves'] = ['all']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)
    fourier = result['Fourier']
    assert 'BA_m_lat_WN' in fourier
    assert fourier['BA_m_lat_WN'][0].shape[2] == nlon // 2


def test_fourier_wave_range_inclusive(synthetic_theta_dataset, minimal_tracer_theta_config):
    """Sum of individual waves 1,2,3 must equal the range '1-3'."""
    config_range = dict(minimal_tracer_theta_config)
    config_range['FourierTransform'] = True
    config_range['Waves'] = ['1-3']

    config_ind = dict(config_range)
    config_ind['Waves'] = ['1', '2', '3']

    result_range, _, _ = tracerTransport(synthetic_theta_dataset, config_range)
    result_ind,   _, _ = tracerTransport(synthetic_theta_dataset, config_ind)

    key = 'BA_m_lat_WN'
    range_vals = result_range['Fourier'][key][0][:, :, 0]
    ind_sum    = (result_ind['Fourier'][key][0][:, :, 0] +
                  result_ind['Fourier'][key][0][:, :, 1] +
                  result_ind['Fourier'][key][0][:, :, 2])

    range_vals = range_vals.magnitude if hasattr(range_vals, 'magnitude') else range_vals
    ind_sum    = ind_sum.magnitude    if hasattr(ind_sum,    'magnitude') else ind_sum

    np.testing.assert_allclose(range_vals, ind_sum, rtol=1e-5)


def test_fourier_open_end_range_no_crash(synthetic_theta_dataset, minimal_tracer_theta_config):
    """Waves=['18-end'] should sum all waves from 18 onwards without crashing (line 163-165)."""
    config = dict(minimal_tracer_theta_config)
    config['FourierTransform'] = True
    config['Waves'] = ['18-end']
    result, _, _ = tracerTransport(synthetic_theta_dataset, config)
    fourier = result['Fourier']
    assert 'BA_m_lat_WN' in fourier
    assert fourier['BA_m_lat_WN'][0].shape[2] == 1


# ---------------------------------------------------------------------------
# mainCalcs — tracerDataInMetFiles=True (lines 184-189)
# ---------------------------------------------------------------------------

def _make_theta_nc_file(tmp_path, filename):
    """NetCDF file with all theta-transport fields (tracer + met in same file)."""
    nlat, nlon, ntheta = 9, 36, 10
    lats   = np.linspace(-80, 80, nlat)
    lons   = np.linspace(0, 350, nlon)
    thetas = np.linspace(300.0, 800.0, ntheta)
    rng = np.random.default_rng(42)
    shape = (ntheta, nlat, nlon)

    press = (1000.0 * np.exp(-thetas[:, None, None] / 700.0)) * np.ones(shape)
    ds = xr.Dataset(
        {
            'V':         (['theta', 'lat', 'lon'],
                          (2.0 * np.cos(np.pi * lats[None, :, None] / 180) * np.ones(shape)
                           + 0.5 * rng.standard_normal(shape)).astype('float32'),
                          {'units': 'm/s'}),
            'THETA_DOT': (['theta', 'lat', 'lon'],
                          (0.01 * rng.standard_normal(shape)).astype('float32'),
                          {'units': 'K/s'}),
            'PRESS':     (['theta', 'lat', 'lon'], press.astype('float32'), {'units': 'hPa'}),
            'BA':        (['theta', 'lat', 'lon'],
                          (1e-6 + 1e-8 * np.sin(2 * np.pi * lons[None, None, :] / 360)
                           * np.ones(shape)).astype('float32'),
                          {'units': 'ppmv'}),
        },
        coords={
            'theta': (['theta'], thetas, {'units': 'K'}),
            'lat':   (['lat'],   lats,   {'units': 'degrees_N'}),
            'lon':   (['lon'],   lons,   {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def _theta_maincalcs_config(output_dir):
    return {
        'outputDirectory': str(output_dir),
        'outPrefix': 'theta_transport_',
        'tracerDataInMetFiles': True,
        'verticalDimensionType': 'theta',
        'vertDim': 'theta',
        'latDim': 'lat',
        'lonDim': 'lon',
        'meridionalWindName': 'V',
        'verticalWindName': 'THETA_DOT',
        'pressureName': 'PRESS',
        'tracerNames': ['BA'],
        'sinksSources': ['0'],
        'massSF': True,
        'FourierTransform': False,
        'Waves': ['1'],
        'binningLat': 1,
        'binningLon': 1,
    }


def _run_theta_maincalcs(pathsAndTime, config, req_vars_with_tracers):
    counter_val = multiprocessing.Value('i', 0)
    init_worker(counter_val)
    ts = pathsAndTime.index[0]
    mainCalcs(config, task_path=(ts, pathsAndTime['Path'].iloc[0]), reqVarsWithTracers=req_vars_with_tracers)


def test_maincalcs_tracer_in_met_files_produces_output(tmp_path):
    """mainCalcs with tracerDataInMetFiles=True reads one file and writes one output (lines 184-189)."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    nc_path = _make_theta_nc_file(input_dir, 'theta_data.nc')
    paths_df = pd.DataFrame(
        {'Path': [str(nc_path)]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00']),
    )
    config = _theta_maincalcs_config(output_dir)
    req_vars = ['V', 'THETA_DOT', 'PRESS', 'BA']

    _run_theta_maincalcs(paths_df, config, req_vars)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 output file, got {len(produced)}: {[p.name for p in produced]}"
    )


# ---------------------------------------------------------------------------
# mainCalcs — tracerDataInMetFiles=False (separate tracer/met files, lines 192-204)
# ---------------------------------------------------------------------------

def _make_tracer_theta_nc(tmp_path, filename):
    """Tracer-only NetCDF already on theta coordinate."""
    nlat, nlon, ntheta = 9, 36, 10
    lats   = np.linspace(-80, 80, nlat)
    lons   = np.linspace(0, 350, nlon)
    thetas = np.linspace(300.0, 800.0, ntheta)
    shape  = (ntheta, nlat, nlon)
    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {'BA': (['theta', 'lat', 'lon'],
                (1e-6 + 1e-8 * rng.standard_normal(shape)).astype('float32'),
                {'units': 'ppmv'})},
        coords={
            'theta': (['theta'], thetas, {'units': 'K'}),
            'lat':   (['lat'],   lats,   {'units': 'degrees_N'}),
            'lon':   (['lon'],   lons,   {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def _make_met_theta_nc(tmp_path, filename):
    """Met-only NetCDF already on theta coordinate."""
    nlat, nlon, ntheta = 9, 36, 10
    lats   = np.linspace(-80, 80, nlat)
    lons   = np.linspace(0, 350, nlon)
    thetas = np.linspace(300.0, 800.0, ntheta)
    shape  = (ntheta, nlat, nlon)
    press  = (1000.0 * np.exp(-thetas[:, None, None] / 700.0)) * np.ones(shape)
    rng = np.random.default_rng(8)
    ds = xr.Dataset(
        {
            'V':         (['theta', 'lat', 'lon'],
                          (2.0 * np.cos(np.pi * lats[None, :, None] / 180)
                           * np.ones(shape)).astype('float32'), {'units': 'm/s'}),
            'THETA_DOT': (['theta', 'lat', 'lon'],
                          (0.01 * rng.standard_normal(shape)).astype('float32'), {'units': 'K/s'}),
            'PRESS':     (['theta', 'lat', 'lon'], press.astype('float32'), {'units': 'hPa'}),
        },
        coords={
            'theta': (['theta'], thetas, {'units': 'K'}),
            'lat':   (['lat'],   lats,   {'units': 'degrees_N'}),
            'lon':   (['lon'],   lons,   {'units': 'degree'}),
        },
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    return path


def test_maincalcs_separate_tracer_met_files_produces_output(tmp_path):
    """mainCalcs with tracerDataInMetFiles=False uses pathDictionary (lines 192-204)."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tracer_path = _make_tracer_theta_nc(input_dir, 'tracer.nc')
    met_path    = _make_met_theta_nc(input_dir, 'met.nc')

    ts = pd.Timestamp('2000-01-01T00:00:00')
    path_dict = {ts: (str(tracer_path), [str(met_path)], [1.0])}

    config = _theta_maincalcs_config(output_dir)
    config['tracerDataInMetFiles'] = False
    config['tracerVertDim'] = 'theta'
    config['tracerLatDim']  = 'lat'
    config['tracerLonDim']  = 'lon'
    config['tracerVerticalDimensionType'] = 'theta'
    config['verticalDimensionType'] = 'theta'

    counter_val = multiprocessing.Value('i', 0)
    init_worker(counter_val)
    req_vars = ['V', 'THETA_DOT', 'PRESS']
    ts_key = list(path_dict.keys())[0]
    entry = (ts_key, *path_dict[ts_key])
    mainCalcs(config, task_entry=entry, reqVars=req_vars)

    produced = list(output_dir.glob('*.nc'))
    assert len(produced) == 1, (
        f"Expected 1 output file, got {len(produced)}: {[p.name for p in produced]}"
    )
