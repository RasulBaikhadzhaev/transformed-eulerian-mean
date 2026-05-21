import numpy as np
import xarray as xr
import pytest


@pytest.fixture
def synthetic_logpress_dataset():
    """Small zonally varying dataset in log-pressure altitude coordinates."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)  # km, log-pressure altitude

    rng = np.random.default_rng(42)

    # Zonal wind: solid-body rotation plus small eddy
    U = 20.0 + 5.0 * np.sin(2 * np.pi * lons[None, None, :] / 360)
    U = np.broadcast_to(U, (nalt, nlat, nlon)).copy()

    # Meridional wind: small wave
    V = 2.0 * np.cos(np.pi * lats[None, :, None] / 180)
    V = np.broadcast_to(V, (nalt, nlat, nlon)).copy()
    V += 0.5 * rng.standard_normal((nalt, nlat, nlon))

    # Potential temperature: increases with altitude, slight wave
    THETA = (300 + alts[:, None, None] * 3.0 +
             2.0 * np.sin(2 * np.pi * lons[None, None, :] / 360))
    THETA = np.broadcast_to(THETA, (nalt, nlat, nlon)).copy()

    # Omega (Pa/s): small values
    OMEGA = 0.01 * rng.standard_normal((nalt, nlat, nlon))

    # Tracer: uniform background plus small wave
    BA = 1e-6 + 1e-8 * np.sin(2 * np.pi * lons[None, None, :] / 360)
    BA = np.broadcast_to(BA, (nalt, nlat, nlon)).copy()

    O3 = 5e-6 + 5e-8 * np.cos(4 * np.pi * lons[None, None, :] / 360)
    O3 = np.broadcast_to(O3, (nalt, nlat, nlon)).copy()

    ds = xr.Dataset(
        {
            'U':     (['alt', 'lat', 'lon'], U,     {'units': 'm/s'}),
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
            'BA':    (['alt', 'lat', 'lon'], BA,    {'units': 'ppmv'}),
            'O3':    (['alt', 'lat', 'lon'], O3,    {'units': 'ppmv'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    return ds


@pytest.fixture
def zonally_uniform_dataset():
    """Dataset with no zonal variation — all primes are identically zero."""
    nlat, nlon, nalt = 9, 36, 10
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    alts = np.linspace(5.0, 50.0, nalt)

    U     = np.ones((nalt, nlat, nlon)) * 20.0
    V     = np.zeros((nalt, nlat, nlon))
    THETA = (300 + alts[:, None, None] * 3.0) * np.ones((nalt, nlat, nlon))
    OMEGA = np.zeros((nalt, nlat, nlon))
    BA    = np.ones((nalt, nlat, nlon)) * 1e-6
    O3    = np.ones((nalt, nlat, nlon)) * 5e-6

    ds = xr.Dataset(
        {
            'U':     (['alt', 'lat', 'lon'], U,     {'units': 'm/s'}),
            'V':     (['alt', 'lat', 'lon'], V,     {'units': 'm/s'}),
            'THETA': (['alt', 'lat', 'lon'], THETA, {'units': 'K'}),
            'OMEGA': (['alt', 'lat', 'lon'], OMEGA, {'units': 'Pa/s'}),
            'BA':    (['alt', 'lat', 'lon'], BA,    {'units': 'ppmv'}),
            'O3':    (['alt', 'lat', 'lon'], O3,    {'units': 'ppmv'}),
        },
        coords={
            'lat': (['lat'], lats, {'units': 'degree'}),
            'lon': (['lon'], lons, {'units': 'degree'}),
            'alt': (['alt'], alts, {'units': 'km'}),
        },
    )
    return ds


@pytest.fixture
def minimal_tem_config():
    return {
        'zonalWindName': 'U',
        'meridionalWindName': 'V',
        'temperatureName': 'THETA',
        'temperatureType': 'theta',
        'verticalWindName': 'OMEGA',
        'verticalWindType': 'omega',
        'saveEddyTerms': False,
        'FourierTransform': False,
        'inputDataDescription': 'test data',
        'Waves': ['1'],
    }


@pytest.fixture
def minimal_tracer_press_config():
    return {
        'meridionalWindName': 'V',
        'verticalWindName': 'OMEGA',
        'verticalWindType': 'omega',
        'temperatureName': 'THETA',
        'temperatureType': 'theta',
        'tracerNames': ['BA'],
        'sinksSources': ['0'],
        'massSF': True,
        'FourierTransform': False,
        'Waves': ['1'],
    }


@pytest.fixture
def synthetic_theta_dataset():
    """Small zonally varying dataset in isentropic (theta) coordinates."""
    nlat, nlon, ntheta = 9, 36, 10
    lats   = np.linspace(-80, 80, nlat)
    lons   = np.linspace(0, 350, nlon)
    thetas = np.linspace(300.0, 800.0, ntheta)  # K, potential temperature levels

    rng = np.random.default_rng(42)

    V     = (2.0 * np.cos(np.pi * lats[None, :, None] / 180) * np.ones((ntheta, nlat, nlon))
             + 0.5 * rng.standard_normal((ntheta, nlat, nlon)))
    THETA_DOT = 0.01 * rng.standard_normal((ntheta, nlat, nlon))

    # Pressure: monotonically decreasing with theta (higher theta = lower pressure)
    PRESS = (1000.0 * np.exp(-thetas[:, None, None] / 700.0)) * np.ones((ntheta, nlat, nlon))

    BA = (1e-6 + 1e-8 * np.sin(2 * np.pi * lons[None, None, :] / 360)) * np.ones((ntheta, nlat, nlon))
    O3 = (5e-6 + 5e-8 * np.cos(4 * np.pi * lons[None, None, :] / 360)) * np.ones((ntheta, nlat, nlon))

    ds = xr.Dataset(
        {
            'V':         (['theta', 'lat', 'lon'], V,         {'units': 'm/s'}),
            'THETA_DOT': (['theta', 'lat', 'lon'], THETA_DOT, {'units': 'K/s'}),
            'PRESS':     (['theta', 'lat', 'lon'], PRESS,     {'units': 'hPa'}),
            'BA':        (['theta', 'lat', 'lon'], BA,        {'units': 'ppmv'}),
            'O3':        (['theta', 'lat', 'lon'], O3,        {'units': 'ppmv'}),
        },
        coords={
            'lat':   (['lat'],   lats,   {'units': 'degree'}),
            'lon':   (['lon'],   lons,   {'units': 'degree'}),
            'theta': (['theta'], thetas, {'units': 'K'}),
        },
    )
    return ds


@pytest.fixture
def zonally_uniform_theta_dataset():
    """Theta-coordinate dataset with no zonal variation — all primes are zero."""
    nlat, nlon, ntheta = 9, 36, 10
    lats   = np.linspace(-80, 80, nlat)
    lons   = np.linspace(0, 350, nlon)
    thetas = np.linspace(300.0, 800.0, ntheta)

    V         = np.zeros((ntheta, nlat, nlon))
    THETA_DOT = np.zeros((ntheta, nlat, nlon))
    PRESS     = (1000.0 * np.exp(-thetas[:, None, None] / 700.0)) * np.ones((ntheta, nlat, nlon))
    BA        = np.ones((ntheta, nlat, nlon)) * 1e-6
    O3        = np.ones((ntheta, nlat, nlon)) * 5e-6

    ds = xr.Dataset(
        {
            'V':         (['theta', 'lat', 'lon'], V,         {'units': 'm/s'}),
            'THETA_DOT': (['theta', 'lat', 'lon'], THETA_DOT, {'units': 'K/s'}),
            'PRESS':     (['theta', 'lat', 'lon'], PRESS,     {'units': 'hPa'}),
            'BA':        (['theta', 'lat', 'lon'], BA,        {'units': 'ppmv'}),
            'O3':        (['theta', 'lat', 'lon'], O3,        {'units': 'ppmv'}),
        },
        coords={
            'lat':   (['lat'],   lats,   {'units': 'degree'}),
            'lon':   (['lon'],   lons,   {'units': 'degree'}),
            'theta': (['theta'], thetas, {'units': 'K'}),
        },
    )
    return ds


@pytest.fixture
def minimal_tracer_theta_config():
    return {
        'meridionalWindName': 'V',
        'verticalWindName':   'THETA_DOT',
        'pressureName':       'PRESS',
        'tracerNames':        ['BA'],
        'sinksSources':       ['0'],
        'massSF':             True,
        'FourierTransform':   False,
        'Waves':              ['1'],
    }
