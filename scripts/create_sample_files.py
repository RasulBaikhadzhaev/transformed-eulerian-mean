"""
Generate synthetic sample .nc files for integration testing.

Produces 18 files in tests/data/:
  era5_sample_000101HH.nc ... era5_sample_000103HH.nc  — ERA5-like met data
      30 hybrid levels, 57 lat, 108 lon; 12 snapshots (00/06/12/18 UTC × Jan 1/2/3)
  clams_theta_sample_00010112.nc / ...0212.nc / ...0312.nc  — CLAMS-like theta-coord tracer
      24 theta levels, 30 lat, 36 lon; 3 daily snapshots (Jan 1/2/3)
  clams_press_sample_00010112.nc / ...0212.nc / ...0312.nc  — CLAMS-like pressure-coord tracer
      24 pressure levels, 30 lat, 36 lon; 3 daily snapshots (Jan 1/2/3)

Run with:
    cd /home/rb/Pixi_folders/git/TEM_pkg
    pixi run python tests/create_sample_files.py
"""

from pathlib import Path
import numpy as np
import xarray as xr

_SAMPLE_ROOT = Path(__file__).parent / "data" / "sample_input"
ERA5_OUT_DIR        = _SAMPLE_ROOT / "ERA5"
CLAMS_THETA_OUT_DIR = _SAMPLE_ROOT / "CLaMS_Theta"
CLAMS_PRESS_OUT_DIR = _SAMPLE_ROOT / "CLaMS_Press"
for _d in (ERA5_OUT_DIR, CLAMS_THETA_OUT_DIR, CLAMS_PRESS_OUT_DIR):
    _d.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dimensions  (~3x the previous single-file samples)
# ---------------------------------------------------------------------------
NHYBRID  = 30   # hybrid levels (was 10; real: 137)
NLAT_ERA = 57   # latitude  (was 19; real: 181)
NLON_ERA = 108  # longitude (was 36; real: 360)

NTHETA   = 24   # theta levels  (was 8;  real: 31)
NPRESS   = 24   # pressure levels (was 8; real: 39)
NLAT_CL  = 30   # CLAMS latitude (was 10; real: 91)
NLON_CL  = 36   # CLAMS longitude (was 12; real: 121)

# ---------------------------------------------------------------------------
# Coordinate arrays
# ---------------------------------------------------------------------------
lats_era  = np.linspace(90.0, -90.0, NLAT_ERA).astype("float32")
lons_era  = np.linspace(0.0, 356.7, NLON_ERA).astype("float32")
hybrid    = np.linspace(0.999, 0.001, NHYBRID).astype("float32")

lats_cl   = np.linspace(-90.0, 90.0, NLAT_CL).astype("float32")
lons_cl   = np.linspace(0.0, 357.0, NLON_CL).astype("float32")

# 24 theta levels spanning the real range [280, 2600] K
theta_levs = np.array([
    280, 290, 300, 310, 320, 330, 340, 350, 360, 370,
    380, 400, 420, 450, 475, 500, 550, 600, 700, 800,
    900, 1000, 1250, 1500,
], dtype="float32")

# 24 pressure levels spanning [1000, 0.1] hPa
press_levs = np.array([
    1000, 900, 850, 681, 562, 464, 383, 316, 261, 215,
     178, 147, 121, 100,  90,  75,  68,  56,  46,  38,
      26,  15,  10,   1,
], dtype="float32")

# ---------------------------------------------------------------------------
# ERA5-like files  (12 snapshots: 00/06/12/18 UTC on 2000-01-01/02/03)
# ---------------------------------------------------------------------------
# Timestamps encode as YYMMDDHH: 00=2000, 01=Jan, DD=day, HH=hour
ERA5_STAMPS = [
    ("00010100", "2000-01-01T00:00:00"),
    ("00010106", "2000-01-01T06:00:00"),
    ("00010112", "2000-01-01T12:00:00"),
    ("00010118", "2000-01-01T18:00:00"),
    ("00010200", "2000-01-02T00:00:00"),
    ("00010206", "2000-01-02T06:00:00"),
    ("00010212", "2000-01-02T12:00:00"),
    ("00010218", "2000-01-02T18:00:00"),
    ("00010300", "2000-01-03T00:00:00"),
    ("00010306", "2000-01-03T06:00:00"),
    ("00010312", "2000-01-03T12:00:00"),
    ("00010318", "2000-01-03T18:00:00"),
]

def _era5_dataset(stamp_str, seed):
    rng = np.random.default_rng(seed)
    shape = (NHYBRID, NLAT_ERA, NLON_ERA)

    # Pressure = hybrid * ps   (simplified sigma)
    ps = 1013.0
    press_3d = (hybrid[:, None, None] * ps * np.ones(shape)).astype("float32")

    # Potential temperature  θ = T * (1000/p)^(2/7),  T ≈ 250 K
    theta_3d = (250.0 * (1000.0 / press_3d) ** (2.0 / 7.0)).astype("float32")

    # Zonal wind: jet + eddy
    U = (20.0
         + 5.0 * np.cos(np.pi * lats_era[None, :, None] / 180.0)
         + 2.0 * np.sin(2 * np.pi * lons_era[None, None, :] / 360.0)
         + 0.5 * rng.standard_normal(shape)).astype("float32")

    # Meridional wind: Hadley-like cell + noise
    V = (1.0 * np.sin(np.pi * lats_era[None, :, None] / 90.0) * np.ones(shape)
         + 0.3 * rng.standard_normal(shape)).astype("float32")

    # Omega (Pa/s)
    OMEGA = (0.05 * rng.standard_normal(shape)).astype("float32")

    # Theta-dot (K/day): diabatic heating
    THETA_DOT = (0.2 * rng.standard_normal(shape)).astype("float32")

    # Temperature (K): T = θ * (p/1000)^(2/7)
    TEMP = (theta_3d * (press_3d / 1000.0) ** (2.0 / 7.0)).astype("float32")

    # Geopotential height (m²/s²): hydrostatic, roughly R*T/g * ln(p0/p)
    GPH = (287.0 * TEMP * np.log(1013.0 / np.maximum(press_3d, 0.01))).astype("float32")

    # Specific humidity (kg/kg): small positive values decreasing with altitude
    SH = (5e-3 * (press_3d / 1013.0) * np.exp(0.1 * rng.standard_normal(shape))).astype("float32")
    SH = np.clip(SH, 0.0, None)

    # 2D tropopause diagnostics (lat×lon)
    shape2d = (NLAT_ERA, NLON_ERA)
    TROP1_Z     = (12.0 + 3.0 * np.cos(np.pi * lats_era[:, None] / 90.0)
                   * np.ones(shape2d)).astype("float32")
    TROP1_TEMP  = (210.0 + 5.0 * rng.standard_normal(shape2d)).astype("float32")
    TROP1_PRESS = (200.0 + 50.0 * np.cos(np.pi * lats_era[:, None] / 90.0)
                   * np.ones(shape2d)).astype("float32")
    TROP1_THETA = (330.0 + 10.0 * rng.standard_normal(shape2d)).astype("float32")

    return xr.Dataset(
        {
            "THETA":         (["hybrid", "lat", "lon"], theta_3d,  {"units": "K"}),
            "PRESS":         (["hybrid", "lat", "lon"], press_3d,  {"units": "hPa"}),
            "U":             (["hybrid", "lat", "lon"], U,          {"units": "m s**-1"}),
            "V":             (["hybrid", "lat", "lon"], V,          {"units": "m s**-1"}),
            "OMEGA":         (["hybrid", "lat", "lon"], OMEGA,      {"units": "Pa s**-1"}),
            "THETA_DOT_TOT": (["hybrid", "lat", "lon"], THETA_DOT,  {"units": "K/day"}),
            "TEMP":          (["hybrid", "lat", "lon"], TEMP,       {"units": "K"}),
            "GPH":           (["hybrid", "lat", "lon"], GPH,        {"units": "m**2 s**-2"}),
            "SH":            (["hybrid", "lat", "lon"], SH,         {"units": "kg kg**-1"}),
            "TROP1_Z":       (["lat", "lon"],           TROP1_Z,    {"units": "km"}),
            "TROP1_TEMP":    (["lat", "lon"],           TROP1_TEMP, {"units": "K"}),
            "TROP1_PRESS":   (["lat", "lon"],           TROP1_PRESS,{"units": "hPa"}),
            "TROP1_THETA":   (["lat", "lon"],           TROP1_THETA,{"units": "K"}),
        },
        coords={
            "hybrid": (["hybrid"], hybrid,   {"units": "1", "long_name": "hybrid levels", "axis": "Z"}),
            "lat":    (["lat"],    lats_era, {"units": "degrees_N", "long_name": "Latitude"}),
            "lon":    (["lon"],    lons_era, {"units": "degrees_E", "long_name": "Longitude"}),
        },
    )

for i, (stamp, _) in enumerate(ERA5_STAMPS):
    ds = _era5_dataset(stamp, seed=i)
    path = ERA5_OUT_DIR / f"era5_sample_{stamp}.nc"
    ds.to_netcdf(path)
    print(f"Written ERA5/{path.name}  ({path.stat().st_size // 1024} kB)")

# ---------------------------------------------------------------------------
# CLAMS theta files  (3 daily snapshots: Jan 1/2/3 at 12 UTC)
# ---------------------------------------------------------------------------
CLAMS_THETA_STAMPS = [("00010112", "2000-01-01T12:00:00"),
                      ("00010212", "2000-01-02T12:00:00"),
                      ("00010312", "2000-01-03T12:00:00")]

def _clams_theta_dataset(stamp_str, seed):
    rng = np.random.default_rng(seed)
    shape = (NTHETA, NLAT_CL, NLON_CL)

    # Pressure decreases with theta (isentropic)
    press = (1000.0 * np.exp(-theta_levs[:, None, None] / 700.0)
             * np.ones(shape)).astype("float32")

    # BA age-of-air: ~0.5 yr at lowest theta, ~6 yr at highest
    ba = (0.5 + 5.5 * (theta_levs[:, None, None] - theta_levs[0])
          / (theta_levs[-1] - theta_levs[0]) * np.ones(shape)
          + 0.05 * rng.standard_normal(shape)).astype("float32")
    ba = np.clip(ba, 0.01, None)

    V = (0.5 * np.sin(np.pi * lats_cl[None, :, None] / 90.0)
         * np.ones(shape)
         + 0.1 * rng.standard_normal(shape)).astype("float32")

    return xr.Dataset(
        {
            "BA":    (["theta", "lat", "lon"], ba,    {"units": "yr",
                      "long_name": "Mean age clock tracer mixing ratio"}),
            "PRESS": (["theta", "lat", "lon"], press, {"units": "hPa"}),
            "V":     (["theta", "lat", "lon"], V,     {"units": "m/s"}),
        },
        coords={
            "theta": (["theta"], theta_levs, {"units": "K",     "long_name": "Potential temperature"}),
            "lat":   (["lat"],   lats_cl,    {"units": "deg N", "long_name": "Latitude"}),
            "lon":   (["lon"],   lons_cl,    {"units": "deg E", "long_name": "Longitude"}),
        },
    )

for i, (stamp, _) in enumerate(CLAMS_THETA_STAMPS):
    ds = _clams_theta_dataset(stamp, seed=100 + i)
    path = CLAMS_THETA_OUT_DIR / f"clams_theta_sample_{stamp}.nc"
    ds.to_netcdf(path)
    print(f"Written CLaMS_Theta/{path.name}  ({path.stat().st_size // 1024} kB)")

# ---------------------------------------------------------------------------
# CLAMS pressure files  (3 daily snapshots: Jan 1/2/3 at 12 UTC)
# ---------------------------------------------------------------------------
CLAMS_PRESS_STAMPS = [("00010112", "2000-01-01T12:00:00"),
                      ("00010212", "2000-01-02T12:00:00"),
                      ("00010312", "2000-01-03T12:00:00")]

def _clams_press_dataset(stamp_str, seed):
    rng = np.random.default_rng(seed)
    shape = (NPRESS, NLAT_CL, NLON_CL)

    press = (press_levs[:, None, None] * np.ones(shape)).astype("float32")

    # BA age-of-air: ~0.5 yr near surface (1000 hPa), ~6 yr at top (1 hPa)
    ba = (0.5 + 5.5 * (np.log(press_levs[0] / press_levs)[:, None, None]
          / np.log(press_levs[0] / press_levs[-1])) * np.ones(shape)
          + 0.05 * rng.standard_normal(shape)).astype("float32")
    ba = np.clip(ba, 0.01, None)

    V = (0.5 * np.sin(np.pi * lats_cl[None, :, None] / 90.0)
         * np.ones(shape)
         + 0.1 * rng.standard_normal(shape)).astype("float32")

    return xr.Dataset(
        {
            "BA":    (["press", "lat", "lon"], ba,    {"units": "yr",
                      "long_name": "Mean age clock tracer mixing ratio"}),
            "PRESS": (["press", "lat", "lon"], press, {"units": "hPa"}),
            "V":     (["press", "lat", "lon"], V,     {"units": "m/s"}),
        },
        coords={
            "press": (["press"], press_levs, {"units": "hPa",   "long_name": "Pressure"}),
            "lat":   (["lat"],   lats_cl,    {"units": "deg N", "long_name": "Latitude"}),
            "lon":   (["lon"],   lons_cl,    {"units": "deg E", "long_name": "Longitude"}),
        },
    )

for i, (stamp, _) in enumerate(CLAMS_PRESS_STAMPS):
    ds = _clams_press_dataset(stamp, seed=200 + i)
    path = CLAMS_PRESS_OUT_DIR / f"clams_press_sample_{stamp}.nc"
    ds.to_netcdf(path)
    print(f"Written CLaMS_Press/{path.name}  ({path.stat().st_size // 1024} kB)")

print(f"\nDone. 18 files written to {_SAMPLE_ROOT}")
