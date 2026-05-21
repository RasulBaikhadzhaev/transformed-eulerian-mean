import sys
import time

import numpy as np
import pandas as pd
import toml
from metpy.units import units


def addRatioUnits():
    '''
    Registers custom atmospheric units (mixing ratios) into the MetPy unit registry.
    This allows for seamless unit conversion between ppmv, ppbv, and mass fractions.
    '''
    try:
        units.ppmv
    except Exception as e:
        if "ppmv" in str(e):
            units.define('ppmv = centim^3/m^3')
            units.define('ppbv = ppmv/1000')
            units.define('pptv = ppbv/1000')

    try:
        units.ppmm
    except Exception as e:
        if "ppmm" in str(e):
            units.define('ppmm = milligram/kilogram')
            units.define('ppbm = ppmm/1000')
            units.define('pptm = ppbm/1000')

    try:
        units.frac
    except Exception as e:
        if "frac" in str(e):
            units.define('frac = kilogram/kilogram')
            units.define('fraction = frac')


def load_and_merge_config(parserArgs):
    """
    Loads a TOML configuration file and overrides values with command-line arguments.
    """
    config = toml.load(open(parserArgs.configFile, 'r'))
    for arg, value in vars(parserArgs).items():
        if value != 'from config file' and arg != 'configFile':
            config[arg] = value
    return config


def format_seconds(seconds):
    """Converts a raw second count into a human-readable 'Hh Mm Ss' format."""
    if seconds < 0:
        return "calculating..."
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")

    return " ".join(parts) if parts else "0s"


def progress_reporter(counter, totalN, timeStart):
    """
    A CLI progress bar that runs in a separate thread to monitor processing.
    """
    while True:
        currentN = counter.value
        elapsed_str = format_seconds(time.time() - timeStart)

        sys.stdout.write(
            f"\rFiles processed: {currentN}/{totalN} ({(currentN/totalN)*100:4.2f}%) | "
            f"Time elapsed: {elapsed_str}\033[K"
        )
        sys.stdout.flush()

        if currentN >= totalN:
            sys.stdout.write("\n")
            sys.stdout.flush()
            break

        time.sleep(1)


def binData(dataset, binningLat, binningLon):
    '''
    Downsample the dataset by applying a spatial block average over latitude and longitude.
    '''
    return dataset.coarsen(lat=binningLat, boundary='trim').mean().coarsen(lon=binningLon, boundary='trim').mean()


def nanGradient1D(y1d, x1d):
    """Helper function to calculate gradient on a 1D slice, ignoring NaNs."""
    valid_mask = ~np.isnan(y1d)
    valid_y = y1d[valid_mask]

    if valid_y.size <= 1:
        return np.full_like(y1d, np.nan)

    valid_x = x1d[valid_mask]
    grad_valid = np.gradient(valid_y, valid_x)

    out = np.full_like(y1d, np.nan)
    out[valid_mask] = grad_valid
    return out


def nanGradient(y_data, x_data, axis=0):
    """
    Calculates the finite difference derivative while handling NaN values.
    Standard np.gradient fails if any NaNs are present; this function masks
    them and calculates the gradient on the remaining valid data.
    """
    y_mag = y_data.magnitude if hasattr(y_data, 'units') else y_data
    x_mag = x_data.magnitude if hasattr(x_data, 'units') else x_data

    if not np.isnan(y_mag).any():
        out_mag = np.gradient(y_mag, x_mag, axis=axis)
    else:
        out_mag = np.apply_along_axis(nanGradient1D, axis=axis, arr=y_mag, x1d=x_mag)

    if hasattr(y_data, 'units') and hasattr(x_data, 'units'):
        return out_mag * units(str(y_data.units / x_data.units))
    elif hasattr(y_data, 'units'):
        return out_mag * units(str(y_data.units))
    elif hasattr(x_data, 'units'):
        return out_mag * units(str(1 / x_data.units))

    return out_mag


def is_equal_or_shorter_than_month(freq):
    f = str(freq).strip()
    f_upper = f.upper()

    month_codes = ['MS', 'ME', 'M', 'BMS', 'BME', 'BM']
    if f_upper in month_codes or 'MONTH' in f_upper:
        return True

    try:
        duration = pd.to_timedelta(f)
    except:
        try:
            duration = pd.to_timedelta('1' + f)
        except:
            return False

    return duration <= pd.Timedelta(days=31)


def is_equal_or_shorter_than_day(freq):
    f = str(freq).strip()
    f_upper = f.upper()

    long_codes = ['MS', 'ME', 'M', 'W', 'Q', 'Y', 'A']
    if any(code == f_upper for code in long_codes) or 'MONTH' in f_upper:
        return False

    if f_upper in ['D', 'B']:
        return True

    try:
        duration = pd.to_timedelta(f)
    except:
        try:
            duration = pd.to_timedelta('1' + f)
        except:
            return False

    return duration <= pd.Timedelta(days=1)
