from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import toml
from metpy.units import units


def addRatioUnits() -> None:
    """
    Register custom atmospheric mixing-ratio units into the MetPy unit registry.

    Defines volumetric (ppmv, ppbv, pptv), mass (ppmm, ppbm, pptm), and
    dimensionless fraction (frac/fraction) units if they are not already present.
    Safe to call multiple times; skips definitions that already exist.
    """
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


def load_and_merge_config(parserArgs: Any) -> dict:
    """
    Load a TOML configuration file and override values with command-line arguments.

    Parameters
    ----------
    parserArgs : argparse.Namespace
        Parsed CLI arguments. ``parserArgs.configFile`` must point to a valid
        TOML file. Any other attribute whose value is not ``'from config file'``
        overwrites the corresponding key in the loaded config dict.

    Returns
    -------
    dict
        Merged configuration mapping.
    """
    config = toml.load(open(parserArgs.configFile, 'r'))
    for arg, value in vars(parserArgs).items():
        if value != 'from config file' and arg != 'configFile':
            config[arg] = value
    return config


def format_seconds(seconds: float) -> str:
    """
    Convert a raw second count into a human-readable string.

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds. Negative values return ``'calculating...'``.

    Returns
    -------
    str
        String of the form ``'1h 4m 7s'``, omitting leading zero components
        (e.g. ``'4m 7s'`` when hours are zero). Always includes seconds.
    """
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


def init_spinner(stop_event: Any, timeStart: float) -> None:
    """
    Print an animated 'Initialisation...' label until *stop_event* is set.

    Intended to run in a daemon thread while file collection and path-matching
    are in progress (before the multiprocessing pool starts).

    Parameters
    ----------
    stop_event : threading.Event
        Set this event from the main thread to stop the spinner.
    timeStart : float
        Start time from ``time.time()``, used to compute elapsed time.
    """
    spinner = ['.  ', '.. ', '...']
    spin_idx = 0
    while not stop_event.is_set():
        elapsed_str = format_seconds(time.time() - timeStart)
        sys.stdout.write(f"\rInitialisation{spinner[spin_idx % 3]} | Time elapsed: {elapsed_str}\033[K")
        sys.stdout.flush()
        spin_idx += 1
        time.sleep(1/3)


def progress_reporter(counter: Any, totalN: int, timeStart: float) -> None:
    """
    Print a CLI progress bar to stdout, updating every second until done.

    Intended to run in a daemon thread alongside a multiprocessing pool.
    Reads ``counter.value`` (a ``multiprocessing.Value('i', ...)`` shared
    integer) and terminates once it reaches *totalN*.

    Parameters
    ----------
    counter : multiprocessing.Value
        Shared integer incremented by worker processes after each file.
    totalN : int
        Total number of files to process.
    timeStart : float
        Start time from ``time.time()``, used to compute elapsed time.
    """
    dots = ['.  ', '.. ', '...']
    dot_idx = 0
    while True:
        currentN = counter.value
        elapsed_str = format_seconds(time.time() - timeStart)
        pct = f"\033[1m{(currentN/totalN)*100:4.2f}%\033[0m"

        sys.stdout.write(
            f"\rProcessing{dots[dot_idx % 3]} | {currentN}/{totalN} files {pct} | "
            f"Time elapsed: {elapsed_str}\033[K"
        )
        sys.stdout.flush()
        dot_idx += 1

        if currentN >= totalN:
            sys.stdout.write("\n")
            sys.stdout.flush()
            break

        time.sleep(1/3)


def binData(dataset: Any, binningLat: int, binningLon: int) -> Any:
    """
    Downsample a dataset by applying a block average over latitude and longitude.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with ``lat`` and ``lon`` dimensions.
    binningLat : int
        Number of latitude grid points to average into one.
    binningLon : int
        Number of longitude grid points to average into one.

    Returns
    -------
    xr.Dataset
        Dataset with reduced spatial resolution. Grid cells that do not fill a
        complete bin are trimmed (``boundary='trim'``).
    """
    return dataset.coarsen(lat=binningLat, boundary='trim').mean().coarsen(lon=binningLon, boundary='trim').mean()


def nanGradient1D(y1d: np.ndarray, x1d: np.ndarray) -> np.ndarray:
    """
    Compute the finite-difference gradient of a 1-D array, skipping NaN values.

    Parameters
    ----------
    y1d : np.ndarray, shape (N,)
        Data values; may contain NaNs.
    x1d : np.ndarray, shape (N,)
        Coordinate values corresponding to *y1d*.

    Returns
    -------
    np.ndarray, shape (N,)
        Gradient array. Positions where *y1d* was NaN remain NaN; positions
        where fewer than two valid neighbours exist are also NaN.
    """
    valid_mask = ~np.isnan(y1d)
    valid_y = y1d[valid_mask]

    if valid_y.size <= 1:
        return np.full_like(y1d, np.nan)

    valid_x = x1d[valid_mask]
    grad_valid = np.gradient(valid_y, valid_x)

    out = np.full_like(y1d, np.nan)
    out[valid_mask] = grad_valid
    return out


def nanGradient(y_data: Any, x_data: Any, axis: int = 0) -> Any:
    """
    Compute a finite-difference derivative along one axis, handling NaN values.

    ``np.gradient`` propagates NaNs; this function isolates valid points per
    slice along *axis* and computes the gradient only on those, leaving NaN
    positions as NaN in the output.

    Parameters
    ----------
    y_data : array-like or pint Quantity, shape (...,)
        Data to differentiate. May contain NaNs.
    x_data : array-like or pint Quantity, 1-D
        Coordinate values along *axis*.
    axis : int, optional
        Axis along which to differentiate (default 0).

    Returns
    -------
    array-like or pint Quantity
        Derivative with the same shape as *y_data*. If both *y_data* and
        *x_data* carry units the result has units ``y_data.units / x_data.units``.
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


def is_equal_or_shorter_than_month(freq: Any) -> bool:
    """
    Return True if *freq* represents a time interval of one month or shorter.

    Accepts pandas offset aliases (e.g. ``'MS'``, ``'3H'``) and timedelta
    strings (e.g. ``'7 days'``). Monthly aliases (``'MS'``, ``'M'``, etc.) are
    treated as exactly one month. Unknown strings return False.

    Parameters
    ----------
    freq : str or pandas offset
        Frequency to test.

    Returns
    -------
    bool
    """
    f = str(freq).strip()
    f_upper = f.upper()

    month_codes = ['MS', 'ME', 'M', 'BMS', 'BME', 'BM']
    if f_upper in month_codes or 'MONTH' in f_upper:
        return True

    try:
        duration = pd.to_timedelta(f)
    except ValueError:
        try:
            duration = pd.to_timedelta('1' + f)
        except ValueError:
            return False

    return duration <= pd.Timedelta(days=31)


def is_equal_or_shorter_than_day(freq: Any) -> bool:
    """
    Return True if *freq* represents a time interval of one day or shorter.

    Monthly, weekly, quarterly, and annual aliases return False. Daily aliases
    (``'D'``, ``'B'``) return True. For numeric timedelta strings the duration
    is compared against 24 hours. Unknown strings return False.

    Parameters
    ----------
    freq : str or pandas offset
        Frequency to test.

    Returns
    -------
    bool
    """
    f = str(freq).strip()
    f_upper = f.upper()

    long_codes = ['MS', 'ME', 'M', 'W', 'Q', 'Y', 'A']
    if any(code == f_upper for code in long_codes) or 'MONTH' in f_upper:
        return False

    if f_upper in ['D', 'B']:
        return True

    try:
        duration = pd.to_timedelta(f)
    except ValueError:
        try:
            duration = pd.to_timedelta('1' + f)
        except ValueError:
            return False
    return duration <= pd.Timedelta(days=1)


def apply_waves_banding(spec: np.ndarray, waves_config: list) -> np.ndarray:
    """
    Reduce a full per-wavenumber array to the bands listed in *waves_config*.

    A single string ``'k'`` selects wavenumber k; ``'k1-k2'`` sums k1 through
    k2 inclusive; ``'k-end'`` sums from k to the last available wavenumber.
    Wavenumber 0 (zonal mean) must already be excluded from *spec* (i.e. the
    first axis position corresponds to k=1).

    Parameters
    ----------
    spec : ndarray, shape (..., N_wn)
        Full per-wavenumber array (last axis = wavenumbers 1 … N_wn).
    waves_config : list of str
        Band descriptors, e.g. ``['1', '2', '6-10', '21-end']``.

    Returns
    -------
    banded : ndarray, shape (..., len(waves_config))
    """
    n_bands = len(waves_config)
    out_shape = spec.shape[:-1] + (n_bands,)
    banded = np.zeros(out_shape, dtype=spec.dtype)
    for i, wave in enumerate(waves_config):
        if '-' not in wave:
            banded[..., i] = spec[..., int(wave) - 1]
        elif 'end' not in wave:
            k1, k2 = int(wave.split('-')[0]), int(wave.split('-')[1])
            banded[..., i] = np.nansum(spec[..., k1 - 1: k2], axis=-1)
        else:
            banded[..., i] = np.nansum(spec[..., int(wave.split('-')[0]) - 1:], axis=-1)
    return banded
