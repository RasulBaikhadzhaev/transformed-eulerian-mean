from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import metpy.interpolate as ip
import numpy as np
import xarray as xr
from metpy.interpolate import interpolate_1d
from metpy.units import units

from .constants import P0, H


def _interp1d_3d(target_1d: np.ndarray, source_3d: np.ndarray, *data_arrays: np.ndarray) -> list[np.ndarray]:
    """Linear interpolation of multiple 3-D arrays along axis 0, sharing weights.

    Computes binary-search indices and linear weights once from source_3d and
    applies them to every data array, avoiding the per-variable overhead of
    calling MetPy interpolate_1d in a loop.

    Parameters
    ----------
    target_1d  : (T,) monotonically increasing target coordinate values
    source_3d  : (S, H, W) source coordinate field (will be sorted if decreasing at [0,0])
    *data_arrays: (S, H, W) data arrays to interpolate, in the same coordinate order

    Returns
    -------
    list of (T, H, W) float32 arrays, one per input data array; out-of-bounds set to NaN
    """
    S, H, W = source_3d.shape
    cols = H * W

    # Ensure source increases along axis 0 (e.g. log-pressure may decrease)
    if source_3d[0, 0, 0] > source_3d[-1, 0, 0]:
        source_3d = source_3d[::-1]
        data_arrays = tuple(d[::-1] for d in data_arrays)

    src = source_3d.reshape(S, cols)  # (S, cols)
    dats = [d.reshape(S, cols).astype('f4') for d in data_arrays]
    col_arange = np.arange(cols)

    src_min = src[0]   # (cols,) — monotone increasing → first row is minimum
    src_max = src[-1]  # (cols,)

    T = len(target_1d)
    out_arrays = [np.empty((T, cols), dtype='f4') for _ in data_arrays]

    for ti in range(T):
        t_val = float(target_1d[ti])
        oob = (t_val < src_min) | (t_val > src_max)

        # Last source index <= t_val for each column
        idx = (src <= t_val).sum(axis=0) - 1
        idx = np.clip(idx, 0, S - 2)

        lo_src = src[idx, col_arange]
        hi_src = src[idx + 1, col_arange]
        denom = hi_src - lo_src
        with np.errstate(divide='ignore', invalid='ignore'):
            wt = np.where(denom != 0, (t_val - lo_src) / denom, 0.5)

        for j, dat in enumerate(dats):
            row = dat[idx, col_arange] + wt * (dat[idx + 1, col_arange] - dat[idx, col_arange])
            row[oob] = np.nan
            out_arrays[j][ti] = row.astype('f4')

    return [a.reshape(T, H, W) for a in out_arrays]


def _build_interpolated_dataset(dataset: xr.Dataset, variables: list[str], target_coord: str, out_coord_values: Any,
                                 interp_target: Any, source_coord: Any, lat_dim: str, lon_dim: str,
                                 interp_fn: Callable, copy_attrs: bool = True, cast_single: bool = True) -> xr.Dataset:
    """
    Interpolate variables from a dataset onto new vertical coordinate values.

    When *source_coord* is 3-D the fast vectorised :func:`_interp1d_3d` path
    is used; when it is 1-D the function falls back to MetPy ``interpolate_1d``.

    Parameters
    ----------
    dataset : xr.Dataset
        Source dataset containing *variables*.
    variables : list of str
        Variable names to interpolate.
    target_coord : str
        Output coordinate name (``'alt'`` or ``'theta'``).
    out_coord_values : array-like or pint Quantity
        Values stored in the output coordinate (e.g. km levels).
    interp_target : array-like or pint Quantity
        Target values passed to the interpolation function. May differ from
        *out_coord_values* (e.g. log-pressure targets when output is in km).
    source_coord : array-like or pint Quantity
        Source coordinate array. Shape ``(S, H, W)`` triggers the 3-D fast
        path; shape ``(S,)`` triggers the MetPy fallback.
    lat_dim : str
        Name of the latitude dimension in *dataset*.
    lon_dim : str
        Name of the longitude dimension in *dataset*.
    interp_fn : Callable
        Interpolation function kept for API compatibility; only used in the
        1-D fallback path.
    copy_attrs : bool
        If True copy the full ``attrs`` dict; if False copy only ``'units'``.
    cast_single : bool
        If True store output as float32; keep original dtype otherwise.

    Returns
    -------
    xr.Dataset
        Dataset on the new vertical grid with coordinates
        (*target_coord*, ``'lat'``, ``'lon'``).
    """
    raw_data = [np.single(np.array(dataset[v])) for v in variables]
    src = source_coord.magnitude if hasattr(source_coord, 'magnitude') else source_coord
    tgt_raw = interp_target.magnitude if hasattr(interp_target, 'magnitude') else interp_target

    if src.ndim == 3:
        tgt_1d = np.asarray(tgt_raw).ravel()
        interped_arrays = _interp1d_3d(tgt_1d, src, *raw_data)
        target_data = {v: (np.single(interped_arrays[i]) if cast_single else interped_arrays[i])
                       for i, v in enumerate(variables)}
    else:
        # 1-D source coordinate: fall back to MetPy for non-standard cases
        inData = {v: np.single(np.array(dataset[v])) * getattr(units, dataset[v].units)
                  for v in variables}
        interped = [interp_fn(interp_target, source_coord, inData[v], axis=0) for v in variables]
        target_data = {v: (np.single(interped[i]) if cast_single else interped[i]) for i, v in enumerate(variables)}

    coord_vals = out_coord_values.magnitude if hasattr(out_coord_values, 'units') else out_coord_values
    coord_units = str(out_coord_values.units) if hasattr(out_coord_values, 'units') else 'km'
    ds = xr.Dataset(coords={target_coord: coord_vals, 'lat': dataset[lat_dim], 'lon': dataset[lon_dim]})
    ds[target_coord].attrs['units'] = coord_units
    for v in variables:
        ds[v] = ((target_coord, 'lat', 'lon'), target_data[v])
        ds[v].attrs = dataset[v].attrs if copy_attrs else {'units': dataset[v].units}
    return ds


def alt2press(x: Any) -> Any:
    """
    Convert log-pressure altitude to pressure.

    Uses ``P = P0 * exp(-z / H)`` with ``H = 7 km``, ``P0 = 1000 hPa``.

    Parameters
    ----------
    x : pint Quantity or array-like
        Log-pressure altitude in km.

    Returns
    -------
    pint Quantity
        Pressure in the same unit system as P0 (hPa).
    """
    return P0 * np.exp(-x / H)


def press2alt(x: Any) -> Any:
    """
    Convert pressure to log-pressure altitude.

    Uses ``z = -H * ln(p / P0)`` with ``H = 7 km``, ``P0 = 1000 hPa``.

    Parameters
    ----------
    x : pint Quantity or array-like
        Pressure in hPa.

    Returns
    -------
    pint Quantity
        Log-pressure altitude in km.
    """
    return -H * np.log(x / P0)


def interpolateToLogPressure(dataset: xr.Dataset, reqVars: list[str], vertDimType: str, targetLevels: Any, vertDimName: str, latDimName: str,
                                lonDimName: str, pressureVarName: str = '', saveInterpolatedZonalMeanVars: list[str] = [], saveZonalMeanVars: list[str] = []) -> xr.Dataset:
    """
    Interpolate a dataset to log-pressure altitude coordinates.

    Three input coordinate types are supported:

    - ``'pressure'`` — the vertical dimension contains pressure values (hPa);
      converted to log-pressure altitude and interpolated to *targetLevels*.
    - ``'other'`` — a 3-D pressure variable (*pressureVarName*) provides the
      source coordinate for interpolation to *targetLevels*.
    - ``'log-pressure'`` — already in log-pressure altitude; dimensions are
      renamed and optionally re-gridded to *targetLevels*.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset.
    reqVars : list of str
        Variables to include in the output.
    vertDimType : str
        Input vertical coordinate type: ``'pressure'``, ``'other'``, or
        ``'log-pressure'``.
    targetLevels : array-like or ``'skip'``
        Target log-pressure altitude levels in km. Pass ``'skip'`` to keep
        the original grid.
    vertDimName : str
        Name of the vertical dimension in *dataset*.
    latDimName : str
        Name of the latitude dimension in *dataset*.
    lonDimName : str
        Name of the longitude dimension in *dataset*.
    pressureVarName : str
        Name of the 3-D pressure variable; required when *vertDimType* is
        ``'other'``.
    saveInterpolatedZonalMeanVars : list of str
        Extra variables to interpolate and then zonal-average.
    saveZonalMeanVars : list of str
        Extra variables to zonal-average without interpolation.

    Returns
    -------
    xr.Dataset
        Dataset on a log-pressure altitude grid with dimensions
        (``'alt'``, ``'lat'``, ``'lon'``).
    """
    if vertDimType == 'pressure':
        datasetLogPress = dataset.rename({vertDimName: 'alt'}).assign_coords({'alt':
            press2alt(np.array(dataset[vertDimName]) * units('hPa'))})
        datasetLogPress['alt'].attrs['long_name'] = 'Log pressure altitude'
        datasetLogPress['alt'].attrs['units'] = 'km'
        datasetLogPress = datasetLogPress.rename({latDimName: 'lat', lonDimName: 'lon'})
        if targetLevels != 'skip':
            datasetLogPress = datasetLogPress.interp(alt=targetLevels, method='linear')

    elif vertDimType == 'other':
        pressureLevels = alt2press(targetLevels * units.km).to('hPa')
        LNPressTarget = np.log(pressureLevels.magnitude)[:, np.newaxis, np.newaxis]

        vert = np.array(dataset[pressureVarName]) * units(dataset[pressureVarName].units)
        LNPressInput = np.log(vert.to('hPa').magnitude)

        datasetLogPress = _build_interpolated_dataset(
            dataset, reqVars + saveInterpolatedZonalMeanVars,
            'alt', targetLevels, LNPressTarget, LNPressInput, latDimName, lonDimName,
            interpolate_1d, cast_single=False,
        )

    elif vertDimType == 'log-pressure':
        datasetLogPress = dataset.rename({vertDimName: 'alt', latDimName: 'lat', lonDimName: 'lon'})
        if targetLevels != 'skip':
            datasetLogPress = datasetLogPress.interp(alt=targetLevels, method='linear')

    else:
        print(f"ERROR: verticalDimensionType parameter can only be set to 'other', 'pressure', or 'log-pressure'.\n\n"
              f"it is currently set to {vertDimType} please check the parameter in the configuration file")
        sys.exit(1)

    for variable in saveInterpolatedZonalMeanVars:
        datasetLogPress[variable] = datasetLogPress[variable].mean(dim='lon', keep_attrs=True)

    for variable in saveZonalMeanVars:
        datasetLogPress[variable] = dataset[variable].mean(dim='lon', keep_attrs=True)

    return datasetLogPress


def interpolateToTheta(dataset: xr.Dataset, reqVarsWithTracers: list[str], tomlConfig: dict) -> xr.Dataset:
    """
    Interpolate a dataset to potential temperature (theta) levels.

    Two input coordinate types are supported:

    - ``'other'`` — a 3-D theta variable provides the source coordinate;
      variables are interpolated to ``tomlConfig['targetLevels']``.
    - ``'theta'`` — already on theta levels; dimensions are renamed and
      optionally re-gridded to ``tomlConfig['targetLevels']``.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset on the original vertical grid.
    reqVarsWithTracers : list of str
        Variables (met + tracers) to include in the output.
    tomlConfig : dict
        Configuration dict; uses ``verticalDimensionType``, ``thetaName``,
        ``targetLevels``, ``vertDim``, ``latDim``, and ``lonDim`` keys.

    Returns
    -------
    xr.Dataset
        Dataset on a theta grid with dimensions
        (``'theta'``, ``'lat'``, ``'lon'``).
    """
    if tomlConfig['verticalDimensionType'] == 'other':
        thetaTargetLevels = tomlConfig['targetLevels'] * units.K
        theta_3D_input = (np.array(dataset[tomlConfig['thetaName']]) *
                          units(dataset[tomlConfig['thetaName']].units)).to('kelvin')

        datasetThetaLevels = _build_interpolated_dataset(
            dataset, reqVarsWithTracers,
            'theta', thetaTargetLevels, thetaTargetLevels, theta_3D_input,
            tomlConfig['latDim'], tomlConfig['lonDim'],
            ip.interpolate_1d,
        )

    elif tomlConfig['verticalDimensionType'] == 'theta':
        datasetThetaLevels = dataset.rename({tomlConfig['vertDim']: 'theta', tomlConfig['latDim']: 'lat', tomlConfig['lonDim']: 'lon'})
        if not isinstance(tomlConfig.get('targetLevels', 'skip'), str):
            datasetThetaLevels = datasetThetaLevels.interp(theta=tomlConfig['targetLevels'], method='linear')

    else:
        print(f"ERROR: verticalDimensionType parameter can only be set to 'other' or 'theta'.\n\n"
              f"it is currently set to {tomlConfig['verticalDimensionType']} please check the parameter in the configuration file")
        sys.exit(1)

    return datasetThetaLevels


def interpolateToThetaAndCombineData(tracerDataset: xr.Dataset, metDataset: xr.Dataset, reqVars: list[str], tomlConfig: dict) -> xr.Dataset:
    """
    Interpolate tracer and met datasets to a common theta grid and merge them.

    Handles all combinations of ``tracerVerticalDimensionType`` (``'other'``
    or ``'theta'``) and ``verticalDimensionType`` (``'other'`` or ``'theta'``).
    After interpolation the met dataset is re-gridded to the tracer theta
    levels with ``interp_like`` before merging.

    Parameters
    ----------
    tracerDataset : xr.Dataset
        Tracer dataset on its original vertical grid.
    metDataset : xr.Dataset
        Meteorological dataset on its original vertical grid.
    reqVars : list of str
        Met variables to include in the output.
    tomlConfig : dict
        Configuration dict with vertical coordinate settings for both tracer
        and met data.

    Returns
    -------
    xr.Dataset
        Merged float32 dataset on a common theta grid with dimensions
        (``'theta'``, ``'lat'``, ``'lon'``).
    """
    if tomlConfig['tracerVerticalDimensionType'] == 'other':
        _tracer_levels = (metDataset[tomlConfig['vertDim']].values if isinstance(tomlConfig.get('targetLevels', 'skip'), str)
                          else tomlConfig['targetLevels'])
        thetaTargetLevels = _tracer_levels * units.K
        theta_3D_input = (np.array(tracerDataset[tomlConfig['tracerThetaName']]) *
                          units(tracerDataset[tomlConfig['tracerThetaName']].units)).to('kelvin')

        tracerDatasetThetaLevels = _build_interpolated_dataset(
            tracerDataset, tomlConfig['tracerNames'],
            'theta', thetaTargetLevels, thetaTargetLevels, theta_3D_input,
            tomlConfig['tracerLatDim'], tomlConfig['tracerLonDim'],
            ip.interpolate_1d, copy_attrs=False,
        )

    elif tomlConfig['tracerVerticalDimensionType'] == 'theta':
        tracerDatasetThetaLevels = tracerDataset.rename({tomlConfig['tracerVertDim']: 'theta', tomlConfig['tracerLatDim']: 'lat',
                                                      tomlConfig['tracerLonDim']: 'lon'})
    else:
        print(f"ERROR: tracerVerticalDimensionType parameter can only be set to 'other' or 'theta'.\n\n"
              f"it is currently set to {tomlConfig['tracerVerticalDimensionType']} please check the parameter in the configuration file")
        sys.exit(1)

    if tomlConfig['verticalDimensionType'] == 'other':
        _met_levels = (tracerDatasetThetaLevels['theta'].values if isinstance(tomlConfig.get('targetLevels', 'skip'), str)
                       else tomlConfig['targetLevels'])
        thetaTargetLevels = _met_levels * units.K
        theta_3D_input = (np.array(metDataset[tomlConfig['thetaName']]) *
                          units(metDataset[tomlConfig['thetaName']].units)).to('kelvin')

        metDatasetThetaLevels = _build_interpolated_dataset(
            metDataset, reqVars,
            'theta', thetaTargetLevels, thetaTargetLevels, theta_3D_input,
            tomlConfig['latDim'], tomlConfig['lonDim'],
            ip.interpolate_1d,
        )

    elif tomlConfig['verticalDimensionType'] == 'theta':
        metDatasetThetaLevels = metDataset.rename({tomlConfig['vertDim']: 'theta', tomlConfig['latDim']: 'lat', tomlConfig['lonDim']: 'lon'})
        if not isinstance(tomlConfig.get('targetLevels', 'skip'), str):
            metDatasetThetaLevels = metDatasetThetaLevels.interp(theta=tomlConfig['targetLevels'], method='linear')

    else:
        print(f"ERROR: verticalDimensionType parameter can only be set to 'other' or 'theta'.\n\n"
              f"it is currently set to {tomlConfig['verticalDimensionType']} please check the parameter in the configuration file")
        sys.exit(1)

    metDatasetThetaLevels = metDatasetThetaLevels.interp_like(tracerDatasetThetaLevels)
    interpolatedDataset = xr.merge([tracerDatasetThetaLevels.astype('single'), metDatasetThetaLevels.astype('single')])

    return interpolatedDataset


def interpolateToPressureAndCombineData(tracerDataset: xr.Dataset, metDataset: xr.Dataset, reqVars: list[str], tomlConfig: dict) -> xr.Dataset:
    """
    Interpolate tracer and met datasets to a common log-pressure grid and merge.

    Handles all combinations of ``tracerVerticalDimensionType`` (``'pressure'``,
    ``'other'``, or ``'log-pressure'``) and ``verticalDimensionType``
    (``'pressure'``, ``'other'``, or ``'log-pressure'``). After interpolation
    the met dataset is re-gridded to the tracer altitude levels with
    ``interp_like`` before merging.

    Parameters
    ----------
    tracerDataset : xr.Dataset
        Tracer dataset on its original vertical grid.
    metDataset : xr.Dataset
        Meteorological dataset on its original vertical grid.
    reqVars : list of str
        Met variables to include in the output.
    tomlConfig : dict
        Configuration dict with vertical coordinate settings for both tracer
        and met data, and ``targetLevels`` (km) for the output grid.

    Returns
    -------
    xr.Dataset
        Merged float32 dataset on a common log-pressure altitude grid with
        dimensions (``'alt'``, ``'lat'``, ``'lon'``).
    """
    if tomlConfig['tracerVerticalDimensionType'] == 'pressure':
        tracerDatasetLogPress = tracerDataset.rename({tomlConfig['tracerVertDim']: 'alt'}).assign_coords({'alt':
            press2alt(np.array(tracerDataset[tomlConfig['tracerVertDim']]) * units('hPa'))})
        tracerDatasetLogPress['alt'].attrs['long_name'] = 'Log pressure altitude'
        tracerDatasetLogPress['alt'].attrs['units'] = 'km'
        tracerDatasetLogPress = tracerDatasetLogPress.rename({tomlConfig['tracerLatDim']: 'lat', tomlConfig['tracerLonDim']: 'lon'})

    elif tomlConfig['tracerVerticalDimensionType'] == 'other':
        pressureLevels = alt2press(tomlConfig['targetLevels'] * units.km).to('hPa')
        LNPressTarget = np.log(pressureLevels.magnitude)[:, np.newaxis, np.newaxis]

        vert = np.array(tracerDataset[tomlConfig['pressureName']]) * units(tracerDataset[tomlConfig['pressureName']].units)
        LNPressInput = np.log(vert.to('hPa').magnitude)

        tracerDatasetLogPress = _build_interpolated_dataset(
            tracerDataset, tomlConfig['tracerNames'],
            'alt', tomlConfig['targetLevels'], LNPressTarget, LNPressInput,
            tomlConfig['tracerLatDim'], tomlConfig['tracerLonDim'],
            ip.interpolate_1d, copy_attrs=False,
        )

    elif tomlConfig['tracerVerticalDimensionType'] == 'log-pressure':
        tracerDatasetLogPress = tracerDataset.rename({tomlConfig['tracerVertDim']: 'alt', tomlConfig['tracerLatDim']: 'lat',
                                                      tomlConfig['tracerLonDim']: 'lon'})
    else:
        print(f"ERROR: tracerVerticalDimensionType parameter can only be set to 'other', 'pressure', or 'log-pressure'.\n\n"
              f"it is currently set to {tomlConfig['tracerVerticalDimensionType']} please check the parameter in the configuration file")
        sys.exit(1)

    if tomlConfig['verticalDimensionType'] == 'pressure':
        metDatasetLogPress = metDataset.rename({tomlConfig['vertDim']: 'alt'}).assign_coords({'alt':
            press2alt(np.array(metDataset[tomlConfig['vertDim']]) * units('hPa'))})
        metDatasetLogPress['alt'].attrs['long_name'] = 'Log pressure altitude'
        metDatasetLogPress['alt'].attrs['units'] = 'km'
        metDatasetLogPress = metDatasetLogPress.rename({tomlConfig['latDim']: 'lat', tomlConfig['lonDim']: 'lon'})

    elif tomlConfig['verticalDimensionType'] == 'other':
        pressureLevels = alt2press(tomlConfig['targetLevels'] * units.km).to('hPa')
        LNPressTarget = np.log(pressureLevels.magnitude)[:, np.newaxis, np.newaxis]

        vert = np.array(metDataset[tomlConfig['pressureName']]) * units(metDataset[tomlConfig['pressureName']].units)
        LNPressInput = np.log(vert.to('hPa').magnitude)

        metDatasetLogPress = _build_interpolated_dataset(
            metDataset, reqVars,
            'alt', tomlConfig['targetLevels'], LNPressTarget, LNPressInput,
            tomlConfig['latDim'], tomlConfig['lonDim'],
            ip.interpolate_1d, cast_single=False,
        )

    elif tomlConfig['verticalDimensionType'] == 'log-pressure':
        metDatasetLogPress = metDataset.rename({tomlConfig['vertDim']: 'alt', tomlConfig['latDim']: 'lat', tomlConfig['lonDim']: 'lon'})

    else:
        print(f"ERROR: verticalDimensionType parameter can only be set to 'other', 'pressure', or 'log-pressure'.\n\n"
              f"it is currently set to {tomlConfig['verticalDimensionType']} please check the parameter in the configuration file")
        sys.exit(1)

    metDatasetLogPress = metDatasetLogPress.interp_like(tracerDatasetLogPress)
    interpolatedDataset = xr.merge([tracerDatasetLogPress.astype('single'), metDatasetLogPress.astype('single')])

    return interpolatedDataset
