import sys

import numpy as np
import xarray as xr
from metpy.units import units
from metpy.interpolate import interpolate_1d
import metpy.interpolate as ip

from .constants import H, P0


def _build_interpolated_dataset(dataset, variables, target_coord, out_coord_values,
                                 interp_target, source_coord, lat_dim, lon_dim,
                                 interp_fn, copy_attrs=True, cast_single=True):
    """
    Interpolate *variables* from *dataset* and return an xr.Dataset at *out_coord_values*.

    target_coord    : str   — output coordinate name ('alt' or 'theta')
    out_coord_values: array — values stored in the output coordinate (e.g. km levels)
    interp_target   : array — target passed to interp_fn (may differ from out_coord_values, e.g. log-pressure)
    source_coord    : array — source coordinate passed to interp_fn
    interp_fn       : callable(interp_target, source_coord, data, axis=0)
    copy_attrs      : if True copy full attrs dict; if False copy only units
    cast_single     : if True store data as float32 (np.single); keep original dtype otherwise
    """
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


def alt2press(x):
    """Converts log-pressure altitude to pressure using the scale height H."""
    return P0 * np.exp(-x / H)


def press2alt(x):
    """Converts pressure to log-pressure altitude using the scale height H."""
    return -H * np.log(x / P0)


def interpolateToLogPressure(dataset, reqVars, vertDimType, targetLevels, vertDimName, latDimName,
                                lonDimName, pressureVarName='', saveInterpolatedZonalMeanVars=[], saveZonalMeanVars=[]):
    '''
    Interpolates a dataset to log-pressure altitude coordinates.

    If vertDimType is 'pressure', the pressure dimension is converted to
    log-pressure altitude and interpolated to targetLevels.  If 'other', a
    3-D pressure variable is used for the interpolation.  If 'log-pressure',
    the dimensions are renamed and the data is optionally re-gridded.
    '''
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


def interpolateToTheta(dataset, reqVarsWithTracers, tomlConfig):
    """
    Interpolates an existing dataset to theta (potential temperature) levels.
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


def interpolateToThetaAndCombineData(tracerDataset, metDataset, reqVars, tomlConfig):
    """
    Interpolates (if necessary) both tracer and met datasets to a common
    potential temperature (theta) vertical grid and merges them.
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


def interpolateToPressureAndCombineData(tracerDataset, metDataset, reqVars, tomlConfig):
    """
    Interpolates both tracer and met datasets to a common log-pressure altitude
    grid and merges them.
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
