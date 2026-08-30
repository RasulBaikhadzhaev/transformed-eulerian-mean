from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def extractTimeFromFileNames(filesPaths: list, timeInfoInFileNames: str) -> pd.DatetimeIndex:
    """
    Parse timestamps from file names based on a user-provided template.

    Accepts ``'YYYY'`` (or ``'YY'`` if century is absent) for year, ``'MM'``
    for month, ``'DD'`` for day, ``'HH'`` for hour, ``'mm'`` for minute, and
    ``'ss'`` for second. At least the year position must be given; include all
    other available time fields. Only one ``'*'`` wildcard is allowed and it
    must be the first or last character; all non-time characters must be
    represented by ``'?'``.

    Parameters
    ----------
    filesPaths : list of Path
        File paths whose names contain embedded timestamps.
    timeInfoInFileNames : str
        Template string describing the position of time tokens in the filename.
        Examples: ``'*YYMMDDHH???'`` for ``era5_10120100.nc``;
        ``'mm?HH?MM?DD?YYYY*'`` for ``50_18_10_25_1990_data.nc``.

    Returns
    -------
    pd.DatetimeIndex
        Parsed timestamps in the same order as *filesPaths*.
    """
    if 'YYYY' in timeInfoInFileNames:
        yearString = 'YYYY'
    elif 'YY' in timeInfoInFileNames:
        yearString = 'YY'
    else:
        print("ERROR: Impossible to extract time information from file names.\n\n"
                "At least position of year (YYYY or YY) must be provided in config.")
        sys.exit(1)

    timeTokens = {'monthString': 'MM', 'dayString': 'DD', 'hourString': 'HH', 'minuteString': 'mm', 'secondString': 'ss'}
    n = len(filesPaths)

    if timeInfoInFileNames.startswith('*'):
        names = pd.Series([str(f) for f in filesPaths])

        def _extract(token: str) -> pd.Series:
            token_len = len(token)
            offset = len(timeInfoInFileNames) - timeInfoInFileNames.find(token) - token_len
            return names.str.slice(-(offset + token_len), -offset if offset > 0 else None)

    else:
        names = pd.Series([f.name for f in filesPaths])

        def _extract(token: str) -> pd.Series:
            start = timeInfoInFileNames.find(token)
            return names.str.slice(start, start + len(token))

    yearInfo = _extract(yearString)
    if yearString == 'YY':
        mask = yearInfo.astype(int) < 50
        yearInfo = mask.map({True: '20', False: '19'}) + yearInfo

    timeInfo = {
        key: _extract(token) if token in timeInfoInFileNames else pd.Series(['00'] * n)
        for key, token in timeTokens.items()
    }

    dateAndtime = pd.to_datetime(
        yearInfo + '-' + timeInfo['monthString'] + '-' + timeInfo['dayString'] + ' '
        + timeInfo['hourString'] + ':' + timeInfo['minuteString'] + ':' + timeInfo['secondString'],
        format='%Y-%m-%d %H:%M:%S'
    )

    return dateAndtime


def _load_and_filter_file_paths(inputDir: str, fileNames: str, timeInfoInFileNames: str,
                                dateStart: str, dateEnd: str, hoursToKeep: list, inputPathType: str) -> pd.DataFrame:
    """
    Shared file-discovery and filtering logic for both collectFileNames variants.

    Parameters
    ----------
    inputDir : str
        Directory path or path to a ``.txt`` file listing NetCDF paths.
    fileNames : str
        Glob pattern used with ``rglob`` when *inputPathType* is ``'directory'``.
    timeInfoInFileNames : str
        Template passed to :func:`extractTimeFromFileNames`.
    dateStart : str
        Exclude files before this date (``'YYYY-MM-DD-HH'`` or ``''``).
    dateEnd : str
        Exclude files after this date (``'YYYY-MM-DD-HH'`` or ``''``).
    hoursToKeep : list of int
        Restrict to these hours of the day; keep all hours if empty.
    inputPathType : str
        ``'directory'`` (rglob search) or ``'.txt'`` (read paths from file).

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``Path`` column and a DatetimeIndex, sorted
        ascending and filtered by date range and *hoursToKeep*.
    """
    if inputPathType in ('directory', 'Directory'):
        if not Path(inputDir).is_dir():
            print(f"ERROR: Directory '{inputDir}' does not exist.\n\n"
                  "Please check if the path is correct. There should be no spaces before or after the path")
            sys.exit(1)
        filesPaths = list(Path(inputDir).rglob(fileNames))
        if not filesPaths:
            print(f"ERROR: File '{fileNames}' does not exist in '{inputDir}'.\n\n"
                  "Please specify NetCDF filename which contains data for TEM calculations")
            sys.exit(1)
    elif inputPathType == '.txt':
        with open(inputDir) as txtFile:
            filesPaths = [Path(line.strip()) for line in txtFile]
        if not filesPaths:
            print("ERROR: No input files found. Please check if the input .txt file exists and\n\n"
                  "includes path(s) to existing NetCDF file(s) with data for TEM calculations")
            sys.exit(1)

    time = extractTimeFromFileNames(filesPaths, timeInfoInFileNames)
    filesPathAndTime = pd.DataFrame(data={'Path': filesPaths}, index=time)

    if dateStart != '' and dateEnd != '':
        filesPathAndTime = filesPathAndTime[(filesPathAndTime.index >= pd.to_datetime(dateStart)) &
                                            (filesPathAndTime.index <= pd.to_datetime(dateEnd))]
    elif dateStart != '':
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index >= pd.to_datetime(dateStart)]
    elif dateEnd != '':
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index <= pd.to_datetime(dateEnd)]

    if filesPathAndTime.empty:
        print(f"ERROR: After filtering by start and end date no suitable files '{fileNames}' exist in '{inputDir}'.\n\n"
              "Please adjust dates or specify NetCDF filename which contains data for TEM calculations")
        sys.exit(1)

    if hoursToKeep:
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index.hour.isin(hoursToKeep)]
    if filesPathAndTime.empty:
        print(f"ERROR: After filtering by hours to keep parameter no suitable files '{fileNames}' exist in '{inputDir}'.\n\n"
              "Please adjust hoursToKeep parameter in the configuration file")
        sys.exit(1)

    return filesPathAndTime.sort_index()


def collectFileNames(inputDir: str, fileNames: str, timeInfoInFileNames: str, outputDir: str = '', dateStart: str = '', dateEnd: str = '',
                     outPrefix: str = '', outDirSkip: int = 0, inputPathType: str = 'directory', hoursToKeep: list = [], outputTemporalMean: str = '') -> tuple[pd.DataFrame, Any, Any]:
    """
    Gather and filter input file paths for the residual-circulation tool.

    Parameters
    ----------
    inputDir : str
        Input directory or ``.txt`` file listing NetCDF paths.
    fileNames : str
        Glob pattern for file discovery (used with ``rglob``).
    timeInfoInFileNames : str
        Template describing time-token positions in file names.
    outputDir : str
        Output directory; used only to identify already-processed timestamps.
    dateStart : str
        Exclude files before this date (``'YYYY-MM-DD-HH'`` or ``''``).
    dateEnd : str
        Exclude files after this date (``'YYYY-MM-DD-HH'`` or ``''``).
    outPrefix : str
        Prefix of output files, used to detect already-processed timestamps.
    outDirSkip : int
        If truthy, skip timestamps whose output file already exists.
    inputPathType : str
        ``'directory'`` or ``'.txt'``.
    hoursToKeep : list of int
        Restrict to these hours of the day; keep all if empty.
    outputTemporalMean : str
        ``'monthly'``, ``'daily'``, or falsy. When set, the already-done skip
        logic is disabled because output filenames differ from per-file names.

    Returns
    -------
    filesPathsAndTime : pd.DataFrame
        DataFrame with a ``Path`` column and a DatetimeIndex.
    missingTimeStamps : pd.DatetimeIndex or pd.DataFrame
        Timestamps absent from the expected regular grid, or an empty
        DataFrame when fewer than three files are found.
    expectedFrequency : str or pd.DateOffset or ``''``
        Modal time step inferred from the file list.
    """
    filesPathAndTime = _load_and_filter_file_paths(
        inputDir, fileNames, timeInfoInFileNames, dateStart, dateEnd, hoursToKeep, inputPathType)

    if len(filesPathAndTime) >= 3:
        expectedFrequency = pd.infer_freq(filesPathAndTime.index)

        if not expectedFrequency:
            all_diffs = filesPathAndTime.index.to_series().diff().dropna().dt.days
            min_gap = all_diffs.min()

            if (28 <= min_gap <= 31) and (len(filesPathAndTime.index.day.unique()) == 1 or filesPathAndTime.index.is_month_end.all()):
                expectedFrequency = pd.DateOffset(months=1)
            else:
                expectedFrequency = filesPathAndTime.index.to_series().diff().dropna().mode()[0]

        completeRange = pd.date_range(
            start=filesPathAndTime.index.min(),
            end=filesPathAndTime.index.max(),
            freq=expectedFrequency
        )
        missingTimeStamps = completeRange.difference(filesPathAndTime.index)
    else:
        expectedFrequency = ''
        missingTimeStamps = pd.DataFrame()

    if outDirSkip and str(outputTemporalMean).lower() not in ['monthly', 'month', 'daily', 'day']:
        filesPaths_inOutDir = list(Path(outputDir).rglob(f'{outPrefix}*.nc'))
        times_InOuttDir = extractTimeFromFileNames(filesPaths_inOutDir, '*YYYY_MM_DD_HH_mm???')
        filesPathAndTime_InOutDir = pd.DataFrame(data={'Path': filesPaths_inOutDir}, index=times_InOuttDir)
        filesPathAndTime = filesPathAndTime[~filesPathAndTime.index.isin(filesPathAndTime_InOutDir.index)]
        if filesPathAndTime.empty:
            print(f"ERROR: Files with prefix '{outPrefix}' are already present in the output directory for all selected timestamps.")
            sys.exit(1)

    return filesPathAndTime, missingTimeStamps, expectedFrequency


def collectFileNamesTTransport(inputDir: str, fileNames: str, timeInfoInFileNames: str, outputDir: str = '', dateStart: str = '', dateEnd: str = '',
                     outPrefix: str = '', outDirSkip: int = 0, inputPathType: str = 'directory', hoursToKeep: list = []) -> tuple[pd.DataFrame, list, Any]:
    """
    Gather and filter input file paths for the tracer-transport tools.

    Same filtering logic as :func:`collectFileNames` but without
    ``outputTemporalMean`` support, and with a simpler missing-timestamp
    estimate (modal time delta rather than ``pd.infer_freq``).

    Parameters
    ----------
    inputDir : str
        Input directory or ``.txt`` file listing NetCDF paths.
    fileNames : str
        Glob pattern for file discovery.
    timeInfoInFileNames : str
        Template describing time-token positions in file names.
    outputDir : str
        Output directory; used only when *outDirSkip* is set.
    dateStart : str
        Exclude files before this date (``'YYYY-MM-DD-HH'`` or ``''``).
    dateEnd : str
        Exclude files after this date (``'YYYY-MM-DD-HH'`` or ``''``).
    outPrefix : str
        Prefix of output files, used to detect already-processed timestamps.
    outDirSkip : int
        If 1, skip timestamps whose output file already exists.
    inputPathType : str
        ``'directory'`` or ``'.txt'``.
    hoursToKeep : list of int
        Restrict to these hours of the day; keep all if empty.

    Returns
    -------
    filesPathsAndTime : pd.DataFrame
        DataFrame with a ``Path`` column and a DatetimeIndex.
    missingTimeStamps : list of pd.Timestamp
        Timestamps absent from the expected regular grid.
    expectedFrequency : pd.Timedelta
        Modal time step inferred from consecutive file timestamps.
    """
    filesPathAndTime = _load_and_filter_file_paths(
        inputDir, fileNames, timeInfoInFileNames, dateStart, dateEnd, hoursToKeep, inputPathType)

    timeDiffs = filesPathAndTime.index.to_series().diff().dropna()
    expectedFrequency = timeDiffs.mode()[0]
    completeRange = pd.date_range(start=filesPathAndTime.index.min(), end=filesPathAndTime.index.max(), freq=expectedFrequency)
    missingTimeStamps = list(completeRange.difference(filesPathAndTime.index))

    if outDirSkip == 1:
        filesPaths_inOutDir = list(Path(outputDir).rglob(f'{outPrefix}*.nc'))
        times_InOuttDir = extractTimeFromFileNames(filesPaths_inOutDir, '*YYYY_MM_DD_HH_mm???')
        filesPathAndTime_InOutDir = pd.DataFrame(data={'Path': filesPaths_inOutDir}, index=times_InOuttDir)
        filesPathAndTime = filesPathAndTime[~filesPathAndTime.index.isin(filesPathAndTime_InOutDir.index)]
        if filesPathAndTime.empty:
            print(f"ERROR: Files with prefix '{outPrefix}' are already present in the output directory for all selected timestamps.")
            sys.exit(1)

    return filesPathAndTime, missingTimeStamps, expectedFrequency


def chunkMetFilesPathsForBinning(metFilesPaths: pd.DataFrame, tracerFilesPaths: pd.DataFrame, MetDataBinningTime: int | str, tracerExpectedFrequency: Any, metExpectedFrequency: Any) -> dict:
    """
    Pair met-data files with tracer files, computing weights for temporal averaging.

    When met data is at a higher temporal frequency than tracer data, multiple
    met files are averaged (with equal or time-proximity weights) to produce
    one met dataset per tracer timestamp.

    Parameters
    ----------
    metFilesPaths : pd.DataFrame
        DataFrame with ``Path`` column and DatetimeIndex for met files.
    tracerFilesPaths : pd.DataFrame
        DataFrame with ``Path`` column and DatetimeIndex for tracer files.
    MetDataBinningTime : int or ``'auto'``
        ``'auto'`` infers the number of met files to bin from the frequency
        ratio; an integer picks the *N* closest met files to each tracer
        timestamp with equal weight ``1/N``.
    tracerExpectedFrequency : pd.Timedelta or pd.DateOffset
        Modal time step of the tracer file series.
    metExpectedFrequency : pd.Timedelta or pd.DateOffset
        Modal time step of the met file series.

    Returns
    -------
    dict
        Mapping ``{tracer_timestamp: [tracer_path, met_paths_array, weights_array]}``.
    """
    pathDictionary = {}

    if MetDataBinningTime == 'auto':
        if tracerExpectedFrequency == metExpectedFrequency:
            if set(tracerFilesPaths.index) & set(metFilesPaths.index):
                pathAndTime = tracerFilesPaths.join(metFilesPaths.rename(columns={'Path': 'metFilesPath'}))
                pathAndTime.dropna(inplace=True)
                pathAndTime['weight'] = 1
                for timestamp, row in pathAndTime.iterrows():
                    pathDictionary[timestamp] = [row.Path, row.metFilesPath, row.weight]
            else:
                # Vectorize window search: compute all lo/hi indices in one searchsorted call
                # instead of scanning the full met DataFrame per tracer timestamp (O(N log M) vs O(N*M))
                half = tracerExpectedFrequency / 2
                lo_arr = metFilesPaths.index.searchsorted(tracerFilesPaths.index - half, side='left')
                hi_arr = metFilesPaths.index.searchsorted(tracerFilesPaths.index + half, side='right')
                for i, timestamp in enumerate(tracerFilesPaths.index):
                    lo, hi = lo_arr[i], hi_arr[i]
                    if hi - lo == 2:
                        met_slice = metFilesPaths.iloc[lo:hi]
                        weight1stFile = np.abs((met_slice.index[0] - timestamp) / tracerExpectedFrequency)
                        weight2ndFile = np.abs((met_slice.index[1] - timestamp) / tracerExpectedFrequency)
                        pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(met_slice.Path), [weight1stFile, weight2ndFile]]

        else:
            # Vectorize window search (O(N log M) vs O(N*M))
            half = tracerExpectedFrequency / 2
            lo_arr = metFilesPaths.index.searchsorted(tracerFilesPaths.index - half, side='left')
            hi_arr = metFilesPaths.index.searchsorted(tracerFilesPaths.index + half, side='right')
            for i, timestamp in enumerate(tracerFilesPaths.index):
                lo, hi = lo_arr[i], hi_arr[i]
                if lo == hi:
                    continue
                metDataPathOfTimestamp = metFilesPaths.iloc[lo:hi].copy()
                metDataPathOfTimestamp.loc[:, 'hour'] = metDataPathOfTimestamp.index.hour
                hourCounts = metDataPathOfTimestamp['hour'].value_counts()
                uniqueHours = len(hourCounts)
                weightsPerRow = (1.0 / uniqueHours) / hourCounts
                metDataPathOfTimestamp['weight'] = metDataPathOfTimestamp['hour'].map(weightsPerRow)
                pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(metDataPathOfTimestamp.Path),
                                             np.array(metDataPathOfTimestamp.weight)]

    elif isinstance(MetDataBinningTime, int):
        # Use searchsorted to find the insertion point, then only argsort a small
        # local window of 2*N candidates rather than the full met index (O(N log M) vs O(N*M))
        insert_arr = metFilesPaths.index.searchsorted(tracerFilesPaths.index)
        for i, timestamp in enumerate(tracerFilesPaths.index):
            center = insert_arr[i]
            lo = max(0, center - MetDataBinningTime)
            hi = min(len(metFilesPaths), center + MetDataBinningTime)
            candidates = metFilesPaths.iloc[lo:hi]
            timeDiffs = abs(candidates.index - timestamp)
            closestRows = candidates.iloc[np.argsort(timeDiffs)[:MetDataBinningTime]]
            pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(closestRows.Path),
                                         np.zeros(MetDataBinningTime) + 1/MetDataBinningTime]

    else:
        print(f"ERROR: MetDataBinningTime in is set to '{MetDataBinningTime}'.\n\n"
              "it can only be set to 'auto' or be integer, please check the parameter in the configuration file")
        sys.exit(1)

    return pathDictionary

def _replace_fill_values(dataset: xr.Dataset, fillValues: list) -> xr.Dataset:
    """Replace user-specified fill values with NaN in all data variables."""
    if not fillValues:
        return dataset
    for var in dataset.data_vars:
        mask = True
        for fv in fillValues:
            mask = mask & (dataset[var] != fv)
        dataset[var] = dataset[var].where(mask)
    return dataset


def readAndTransposeData(
    filePath: str, reqVars: list[str], vertDimName: str, latDimName: str, lonDimName: str,
    saveInterpolatedZonalMeanVars: list[str] = [], saveZonalMeanVars: list[str] = [],
    timeDimName: str = '', fillValues: list = [],
) -> xr.Dataset:
    """
    Read a NetCDF file and standardize dimension order to (vertical, lat, lon).

    Squeezes any length-1 dimensions (e.g. a single time step). If *timeDimName*
    is provided and that dimension remains after squeezing, it is placed first.

    Parameters
    ----------
    filePath : str or Path
        Path to the NetCDF file.
    reqVars : list of str
        Variables to load.
    vertDimName : str
        Name of the vertical dimension in the file.
    latDimName : str
        Name of the latitude dimension in the file.
    lonDimName : str
        Name of the longitude dimension in the file.
    saveInterpolatedZonalMeanVars : list of str
        Additional variables to load for zonal-mean output after interpolation.
    saveZonalMeanVars : list of str
        Additional variables to load for direct zonal-mean output.
    timeDimName : str
        Name of the time dimension in the file. Pass ``''`` (default) if files
        have no time dimension or it should be squeezed away.
    fillValues : list
        Fill/missing values to replace with NaN on input. Default is ``[]`` (disabled).

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ordered as (vertical, lat, lon), or
        (timeDimName, vertical, lat, lon) when a time dimension is present.
    """
    with xr.open_dataset(filePath) as ds:
        dataset = ds[reqVars + saveInterpolatedZonalMeanVars + saveZonalMeanVars].squeeze().load()
    dataset = _replace_fill_values(dataset, fillValues)
    if timeDimName and timeDimName in dataset.dims:
        dataset = dataset.transpose(timeDimName, vertDimName, latDimName, lonDimName)
    else:
        dataset = dataset.transpose(vertDimName, latDimName, lonDimName)
    return dataset


def compute_temporal_mean(
    paths: list[str],
) -> xr.Dataset:
    """
    Compute the temporal mean of a set of NetCDF files by accumulating their sum.

    Parameters
    ----------
    paths : list of str
        Paths to the NetCDF files to average.
    Returns
    -------
    xr.Dataset
        Dataset containing the mean of all files.
    """
    total: xr.Dataset | None = None
    count = 0
    with xr.set_options(keep_attrs=True):
        for path in paths:
            ds = xr.open_dataset(path)
            total = ds if total is None else total + ds
            count += 1
        if total is None or count == 0:
            raise ValueError("No files were provided.")
        
        temporalMeanDS = total / count
    return temporalMeanDS


def readDataAndGetWeightedAverage(filesPaths: np.ndarray, weights: np.ndarray, reqVars: list[str], vertDimName: str, latDimName: str, lonDimName: str, fillValues: list = []) -> xr.Dataset:
    """
    Read multiple NetCDF files and compute a weighted average across them.

    Used to temporally bin met-data files to match a lower-frequency tracer
    dataset. Weights must sum to 1 for a proper average.

    Parameters
    ----------
    filesPaths : np.ndarray of str/Path
        Paths to the NetCDF files to average.
    weights : np.ndarray of float
        Per-file weights; must have the same length as *filesPaths*.
    reqVars : list of str
        Variables to load and average.
    vertDimName : str
        Name of the vertical dimension, used to set final transpose order.
    latDimName : str
        Name of the latitude dimension.
    lonDimName : str
        Name of the longitude dimension.
    fillValues : list
        Fill/missing values to replace with NaN on input. Default is ``[]`` (disabled).

    Returns
    -------
    xr.Dataset
        Weighted-mean dataset with dimensions ordered as (vertical, lat, lon).
    """
    for index, path in enumerate(filesPaths):
        with xr.open_dataset(path) as ds:
            dataset = ds[reqVars].squeeze().load()
        dataset = _replace_fill_values(dataset, fillValues)
        if index == 0:
            weightedMeanDataset = dataset * weights[index]
            attrs = {v: dataset[v].attrs for v in reqVars}
        else:
            weightedMeanDataset = weightedMeanDataset + (dataset * weights[index])
    for variable in reqVars:
        weightedMeanDataset[variable].attrs = attrs[variable]
        weightedMeanDataset[variable] = weightedMeanDataset[variable].transpose(vertDimName, latDimName, lonDimName)
    return weightedMeanDataset


_THETA_VERT_COORD = {'name': 'theta', 'long_name': 'Potential temperature levels', 'units': 'K'}
_ALT_VERT_COORD   = {'name': 'alt',   'long_name': 'Log-pressure altitude',        'units': 'm'}


def saveOut(dataToSave: dict, tomlConfig: dict, timeStamp: Any, lats: np.ndarray,
            vertLevels: Any, vertCoord: dict = _THETA_VERT_COORD) -> None:
    """
    Package tracer-transport results into an xarray Dataset and write to NetCDF.

    Parameters
    ----------
    dataToSave : dict
        Mapping ``{var_name: [data_array, long_name, units]}``. An optional
        ``'Fourier'`` key holds a nested dict of the same structure for wave
        decomposition fields.
    tomlConfig : dict
        Configuration dict; uses ``outputDirectory``, ``outPrefix``, and
        ``Waves`` keys.
    timeStamp : pd.Timestamp
        Timestamp for the output file name and the ``time`` coordinate.
    lats : np.ndarray
        Latitude coordinate values in degrees.
    vertLevels : array-like
        Vertical coordinate values.
    vertCoord : dict, optional
        Metadata for the vertical coordinate: ``{'name', 'long_name', 'units'}``.
        Defaults to potential-temperature levels (theta, K).
    """
    fnout = tomlConfig['outputDirectory'] + '/' + tomlConfig['outPrefix'] + str(timeStamp)[:-3].replace('-', '_').replace(' ', '_').replace(':', '_') + '.nc'

    if 'Fourier' in dataToSave.keys():
        Fourier = dataToSave['Fourier']
        FourierToSave = True
        del dataToSave['Fourier']
    else:
        FourierToSave = False

    vertName = vertCoord['name']
    dsOut = xr.Dataset()

    for variable in dataToSave.keys():
        dsOut[variable] = ((vertName, 'lat'), np.single(dataToSave[variable][0]))
        getattr(dsOut, variable).attrs['long_name'] = dataToSave[variable][1]
        getattr(dsOut, variable).attrs['units'] = dataToSave[variable][2]

    if FourierToSave:
        for variable in Fourier.keys():
            dsOut[variable] = ((vertName, 'lat', 'waveN'),
                               np.single(Fourier[variable][0]))
            getattr(dsOut, variable).attrs['long_name'] = Fourier[variable][1]
            getattr(dsOut, variable).attrs['units'] = Fourier[variable][2]

        if tomlConfig['Waves'] == ['all'] or tomlConfig['Waves'] == ['All']:
            waveNumbers = list(range(1, Fourier[list(Fourier.keys())[0]][0].shape[2] + 1))
        else:
            waveNumbers = tomlConfig['Waves']

        dsOut.coords['waveN'] = waveNumbers
        dsOut.waveN.attrs['long_name'] = 'wave number'

    dsOut.coords[vertName] = vertLevels
    getattr(dsOut, vertName).attrs['long_name'] = vertCoord['long_name']
    getattr(dsOut, vertName).attrs['units'] = vertCoord['units']
    dsOut.coords['lat'] = lats
    dsOut.lat.attrs['long_name'] = 'latitude'
    dsOut.lat.attrs['units'] = 'degree_N'
    dsOut.coords['time'] = [timeStamp]

    dsOut.to_netcdf(fnout)
